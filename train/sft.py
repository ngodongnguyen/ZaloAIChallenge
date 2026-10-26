"""LoRA SFT on multi-turn trajectories; loss only on assistant turns. Merges and saves the full model."""
import argparse
import json
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import BASE_LLM, seed_everything  # noqa: E402


class SFTData(Dataset):
    def __init__(self, path, tok, max_len, holdout_ids=()):
        self.rows = []
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r["id"] in holdout_ids:
                continue
            enc = self.encode(r["messages"], tok, max_len)
            if enc:
                self.rows.append(enc)

    @staticmethod
    def encode(messages, tok, max_len):
        full = tok.apply_chat_template(messages, tokenize=True)
        full = full["input_ids"] if isinstance(full, dict) or hasattr(full, "keys") else full
        labels = [-100] * len(full)
        for i, m in enumerate(messages):
            if m["role"] != "assistant":
                continue
            pre = tok.apply_chat_template(messages[:i], add_generation_prompt=True, tokenize=True)
            pre = pre["input_ids"] if hasattr(pre, "keys") else pre
            upto = tok.apply_chat_template(messages[: i + 1], tokenize=True)
            upto = upto["input_ids"] if hasattr(upto, "keys") else upto
            for j in range(len(pre), min(len(upto), len(full))):
                labels[j] = full[j]
        if len(full) > max_len:
            return None  # drop over-long examples rather than truncating the answer
        return {"input_ids": full, "labels": labels}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(batch, pad_id):
    m = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), m), pad_id)
    lab = torch.full((len(batch), m), -100)
    att = torch.zeros((len(batch), m), dtype=torch.long)
    for k, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[k, :n] = torch.tensor(b["input_ids"])
        lab[k, :n] = torch.tensor(b["labels"])
        att[k, :n] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train/sft.jsonl")
    ap.add_argument("--base", default=BASE_LLM)
    ap.add_argument("--out", default="models/lifemate-llm")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=12000)
    ap.add_argument("--holdout", default=None, help="file with sample ids (one per line) kept out for validation")
    args = ap.parse_args()

    seed_everything()
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))

    holdout = set(open(args.holdout).read().split()) if args.holdout else set()
    ds = SFTData(args.data, tok, args.max_len, holdout)
    print("train examples:", len(ds))

    trainer = Trainer(
        model=model,
        train_dataset=ds,
        data_collator=lambda b: collate(b, tok.pad_token_id),
        args=TrainingArguments(
            output_dir=args.out + "_ckpt", num_train_epochs=args.epochs, learning_rate=args.lr,
            per_device_train_batch_size=1, gradient_accumulation_steps=16, warmup_ratio=0.05,
            lr_scheduler_type="cosine", logging_steps=10, save_strategy="no", bf16=True,
            seed=42, report_to="none", remove_unused_columns=False),
    )
    trainer.train()
    model = model.merge_and_unload()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)


if __name__ == "__main__":
    main()
