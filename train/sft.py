"""LoRA SFT on multi-turn trajectories; loss only on assistant turns. Merges and saves the full model."""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
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
        full = tok.apply_chat_template(messages, tokenize=True, enable_thinking=False)
        full = full["input_ids"] if isinstance(full, dict) or hasattr(full, "keys") else full
        labels = [-100] * len(full)
        for i, m in enumerate(messages):
            if m["role"] != "assistant":
                continue
            pre = tok.apply_chat_template(messages[:i], add_generation_prompt=True, tokenize=True, enable_thinking=False)
            pre = pre["input_ids"] if hasattr(pre, "keys") else pre
            upto = tok.apply_chat_template(messages[: i + 1], tokenize=True, enable_thinking=False)
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


class MaskedLossTrainer(Trainer):
    """Compute logits only at supervised (assistant) positions.

    The stock loss materialises [seq_len, vocab] logits in fp32 (~6 GiB at 8k tokens x 152k vocab), which OOMs a T4.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inner = model.base_model.model if hasattr(model, "base_model") else model  # Qwen2ForCausalLM under peft
        hidden = inner.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).last_hidden_state
        target = inputs["labels"][:, 1:]
        keep = target != -100
        h = hidden[:, :-1][keep]
        logits = inner.lm_head(h.to(inner.lm_head.weight.dtype)).float()
        if num_items_in_batch is not None:
            # Trainer sums micro-batch losses and expects them pre-normalised by the supervised-token count of the
            # whole accumulated batch; a plain mean here would inflate gradients by grad_accum.
            loss = F.cross_entropy(logits, target[keep], reduction="sum") / num_items_in_batch
        else:
            loss = F.cross_entropy(logits, target[keep])  # Trainer divides by grad_accum itself in this case
        return (loss, None) if return_outputs else loss


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
    ap.add_argument("--fp16", action="store_true", help="use fp16 instead of bf16 (T4)")
    ap.add_argument("--qlora", action="store_true", help="4-bit base weights (needs bitsandbytes)")
    ap.add_argument("--holdout", default=None, help="file with sample ids (one per line) kept out for validation")
    args = ap.parse_args()

    seed_everything()
    tok = AutoTokenizer.from_pretrained(args.base)
    dtype = torch.float16 if args.fp16 else torch.bfloat16
    kw = {}
    if args.qlora:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        kw["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype, **kw)
    if args.qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))

    holdout = set(open(args.holdout).read().split()) if args.holdout else set()
    ds = SFTData(args.data, tok, args.max_len, holdout)
    print("train examples:", len(ds))

    steps = -(-int(len(ds) * args.epochs) // 16)  # optimizer steps (grad-accum 16); warmup_ratio was removed in transformers 5
    trainer = MaskedLossTrainer(
        model=model,
        train_dataset=ds,
        data_collator=lambda b: collate(b, tok.pad_token_id),
        args=TrainingArguments(
            output_dir=args.out + "_ckpt", num_train_epochs=args.epochs, learning_rate=args.lr,
            per_device_train_batch_size=1, gradient_accumulation_steps=16, warmup_steps=max(1, int(0.05 * steps)),
            lr_scheduler_type="cosine", logging_steps=10, save_strategy="no", bf16=not args.fp16, fp16=args.fp16,
            seed=42, report_to="none", remove_unused_columns=False),
    )
    trainer.train()
    model.save_pretrained(args.out + "_adapter")
    del trainer, model
    torch.cuda.empty_cache()
    # merge into a clean (non-quantized) base so the final model needs no bitsandbytes
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, args.out + "_adapter").merge_and_unload()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)


if __name__ == "__main__":
    main()
