"""LoRA SFT on multi-turn trajectories; loss only on assistant turns. Merges and saves the full model.

One GPU: python train/sft.py ...    Several GPUs (data parallel): torchrun --nproc_per_node 2 train/sft.py ...
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import BASE_LLM, seed_everything  # noqa: E402


def no_gqa_sdpa():
    """transformers >= 4.5x passes enable_gqa=True to sdpa when there is no mask; on a T4 (no flash attention) that
    falls back to the math kernel, which builds the full heads x seq x seq matrix (4 GiB at 8k tokens for Qwen3-4B).
    Repeating k/v instead keeps the memory-efficient kernel. No-op on transformers versions without this switch."""
    try:
        from transformers.integrations import sdpa_attention
    except ImportError:
        return
    if hasattr(sdpa_attention, "use_gqa_in_sdpa"):
        sdpa_attention.use_gqa_in_sdpa = lambda *args, **kwargs: False


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


class MaskedLossLM(torch.nn.Module):
    """Compute logits only at supervised (assistant) positions.

    The stock loss materialises [seq_len, vocab] logits in fp32 (~6 GiB at 8k tokens x 152k vocab), which OOMs a T4.
    The loss lives in forward() rather than in Trainer.compute_loss: under DDP (torchrun, one model copy per GPU)
    gradients are only all-reduced for a forward that went through the DDP wrapper, and Trainer wraps this module.
    forward() takes no loss kwargs, so Trainer divides the per-sample mean loss by grad_accum itself.
    """

    def __init__(self, peft_model):
        super().__init__()
        self.peft_model = peft_model

    def forward(self, input_ids, attention_mask, labels):
        inner = self.peft_model.base_model.model  # Qwen2ForCausalLM under peft
        if bool(attention_mask.all()):
            attention_mask = None  # no padding (batch of 1): let sdpa take its is_causal fast path
        hidden = inner.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        target = labels[:, 1:]
        keep = target != -100
        h = hidden[:, :-1][keep]
        tgt = target[keep]
        # Even supervised-only logits (2k tokens x 152k vocab in fp32, plus their grad) OOM a T4 that holds a
        # full model copy, so build them CHUNK tokens at a time and recompute them in backward.
        total = 0.0
        for i in range(0, tgt.numel(), CHUNK):
            total = total + checkpoint(_chunk_loss, inner.lm_head, h[i:i + CHUNK], tgt[i:i + CHUNK],
                                       use_reentrant=False)
        return {"loss": total / max(1, tgt.numel())}


CHUNK = 1024


class StopOnNanLoss(TrainerCallback):
    """Fail fast if the logged loss goes NaN/inf (fp16 overflow): hours of GPU would otherwise be wasted.
    A NaN grad_norm in the first steps is normal (the fp16 grad scaler skips those steps); a NaN loss is not."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        loss = (logs or {}).get("loss")
        if loss is not None and not math.isfinite(loss):
            raise RuntimeError(f"loss is {loss} at step {state.global_step}; stopping")


def _chunk_loss(lm_head, h, tgt):
    logits = lm_head(h.to(lm_head.weight.dtype)).float()
    return F.cross_entropy(logits, tgt, reduction="sum")


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
    ap.add_argument("--no-merge", action="store_true", help="save only the LoRA adapter (<out>_adapter)")
    args = ap.parse_args()

    seed_everything()
    no_gqa_sdpa()
    # Under torchrun every GPU holds a full model copy and trains on its share of the data (DDP).
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    tok = AutoTokenizer.from_pretrained(args.base)
    dtype = torch.float16 if args.fp16 else torch.bfloat16
    kw = {}
    if args.qlora:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        kw["device_map"] = {"": local_rank}
    # sdpa with no mask runs the memory-efficient causal kernel; a full fp32 attention matrix (eager) OOMs a T4.
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype, attn_implementation="sdpa", **kw)
    # Non-reentrant checkpointing: the reentrant variant breaks DDP gradient sync.
    ckpt = {"use_reentrant": False}
    if args.qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs=ckpt)
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=ckpt)
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    if args.fp16:
        # Trainer's fp16 grad scaler refuses fp16 trainable params, so keep the LoRA weights in fp32.
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()

    holdout = set(open(args.holdout).read().split()) if args.holdout else set()
    ds = SFTData(args.data, tok, args.max_len, holdout)
    if local_rank == 0:
        print("train examples:", len(ds), "gpus:", world, flush=True)

    accum = max(1, 16 // world)  # effective batch stays 16 whatever the GPU count
    steps = -(-int(len(ds) * args.epochs) // 16)  # optimizer steps; warmup_ratio was removed in transformers 5
    trainer = Trainer(
        model=MaskedLossLM(model),
        train_dataset=ds,
        data_collator=lambda b: collate(b, tok.pad_token_id),
        args=TrainingArguments(
            output_dir=args.out + "_ckpt", num_train_epochs=args.epochs, learning_rate=args.lr,
            per_device_train_batch_size=1, gradient_accumulation_steps=accum, warmup_steps=max(1, int(0.05 * steps)),
            lr_scheduler_type="cosine", logging_steps=1, save_strategy="no", bf16=not args.fp16, fp16=args.fp16,
            seed=42, report_to="none", remove_unused_columns=False, ddp_find_unused_parameters=False),
        callbacks=[StopOnNanLoss()],
    )
    trainer.train()
    if local_rank != 0:
        return
    model.save_pretrained(args.out + "_adapter")
    del trainer, model
    torch.cuda.empty_cache()
    if args.no_merge:
        return
    # merge into a clean (non-quantized) base so the final model needs no bitsandbytes
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, args.out + "_adapter").merge_and_unload()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)


if __name__ == "__main__":
    main()
