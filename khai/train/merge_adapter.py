"""Merge a saved LoRA adapter into its base model and save the full model (the last step of sft.py, for when it died on disk space).

python train/merge_adapter.py --base Qwen/Qwen3.5-4B --adapter models/lifemate-llm-para-q35e3_adapter --out models/lifemate-llm-para-q35e3
"""
import argparse
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

if not (Path(args.adapter) / "adapter_config.json").exists():
    raise SystemExit(f"no adapter at {args.adapter}")
if Path(args.out).exists():  # a half-written model (config.json but no weights) would make run_e16.sh skip training and merging
    shutil.rmtree(args.out)

base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
model.save_pretrained(args.out)
AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
print("saved", args.out)
