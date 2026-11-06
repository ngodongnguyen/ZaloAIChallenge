"""Build the two merged models the Docker image ships, the same way the Kaggle eval kernel did (fp16 merge).

  python scripts/build_models.py --asr-adapter models/adapters/asr_lora --llm-adapter models/adapters/llm_lora

  models/asr          <- vinai/PhoWhisper-large + ASR LoRA (a002)
  models/lifemate-llm <- Qwen/Qwen3-4B-Instruct-2507 + LLM LoRA (r018, or a soup written by research/loop/make_soup.py)

Run scripts/download_models.sh first: it puts the base models in models/asr-base and models/base-llm.
vinai/PhoWhisper-large ships only pytorch_model.bin, which transformers 4.57 loads only with torch >= 2.6; on an
older torch, merge the ASR where torch is newer (the Kaggle kernel did) or copy an already merged ASR into
models/asr and pass --only llm.
"""
import argparse
import shutil
from pathlib import Path

import torch
from peft import PeftModel


def merge(kind, base, adapter, out):
    if kind == "asr":
        from transformers import WhisperForConditionalGeneration as M, WhisperProcessor as P
    else:
        from transformers import AutoModelForCausalLM as M, AutoTokenizer as P
    if not (Path(adapter) / "adapter_config.json").exists():
        raise SystemExit(f"no adapter at {adapter}")
    if Path(out).exists():
        shutil.rmtree(out)
    model = M.from_pretrained(base, torch_dtype=torch.float16)
    PeftModel.from_pretrained(model, adapter).merge_and_unload().save_pretrained(out)
    P.from_pretrained(base).save_pretrained(out)
    print(f"merged {adapter} -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr-base", default="models/asr-base")
    ap.add_argument("--llm-base", default="models/base-llm")
    ap.add_argument("--asr-adapter", default="models/adapters/asr_lora")
    ap.add_argument("--llm-adapter", default="models/adapters/llm_lora")
    ap.add_argument("--asr-out", default="models/asr")
    ap.add_argument("--llm-out", default="models/lifemate-llm")
    ap.add_argument("--only", choices=["asr", "llm"], default=None)
    args = ap.parse_args()
    if args.only != "llm":
        merge("asr", args.asr_base, args.asr_adapter, args.asr_out)
    if args.only != "asr":
        merge("llm", args.llm_base, args.llm_adapter, args.llm_out)


if __name__ == "__main__":
    main()
