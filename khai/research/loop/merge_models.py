"""Merge the Kaggle-trained LoRA adapters into their base models for local use.

  python loop/merge_models.py
Writes models/asr_ft (PhoWhisper-large + a002 adapter) and models/llm_sft (Qwen2.5-3B-Instruct + r011c adapter).
Both are fp16 and live outside zalo_ai.
"""
import sys
from pathlib import Path

import torch
from peft import PeftModel

RESEARCH = Path("C:/Project/zalo_research")
OUT = RESEARCH / "models"
ASR_ADAPTER = RESEARCH / "loop/runs_work/a002-asr-ft/output/out/asr_lora"
LLM_ADAPTER = RESEARCH / "loop/runs_work/r011c-sft-asr/output/out/lora"


def merge_asr():
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    base = "vinai/PhoWhisper-large"
    model = WhisperForConditionalGeneration.from_pretrained(base, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, str(ASR_ADAPTER)).merge_and_unload()
    model.save_pretrained(OUT / "asr_ft")
    WhisperProcessor.from_pretrained(base).save_pretrained(OUT / "asr_ft")
    print("asr_ft saved", flush=True)


def merge_llm():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = "Qwen/Qwen2.5-3B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, str(LLM_ADAPTER)).merge_and_unload()
    model.save_pretrained(OUT / "llm_sft")
    AutoTokenizer.from_pretrained(base).save_pretrained(OUT / "llm_sft")
    print("llm_sft saved", flush=True)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    which = sys.argv[1:] or ["asr", "llm"]
    if "asr" in which:
        merge_asr()
    if "llm" in which:
        merge_llm()
