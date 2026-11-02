"""Evaluate fp16 on one GPU on the same locked dev split, to compare with the bf16 baseline (r000).

Run after r000-baseline is recorded as done. Same code (workspace HEAD), only the LLM dtype and
GPU placement change, so the score difference comes from precision and placement alone.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path("C:/Project/zalo_research")))

from harness.budget import Budget
from harness.evalrun import RESEARCH, evaluate
from harness.registry import Registry
from harness.split import load_split

SPLIT = RESEARCH / "splits" / "lifemate_split.json"
WORKSPACE = RESEARCH / "workspace"


def main():
    dev = load_split(SPLIT)["dev"]
    params = {"ids": dev, "llm_dtype": "fp16", "llm_device": "1gpu", "out_name": "r000b"}
    cfg = {
        "hypothesis": "fp16 on one GPU on the same dev split, to compare with the bf16 baseline.",
        "parent": "r000-baseline", "llm_dtype": "fp16", "llm_device": "1gpu",
        "n_samples": len(dev), "split": "dev",
    }
    reg = Registry(RESEARCH)
    budget = Budget(RESEARCH / "budget.json")
    run_id, metrics = evaluate("r000b-fp16", params, SPLIT, WORKSPACE, reg, budget,
                               est_s=2400, cfg=cfg)
    reg.rebuild_index()
    print("FP16", run_id, metrics, flush=True)


if __name__ == "__main__":
    main()
