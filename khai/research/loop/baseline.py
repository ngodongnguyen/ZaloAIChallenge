"""Round 0: evaluate the unmodified repo (workspace HEAD) on the locked dev split. Run once."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("C:/Project/zalo_research")))

from harness.budget import Budget
from harness.evalrun import RESEARCH, collect, evaluate
from harness.registry import Registry
from harness.split import load_split

SPLIT = RESEARCH / "splits" / "lifemate_split.json"
WORKSPACE = RESEARCH / "workspace"


def main():
    split = load_split(SPLIT)
    dev = split["dev"]
    params = {"ids": dev, "llm_dtype": "bf16", "llm_device": "2gpu", "out_name": "r000"}
    cfg = {
        "hypothesis": "Baseline: unmodified repo (bf16, both GPUs) on the locked dev split.",
        "parent": None, "llm_dtype": "bf16", "llm_device": "2gpu",
        "n_samples": len(dev), "split": "dev",
    }
    reg = Registry(RESEARCH)
    budget = Budget(RESEARCH / "budget.json")
    if "--collect" in sys.argv:
        # The kernel was already pushed by an earlier attempt; wait for it and record the result.
        info = json.loads((reg.runs_dir / "r000-baseline" / "run.json").read_text(encoding="utf-8"))
        run_id, metrics = collect("r000-baseline", params, SPLIT, reg, budget, since=info["created"])
    else:
        run_id, metrics = evaluate("r000-baseline", params, SPLIT, WORKSPACE, reg, budget,
                                   est_s=3600, cfg=cfg)
    reg.rebuild_index()
    print("BASELINE", run_id, metrics, flush=True)


if __name__ == "__main__":
    main()
