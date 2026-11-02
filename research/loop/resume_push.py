"""Resume a run whose code dataset was uploaded but whose kernel was never pushed (e.g. the launcher
stopped while the new dataset was still processing): wait for the files, push the kernel, collect.

  python loop/resume_push.py --account acc3 --label r015-4b-histasr --base-llm Qwen/Qwen3-4B-Instruct-2507
Builds the inference-only parameters of loop/infer_run.py.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path("C:/Project/zalo_research")))

from harness import kaggle_io
from harness.budget import Budget
from harness.evalrun import ACCOUNTS, RESEARCH, collect, write_kernel
from harness.loop_config import eval_params
from harness.registry import Registry
from harness.split import load_split

SPLIT = RESEARCH / "splits" / "lifemate_split.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--base-llm", default=None)
    args = ap.parse_args()

    split = load_split(SPLIT)
    params = eval_params(split, args.label)  # same parameters loop/infer_run.py builds
    params.pop("sft")
    params.update({"test": True, "asr_history": True, "asr_adapter": "asr_lora", "llm_adapter": "llm_lora"})
    if args.base_llm:
        params["base_llm"] = args.base_llm
        params["transformers"] = "4.57.6"

    workdir = RESEARCH / "loop" / "runs_work" / args.label
    snap = workdir / "snapshot"
    expected = {f"{d}/{p.name}": p.stat().st_size for d in ("src", "train") for p in (snap / d).glob("*.py")}
    for w in (snap / "adapters").glob("*/adapter_model.safetensors"):
        expected[w.relative_to(snap).as_posix()] = w.stat().st_size
    kaggle_io.wait_dataset_files(ACCOUNTS[args.account]["code_dataset"], expected, account=args.account,
                                 timeout_s=3600)
    kdir = write_kernel(workdir / "kernel", params, args.account)
    kaggle_io.push_kernel(kdir, args.account)
    print("pushed", ACCOUNTS[args.account]["kernel"], flush=True)
    reg = Registry(RESEARCH)
    budget = Budget(RESEARCH / f"budget_{args.account}.json")
    run_id, metrics = collect(args.label, params, SPLIT, reg, budget, since=time.time(), timeout_s=12 * 3600,
                              account=args.account)
    reg.rebuild_index()
    print("INFER RUN", run_id, metrics, flush=True)


if __name__ == "__main__":
    main()
