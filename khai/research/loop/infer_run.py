"""Inference only (no SFT) with LoRA adapters trained earlier: dev eval + full public test on Kaggle.

  python loop/infer_run.py --account acc1 --label r014-v3-histasr --llm-run r012c-v3

History user turns are transcribed with ASR (test history has audio only). On dev the gold history text
is replaced by ASR text too, so dev mimics the test condition.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("C:/Project/zalo_research")))

from harness.budget import Budget
from harness.evalrun import RESEARCH, collect, evaluate
from harness.loop_config import eval_params
from harness.registry import Registry
from harness.split import load_split

SPLIT = RESEARCH / "splits" / "lifemate_split.json"
WORKSPACE = RESEARCH / "workspace"
RUNS = RESEARCH / "loop" / "runs_work"
ASR_ADAPTER = RUNS / "a002-asr-ft" / "output" / "out" / "asr_lora"


def budget_path(account):
    return RESEARCH / ("budget.json" if account == "acc1" else f"budget_{account}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", required=True, choices=["acc1", "acc2", "acc3", "acc4", "acc5"])
    ap.add_argument("--label", required=True)
    ap.add_argument("--llm-run", required=True, help="run whose output/out/lora adapter to load")
    ap.add_argument("--base-llm", default=None)
    ap.add_argument("--est-s", type=int, default=4 * 3600)
    ap.add_argument("--split", default=None, help="split file to score instead of the default dev")
    ap.add_argument("--e16", action="store_true", help="E16 agent settings, as used by sft_run.py --e16")
    ap.add_argument("--no-test", action="store_true", help="skip the public test")
    ap.add_argument("--gold-text", action="store_true", help="feed gold transcripts instead of ASR, to measure the LLM alone")
    ap.add_argument("--asr-run", default=None, help="run whose asr_lora adapter to use instead of a002")
    ap.add_argument("--transformers", default="4.57.6", help="transformers version when --base-llm is set")
    ap.add_argument("--plain-template", action="store_true", help="replace the model's chat template, see run_eval.py")
    ap.add_argument("--collect", action="store_true")
    args = ap.parse_args()

    split_path = Path(args.split) if args.split else SPLIT
    split = load_split(split_path)
    params = eval_params(split, args.label)
    params.pop("sft")
    params.update({"test": not args.no_test, "asr_history": not args.gold_text, "gold_text": args.gold_text,
                   "asr_adapter": "asr_lora", "llm_adapter": "llm_lora"})
    if args.e16:
        params["env"] = {"MONEY_HINT": "0", "RESULT_DIGEST": "0", "TIME_BUDGET_S": "120"}
    if args.base_llm:
        params["base_llm"] = args.base_llm
        params["transformers"] = args.transformers
    params["plain_template"] = args.plain_template
    asr = RUNS / args.asr_run / "output" / "out" / "asr_lora" if args.asr_run else ASR_ADAPTER
    adapters = {"asr_lora": asr, "llm_lora": RUNS / args.llm_run / "output" / "out" / "lora"}
    cfg = {"hypothesis": f"inference only with {args.llm_run} adapters; history user turns transcribed by ASR",
           "parent": args.llm_run, "llm_dtype": "fp16", "llm_device": "1gpu", "base_llm": args.base_llm,
           "n_samples": len(split["dev"]), "split": "dev", "asr_history": True}
    reg = Registry(RESEARCH)
    budget = Budget(budget_path(args.account))
    if args.collect:
        info = json.loads((reg.runs_dir / args.label / "run.json").read_text(encoding="utf-8"))
        run_id, metrics = collect(args.label, params, split_path, reg, budget, since=info["created"],
                                  timeout_s=12 * 3600, account=args.account)
    else:
        run_id, metrics = evaluate(args.label, params, split_path, WORKSPACE, reg, budget, est_s=args.est_s, cfg=cfg,
                                   account=args.account, timeout_s=12 * 3600, adapters=adapters)
    reg.rebuild_index()
    print("INFER RUN", run_id, metrics, flush=True)


if __name__ == "__main__":
    main()
