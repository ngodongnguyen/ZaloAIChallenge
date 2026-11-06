"""Generate the E16 paraphrases on Kaggle (no training, no scoring): rewrites of train utterances by a local teacher.

  python loop/para_run.py --account acc1 --label p001-para
  python loop/para_run.py --account acc1 --label p001-para --collect

The result lands in loop/runs_work/<label>/output/out/paraphrases.jsonl.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path("C:/Project/zalo_research")))

from harness import kaggle_io
from harness.budget import Budget
from harness.evalrun import ACCOUNTS, RESEARCH, snapshot_workspace, write_kernel

WORKSPACE = RESEARCH / "workspace"


def budget_path(account):
    return RESEARCH / ("budget.json" if account == "acc1" else f"budget_{account}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", required=True, choices=["acc1", "acc2", "acc3"])
    ap.add_argument("--label", required=True)
    ap.add_argument("--teacher", default="Qwen/Qwen3-14B")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--timeout-s", type=int, default=int(3.7 * 3600), help="stop generating and keep the partial file")
    ap.add_argument("--seed", default=None, help="folder with a partial paraphrases.jsonl to continue from")
    ap.add_argument("--collect", action="store_true", help="resume: wait for an already-pushed kernel")
    args = ap.parse_args()

    workdir = RESEARCH / "loop" / "runs_work" / args.label
    workdir.mkdir(parents=True, exist_ok=True)
    pushed = workdir / "pushed_at.txt"
    kernel = ACCOUNTS[args.account]["kernel"]
    if not args.collect:
        snap = snapshot_workspace(WORKSPACE, workdir / "snapshot", args.account, {"seed": args.seed} if args.seed else None)
        kaggle_io.new_dataset_version(snap, args.label, account=args.account)
        expected = {f"{d}/{p.name}": p.stat().st_size for d in ("src", "train") for p in (snap / d).glob("*.py")}
        if args.seed:
            seed_file = snap / "adapters" / "seed" / "paraphrases.jsonl"
            expected[seed_file.relative_to(snap).as_posix()] = seed_file.stat().st_size
        kaggle_io.wait_dataset_files(ACCOUNTS[args.account]["code_dataset"], expected, account=args.account)
        params = {"out_name": args.label, "transformers": "4.57.6",
                  "paraphrase": {"teacher": args.teacher, "n": args.n, "timeout_s": args.timeout_s,
                                 "seed": "seed" if args.seed else None}}
        kdir = write_kernel(workdir / "kernel", params, args.account)
        kaggle_io.push_kernel(kdir, args.account)
        pushed.write_text(str(time.time()))
        print("pushed", kernel, flush=True)
    since = float(pushed.read_text())
    status, elapsed = kaggle_io.wait_kernel(kernel, 5 * 3600, since=since, account=args.account)
    Budget(budget_path(args.account)).record(args.label, elapsed, since, time.time())
    print("kernel", status, round(elapsed), flush=True)
    kaggle_io.download_kernel_output(kernel, workdir / "output", account=args.account)
    rows = [json.loads(line) for line in open(workdir / "output" / "out" / "paraphrases.jsonl", encoding="utf-8")]
    print("PARA", len(rows), "kept", sum(r["ok"] for r in rows), flush=True)


if __name__ == "__main__":
    main()
