"""LoRA fine-tune PhoWhisper on the train audio (outside dev and holdout) on a Kaggle account, report dev WER.

  python loop/asr_ft_run.py --account acc1 --label a001-asr-ft
  python loop/asr_ft_run.py --account acc1 --label a001-asr-ft --collect   # resume waiting

The result json (WER base vs tuned, per-clip transcripts) and the adapter land in loop/runs_work/<label>/output.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path("C:/Project/zalo_research")))

from harness import kaggle_io
from harness.budget import Budget
from harness.evalrun import ACCOUNTS, RESEARCH, write_kernel
from harness.split import load_split

SPLIT = RESEARCH / "splits" / "lifemate_split.json"
TEMPLATE = RESEARCH / "kernel_template" / "run_asr_ft.py"
ASR_FT = {"r": 32, "lr": 2e-4, "epochs": 3, "batch": 4, "accum": 4, "seed": 42}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", required=True, choices=["acc1", "acc2", "acc3", "acc4", "acc5"])
    ap.add_argument("--label", required=True)
    ap.add_argument("--epochs", type=int, default=ASR_FT["epochs"])
    ap.add_argument("--history", action="store_true", help="also train on the audio of user history turns")
    ap.add_argument("--collect", action="store_true")
    args = ap.parse_args()

    split = load_split(SPLIT)
    params = {"ids": split["dev"], "exclude_ids": sorted(set(split["dev"]) | set(split["holdout"])),
              "asr_ft": dict(ASR_FT, epochs=args.epochs), "out_name": args.label, "history_clips": args.history}
    workdir = RESEARCH / "loop" / "runs_work" / args.label
    kernel = ACCOUNTS[args.account]["kernel"]
    budget = Budget(RESEARCH / ("budget.json" if args.account == "acc1" else f"budget_{args.account}.json"))
    marker = workdir / "pushed_at.txt"
    if args.collect:
        since = float(marker.read_text())
    else:
        kdir = write_kernel(workdir / "kernel", params, args.account, template=TEMPLATE)
        kaggle_io.push_kernel(kdir, args.account)
        since = time.time()
        marker.write_text(str(since))
    status, elapsed = kaggle_io.wait_kernel(kernel, 6 * 3600, since=since, account=args.account)
    budget.record(args.label, elapsed, since, time.time())
    if status != "COMPLETE":
        raise RuntimeError(f"kernel finished with {status}")
    out = workdir / "output"
    kaggle_io.download_kernel_output(kernel, out, account=args.account)
    result = json.loads((out / "out" / f"{args.label}.json").read_text(encoding="utf-8"))
    print("ASR FT", args.label, {k: result[k] for k in ("wer_base", "wer_tuned", "train_clips", "seconds")}, flush=True)


if __name__ == "__main__":
    main()
