"""One SFT + dev evaluation on a chosen Kaggle account.

  python loop/sft_run.py --account acc2 --label r001-sft
  python loop/sft_run.py --account acc1 --label r002-sft-large --n-samples 1000
  python loop/sft_run.py --account acc2 --label r012-v3 --v3

Leakage guard: the SFT pool excludes every id in the split file (dev and holdout). --v3 trains on holdout
too (dev stays excluded and is what gets scored), 2 epochs on 2 GPUs, with the fine-tuned PhoWhisper (a002).
--v4 is v3 plus token-weighted loss, history ASR on dev and test, and a raised time budget (T4 cut 34/634 test
samples at 26 s). The prompt comes from the workspace commit (v1 since the v4 commit).
--e16 reruns Nguyen's E16 with the code of workspace commit "Sync E16": 1 epoch on gold + 2 paraphrases per sample,
max-len 6144, his holdout excluded (so 98 of our 127 dev ids are in training and the dev score is not clean), plus
history ASR. T4 forces fp16 and 2 GPUs; a002 replaces his asr_final. --final also trains on his holdout, so the dev
score means nothing there; only the public test score counts.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("C:/Project/zalo_research")))

from harness.budget import Budget
from harness.evalrun import RESEARCH, collect, evaluate
from harness.registry import Registry
from harness.split import load_split
from harness.loop_config import SFT_CONFIG, eval_params

SPLIT = RESEARCH / "splits" / "lifemate_split.json"
ASR_ADAPTER = RESEARCH / "loop" / "runs_work" / "a002-asr-ft" / "output" / "out" / "asr_lora"
# v3: every train sample except dev (1418), all trajectories fit in max_len (longest is 7250 tokens),
# gold transcripts only (mixing ASR text did not help), 2 epochs on 2 T4s in data parallel.
V3_SFT = {"n_samples": 100000, "epochs": 2, "lr": 2e-4, "max_len": 7500, "seed": 42,
          "use_asr_text": False, "gpus": 2}
V4_SFT = dict(V3_SFT, token_loss=True)
E16_SFT = dict(V4_SFT, epochs=1, max_len=6144, para="para")
E16_HOLDOUT = RESEARCH / "splits" / "nguyen_holdout_all.txt"
PARA_DIR = RESEARCH / "loop" / "runs_work" / "p002-para" / "output" / "out"
WORKSPACE = RESEARCH / "workspace"


def budget_path(account):
    return RESEARCH / ("budget.json" if account == "acc1" else f"budget_{account}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", required=True, choices=["acc1", "acc2", "acc3", "acc4", "acc5"])
    ap.add_argument("--label", required=True)
    ap.add_argument("--n-samples", type=int, default=SFT_CONFIG["n_samples"])
    ap.add_argument("--est-s", type=int, default=5400)
    ap.add_argument("--collect", action="store_true", help="resume: wait for an already-pushed kernel")
    ap.add_argument("--v3", action="store_true", help="v3 recipe (see module docstring)")
    ap.add_argument("--v4", action="store_true", help="v4 recipe (see module docstring)")
    ap.add_argument("--e16", action="store_true", help="E16 rerun (see module docstring)")
    ap.add_argument("--final", action="store_true", help="with --e16: train on all of train, Nguyen's holdout included")
    ap.add_argument("--aug-frac", type=float, default=0.0, help="with --e16: rename tools in this fraction of paraphrases")
    ap.add_argument("--seed", type=int, default=None, help="with --e16: training seed")
    ap.add_argument("--synth", action="store_true", help="with --e16: add the synthetic scenarios of synth/out")
    ap.add_argument("--synth-dir", default="out", help="with --synth: folder under synth/ holding items.jsonl and catalog.json")
    ap.add_argument("--split", default=None, help="split file to score instead of the default dev")
    ap.add_argument("--base-llm", default=None, help="HF id of the base LLM (default Qwen2.5-3B-Instruct)")
    ap.add_argument("--transformers", default="4.57.6", help="transformers version when --base-llm is set")
    ap.add_argument("--plain-template", action="store_true", help="replace the model's chat template, see run_eval.py")
    ap.add_argument("--smoke", action="store_true", help="with --e16: 60 samples plus 5 long ones, no public test")
    ap.add_argument("--epochs", type=float, default=None, help="with --e16: epochs instead of 1")
    ap.add_argument("--no-test", action="store_true", help="skip the public test")
    args = ap.parse_args()

    sft = E16_SFT if args.e16 else V4_SFT if args.v4 else V3_SFT if args.v3 else dict(SFT_CONFIG, n_samples=args.n_samples)
    if args.aug_frac:
        sft = dict(sft, aug_frac=args.aug_frac)
    if args.seed is not None:
        sft = dict(sft, seed=args.seed)
    if args.synth:
        sft = dict(sft, synth="synth")
    if args.smoke:
        sft = dict(sft, n_samples=60, longest=5)
    if args.epochs:
        sft = dict(sft, epochs=args.epochs)
    split_path = Path(args.split) if args.split else SPLIT
    split = load_split(split_path)
    params = eval_params(split, args.label)
    params["sft"] = sft
    params["test"] = not (args.smoke or args.no_test)  # also run the full public test and keep submission.jsonl
    params["plain_template"] = args.plain_template
    adapters = None
    if args.v3 or args.v4 or args.e16:
        params["exclude_ids"] = sorted(split["dev"])
        params["asr_adapter"] = "asr_lora"
        params["test_deadline_s"] = int(8.5 * 3600)  # past this, keep the adapter and run test in another kernel
        adapters = {"asr_lora": ASR_ADAPTER}
    if args.v4:
        params["asr_history"] = True
        params["time_budget_s"] = 60
    if args.e16:
        params["exclude_ids"] = [] if args.final else sorted(E16_HOLDOUT.read_text().split())
        params["asr_history"] = True
        params["env"] = {"MONEY_HINT": "0", "RESULT_DIGEST": "0", "TIME_BUDGET_S": "120"}
        params["test_deadline_s"] = int(9.5 * 3600)
        adapters["para"] = PARA_DIR
        if args.synth:
            adapters["synth"] = RESEARCH / "synth" / args.synth_dir
    if args.base_llm:
        params["base_llm"] = args.base_llm
        params["transformers"] = args.transformers  # Qwen3 needs >= 4.51; SFT memory checked locally on 4.57.6
    cfg = {
        "hypothesis": f"speed and memory check of {args.base_llm} against r017" if args.smoke else
                      f"r021 recipe with {args.epochs} epochs: does the model underfit" if args.epochs else
                      "v3: Nguyen prompt v2 + agent fixes, SFT 2 epochs on all train except dev (2-GPU DDP), "
                      "fine-tuned PhoWhisper, transcribe-first test" if args.v3 else
                      "v4: prompt v1 (the 50-point commit), token-weighted loss, history ASR, time budget 60 s"
                      if args.v4 else
                      f"r021 recipe with synth/{args.synth_dir} (round-2 scenarios added): does more targeted synthetic "
                      "data help on unseen apps" if args.synth and args.synth_dir != "out" else
                      "r017 recipe plus synthetic apps from gpt-6.1-sol: does it help on unseen apps"
                      if args.synth else
                      f"r017 recipe with tools renamed in {args.aug_frac} of paraphrases: does it help on unseen apps"
                      if args.aug_frac else
                      f"r018 recipe with seed {args.seed}: how much of a score gap is noise"
                      if args.final and args.seed is not None else
                      "r017 recipe trained on all of train: does adding Nguyen's holdout raise the public score"
                      if args.final else
                      "E16 rerun on Kaggle: paraphrases, 1 epoch, max-len 6144, Nguyen's holdout, plus history ASR"
                      if args.e16 else
                      f"LoRA SFT on {args.n_samples} train samples outside dev and holdout, then dev eval "
                      f"in fp16 one GPU ({args.account}).",
        "parent": "r015-4b-histasr" if args.v4 else "r011c-sft-asr" if args.v3 else "r000b-fp16", "llm_dtype": "fp16", "llm_device": "1gpu",
        "n_samples": len(split["dev"]), "split": "dev", "sft": sft, "base_llm": args.base_llm,
    }
    reg = Registry(RESEARCH)
    budget = Budget(budget_path(args.account))
    if args.collect:
        info = json.loads((reg.runs_dir / args.label / "run.json").read_text(encoding="utf-8"))
        run_id, metrics = collect(args.label, params, split_path, reg, budget, since=info["created"], timeout_s=12 * 3600,
                                  account=args.account)
    else:
        run_id, metrics = evaluate(args.label, params, split_path, WORKSPACE, reg, budget,
                                   est_s=args.est_s, cfg=cfg, account=args.account, timeout_s=12 * 3600,
                                   adapters=adapters)
    reg.rebuild_index()
    print("SFT RUN", run_id, metrics, flush=True)


if __name__ == "__main__":
    main()
