#!/usr/bin/env bash
# E15 end to end: SFT data (money hint + per-floor device digest) -> train -> holdout eval -> public_test submission.
# Each stage is skipped when its output exists, so a re-run resumes where it stopped.
#   bash scripts/run_e15.sh            # holdout model (measure on holdout, also writes a public_test submission)
#   FINAL=1 bash scripts/run_e15.sh    # all train data (no holdout, no eval), writes the final submission
# Roughly 5 h train + 1-2 h inference.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE=${BASE:-Qwen/Qwen3-4B-Instruct-2507}
ASR_EVAL=${ASR_EVAL:-models/asr_ft}        # not trained on the holdout, so holdout numbers stay clean
ASR_SUB=${ASR_SUB:-models/asr_final}       # trained on all train audio, for the submission
# the digest must be on for data building, training and inference alike
export RESULT_DIGEST=1
# the 26 s per-sample budget is for the A30; do not cut samples short when measuring on a slower GPU
export TIME_BUDGET_S=${TIME_BUDGET_S:-120}

if [ "${FINAL:-0}" = "1" ]; then
  TAG=digest-final; HOLD=""
else
  TAG=digest; HOLD="--holdout cache/holdout_all.txt"
fi
LLM=models/lifemate-llm-$TAG
SFT=train/sft_$TAG.jsonl
SUB=out/submission_$TAG.jsonl

for f in data/train/train_tools_catalog.json data/train/zaic_tools; do
  [ -e "$f" ] || { echo "missing $f (copy data/ first)"; exit 1; }
done
if [ -z "$HOLD" ]; then ASR_CHECK=$ASR_SUB; else ASR_CHECK=$ASR_EVAL; fi
[ -d "$ASR_CHECK" ] || { echo "missing $ASR_CHECK"; exit 1; }

echo "== 1/4 SFT data"
[ -f "$SFT" ] || python train/build_sft.py --root data --out "$SFT"

echo "== 2/4 train -> $LLM"
if [ ! -f "$LLM/config.json" ]; then
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train/sft.py --data "$SFT" --base "$BASE" \
    --out "$LLM" --epochs 2 --max-len 6144 $HOLD
fi

if [ -z "$HOLD" ]; then
  echo "== 3/4 holdout eval skipped (FINAL: holdout is in the training data)"
else
  echo "== 3/4 holdout eval (ASR_MODEL=$ASR_EVAL)"
  for part in unseen seen; do
    P=out/pred_${part}_$TAG.jsonl
    [ -f "$P" ] || ASR_MODEL=$ASR_EVAL LLM_MODEL=$LLM python src/infer.py --root data --split train \
      --ids cache/holdout_$part.txt --out "$P"
    echo "-- $part"; python src/eval.py --pred "$P" --root data --split train
  done
  echo "-- unseen error breakdown"; python src/analyze.py --pred out/pred_unseen_$TAG.jsonl --root data --split train --show 15
fi

echo "== 4/4 public_test submission (ASR_MODEL=$ASR_SUB)"
[ -f "$SUB" ] || ASR_MODEL=$ASR_SUB LLM_MODEL=$LLM python src/infer.py --root data --split public_test --out "$SUB"
echo "lines: $(wc -l < "$SUB") (expected 634)"
echo "done: $SUB"
