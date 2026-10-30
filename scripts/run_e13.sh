#!/usr/bin/env bash
# E13 end to end: thoughts (teacher) -> SFT data -> train -> holdout eval -> public_test submission.
# Each stage is skipped when its output exists, so a re-run resumes where it stopped.
#   bash scripts/run_e13.sh            # holdout model (measure on holdout, also writes a public_test submission)
#   FINAL=1 bash scripts/run_e13.sh    # all train data (no holdout, no eval), writes the final submission
# Run inside tmux; the whole thing takes roughly 3 h (thoughts) + 5 h (train) + 1.5 h (inference).
set -euo pipefail
cd "$(dirname "$0")/.."

TEACHER=${TEACHER:-Qwen/Qwen3-14B}
BASE=${BASE:-Qwen/Qwen3-4B-Instruct-2507}
ASR_EVAL=${ASR_EVAL:-models/asr_ft}        # not trained on the holdout, so holdout numbers stay clean
ASR_SUB=${ASR_SUB:-models/asr_final}       # trained on all train audio, for the submission
THOUGHTS=train/thoughts_full.jsonl

if [ "${FINAL:-0}" = "1" ]; then
  TAG=thought-final; HOLD=""
else
  TAG=thought; HOLD="--holdout cache/holdout_all.txt"
fi
LLM=models/lifemate-llm-$TAG
SFT=train/sft_$TAG.jsonl
SUB=out/submission_$TAG.jsonl

echo "== 1/5 thoughts ($TEACHER)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train/gen_thoughts.py --root data --model "$TEACHER" --quant 4bit --out "$THOUGHTS"

echo "== 2/5 SFT data"
python train/build_sft.py --root data --thoughts "$THOUGHTS" --out "$SFT"

echo "== 3/5 train -> $LLM"
if [ ! -f "$LLM/config.json" ]; then
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train/sft.py --data "$SFT" --base "$BASE" \
    --out "$LLM" --epochs 2 --max-len 6144 $HOLD
fi

if [ -z "$HOLD" ]; then
  echo "== 4/5 holdout eval skipped (FINAL: holdout is in the training data)"
else
  echo "== 4/5 holdout eval (ASR_MODEL=$ASR_EVAL)"
  for part in unseen seen; do
    P=out/pred_${part}_$TAG.jsonl
    [ -f "$P" ] || ASR_MODEL=$ASR_EVAL LLM_MODEL=$LLM python src/infer.py --root data --split train \
      --ids cache/holdout_$part.txt --out "$P"
    echo "-- $part"; python src/eval.py --pred "$P" --root data --split train
  done
  echo "-- unseen error breakdown"; python src/analyze.py --pred out/pred_unseen_$TAG.jsonl --root data --split train --show 15
fi

echo "== 5/5 public_test submission (ASR_MODEL=$ASR_SUB)"
[ -f "$SUB" ] || ASR_MODEL=$ASR_SUB LLM_MODEL=$LLM python src/infer.py --root data --split public_test --out "$SUB"
echo "lines: $(wc -l < "$SUB") (expected 634)"
echo "done: $SUB"
