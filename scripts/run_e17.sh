#!/usr/bin/env bash
# E17 end to end: real ASR transcripts of the train audio as extra training text -> SFT data -> train -> holdout eval -> submission.
# Until now the LLM only saw clean (gold) text in training but gets ASR output at test time. The base PhoWhisper is used for
# the transcripts: asr_ft has memorised the train audio, so its errors would not look like those on new speakers.
# Each stage is skipped when its output exists, so a re-run resumes where it stopped.
#   bash scripts/run_e17.sh                 # holdout model (also writes a public_test submission)
#   FINAL=1 bash scripts/run_e17.sh         # all train data (no holdout, no eval), writes the final submission
#   USE_PARA=1 bash scripts/run_e17.sh      # also add the E16 rewrites (needs train/paraphrases.jsonl)
# Roughly 0.5 h transcribing + 5 h train (1 epoch over ~2x data) + 1-2 h inference.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE=${BASE:-Qwen/Qwen3-4B-Instruct-2507}
ASR_TRAIN=${ASR_TRAIN:-models/asr}         # base PhoWhisper: realistic noise for unseen speakers
ASR_EVAL=${ASR_EVAL:-models/asr_ft}        # not trained on the holdout, so holdout numbers stay clean
ASR_SUB=${ASR_SUB:-models/asr_final}       # trained on all train audio, for the submission
EPOCHS=${EPOCHS:-1}
CACHE=cache/asr_base_train.jsonl
# same prompt as E8 (no E14 hint, no E15 digest)
export MONEY_HINT=0
export RESULT_DIGEST=0
export TIME_BUDGET_S=${TIME_BUDGET_S:-120}

PARA_ARG=""; NAME=asr
if [ "${USE_PARA:-0}" = "1" ]; then
  [ -f train/paraphrases.jsonl ] || { echo "missing train/paraphrases.jsonl (run scripts/run_e16.sh first)"; exit 1; }
  PARA_ARG="--para train/paraphrases.jsonl"; NAME=asr-para
fi
if [ "${FINAL:-0}" = "1" ]; then
  TAG=$NAME-final; HOLD=""; ASR_CHECK=$ASR_SUB
else
  TAG=$NAME; HOLD="--holdout cache/holdout_all.txt"; ASR_CHECK=$ASR_EVAL
fi
LLM=models/lifemate-llm-$TAG
SFT=train/sft_$TAG.jsonl
SUB=out/submission_$TAG.jsonl

for f in data/train/train_tools_catalog.json data/train/zaic_tools "$ASR_TRAIN" "$ASR_CHECK"; do
  [ -e "$f" ] || { echo "missing $f"; exit 1; }
done
if [ -n "$HOLD" ]; then
  for f in cache/holdout_all.txt cache/holdout_unseen.txt cache/holdout_seen.txt; do
    [ -e "$f" ] || { echo "missing $f (copy cache/holdout_*.txt from the machine that made them)"; exit 1; }
  done
fi

echo "== 1/5 transcribe train audio ($ASR_TRAIN)"
[ -f "$CACHE" ] || ASR_MODEL=$ASR_TRAIN python src/transcribe_split.py --root data --split train --out "$CACHE"
echo "transcripts: $(wc -l < "$CACHE") (expected 1545)"

echo "== 2/5 SFT data"
[ -f "$SFT" ] || python train/build_sft.py --root data --asr-cache "$CACHE" $PARA_ARG --out "$SFT"

echo "== 3/5 train -> $LLM"
if [ ! -f "$LLM/config.json" ]; then
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train/sft.py --data "$SFT" --base "$BASE" \
    --out "$LLM" --epochs "$EPOCHS" --max-len 6144 $HOLD
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
