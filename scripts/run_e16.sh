#!/usr/bin/env bash
# E16 end to end: reworded utterances (local teacher) -> SFT data -> train -> holdout eval -> public_test submission.
# Same labels, more ways of saying them. Each stage is skipped when its output exists, so a re-run resumes where it stopped.
#   bash scripts/run_e16.sh            # holdout model (measure on holdout, also writes a public_test submission)
#   FINAL=1 bash scripts/run_e16.sh    # all train data (no holdout, no eval), writes the final submission
# Roughly 1-2 h rewriting + 6-8 h train (3x the data, 1 epoch) + 1-2 h inference.
set -euo pipefail
cd "$(dirname "$0")/.."

TEACHER=${TEACHER:-Qwen/Qwen3-14B}
BASE=${BASE:-Qwen/Qwen3-4B-Instruct-2507}
ASR_EVAL=${ASR_EVAL:-models/asr_ft}        # not trained on the holdout, so holdout numbers stay clean
ASR_SUB=${ASR_SUB:-models/asr_final}       # trained on all train audio, for the submission
PARA_N=${PARA_N:-2}                        # rewrites per sample
EPOCHS=${EPOCHS:-1}                        # (1 + PARA_N) x the data per epoch
PARAS=train/paraphrases.jsonl
# E16 isolates the wording data: no E14 money hint, no E15 digest (the E8 baseline had neither)
export MONEY_HINT=0
export RESULT_DIGEST=0
# the 26 s per-sample budget is for the A30; do not cut samples short when measuring on a slower GPU
export TIME_BUDGET_S=${TIME_BUDGET_S:-120}

if [ "${FINAL:-0}" = "1" ]; then
  TAG=para-final; HOLD=""; ASR_CHECK=$ASR_SUB
else
  TAG=para; HOLD="--holdout cache/holdout_all.txt"; ASR_CHECK=$ASR_EVAL
fi
LLM=models/lifemate-llm-$TAG
SFT=train/sft_$TAG.jsonl
SUB=out/submission_$TAG.jsonl

for f in data/train/train_tools_catalog.json data/train/zaic_tools "$ASR_CHECK"; do
  [ -e "$f" ] || { echo "missing $f"; exit 1; }
done
if [ -n "$HOLD" ]; then
  for f in cache/holdout_all.txt cache/holdout_unseen.txt cache/holdout_seen.txt; do
    [ -e "$f" ] || { echo "missing $f (copy cache/holdout_*.txt from the machine that made them)"; exit 1; }
  done
fi

echo "== 1/5 rewrites ($TEACHER, n=$PARA_N)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train/paraphrase.py --root data --model "$TEACHER" --quant 4bit \
  --n "$PARA_N" --out "$PARAS"
python - <<EOF
import json
rows = [json.loads(l) for l in open("$PARAS", encoding="utf-8")]
print(f"rewrites: {len(rows)}, kept {sum(r['ok'] for r in rows)}")
EOF

echo "== 2/5 SFT data"
[ -f "$SFT" ] || python train/build_sft.py --root data --para "$PARAS" --out "$SFT"

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
