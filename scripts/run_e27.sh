#!/usr/bin/env bash
# E27 = E22 with a stronger labeller: Qwen3.5-9B (4-bit, thinking, 3 retrieved few-shot train examples) answers the same
# synthetic requests for the 11 public_test-only apps instead of the student. Everything else as E22: base data
# sft_asr-para-cond-q35nc + 2 x kept trajectories, Qwen3.5-4B, 1 epoch, holdout excluded, COND_HINT=1.
# Each stage is skipped when its output exists, so a re-run resumes.
#   RUNS=2 bash scripts/run_e27.sh     # RUNS=1: keep every error-free, non-asking run (no agreement filter)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
RUNS=${RUNS:-2}
TAG=asr-para-cond-teach9b
SYN=train/synth_sft_teach9b.jsonl
DONE=train/synth_done_teach9b.txt
SFT=train/sft_$TAG.jsonl
LLM=models/lifemate-llm-$TAG
SUB=out/submission_$TAG.jsonl
export MONEY_HINT=0 RESULT_DIGEST=0 COND_HINT=1 TIME_BUDGET_S=400

echo "== 1/4 teacher rollouts (runs=$RUNS) -> $SYN"
LLM_4BIT=1 LLM_THINK=1 FEWSHOT=3 $PY train/synth_unseen.py --stage roll --llm Qwen/Qwen3.5-9B \
  --runs "$RUNS" --temp 0.6 --out "$SYN" --done "$DONE"
echo "kept: $(wc -l < "$SYN")"

echo "== 2/4 SFT data -> $SFT"
[ -f "$SFT" ] || cat train/sft_asr-para-cond-q35nc.jsonl "$SYN" "$SYN" > "$SFT"
wc -l "$SFT"

echo "== 3/4 train 1 epoch -> $LLM"
if [ ! -f "$LLM/config.json" ]; then
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY train/sft.py --data "$SFT" --base Qwen/Qwen3.5-4B \
    --out "$LLM" --epochs 1 --max-len 6144 --holdout cache/holdout_all.txt
fi

echo "== 4/4 public_test submission, then holdout unseen"
export TIME_BUDGET_S=120  # as the E20/E22 submissions (run_e17.sh default)
[ -f "$SUB" ] || ASR_MODEL=models/asr_final LLM_MODEL=$LLM $PY src/infer.py --root data --split public_test --out "$SUB"
echo "lines: $(wc -l < "$SUB") (expected 634)"
P=out/pred_unseen_$TAG.jsonl
[ -f "$P" ] || ASR_MODEL=models/asr_ft LLM_MODEL=$LLM $PY src/infer.py --root data --split train --ids cache/holdout_unseen.txt --out "$P"
$PY src/eval.py --pred "$P" --root data --split train
echo "done: $SUB"
