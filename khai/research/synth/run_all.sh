#!/usr/bin/env bash
# Four parallel generators, each with its own random app groups.
cd /c/Project/zalo_research || exit 1
PY=C:/Users/Khai/venvs/zaic/Scripts/python.exe
export PYTHONIOENCODING=utf-8
for s in 1 2 3 4; do
  "$PY" -u synth/gen.py scenarios --apps synth/apps.json --out "synth/scen_$s.jsonl" --per-batch 14 --batches 12 --seed "$s" > "synth/gen_$s.log" 2>&1 &
done
wait
