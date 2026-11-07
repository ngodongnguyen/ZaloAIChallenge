"""Samples where the gold-transcript run has the right state hash and the ASR run does not: print gold vs ASR text.
   ASR_MODEL=models/asr_ft python scripts/asr_loss.py out/detail_unseen_gold.jsonl out/detail_unseen_para-q35e3.jsonl"""
import json
import sys

sys.path.insert(0, "src")
from asr import ASR

gold_detail, asr_detail = sys.argv[1], sys.argv[2]
load = lambda p: {r["id"]: r for r in map(json.loads, open(p, encoding="utf-8"))}
g, a = load(gold_detail), load(asr_detail)
rows = {}
for line in open("data/train/train.jsonl", encoding="utf-8"):
    r = json.loads(line)
    rows[r["id"]] = r
lost = [i for i in g if g[i]["hash_ok"] and i in a and not a[i]["hash_ok"]]
won = [i for i in g if not g[i]["hash_ok"] and i in a and a[i]["hash_ok"]]
print(f"gold ok / ASR wrong: {len(lost)}   gold wrong / ASR ok: {len(won)}")
asr = ASR()
for i in lost:
    r = rows[i]
    print(i)
    print("  GOLD:", r["audio_text"])
    print("  ASR :", asr.transcribe(f"data/train/{r['audio']}", i))
