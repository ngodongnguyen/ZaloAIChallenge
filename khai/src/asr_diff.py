"""Which words does the ASR get wrong? Transcribe the holdout samples, save the text, and count the word-level substitutions
against the gold text (cache/asr_gold.jsonl), with the most frequent (gold -> heard) pairs and examples.

ASR_MODEL=models/asr_ft python src/asr_diff.py --ids cache/holdout_all.txt --out cache/asr_ft_holdout.jsonl
"""
import argparse
import difflib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from asr import ASR  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data")
ap.add_argument("--split", default="train")
ap.add_argument("--ids", default="cache/holdout_all.txt")
ap.add_argument("--gold", default="cache/asr_gold.jsonl")
ap.add_argument("--out", default="cache/asr_ft_holdout.jsonl")
ap.add_argument("--top", type=int, default=30)
args = ap.parse_args()

ids = set(open(args.ids).read().split())
gold = {json.loads(l)["id"]: json.loads(l)["text"] for l in open(args.gold, encoding="utf-8")}
heard = {}
if Path(args.out).exists():
    heard = {json.loads(l)["id"]: json.loads(l)["text"] for l in open(args.out, encoding="utf-8")}

asr = ASR()
root = Path(args.root) / args.split
with open(root / f"{args.split}.jsonl", encoding="utf-8") as fin, open(args.out, "a", encoding="utf-8") as fout:
    for line in fin:
        item = json.loads(line)
        if item["id"] not in ids or item["id"] in heard:
            continue
        heard[item["id"]] = asr.transcribe(root / item["audio"], item["id"])
        fout.write(json.dumps({"id": item["id"], "text": heard[item["id"]]}, ensure_ascii=False) + "\n")
        fout.flush()


def words(s):
    return re.findall(r"\w+", s.lower())


pairs, ex = Counter(), defaultdict(list)
errs = total = digits = 0
for i, g in gold.items():
    if i not in heard:
        continue
    a, b = words(g), words(heard[i])
    total += len(a)
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        errs += max(i2 - i1, j2 - j1)
        key = (" ".join(a[i1:i2]) or "-", " ".join(b[j1:j2]) or "-")
        pairs[key] += 1
        if len(ex[key]) < 1:
            ex[key].append(i)
    digits += sum(1 for w in b if w.isdigit()) != sum(1 for w in a if w.isdigit())
print(f"samples {len(heard)} | word error rate {errs / max(total, 1):.3f} | samples whose digit count differs {digits}")
for (g, h), v in pairs.most_common(args.top):
    print(f"{v:3d}  gold '{g}'  ->  heard '{h}'   ({ex[(g, h)][0]})")
