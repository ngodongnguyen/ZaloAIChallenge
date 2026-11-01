"""Where do the arguments differ? For samples whose predicted tool sequence equals the gold one, count (tool, key, kind) of the
differences (missing / extra / value) and print the most frequent with examples.

python src/arg_diff.py --pred out/pred_unseen_para-q35b.jsonl --root data --split train
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval import f1, fuzzy_params, norm  # noqa: E402  (the same matching rules as the scorer)

ap = argparse.ArgumentParser()
ap.add_argument("--pred", required=True)
ap.add_argument("--root", default="data")
ap.add_argument("--split", default="train")
ap.add_argument("--top", type=int, default=20)
args = ap.parse_args()

fuzzy = fuzzy_params(json.load(open(Path(args.root) / args.split / f"{args.split}_tools_catalog.json", encoding="utf-8")))
gold = {}
for line in open(Path(args.root) / args.split / f"{args.split}.jsonl", encoding="utf-8"):
    r = json.loads(line)
    gold[r["id"]] = r["label"]["tool_calls"]

n = same_seq = exact = 0
cnt, ex = Counter(), defaultdict(list)
for line in open(args.pred, encoding="utf-8"):
    p = json.loads(line)
    g, c = gold.get(p["id"]), p["tool_calls"]
    n += 1
    if g is None or [x["name"] for x in g] != [x["name"] for x in c]:
        continue
    same_seq += 1
    wrong = False
    for gc, pc in zip(g, c):
        ga, pa = gc.get("arguments", {}), pc.get("arguments", {})
        for k in sorted(set(ga) | set(pa)):
            if k not in pa:
                kind = "missing"
            elif k not in ga:
                kind = "extra"
            elif k in fuzzy.get(gc["name"], set()):
                if f1(pa[k], ga[k]) > 0.6:  # the scorer accepts this
                    continue
                kind = "value (fuzzy <= 0.6)"
            elif norm(ga[k]) != norm(pa[k]):
                kind = "value"
            else:
                continue
            wrong = True
            key = (gc["name"], k, kind)
            cnt[key] += 1
            if len(ex[key]) < 2:
                ex[key].append((p["id"], ga.get(k), pa.get(k)))
    exact += not wrong
print(f"samples {n} | same tool sequence {same_seq} | of those the scorer accepts every argument {exact}")
for (tool, k, kind), v in cnt.most_common(args.top):
    print(f"{v:3d}  {tool}.{k}  [{kind}]")
    for sid, gv, pv in ex[(tool, k, kind)]:
        print(f"       {sid}: gold={json.dumps(gv, ensure_ascii=False)[:70]}  pred={json.dumps(pv, ensure_ascii=False)[:70]}")
