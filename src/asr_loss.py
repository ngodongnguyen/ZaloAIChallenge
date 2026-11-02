"""Which ASR mistakes cost points? Lists the samples that the LLM gets right (final state == gold) from the gold text but
wrong from the ASR text, with both texts and the first tool call that differs.

python src/asr_loss.py --pred-asr out/pred_unseen_para-q35e3.jsonl --pred-gold out/pred_unseen_gold.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--pred-asr", required=True)
ap.add_argument("--pred-gold", required=True)
ap.add_argument("--asr-text", default="cache/asr_ft_holdout.jsonl")
ap.add_argument("--gold-text", default="cache/asr_gold.jsonl")
ap.add_argument("--root", default="data")
ap.add_argument("--split", default="train")
ap.add_argument("--top", type=int, default=40)
args = ap.parse_args()

base = Path(args.root) / args.split
sys.path.insert(0, str(base))
from zaic_tools import ToolEnv  # noqa: E402


def load(path, key=None):
    return {json.loads(l)["id"]: (json.loads(l)[key] if key else json.loads(l)) for l in open(path, encoding="utf-8")}


gold = load(base / f"{args.split}.jsonl")
pa, pg = load(args.pred_asr), load(args.pred_gold)
ta, tg = load(args.asr_text, "text"), load(args.gold_text, "text")


def gold_hash(sid):
    env = ToolEnv(args.split, sid, root=args.root)
    for c in gold[sid]["label"]["tool_calls"]:
        env.call(c["name"], c.get("arguments", {}))
    return env.get_state_hash()


lost = []
for sid in sorted(set(pa) & set(pg)):
    h = gold_hash(sid)
    if pg[sid]["state_hash"] == h and pa[sid]["state_hash"] != h:
        lost.append(sid)
print(f"right with gold text but wrong with ASR text: {len(lost)} of {len(set(pa) & set(pg))}")
for sid in lost[:args.top]:
    ca, cg = pa[sid]["tool_calls"], pg[sid]["tool_calls"]
    diff = next(((i, x, y) for i, (x, y) in enumerate(zip(cg, ca)) if x != y), None)
    print(f"--- {sid}\n  gold text: {tg.get(sid)}\n  ASR text:  {ta.get(sid)}")
    if diff:
        i, x, y = diff
        print(f"  call {i} gold-run: {json.dumps(x, ensure_ascii=False)[:140]}\n  call {i} asr-run:  {json.dumps(y, ensure_ascii=False)[:140]}")
    else:
        print(f"  calls: gold-run {len(cg)} vs asr-run {len(ca)}")
