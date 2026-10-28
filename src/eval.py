"""Offline scorer approximating the official tool-call metric (state_hash is not checked).

python src/eval.py --pred submission.jsonl --root . --split train
Official formula was an image in the brief, so this reports P, R and F1 = 2PR/(P+R)
with P = M/(M+U), R = M/|G|; gold-empty samples score 1 only for an empty submission.
"""
import argparse
import sys
import json
import re
import unicodedata
from collections import Counter


def toks(s):
    return re.findall(r"\w+", unicodedata.normalize("NFC", str(s)).lower())


def f1(a, b):
    ta, tb = toks(a), toks(b)
    if not ta and not tb:
        return 1.0
    common = sum((Counter(ta) & Counter(tb)).values())
    if not common:
        return 0.0
    p, r = common / len(ta), common / len(tb)
    return 2 * p * r / (p + r)


def fuzzy_params(catalog):
    out = {}
    for app in catalog["apps"]:
        for t in app["tools"]:
            props = (t.get("parameters") or {}).get("properties", {}) or {}
            out[t["name"]] = {k for k, v in props.items() if v.get("x-match") == "fuzzy"}
    return out


def norm(v):
    return json.dumps(v, sort_keys=True, ensure_ascii=False)


def match(sub, gold, fuzzy):
    if sub["name"] != gold["name"]:
        return False
    sa, ga = sub.get("arguments", {}), gold.get("arguments", {})
    if set(sa) != set(ga):
        return False
    fz = fuzzy.get(gold["name"], set())
    for k in ga:
        if k in fz:
            if f1(sa[k], ga[k]) <= 0.6:
                return False
        elif norm(sa[k]) != norm(ga[k]):
            return False
    return True


def score(sub, gold, fuzzy):
    if not gold:
        return 1.0 if not sub else 0.0, 0, 0
    used, m = set(), 0
    for g in gold:
        for i, s in enumerate(sub):
            if i not in used and match(s, g, fuzzy):
                used.add(i)
                m += 1
                break
    u = len(sub) - m
    p = m / (m + u) if (m + u) else 0.0
    r = m / len(gold)
    return (2 * p * r / (p + r) if p + r else 0.0), m, u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    base = f"{args.root}/{args.split}"
    fuzzy = fuzzy_params(json.load(open(f"{base}/{args.split}_tools_catalog.json", encoding="utf-8")))
    gold = {}
    for line in open(f"{base}/{args.split}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        gold[r["id"]] = r["label"]["tool_calls"]

    sys.path.insert(0, base)  # zaic_tools of the split: gold hash = replay of gold calls
    from zaic_tools import ToolEnv

    def gold_hash(sid):
        env = ToolEnv(args.split, sid, root=args.root)
        for c in gold[sid]:
            env.call(c["name"], c.get("arguments", {}))
        return env.get_state_hash()

    scores, empty_ok, empty_n, hash_ok, tool_scores = [], 0, 0, 0, []
    for line in open(args.pred, encoding="utf-8"):
        p = json.loads(line)
        if p["id"] not in gold:
            continue
        s, _, _ = score(p["tool_calls"], gold[p["id"]], fuzzy)
        tool_scores.append(s)
        if p.get("state_hash") != gold_hash(p["id"]):
            s = 0.0  # official rule: wrong final state scores 0
        else:
            hash_ok += 1
        scores.append(s)
        if not gold[p["id"]]:
            empty_n += 1
            empty_ok += s
    n = max(len(scores), 1)
    print(f"samples={len(scores)}  mean_score={sum(scores) / n:.4f}  (tool-call only: {sum(tool_scores) / n:.4f})")
    print(f"state_hash matches gold: {hash_ok}/{len(scores)}")
    if empty_n:
        print(f"out-of-scope: {int(empty_ok)}/{empty_n} correct")


if __name__ == "__main__":
    main()
