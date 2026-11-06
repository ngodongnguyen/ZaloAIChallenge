"""Break down where a prediction file loses points, and print the worst samples.

python src/analyze.py --pred out/pred_seen_ft.jsonl --root data --split train [--show 15]
Categories (first that applies): empty_pred (gold not empty), spurious (gold empty, pred not),
wrong_tool (a gold tool never called), arg_mismatch (right tools, wrong arguments), extra_calls
(all gold matched but extra calls), hash_only (calls match but final state differs), ok.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval import fuzzy_params, match, score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--root", default="data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--show", type=int, default=15)
    args = ap.parse_args()

    base = Path(args.root) / args.split
    fuzzy = fuzzy_params(json.load(open(base / f"{args.split}_tools_catalog.json", encoding="utf-8")))
    gold = {}
    for line in open(base / f"{args.split}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        gold[r["id"]] = r["label"]["tool_calls"]

    sys.path.insert(0, str(base))
    from zaic_tools import ToolEnv

    def gold_hash(sid):
        env = ToolEnv(args.split, sid, root=args.root)
        for c in gold[sid]:
            env.call(c["name"], c.get("arguments", {}))
        return env.get_state_hash()

    cats, per_app, bad = Counter(), defaultdict(lambda: [0, 0.0]), []
    for line in open(args.pred, encoding="utf-8"):
        p = json.loads(line)
        g, s = gold[p["id"]], p["tool_calls"]
        sc, m, u = score(s, g, fuzzy)
        hash_ok = p.get("state_hash") == gold_hash(p["id"])
        final = sc if hash_ok else 0.0
        if not g:
            cat = "ok" if not s else "spurious"
        elif not s:
            cat = "empty_pred"
        elif {c["name"] for c in g} - {c["name"] for c in s}:
            cat = "wrong_tool"
        elif m < len(g):
            cat = "arg_mismatch"
        elif u:
            cat = "extra_calls"
        else:
            cat = "ok" if hash_ok else "hash_only"
        cats[cat] += 1
        for app in {c["name"].split(".")[0] for c in g if not c["name"].startswith("sys.")} or {"(none)"}:
            per_app[app][0] += 1
            per_app[app][1] += final
        if final < 1.0:
            bad.append((final, p["id"], cat, g, s))

    n = sum(cats.values())
    print(f"samples={n}")
    for c, k in cats.most_common():
        print(f"  {c:<12} {k:>4}  ({k / n:.0%})")
    print("\nper app (mean final score, n):")
    for app, (k, tot) in sorted(per_app.items(), key=lambda x: x[1][1] / x[1][0]):
        print(f"  {app:<18} {tot / k:.2f}  n={k}")
    print(f"\nworst {args.show} samples:")
    for final, sid, cat, g, s in sorted(bad, key=lambda x: x[0])[: args.show]:
        print(f"--- {sid} [{cat}] score={final:.2f}")
        print("  gold:", json.dumps(g, ensure_ascii=False)[:400])
        print("  pred:", json.dumps(s, ensure_ascii=False)[:400])


if __name__ == "__main__":
    main()
