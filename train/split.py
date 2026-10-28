"""App-level validation split.

Hold out whole apps (every sample whose gold calls touch them) to measure generalisation to tools never
seen in training, plus a random 'seen' set. Training must exclude both (pass the combined file to sft.py --holdout).
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data")
ap.add_argument("--split", default="train")
ap.add_argument("--out-dir", default="cache")
ap.add_argument("--n-apps", type=int, default=6, help="apps held out entirely")
ap.add_argument("--n-seen", type=int, default=100)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

rows = [json.loads(l) for l in open(Path(args.root) / args.split / f"{args.split}.jsonl", encoding="utf-8")]


def apps_of(r):
    return {c["name"].split(".")[0] for c in r["label"]["tool_calls"] if not c["name"].startswith("sys.")}


use = Counter(a for r in rows for a in apps_of(r))
rng = random.Random(args.seed)
# mid-frequency apps: enough samples to evaluate (>=20) yet cheap to lose from training
pool = sorted(a for a, n in use.items() if 20 <= n <= 40)
held_apps = set(rng.sample(pool, min(args.n_apps, len(pool))))
unseen = [r["id"] for r in rows if apps_of(r) & held_apps]
rest = [r["id"] for r in rows if r["id"] not in set(unseen)]
seen = rng.sample(rest, args.n_seen)

out = Path(args.out_dir)
out.mkdir(exist_ok=True, parents=True)
(out / "holdout_unseen.txt").write_text("\n".join(unseen))
(out / "holdout_seen.txt").write_text("\n".join(seen))
(out / "holdout_all.txt").write_text("\n".join(unseen + seen))
print("held-out apps:", sorted(held_apps))
print(f"unseen-app samples: {len(unseen)}  seen holdout: {len(seen)}  train left: {len(rows) - len(unseen) - len(seen)}")
