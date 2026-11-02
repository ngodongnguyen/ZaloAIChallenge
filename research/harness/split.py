"""Create the dev/holdout split once and compute gold state hashes by replaying gold tool calls.

Grouping: a sample's group is the app of its first gold call ("none" if the gold list is empty).
Whole groups go to either dev or holdout, so no app group leaks between tuning and the locked set.
Split and hashes are written once. `build_split` refuses to overwrite an existing split file.
"""
import hashlib
import json
import random
import sys
from pathlib import Path

DATA_ROOT = Path("C:/Project/zalo_ai/data/LifeMate/train+public_test")
TRAIN_JSONL = DATA_ROOT / "train" / "train.jsonl"


def load_train(path=TRAIN_JSONL):
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def group_of(row):
    calls = row["label"]["tool_calls"]
    if not calls:
        return "none"
    return calls[0]["name"].split(".", 1)[0]


def gold_state_hash(row, root=DATA_ROOT):
    """Replay gold calls in order with the matching zaic_tools, as the organisers do."""
    train_dir = str(root / "train")
    if train_dir not in sys.path:
        sys.path.insert(0, train_dir)
    from zaic_tools import ToolEnv  # imported here so the module loads without the data present
    env = ToolEnv("train", row["id"], root=root)
    for c in row["label"]["tool_calls"]:
        env.call(c["name"], c["arguments"])
    return env.get_state_hash()


def build_split(out_path, dev_target=120, holdout_target=200, seed=42, rows=None):
    """Whole groups go to dev until dev_target, then to holdout until holdout_target.
    Groups left over are unused, so the locked set stays small enough to run once at the end."""
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(f"split already exists, refusing to overwrite: {out_path}")
    rows = rows if rows is not None else load_train()
    groups = {}
    for r in rows:
        groups.setdefault(group_of(r), []).append(r["id"])
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    dev_groups, dev_ids, holdout_groups, holdout_ids = [], [], [], []
    for g in names:
        if len(dev_ids) < dev_target:
            dev_groups.append(g)
            dev_ids.extend(groups[g])
        elif len(holdout_ids) < holdout_target:
            holdout_groups.append(g)
            holdout_ids.extend(groups[g])
    overlap = set(dev_ids) & set(holdout_ids)
    if overlap:
        raise AssertionError(f"dev and holdout overlap: {sorted(overlap)[:5]}")

    by_id = {r["id"]: r for r in rows}
    gold = {i: gold_state_hash(by_id[i]) for i in dev_ids + holdout_ids}
    payload = {
        "seed": seed, "group_key": "first gold call app",
        "dev_groups": dev_groups, "holdout_groups": holdout_groups,
        "dev": dev_ids, "holdout": holdout_ids,
        "gold_state_hash": gold,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    payload["sha256"] = hashlib.sha256(blob).hexdigest()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return payload


def load_split(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def sha_check(path):
    """Hash of the split file's bytes. The loop stops if this changes after it started."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
