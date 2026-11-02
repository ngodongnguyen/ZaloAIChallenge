"""Score a run's output file against the locked split and the train labels.

Gold calls come from train/train.jsonl label.tool_calls. Gold state hashes come from the split file
(computed once by replaying gold calls with zaic_tools). Fuzzy argument names come from the catalog.
The score is approximate (see harness/metric.py).
"""
import json
from pathlib import Path

from harness.metric import sample_score, mean_score
from harness.split import load_split, load_train, DATA_ROOT


def fuzzy_by_tool(catalog_path):
    with open(catalog_path, encoding="utf-8-sig") as f:
        catalog = json.load(f)
    out = {}
    for app in catalog["apps"]:
        for t in app["tools"]:
            props = (t.get("parameters") or {}).get("properties") or {}
            out[t["name"]] = {k for k, v in props.items() if v.get("x-match") == "fuzzy"}
    return out


def score_run(output_jsonl, split_path, root=DATA_ROOT):
    split = load_split(split_path)
    gold_hash = split["gold_state_hash"]
    dev_ids = split["dev"]
    labels = {r["id"]: r["label"]["tool_calls"] for r in load_train(root / "train" / "train.jsonl")}
    fz = fuzzy_by_tool(root / "train" / "train_tools_catalog.json")

    got = {}
    with open(output_jsonl, encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                got[r["id"]] = r
    missing = [i for i in dev_ids if i not in got]
    if missing:
        raise ValueError(f"{len(missing)} dev ids missing from output, e.g. {missing[:3]}")

    per = {}
    for i in dev_ids:
        r = got[i]
        per[i] = sample_score(r["tool_calls"], labels[i], r["state_hash"], gold_hash[i], fz)
    scores = [per[i] for i in dev_ids]
    return {
        "score": round(mean_score(scores), 6),
        "n": len(dev_ids),
        "zero_samples": sum(1 for s in scores if s == 0.0),
        "per_sample": per,
    }
