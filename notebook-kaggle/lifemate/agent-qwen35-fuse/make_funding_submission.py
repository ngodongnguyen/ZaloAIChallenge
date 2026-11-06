"""lm-B-057: replay a finished submission through fresh ToolEnv sessions and choose the paying source from balances already read.

    python make_funding_submission.py --data-root dataset/lifemate/train+public_test --base submission.jsonl --traces dir --out dir

No call is added, removed or reordered. A control replay must reproduce the base calls and hash before any edit is kept.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from funding_rule import rule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.data_root).resolve()
    sys.path.insert(0, str(root / "public_test"))
    from zaic_tools import ToolEnv

    base = [json.loads(line) for line in Path(a.base).read_text(encoding="utf-8").splitlines() if line.strip()]
    requests = {}
    for path in Path(a.traces).glob("traces.shard*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if "id" in row:
                requests[row["id"]] = row["request"]
    out, changes = [], []
    for row in base:
        control = ToolEnv("public_test", row["id"], root=root)
        for c in row["tool_calls"]:
            control.call(c["name"], c.get("arguments") or {})
        if control.get_state_hash() != row["state_hash"]:
            raise SystemExit(f"control replay differs: {row['id']}")
        env = ToolEnv("public_test", row["id"], root=root)
        steps, edits = [], []
        for c in row["tool_calls"]:
            args = c.get("arguments") or {}
            new = rule(requests[row["id"]], steps, c["name"], args)
            if new != args:
                edits.append({"name": c["name"], "before": args, "after": new})
            steps.append({"name": c["name"], "arguments": new, "result": env.call(c["name"], new)})
        if edits:
            changes.append({"id": row["id"], "request": requests[row["id"]], "edits": edits, "hash_changed": env.get_state_hash() != row["state_hash"]})
        out.append({"id": row["id"], "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash()})
    target = Path(a.out)
    target.mkdir(parents=True, exist_ok=True)
    path = target / "submission.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    report = {"n": len(out), "changed_samples": len(changes), "changed_hashes": sum(c["hash_changed"] for c in changes),
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "changes": changes}
    (target / "funding_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print({k: report[k] for k in ("n", "changed_samples", "changed_hashes", "sha256")})
    for c in changes:
        print(c["id"], c["hash_changed"], c["request"][:110], [(e["before"], e["after"]) for e in c["edits"]][:2])


if __name__ == "__main__":
    main()
