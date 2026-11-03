"""Replay saved model proposals through fresh ToolEnv sessions, normalizing before env.call.

Example local public run:
python replay_actions.py --baseline-dir /path/to/lm_whisper_full/public \
    --out-dir /tmp/action-fix --data-root dataset/lifemate/train+public_test

No labels are used for public inference. No calls are deleted or inserted. Every
candidate call and final hash are taken from that candidate session's own ToolEnv.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from action_normalizer import normalize_action


def collect(directory, prefix):
    records, hashes = {}, set()
    paths = sorted(Path(directory).glob(prefix + ".shard*.jsonl"))
    if not paths:
        raise ValueError(f"No {prefix} shards under {directory}")
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows or not rows[0].get("_header", {}).get("run_hash"):
            raise ValueError(f"Missing provenance header: {path}")
        hashes.add(rows[0]["_header"]["run_hash"])
        for row in rows[1:]:
            if row.get("status") != "completed" or row["id"] in records:
                raise ValueError(f"Incomplete/duplicate ID in {path}")
            records[row["id"]] = row
    if len(hashes) != 1:
        raise ValueError("Mixed source run hashes")
    return records, next(iter(hashes))


def replay_episode(env, row, request, calls, fix=True):
    inference = copy.deepcopy(row)
    inference.pop("label", None)
    inference.pop("audio_text", None)
    for turn in inference.get("history", []):
        turn.pop("audio_text", None)
    steps = []
    for call in calls:
        args, audit = (normalize_action(inference, request, steps, call["name"], call.get("arguments") or {})
                       if fix else (copy.deepcopy(call.get("arguments") or {}), []))
        check = env.validate(call["name"], args)
        if not check.get("ok"):
            raise ValueError(f"{row['id']}: invalid proposal: {check}")
        result = env.call(call["name"], args)
        steps.append({"kind": "call", "name": call["name"], "arguments": args, "result": result,
                      "normalization": audit})
    actual = env.get_tool_calls()
    if len(actual) != len(calls) or [c["name"] for c in actual] != [c["name"] for c in calls]:
        raise ValueError("ToolEnv trace differs from executed proposals")
    return {"id": row["id"], "tool_calls": actual, "state_hash": env.get_state_hash()}, steps


def validate_request_source(baseline_dir, rows, traces):
    baseline_dir = Path(baseline_dir)
    config = json.loads((baseline_dir / "config.json").read_text())
    identity = config.get("asr_model_identity")
    if not identity:
        raise ValueError("Source config has no verified ASR model identity")
    paths = sorted((baseline_dir.parent / "asr").glob(f"asr_{config['split']}_{config['asr_backend']}.shard*.jsonl"))
    if not paths:
        raise ValueError("Missing source ASR cache")
    transcripts, hashes = {}, []
    for path in paths:
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not records or records[0].get("_header", {}).get("run_hash") != identity:
            raise ValueError("Source ASR identity mismatch")
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        for record in records[1:]:
            if record.get("status") != "completed" or record["id"] in transcripts:
                raise ValueError("Incomplete/duplicate source ASR")
            transcripts[record["id"]] = record["text"]
    for qid, trace in traces.items():
        if transcripts.get(rows[qid]["audio"]) != trace.get("request"):
            raise ValueError(f"Trace request differs from source ASR: {qid}")
    return {"asr_model_identity": identity, "asr_cache_sha256": hashes}


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    temporary.replace(path)


def validate_cli(split_dir, submission, expected):
    run = subprocess.run([sys.executable, "-m", "zaic_tools", "--split", split_dir.name, "--root", "..",
                          "--validate", str(Path(submission).resolve())], cwd=split_dir,
                         capture_output=True, text=True)
    matches = re.findall(r"(?m)^(\d+) lines checked, (\d+) schema errors\s*$", run.stdout)
    if run.returncode or len(matches) != 1 or tuple(map(int, matches[0])) != (expected, 0):
        raise ValueError(f"Official schema validation failed: {run.stdout} {run.stderr}")
    return run.stdout.strip()


def run_replay(data_root, baseline_dir, out_dir, split="public_test"):
    data_root, out = Path(data_root).resolve(), Path(out_dir).resolve()
    split_dir = data_root / split
    rows = [json.loads(line) for line in (split_dir / (split + ".jsonl")).read_text().splitlines() if line.strip()]
    by_id = {r["id"]: r for r in rows}
    predictions, source_hash = collect(baseline_dir, "preds")
    traces, trace_hash = collect(baseline_dir, "traces")
    if source_hash != trace_hash or set(predictions) != set(traces):
        raise ValueError("Prediction/trace provenance or IDs differ")
    if split == "public_test" and set(predictions) != set(by_id):
        raise ValueError("Public source is not complete")
    if not set(predictions) <= set(by_id):
        raise ValueError("Unknown source IDs")
    asr_provenance = validate_request_source(baseline_dir, by_id, traces)
    # ToolEnv itself reads the simulator data. The agent never opens state files.
    sys.path.insert(0, str(split_dir))
    for name in list(sys.modules):
        if name == "zaic_tools" or name.startswith("zaic_tools."):
            del sys.modules[name]
    from zaic_tools import ToolEnv

    normalizer_sha = hashlib.sha256(Path(__file__).with_name("action_normalizer.py").read_bytes()).hexdigest()
    run_hash = hashlib.sha256((source_hash + normalizer_sha).encode()).hexdigest()
    candidates, candidate_traces, changes = [], [], []
    for index, qid in enumerate(sorted(predictions)):
        row, proposal, trace = by_id[qid], predictions[qid], traces[qid]
        # A separate control replay verifies that source predictions reproduce on
        # this official dataset/version. Candidate inference gets one fresh env.
        control, _ = replay_episode(ToolEnv(split, qid, root=data_root), row, trace["request"], proposal["tool_calls"], False)
        if control["state_hash"] != proposal["state_hash"] or control["tool_calls"] != proposal["tool_calls"]:
            raise ValueError(f"Source prediction does not reproduce: {qid}")
        candidate, steps = replay_episode(ToolEnv(split, qid, root=data_root), row, trace["request"], proposal["tool_calls"])
        candidates.append(candidate)
        candidate_traces.append({"id": qid, "status": "completed", "request": trace["request"], "steps": steps,
                                 "source_state_hash": proposal["state_hash"], "state_hash": candidate["state_hash"]})
        edits = [s for s in steps if s["normalization"]]
        if edits:
            changes.append({"id": qid, "request": trace["request"], "state_hash_changed": candidate["state_hash"] != proposal["state_hash"],
                            "edits": [{"name": s["name"], "normalization": s["normalization"]} for s in edits]})
        if index % 100 == 0:
            print(f"[action fix] {index}/{len(predictions)}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "traces.shard0.jsonl", [{"_header": {"run_hash": run_hash}}] + candidate_traces)
    write_jsonl(out / "preds.shard0.jsonl", [{"_header": {"run_hash": run_hash}}] + [dict(r, status="completed") for r in candidates])
    candidate_path = out / "candidate.jsonl"
    write_jsonl(candidate_path, candidates)
    cli = validate_cli(split_dir, candidate_path, len(candidates))
    candidate_path.replace(out / "submission.jsonl")
    report = {"n": len(candidates), "complete": True, "source_run_hash": source_hash, "normalizer_sha256": normalizer_sha,
              "submission_sha256": hashlib.sha256((out / "submission.jsonl").read_bytes()).hexdigest(),
              "changed_samples": len(changes), "changed_state_hashes": sum(x["state_hash_changed"] for x in changes),
              "calls": sum(len(r["tool_calls"]) for r in candidates), "official_validation": cli,
              "changes": changes, **asr_provenance,
              "method": "cached proposals, fresh control/candidate ToolEnv sessions, normalize before call"}
    (out / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "changes"}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="dataset/lifemate/train+public_test")
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", choices=["train", "public_test"], default="public_test")
    args = parser.parse_args()
    run_replay(args.data_root, args.baseline_dir, args.out_dir, args.split)
