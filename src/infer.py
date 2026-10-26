"""Run the agent over a split and write submission.jsonl.

Local:   python src/infer.py --root . --split public_test --out submission.jsonl
Docker:  python src/infer.py --data-dir /data --out /app/result/submission.jsonl
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import seed_everything  # noqa: E402


def resolve_root(args):
    """Return (root, split). In Docker the split folder itself is mounted at /data."""
    if args.data_dir:
        d = Path(args.data_dir).resolve()
        jsonls = [p for p in d.glob("*.jsonl") if p.name != "sample_submission.jsonl"]
        split = jsonls[0].stem if jsonls else "private_test"
        if (d / split).is_dir():
            return d, split
        root = Path(tempfile.mkdtemp())
        os.symlink(d, root / split)
        return root, split
    return Path(args.root).resolve(), args.split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--split", default="public_test")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default="submission.jsonl")
    ap.add_argument("--asr-cache", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    seed_everything()
    root, split = resolve_root(args)
    sys.path.insert(0, str(root / split))  # zaic_tools of the split
    from zaic_tools import ToolEnv  # noqa: E402
    from agent import run_agent  # noqa: E402
    from asr import ASR  # noqa: E402
    from llm import LLM  # noqa: E402

    with open(root / split / f"{split}_tools_catalog.json", encoding="utf-8") as f:
        catalog = json.load(f)
    asr, llm = ASR(cache_path=args.asr_cache), LLM()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(root / split / f"{split}.jsonl", encoding="utf-8") as fin, \
            open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            item = json.loads(line)
            t0 = time.time()
            env = ToolEnv(split, item["id"], root=root)  # exactly one env per sample
            try:
                text = asr.transcribe(root / split / item["audio"], item["id"])
                run_agent(env, item, text, catalog, llm, t0)
            except Exception as e:  # keep whatever was logged so far
                print(f"[warn] {item['id']}: {e!r}", file=sys.stderr)
            fout.write(json.dumps(
                {"id": item["id"], "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash()},
                ensure_ascii=False) + "\n")
            fout.flush()
            n += 1
            print(f"{item['id']} {time.time() - t0:.1f}s", file=sys.stderr)
            if args.limit and n >= args.limit:
                break


if __name__ == "__main__":
    main()
