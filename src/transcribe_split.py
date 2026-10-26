"""Pre-transcribe a split's audio into a cache (id -> text). Useful to mix real ASR output into SFT data."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from asr import ASR  # noqa: E402
from config import seed_everything  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
ap.add_argument("--split", default="train")
ap.add_argument("--out", required=True)
args = ap.parse_args()

seed_everything()
base = Path(args.root) / args.split
asr = ASR()
with open(base / f"{args.split}.jsonl", encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
    for line in fin:
        item = json.loads(line)
        text = asr.transcribe(base / item["audio"])
        fout.write(json.dumps({"id": item["id"], "text": text}, ensure_ascii=False) + "\n")
        fout.flush()
