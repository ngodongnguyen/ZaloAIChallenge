"""PhoWhisper transcripts for every train audio, written to data/asr_train.jsonl ({id, text}).

Used as --asr-cache for train/build_sft.py so SFT also sees ASR-noisy inputs. Resumable: ids already in
the output file are skipped. Run with the zaic venv:  python loop/asr_train.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "C:/Project/zalo_ai/src")
sys.path.insert(0, "C:/Project/zalo_research")

from harness.score import DATA_ROOT, load_train  # noqa: E402
from asr import ASR  # noqa: E402

OUT = Path("C:/Project/zalo_research/data/asr_train.jsonl")


def main():
    OUT.parent.mkdir(exist_ok=True)
    done = set()
    if OUT.exists():
        done = {json.loads(l)["id"] for l in open(OUT, encoding="utf-8") if l.strip()}
    rows = [r for r in load_train(DATA_ROOT / "train" / "train.jsonl") if r["id"] not in done]
    asr = ASR()
    with open(OUT, "a", encoding="utf-8") as f:
        for k, r in enumerate(rows, 1):
            text = asr.transcribe(DATA_ROOT / "train" / r["audio"], r["id"])
            f.write(json.dumps({"id": r["id"], "text": text}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{k}/{len(rows)}] {r['id']}", flush=True)


if __name__ == "__main__":
    main()
