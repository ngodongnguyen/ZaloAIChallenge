"""Measure ASR word/char error rate against the gold `audio_text` of a labelled split.

python src/asr_eval.py --root data --models models/asr openai/whisper-large-v3-turbo --limit 150
Writes hypotheses to <out-dir>/<model>.jsonl so they can be inspected or reused as an ASR cache.
"""
import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from asr import ASR  # noqa: E402
from config import seed_everything  # noqa: E402


def norm(s):
    s = unicodedata.normalize("NFC", s).lower()
    return re.findall(r"\w+", s)


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--out-dir", default="out/asr_eval")
    args = ap.parse_args()

    seed_everything()
    base = Path(args.root) / args.split
    rows = [json.loads(l) for l in open(base / f"{args.split}.jsonl", encoding="utf-8")][: args.limit]
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for model in args.models:
        asr = ASR(model)
        werr = wn = cerr = cn = 0
        path = Path(args.out_dir) / (model.strip("/").replace("/", "_") + ".jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                hyp = asr.transcribe(base / r["audio"])
                ref_w, hyp_w = norm(r["audio_text"]), norm(hyp)
                werr += edit_distance(ref_w, hyp_w)
                wn += len(ref_w)
                ref_c, hyp_c = "".join(ref_w), "".join(hyp_w)
                cerr += edit_distance(ref_c, hyp_c)
                cn += len(ref_c)
                f.write(json.dumps({"id": r["id"], "ref": r["audio_text"], "text": hyp}, ensure_ascii=False) + "\n")
        print(f"{model}: WER={werr / wn:.3f}  CER={cerr / cn:.3f}  (n={len(rows)})  -> {path}", flush=True)
        del asr


if __name__ == "__main__":
    main()
