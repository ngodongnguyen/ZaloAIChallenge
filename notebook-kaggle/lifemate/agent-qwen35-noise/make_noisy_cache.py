"""lm-sft-064: train transcripts with real ASR errors in the format the agent reads at run time.

    python make_noisy_cache.py --train-dir dataset/lifemate/train+public_test/train --oof asr_train_gipoof.shard0.jsonl --out dir

Each disagreeing span of the reference text takes the words of the out-of-fold Gipformer transcript; case, punctuation and digits
stay as in the reference wherever the words agree, as in a fused transcript. Public data is never read.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "agent-qwen35-fuse"))
from asr_fuse import TrigramLM, fuse, spoken_text
from make_fuse_cache import read_cache, read_rows, train_texts
from train_fuse_model import edit_distance


def gold_by_audio(rows):
    return {t["audio"]: t["audio_text"] for r in rows for t in [r] + [h for h in r.get("history", []) if h.get("role") == "user"]}


def noisy_text(gold, oof, lm):
    return fuse(gold, oof, lm, set(), bias=-1e9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--oof", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = read_rows(Path(a.train_dir) / "train.jsonl")
    gold = gold_by_audio(rows)
    oof_id, oof = read_cache(a.oof)
    if set(oof) != set(gold) or len(gold) != 5036:
        raise SystemExit(f"out-of-fold cache covers {len(oof)} of {len(gold)} train audio")
    lm = TrigramLM(train_texts(rows))
    texts = {k: noisy_text(gold[k], oof[k], lm) for k in sorted(gold)}
    if any(not t.strip() for t in texts.values()):
        raise SystemExit("empty noisy transcript")
    wer = lambda hyp: sum(edit_distance(spoken_text(gold[k]), spoken_text(hyp[k])) for k in gold) / sum(len(spoken_text(gold[k])) for k in gold)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    identity = "lm-sft-064-noisy-" + hashlib.sha256(json.dumps([oof_id, Path(__file__).read_bytes().hex()]).encode()).hexdigest()[:16]
    cache = out / "asr_train_noisy.shard0.jsonl"
    cache.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in [{"_header": {"run_hash": identity}}] + [
        {"id": k, "status": "completed", "text": v} for k, v in texts.items()]), encoding="utf-8")
    report = {"identity": identity, "n_audio": len(texts), "source": oof_id, "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
              "wer_oof": wer(oof), "wer_noisy": wer(texts), "unchanged": sum(texts[k] == gold[k] for k in gold)}
    (out / "noisy_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
