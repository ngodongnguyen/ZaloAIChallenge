"""Dev experiment: bias Whisper toward the sample's own entity names via prompt_ids.

  python loop/asr_bias_exp.py [--model models/asr_ft] [--budget 150]
Entity names come from <split>_environment/<id>/state.json (the same data env.get_state() returns).
Prints dev WER without and with the entity prompt, and an entity-only error rate.
"""
import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, "C:/Project/zalo_research")
from harness.score import DATA_ROOT, load_train  # noqa: E402
from harness.split import load_split  # noqa: E402

KEYS = {"name", "title", "artist", "location", "members", "label", "bank", "provider", "tags", "author",
        "sender", "subject", "headline", "source", "channels", "channel", "city", "station", "from", "to",
        "relation", "content", "category", "topic"}
OUT = Path("C:/Project/zalo_research/loop/runs_work/asr_bias")


def entities(state):
    """Distinct entity-like strings from the app state; proper-noun-like ones (with capitals) first."""
    seen, out = set(), []

    def walk(o, key=None):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, k)
        elif isinstance(o, list):
            for v in o:
                walk(v, key)
        elif isinstance(o, str) and key in KEYS and re.search(r"[^\W\d_]", o) and len(o) <= 80:
            s = o.strip()
            if s.lower() not in seen:
                seen.add(s.lower())
                out.append(s)

    for app, v in state.items():
        if app != "_meta":
            walk(v)
    return sorted(out, key=lambda s: (not any(c.isupper() for c in s), len(s)))


def build_prompt(names, tok, budget):
    picked = []
    for n in names:
        if len(tok(", ".join(picked + [n]), add_special_tokens=False).input_ids) > budget:
            break
        picked.append(n)
    return ", ".join(picked)


def norm(s):
    s = unicodedata.normalize("NFC", s or "").lower()
    return re.sub(r"[^\w\s]", " ", s).split()


def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[:], i
        for j in range(1, len(b) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
    return d[-1]


def last_text(x):
    return (x[-1] if x else "") if isinstance(x, list) else (x or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="C:/Project/zalo_research/models/asr_ft")
    ap.add_argument("--budget", type=int, default=150)
    ap.add_argument("--tag", default="ft")
    args = ap.parse_args()

    split = load_split(Path("C:/Project/zalo_research/splits/lifemate_split.json"))
    rows = {r["id"]: r for r in load_train(DATA_ROOT / "train" / "train.jsonl")}
    proc = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda").eval()

    OUT.mkdir(parents=True, exist_ok=True)
    res = []
    for k, i in enumerate(split["dev"], 1):
        r = rows[i]
        audio, sr = sf.read(str(DATA_ROOT / "train" / r["audio"]), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        feats = proc.feature_extractor(audio, sampling_rate=sr, return_tensors="pt").input_features
        feats = feats.to("cuda", torch.float16)
        state = json.load(open(DATA_ROOT / "train" / "train_environment" / i / "state.json", encoding="utf-8"))
        prompt = build_prompt(entities(state), proc.tokenizer, args.budget)
        hyps = {}
        with torch.no_grad():
            for mode in ("plain", "bias"):
                kw = {}
                if mode == "bias" and prompt:
                    kw["prompt_ids"] = torch.tensor(proc.get_prompt_ids(prompt)).to("cuda")
                out = model.generate(feats, language="vi", task="transcribe", do_sample=False, num_beams=1,
                                     max_new_tokens=200, **kw)
                text = proc.batch_decode(out, skip_special_tokens=True)[0].strip()
                if mode == "bias" and prompt and text.startswith(prompt):  # some versions echo the prompt
                    text = text[len(prompt):].strip()
                hyps[mode] = text
        res.append({"id": i, "ref": last_text(r["audio_text"]), "prompt": prompt, **hyps})
        print(k, flush=True)

    json.dump(res, open(OUT / f"{args.tag}_b{args.budget}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for mode in ("plain", "bias"):
        e = sum(edit_distance(norm(x["ref"]), norm(x[mode])) for x in res)
        n = sum(len(norm(x["ref"])) for x in res)
        print(f"{args.tag} {mode}: WER {e / n:.4f}")


if __name__ == "__main__":
    main()
