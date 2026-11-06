"""Fit the span chooser of asr_fuse on out-of-fold transcripts of train audio. Public data is never read.

    python train_fuse_model.py --train-dir dataset/lifemate/train+public_test/train --primary a.jsonl --secondary b.jsonl --out fuse_model.json

The language model counts only train rows whose audio the two caches do not cover, so span features match what the
chooser meets on unseen audio. --holdout-half also reports WER on the half of the audio not used for fitting.
"""
import argparse
import difflib
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from asr_fuse import SpanModel, TrigramLM, fuse, span_features, spoken_text
from make_fuse_cache import catalog_words, context_words, read_cache, read_rows


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
    return row[-1]


def examples(audios, primary, secondary, gold, lm, context, app_words):
    X, Y = [], []
    for audio in audios:
        a, b, ref = spoken_text(primary[audio]), spoken_text(secondary[audio]), gold[audio]
        out, left = [], ["<s>", "<s>"]
        for op, i, j, u, v in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
            if op == "equal":
                out += a[i:j]
                left += a[i:j]
                continue
            ca, cb = a[i:j], b[u:v]
            right = a[j:j + 2] + (["</s>"] if j + 2 > len(a) else [])
            da, db = edit_distance(ref, out + ca + a[j:]), edit_distance(ref, out + cb + a[j:])
            if da != db:
                X.append(span_features(lm, left, ca, cb, right, context[audio], app_words[audio]))
                Y.append(float(db < da))
            pick = ca if da <= db else cb
            out += pick
            left += pick
    return np.array(X), np.array(Y)


def fit(X, Y, l2=1.0, steps=400):
    mean, scale = X.mean(0), X.std(0) + 1e-6
    mean[-1], scale[-1] = 0.0, 1.0  # the intercept stays 1
    Z = (X - mean) / scale
    w = np.zeros(Z.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-Z @ w))
        grad = Z.T @ (p - Y) / len(Y) + l2 * w / len(Y)
        hess = (Z * (p * (1 - p))[:, None]).T @ Z / len(Y) + np.eye(len(w)) * (l2 / len(Y) + 1e-6)
        w -= np.linalg.solve(hess, grad)
    return SpanModel(w.tolist(), mean.tolist(), scale.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--primary", required=True)
    ap.add_argument("--secondary", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.train_dir)
    _, primary = read_cache(a.primary)
    _, secondary = read_cache(a.secondary)
    rows = read_rows(root / "train.jsonl")
    app_text, catalog_text = catalog_words(json.loads((root / "train_tools_catalog.json").read_text(encoding="utf-8")))
    gold, context, app_words, fit_texts, final = {}, {}, {}, [], set()
    for row in rows:
        turns = [row] + [t for t in row.get("history", []) if t.get("role") == "user"]
        words, installed = context_words(row), {w for app in row["context"].get("installed_apps", {}) for w in app_text.get(app, ())}
        final.add(row["audio"])
        covered = any(t["audio"] in primary for t in turns)
        for t in turns:
            gold[t["audio"]] = spoken_text(t["audio_text"])
            context[t["audio"]], app_words[t["audio"]] = words, installed
            if not covered:
                fit_texts.append(t["audio_text"])
    lm = TrigramLM(fit_texts + catalog_text)
    audios = sorted(set(primary) & set(secondary) & set(gold))
    half = lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % 2
    tune, test = [x for x in audios if not half(x)], [x for x in audios if half(x)]
    def wer(items, texts):
        return sum(edit_distance(gold[x], spoken_text(texts[x])) for x in items) / sum(len(gold[x]) for x in items)
    check = fit(*examples(tune, primary, secondary, gold, lm, context, app_words))
    fused = {x: fuse(primary[x], secondary[x], lm, context[x], model=check, app_words=app_words[x]) for x in test}
    rule = {x: fuse(primary[x], secondary[x], TrigramLM(fit_texts), context[x]) for x in test}
    last = [x for x in test if x in final]
    report = {"n_fit_texts": len(fit_texts), "n_audio": len(audios), "test_primary": wer(test, primary), "test_secondary": wer(test, secondary), "test_rule": wer(test, rule),
              "test_model": wer(test, fused), "final_primary": wer(last, primary), "final_rule": wer(last, rule), "final_model": wer(last, fused)}
    X, Y = examples(audios, primary, secondary, gold, lm, context, app_words)
    model = fit(X, Y)
    report["n_spans_all"] = len(Y)
    Path(a.out).write_text(json.dumps({"weights": model.weights, "mean": model.mean, "scale": model.scale, "report": report}, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
