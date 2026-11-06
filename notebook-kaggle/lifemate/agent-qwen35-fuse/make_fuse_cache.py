"""lm-A-054: fuse two finished ASR caches of public_test into one cache the agent reads like any single transcript.

    python make_fuse_cache.py --data-root dataset/lifemate/train+public_test --primary a.jsonl --secondary b.jsonl --out dir

The language model counts train audio_text only. Context words are what the agent has at run time; public labels do not exist.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from asr_fuse import SpanModel, TrigramLM, fuse, spoken_text


def read_cache(path):
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return records[0]["_header"]["run_hash"], {r["id"]: r["text"] for r in records[1:]}


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def strings(x, out):
    if isinstance(x, dict):
        for v in x.values():
            strings(v, out)
    elif isinstance(x, list):
        for v in x:
            strings(v, out)
    elif isinstance(x, str) and len(x) <= 80:
        out.append(x)
    return out


def context_words(row):
    texts = strings((row.get("context") or {}).get("user_profile"), [])
    for turn in row.get("history", []):
        if turn.get("role") != "user":
            texts += [turn["text"]] if turn.get("text") else []
            strings(turn.get("tool_results"), texts)
    return {w for text in texts for w in spoken_text(text)}


def catalog_words(catalog):
    """{app id: its spoken words}, and every catalog text for the language model."""
    words, texts = {}, []
    for app in catalog["apps"]:
        own = [app.get("name_vi", ""), app.get("description", "")]
        for tool in app["tools"]:
            own.append(tool.get("description", ""))
            own += [p.get("description", "") for p in (tool.get("parameters") or {}).get("properties", {}).values()]
        own = [t for t in own if t]
        words[app["id"]] = {w for t in own for w in spoken_text(t)}
        texts += own
    return words, texts


def train_texts(rows):
    return [t.get("audio_text") for r in rows for t in [r] + [h for h in r.get("history", []) if h.get("role") == "user"] if t.get("audio_text")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--primary", required=True)
    ap.add_argument("--secondary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", help="fuse_model.json from train_fuse_model.py; without it the fixed rule is used")
    a = ap.parse_args()
    root, out = Path(a.data_root), Path(a.out)
    primary_id, primary = read_cache(a.primary)
    secondary_id, secondary = read_cache(a.secondary)
    public = read_rows(root / "public_test/public_test.jsonl")
    model = None
    if a.model:
        saved = json.loads(Path(a.model).read_text(encoding="utf-8"))
        model = SpanModel(saved["weights"], saved["mean"], saved["scale"])
    app_text, catalog_text = catalog_words(json.loads((root / "public_test/public_test_tools_catalog.json").read_text(encoding="utf-8")))
    lm = TrigramLM(train_texts(read_rows(root / "train/train.jsonl")) + (catalog_text if model else []))
    context, final, installed = {}, set(), {}
    for row in public:
        words = context_words(row)
        final.add(row["audio"])
        for audio in [row["audio"]] + [t["audio"] for t in row.get("history", []) if t.get("role") == "user"]:
            context[audio] = words
            installed[audio] = {w for app in row["context"].get("installed_apps", {}) for w in app_text.get(app, ())}
    if not (set(primary) == set(secondary) == set(context)) or len(context) != 1933:
        raise SystemExit("caches do not cover the same 1933 public audio")
    texts = {k: fuse(primary[k], secondary[k], lm, context[k], model=model, app_words=installed[k]) for k in sorted(primary)}
    if any(not t.strip() for t in texts.values()):
        raise SystemExit("empty fused transcript")
    code = hashlib.sha256((Path(__file__).with_name("asr_fuse.py")).read_bytes()).hexdigest()
    model_sha = hashlib.sha256(Path(a.model).read_bytes()).hexdigest() if a.model else None
    identity = "lm-A-054-fuse-" + hashlib.sha256(json.dumps([primary_id, secondary_id, code] + ([model_sha] if model_sha else [])).encode()).hexdigest()[:16]
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "asr_public_test_whisper_full.shard0.jsonl"
    cache.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in [{"_header": {"run_hash": identity}}] + [
        {"id": k, "status": "completed", "text": v} for k, v in texts.items()]), encoding="utf-8")
    report = {"identity": identity, "n_audio": len(texts), "primary": primary_id, "secondary": secondary_id, "fuse_code_sha256": code, "model_sha256": model_sha,
              "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
              "changed_all": sum(texts[k] != primary[k] for k in texts),
              "changed_final": sum(texts[k] != primary[k] for k in final)}
    (out / "fuse_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
