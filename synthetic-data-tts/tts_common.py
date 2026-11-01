"""LifeMate development audio only. No training labels are generated here."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import wave


BASE = Path(__file__).resolve().parent
PROJECT = BASE
FLUX_VI_MODELS = ("flux-hoa-vi", "flux-minh-vi")
SAMPLE_TEXTS = [{'id': 'smoke_001', 'source_id': 'smoke_only', 'text': 'Xin chào, đây là câu thử giọng nói tiếng Việt.'}, {'id': 'smoke_002', 'source_id': 'smoke_only', 'text': 'Nhắn cho mẹ là tối nay con về muộn mười phút.'}, {'id': 'smoke_003', 'source_id': 'smoke_only', 'text': 'Đặt báo thức lúc sáu giờ ba mươi sáng mai.'}, {'id': 'smoke_004', 'source_id': 'smoke_only', 'text': 'Chuyển cho chị Huyền một trăm năm mươi nghìn đồng.'}, {'id': 'smoke_005', 'source_id': 'smoke_only', 'text': 'Bật đèn phòng khách rồi tắt điều hòa phòng ngủ.'}]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def read_input(path):
    """JSONL: unique string id, text; optional text_tts, source_id and turn."""
    rows, seen = [], set()
    with Path(path).open(encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}") from e
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_no}: expected an object")
            for key in ("id", "text"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"Line {line_no}: {key} must be non-empty text")
            if "text_tts" in row and (not isinstance(row["text_tts"], str) or not row["text_tts"].strip()):
                raise ValueError(f"Line {line_no}: invalid text_tts")
            if row["id"] in seen:
                raise ValueError(f"duplicate id: {row['id']}")
            seen.add(row["id"])
            rows.append(row)
    if not rows:
        raise ValueError("Input contains no rows")
    return rows


def inspect_wav(path):
    with wave.open(str(path), "rb") as f:
        channels, width, rate, frames = f.getnchannels(), f.getsampwidth(), f.getframerate(), f.getnframes()
        if (channels, width, rate) != (1, 2, 16000) or frames <= 0:
            raise ValueError("Expected non-empty mono PCM16 WAV at 16000 Hz")
        # Read all PCM to detect truncated files rather than trusting the header.
        data = f.readframes(frames)
        if len(data) != frames * width:
            raise ValueError("Truncated WAV")
    return {"sample_rate": rate, "channels": channels, "duration_s": frames / rate}


def ffmpeg_binary():
    binary = shutil.which("ffmpeg")
    if binary:
        return binary
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as e:
        raise RuntimeError("Install requirements.txt (imageio-ffmpeg) or ffmpeg") from e


def normalize_audio(source, destination):
    """Decode and resample; never pretend MP3/WebM bytes are WAV."""
    result = subprocess.run([ffmpeg_binary(), "-nostdin", "-v", "error", "-y", "-i", str(source),
                             "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(destination)],
                            capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError("ffmpeg could not decode audio; check the source file")
    inspect_wav(destination)


def append_jsonl(path, value):
    path = Path(path)
    needs_newline = False
    if path.exists() and path.stat().st_size:
        with path.open("rb") as existing:
            existing.seek(-1, os.SEEK_END)
            needs_newline = existing.read(1) != b"\n"
    with Path(path).open("a", encoding="utf-8") as f:
        if needs_newline:
            f.write("\n")
        f.write(json.dumps(value, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_manifest(path):
    records = {}
    if Path(path).exists():
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    records[r["key"]] = r
                except (json.JSONDecodeError, KeyError, TypeError):
                    # An interrupted final append is harmless; its audio is regenerated.
                    continue
    return records


def recording_audio(out, source_audio_sha256, text, recording_id):
    """Find this upload, including a resumed older upload in an append-only manifest."""
    records = load_manifest(Path(out) / "manifest.jsonl")
    for record in reversed(list(records.values())):
        if (record.get("provider") == "human" and record.get("id") == recording_id
                and record.get("text") == text
                and record.get("metadata", {}).get("source_audio_sha256") == source_audio_sha256):
            return record["audio"]
    raise ValueError("No successful audio found for this recording")


def generate(rows, out, *, provider, voice, model, synth, metadata=None, limit=None):
    """synth(text, raw_path) writes a decodable audio file. Manifest is append-only."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    audio_dir = out / "audio"
    audio_dir.mkdir(exist_ok=True)
    manifest = out / "manifest.jsonl"
    existing = load_manifest(manifest)
    stats = {"generated": 0, "skipped": 0, "failed": 0}
    metadata = metadata or {}
    for row in rows[:limit] if limit is not None else rows:
        identity = {"id": row["id"], "text": row["text"], "text_tts": row.get("text_tts", row["text"]),
                    "provider": provider, "voice": voice, "model": model, "metadata": metadata,
                    "output_format": "mono-pcm16-16000-v1"}
        for name in ("source_id", "turn", "split", "episode_id"):
            if name in row:
                identity[name] = row[name]
        key = digest(identity)
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", row["id"])[:48] or "sample"
        destination = audio_dir / f"{slug}-{key[:20]}.wav"
        previous = existing.get(key)
        if previous and destination.is_file():
            try:
                inspect_wav(destination)
                if file_hash(destination) == previous.get("audio_sha256"):
                    stats["skipped"] += 1
                    continue
            except (ValueError, wave.Error, EOFError, OSError):
                pass
        start = time.monotonic()
        try:
            with tempfile.TemporaryDirectory(dir=out, prefix=".tts-") as tmp:
                raw, normalized = Path(tmp) / "raw.audio", Path(tmp) / "normalized.wav"
                synth(identity["text_tts"], raw)
                normalize_audio(raw, normalized)
                audio_info = inspect_wav(normalized)
                os.replace(normalized, destination)
            record = {**identity, "key": key, "audio": destination.relative_to(out).as_posix(),
                      "audio_sha256": file_hash(destination), **audio_info,
                      "generation_s": round(time.monotonic() - start, 3),
                      "qa_status": "unreviewed"}
            for name in ("source_id", "turn", "split", "episode_id"):
                if name in row:
                    record[name] = row[name]
            append_jsonl(manifest, record)
            existing[key] = record
            stats["generated"] += 1
        except Exception as e:
            # Provider exceptions never include Authorization headers in our adapters.
            message = str(e)
            secret = os.environ.get("DEEPGRAM_API_KEY", "")
            if secret:
                message = message.replace(secret, "[REDACTED]")
            append_jsonl(out / "errors.jsonl", {"id": row["id"], "provider": provider,
                                              "voice": voice, "key": key,
                                              "error": type(e).__name__ + ": " + message})
            stats["failed"] += 1
    return stats


def vietnamese_models(data):
    found = []
    for model in data.get("tts", []):
        langs = model.get("languages", [])
        if isinstance(langs, str):
            langs = [langs]
        if any(str(lang).lower().replace("_", "-").split("-")[0] == "vi" for lang in langs):
            found.append(model)
    return found


def deepgram_models(key):
    import requests
    response = requests.get("https://api.deepgram.com/v1/models",
                            headers={"Authorization": "Token " + key}, timeout=(10, 30))
    if response.status_code != 200:
        raise RuntimeError(f"Deepgram models HTTP {response.status_code}; check key/permissions")
    # /v1/models lists Aura, not the Flux v2 catalog. Flux IDs are documented at:
    # https://developers.deepgram.com/docs/flux-tts/voices (checked 2026-10-30).
    flux = [{"name": "Hoa", "canonical_name": FLUX_VI_MODELS[0], "languages": ["vi"], "endpoint": "/v2/speak"},
            {"name": "Minh", "canonical_name": FLUX_VI_MODELS[1], "languages": ["vi"], "endpoint": "/v2/speak"}]
    return flux + vietnamese_models(response.json())


def deepgram_synth(text, destination, model, key, session=None):
    if session is None:
        import requests
        session = requests
    version = "v2" if model.startswith("flux-") else "v1"
    response = session.post(f"https://api.deepgram.com/{version}/speak",
                            headers={"Authorization": "Token " + key, "Content-Type": "application/json"},
                            params={"model": model, "encoding": "linear16", "container": "wav", "sample_rate": 16000},
                            json={"text": text}, timeout=(10, 90))
    if response.status_code != 200:
        # Do not log provider bodies, which can echo request details.
        raise RuntimeError(f"Deepgram synthesis HTTP {response.status_code}; check model/key/quota")
    if not response.content.startswith(b"RIFF"):
        raise RuntimeError("Deepgram returned non-WAV data")
    Path(destination).write_bytes(response.content)


def load_env(path=None):
    """Read just supported keys, never execute shell or print values."""
    p = Path(path) if path else PROJECT / "deepgram" / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in {"DEEPGRAM_API_KEY", "DEEPGRAM_MODELS"}:
            os.environ.setdefault(key, value.strip().strip("\"'"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("deepgram",):
        p = sub.add_parser(name)
        p.add_argument("--input", type=Path, help="JSONL id,text; omit to use built-in smoke texts")
        p.add_argument("--out", type=Path, default=PROJECT / name / "outputs")
        p.add_argument("--limit", type=int, default=2, help="Smoke default: 2 rows; use --all for all rows")
        p.add_argument("--all", action="store_true")
        p.add_argument("--dry-run", action="store_true", help="Validate inputs and show character count; no TTS calls")
        p.add_argument("--models", nargs="+", help="Exact IDs; default: flux-hoa-vi and flux-minh-vi")
    sub.add_parser("deepgram-voices")
    p = sub.add_parser("import-recording")
    p.add_argument("audio", type=Path)
    p.add_argument("--text", required=True, help="Exact words you actually recorded")
    p.add_argument("--id", default="my_voice_reference")
    p.add_argument("--out", type=Path, default=PROJECT / "colab" / "reference")
    args = parser.parse_args(argv)
    load_env()
    if args.command == "import-recording":
        source = args.audio.resolve()
        def copy_recording(text, destination):
            shutil.copyfile(source, destination)
        result = generate([{"id": args.id, "text": args.text}], args.out, provider="human",
                          voice="my_voice", model="microphone", synth=copy_recording,
                          metadata={"source_audio_sha256": file_hash(source)})
        if not result["failed"]:
            result["audio"] = recording_audio(args.out, file_hash(source), args.text, args.id)
        print(json.dumps(result, ensure_ascii=False))
        return int(result["failed"] > 0)
    if args.command != "deepgram-voices":
        if args.limit <= 0:
            parser.error("--limit must be positive")
        rows = read_input(args.input) if args.input else SAMPLE_TEXTS
        selected = rows if args.all else rows[:args.limit]
        if args.dry_run:
            print(json.dumps({"rows": len(selected), "characters_per_voice": sum(len(r.get("text_tts", r["text"])) for r in selected),
                              "provider": args.command, "network_calls": 0}, ensure_ascii=False))
            return 0
    key = os.environ.get("DEEPGRAM_API_KEY")
    if not key:
        parser.error("Set DEEPGRAM_API_KEY in environment or local .env")
    if args.command == "deepgram-voices":
        print(json.dumps(deepgram_models(key), ensure_ascii=False, indent=2))
        return 0
    models = args.models or os.environ.get("DEEPGRAM_MODELS", "").split(",")
    models = list(dict.fromkeys(x.strip() for x in models if x.strip()))
    if not models:
        models = list(FLUX_VI_MODELS)
    if not models:
        raise RuntimeError("No Vietnamese TTS model returned. Use --models with the exact IDs from your console; no English fallback.")
    failed = 0
    for model in models:
        result = generate(selected, args.out, provider="deepgram", voice=model, model=model,
                          synth=lambda text, dst, model=model: deepgram_synth(text, dst, model, key))
        print(json.dumps({"model": model, **result}))
        failed += result["failed"]
    return int(failed > 0)


def cli(argv=None):
    try:
        return main(argv)
    except (ValueError, RuntimeError, OSError) as error:
        # Short errors; do not dump environments or request objects.
        print(f"Error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
