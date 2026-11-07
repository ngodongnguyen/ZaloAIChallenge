"""WER of Gipformer 1.5 (68M Zipformer transducer, MIT) on the 150 holdout clips asr_finetune.py reports on (WER 0.098 for asr_ft).

pip install sherpa-onnx huggingface_hub soundfile   (sherpa-onnx 1.13+ also needs sherpa-onnx-core)
python scripts/gipformer_wer.py --holdout cache/holdout_all.txt [--compare models/asr_ft]

Gold transcripts sometimes write numbers as digits ("127 triệu") while Gipformer speaks them out, which costs it several
words per number; the score is therefore also reported on clips whose gold has no digit. Hypotheses go to
out/asr_eval/gipformer1.5.jsonl ({id, ref, text}) and can be used as an --asr-cache."""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from asr_eval import edit_distance, norm  # noqa: E402

REPO = "g-group-ai-lab/gipformer1.5-68M-rnnt"


def make_recognizer(model_dir=None, int8=False, threads=4):
    import sherpa_onnx
    from huggingface_hub import hf_hub_download
    suffix = ".int8.onnx" if int8 else ".onnx"
    names = {k: k + suffix for k in ("encoder", "decoder", "joiner")}
    path = lambda f: str(Path(model_dir) / f) if model_dir else hf_hub_download(REPO, f)
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=path(names["encoder"]), decoder=path(names["decoder"]), joiner=path(names["joiner"]),
        tokens=path("tokens.txt"), num_threads=threads, sample_rate=16000, feature_dim=80,
        decoding_method="modified_beam_search")


def transcribe(rec, wav):
    audio, sr = sf.read(str(wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    stream = rec.create_stream()
    stream.accept_waveform(sr, audio)  # the files are 16 kHz (asr_finetune.py asserts it)
    rec.decode_stream(stream)
    return stream.result.text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--model-dir", default=None, help="local folder with the onnx files and tokens.txt instead of downloading")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--compare", default=None, help="a PhoWhisper-style model dir (e.g. models/asr_ft) run on the same clips")
    ap.add_argument("--out", default="out/asr_eval/gipformer1.5.jsonl")
    args = ap.parse_args()

    base = Path(args.root) / args.split
    hold = set(open(args.holdout).read().split())
    rows = [json.loads(l) for l in open(base / f"{args.split}.jsonl", encoding="utf-8")]
    val = [(r["id"], base / r["audio"], r["audio_text"]) for r in rows if r["id"] in hold]
    random.Random(0).shuffle(val)  # same selection as train/asr_finetune.py
    val = val[: args.n]

    rec = make_recognizer(args.model_dir, args.int8)
    other = None
    if args.compare:
        from asr import ASR
        other = ASR(args.compare)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    stats = {"gip": [0, 0], "gip_nodigit": [0, 0], "cmp": [0, 0], "cmp_nodigit": [0, 0], "best": [0, 0]}
    with open(args.out, "w", encoding="utf-8") as f:
        for sid, wav, ref in val:
            hyp = transcribe(rec, wav)
            f.write(json.dumps({"id": sid, "ref": ref, "text": hyp}, ensure_ascii=False) + "\n")
            r = norm(ref)
            eg = edit_distance(r, norm(hyp))
            nodigit = not re.search(r"\d", ref)
            stats["gip"][0] += eg
            stats["gip"][1] += len(r)
            if nodigit:
                stats["gip_nodigit"][0] += eg
                stats["gip_nodigit"][1] += len(r)
            if other:
                ec = edit_distance(r, norm(other.transcribe(wav)))
                stats["cmp"][0] += ec
                stats["cmp"][1] += len(r)
                if nodigit:
                    stats["cmp_nodigit"][0] += ec
                    stats["cmp_nodigit"][1] += len(r)
                stats["best"][0] += min(eg, ec)  # oracle that picks the better of the two per clip
                stats["best"][1] += len(r)
    w = lambda k: stats[k][0] / max(stats[k][1], 1)
    print(f"clips: {len(val)}")
    print(f"Gipformer 1.5: WER {w('gip'):.3f}  | clips without digits in gold: {w('gip_nodigit'):.3f}")
    if other:
        print(f"{args.compare}: WER {w('cmp'):.3f}  | clips without digits in gold: {w('cmp_nodigit'):.3f}")
        print(f"oracle (best of the two per clip): {w('best'):.3f}")
    print("hypotheses:", args.out)


if __name__ == "__main__":
    main()
