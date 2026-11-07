"""Fine-tune Gipformer 1.5 (icefall zipformer transducer, MIT) on the labelled competition audio.

Needs its own environment (torch 2.8 + k2 + kaldifeat, see docs below) and an icefall clone; icefall's lhotse import is mocked
exactly as in gipformer/infer_pytorch.py. Same clips and holdout as train/asr_finetune.py (final turn + user history turns,
holdout ids excluded, WER on the same 150 holdout clips).

  python train/gipformer_ft.py --holdout cache/holdout_all.txt --out models/gipformer_ft            # train
  python train/gipformer_ft.py --holdout cache/holdout_all.txt --model-dir models/gipformer_ft \\
         --dump out/asr_gipformer_ft_holdout.jsonl                                                  # transcripts for infer.py --asr-cache

Targets are the gold text in spoken form (src/vinum.py: digits read out, punctuation dropped), upper-cased like the model's
own output. Loss, optimizer (ScaledAdam), schedule (Eden) and batch-count handling follow icefall zipformer/finetune.py."""
import argparse
import importlib.abc
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
import types
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from vinum import spoken  # noqa: E402

REPO = "g-group-ai-lab/gipformer1.5-68M-rnnt"
SR = 16000


# ---- lhotse mock, as in gipformer/infer_pytorch.py (icefall imports it, the model code never uses it) ----
class _MockModule(types.ModuleType):
    class _Dummy:
        def __init__(self, *a, **kw):
            pass

        def __call__(self, *a, **kw):
            return self

        def __getattr__(self, name):
            return type(self)()

    def __getattr__(self, name):
        return self._Dummy


class _LhotseFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "lhotse" or fullname.startswith("lhotse."):
            return importlib.util.spec_from_loader(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        mod = _MockModule(spec.name)
        mod.__path__ = []
        mod.__file__ = "<mocked>"
        mod.__version__ = "0.0.0"
        return mod

    def exec_module(self, module):
        pass


def setup_icefall(icefall_dir):
    sys.meta_path.insert(0, _LhotseFinder())
    for p in (icefall_dir, icefall_dir / "egs/librispeech/ASR", icefall_dir / "egs/librispeech/ASR/zipformer"):
        sys.path.insert(0, str(p))


def read_audio(path):
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert sr == SR, f"{path}: expected {SR} Hz, got {sr}"
    return audio


def collect(base, rows, skip_ids):
    """(wav, gold text) for the final turn and every user history turn of non-held-out samples (same as asr_finetune.py)."""
    clips = []
    for r in rows:
        if r["id"] in skip_ids:
            continue
        clips.append((base / r["audio"], r["audio_text"]))
        for h in r.get("history") or []:
            if h.get("role") == "user" and h.get("audio") and h.get("audio_text"):
                clips.append((base / h["audio"], h["audio_text"]))
    return clips


def spec_augment(f, rng):
    """f: (T, 80) log-mel. Two frequency masks and about one time mask per 1.2 s, filled with the mean."""
    f = f.clone()
    T = f.size(0)
    fill = f.mean()
    for _ in range(2):
        w = rng.randint(0, 27)
        s = rng.randint(0, 80 - w)
        f[:, s:s + w] = fill
    for _ in range(max(1, T // 120)):
        w = rng.randint(0, max(1, min(40, T // 5)))
        s = rng.randint(0, max(0, T - w))
        f[s:s + w, :] = fill
    return f


def make_batches(clips, durations, max_duration, rng):
    order = list(range(len(clips)))
    rng.shuffle(order)
    batches = []
    for i in range(0, len(order), 400):  # sort inside shuffled windows: little padding, still random batches
        chunk = sorted(order[i:i + 400], key=lambda k: durations[k])
        cur, total = [], 0.0
        for k in chunk:
            if cur and total + durations[k] > max_duration:
                batches.append(cur)
                cur, total = [], 0.0
            cur.append(k)
            total += durations[k]
        if cur:
            batches.append(cur)
    rng.shuffle(batches)
    return batches


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--icefall-dir", default=str(Path.home() / "icefall"))
    pre_args, _ = pre.parse_known_args()
    setup_icefall(Path(pre_args.icefall_dir))

    import k2
    import kaldifeat
    import sentencepiece as spm
    from beam_search import modified_beam_search
    from huggingface_hub import hf_hub_download
    from icefall.utils import create_grad_scaler, get_parameter_groups_with_lrs, torch_autocast
    from optim import Eden, ScaledAdam
    from torch.nn.utils.rnn import pad_sequence
    from train import add_model_arguments, get_model, get_params

    from asr_eval import edit_distance

    ap = argparse.ArgumentParser()
    ap.add_argument("--icefall-dir", default=str(Path.home() / "icefall"))
    ap.add_argument("--root", default="data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--model-dir", default=None, help="folder with model.pt, bpe.model, tokens.txt (default: download Gipformer 1.5)")
    ap.add_argument("--out", default="models/gipformer_ft")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--base-lr", type=float, default=0.0005, help="icefall trains from scratch at 0.0045; fine-tuning uses less")
    ap.add_argument("--max-duration", type=float, default=200.0, help="seconds of audio per batch")
    ap.add_argument("--eval-n", type=int, default=150)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--no-specaug", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: use only the first N training clips")
    ap.add_argument("--context-size", type=int, default=2)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--dump", default=None, help="skip training; write {id, text} for every holdout sample (--asr-cache for infer.py)")
    add_model_arguments(ap)
    args = ap.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = not args.no_fp16 and device.type == "cuda"

    files = {}
    for name in ("model.pt", "bpe.model", "tokens.txt"):
        files[name] = str(Path(args.model_dir) / name) if args.model_dir else hf_hub_download(REPO, name)

    params = get_params()
    params.update(vars(args))
    table = k2.SymbolTable.from_file(files["tokens.txt"])
    params.blank_id, params.unk_id = table["<blk>"], table["<unk>"]
    n = sum(1 for s in table.symbols if not s.startswith("#"))
    params.vocab_size = n - (1 if table["<blk>"] == 0 else 0) + 1
    sp = spm.SentencePieceProcessor()
    sp.load(files["bpe.model"])

    model = get_model(params)
    ckpt = torch.load(files["model.pt"], map_location="cpu", weights_only=False)
    print(model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt, strict=False))
    model.to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}  device: {device}  fp16: {use_fp16}", flush=True)

    opts = kaldifeat.FbankOptions()
    opts.device = device
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = SR
    opts.mel_opts.num_bins = 80
    opts.mel_opts.high_freq = -400
    fbank = kaldifeat.Fbank(opts)

    def features(waves, augment=False, rng=None):
        feats = fbank([torch.from_numpy(w).to(device) for w in waves])
        if augment:
            feats = [spec_augment(f, rng) for f in feats]
        lens = torch.tensor([f.size(0) for f in feats], device=device)
        return pad_sequence(feats, batch_first=True, padding_value=math.log(1e-10)), lens

    @torch.no_grad()
    def transcribe(waves):
        model.eval()
        feats, lens = features(waves)
        enc, enc_lens = model.forward_encoder(feats, lens)
        hyps = modified_beam_search(model=model, encoder_out=enc, encoder_out_lens=enc_lens, beam=args.beam)
        return [sp.decode(h) for h in hyps]

    base = Path(args.root) / args.split
    rows = [json.loads(l) for l in open(base / f"{args.split}.jsonl", encoding="utf-8")]
    hold = set(open(args.holdout).read().split())

    if args.dump:  # transcripts in the style the LLM saw in training: lower case, capital first letter, final dot
        Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
        todo = [(r["id"], base / r["audio"]) for r in rows if r["id"] in hold]
        with open(args.dump, "w", encoding="utf-8") as f:
            for i in range(0, len(todo), 8):
                chunk = todo[i:i + 8]
                for (sid, _), t in zip(chunk, transcribe([read_audio(p) for _, p in chunk])):
                    t = t.lower()
                    f.write(json.dumps({"id": sid, "text": t[:1].upper() + t[1:] + "."}, ensure_ascii=False) + "\n")
        print("wrote", args.dump)
        return

    val = [(base / r["audio"], r["audio_text"]) for r in rows if r["id"] in hold]
    random.Random(0).shuffle(val)  # same 150 clips as train/asr_finetune.py and scripts/gipformer_wer.py
    val = val[: args.eval_n]

    def wer():
        """(WER against the spoken-form gold, WER on the clips whose gold has no digit: comparable with asr_ft's 0.094)."""
        e = w = e2 = w2 = 0
        for i in range(0, len(val), 8):
            chunk = val[i:i + 8]
            for (_, ref), hyp in zip(chunk, transcribe([read_audio(p) for p, _ in chunk])):
                r, h = spoken(ref).split(), spoken(hyp).split()
                d = edit_distance(r, h)
                e, w = e + d, w + len(r)
                if not any(c.isdigit() for c in ref):
                    e2, w2 = e2 + d, w2 + len(r)
        return e / max(w, 1), e2 / max(w2, 1)

    train = collect(base, rows, hold)
    if args.limit:
        train = train[: args.limit]
    durations = [sf.info(str(p)).duration for p, _ in train]
    print(f"train clips: {len(train)} ({sum(durations) / 3600:.1f} h)  eval clips: {len(val)}", flush=True)
    a, b = wer()
    print(f"WER before: {a:.3f}  (no-digit clips: {b:.3f})", flush=True)

    optimizer = ScaledAdam(get_parameter_groups_with_lrs(model, lr=args.base_lr, include_names=True), lr=args.base_lr, clipping_scale=2.0)
    scheduler = Eden(optimizer, 100000.0, 100.0)  # decay stays negligible over a short fine-tune
    scaler = create_grad_scaler(enabled=use_fp16, init_scale=1.0)
    rng = random.Random(1)
    step, t0, best = 0, time.time(), 9.0

    def set_batch_count(count):  # finetune.py adds a large constant so every ScheduledFloat sits at its end value
        for name, module in model.named_modules():
            if hasattr(module, "batch_count"):
                module.batch_count = count
            if hasattr(module, "name"):
                module.name = name

    for ep in range(1, args.epochs + 1):
        model.train()
        batches = make_batches(train, durations, args.max_duration, rng)
        run = 0.0
        for bi, idx in enumerate(batches, 1):
            step += 1
            if bi % 10 == 1:
                set_batch_count(100000 + step)
            waves = [read_audio(train[k][0]) for k in idx]
            texts = [spoken(train[k][1]).upper() for k in idx]
            feats, lens = features(waves, augment=not args.no_specaug, rng=rng)
            y = k2.RaggedTensor(sp.encode(texts, out_type=int))
            with torch_autocast(enabled=use_fp16):
                simple_loss, pruned_loss = model(x=feats, x_lens=lens, y=y, prune_range=5, am_scale=0.0, lm_scale=0.25)[:2]
                loss = 0.5 * simple_loss + pruned_loss  # warm-up is over for a pretrained model: scales at their final values
            scaler.scale(loss).backward()
            scheduler.step_batch(step)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if use_fp16:  # finetune.py keeps the grad scale from collapsing
                cur = scaler._scale.item()
                if cur < 8.0 or (cur < 32.0 and step % 400 == 0):
                    scaler.update(cur * 2.0)
            frames = int((lens // 4).sum())
            run += loss.item() / max(frames, 1)
            if bi % 20 == 0:
                el = time.time() - t0
                print(f"epoch {ep} batch {bi}/{len(batches)} loss/frame {run / 20:.4f}  {el / 60:.1f}m", flush=True)
                run = 0.0
        a, b = wer()
        print(f"epoch {ep} WER: {a:.3f}  (no-digit clips: {b:.3f})", flush=True)
        if a < best:
            best = a
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict()}, out / "model.pt")
            shutil.copy(files["bpe.model"], out / "bpe.model")
            shutil.copy(files["tokens.txt"], out / "tokens.txt")
            print(f"saved best (WER {best:.3f}) to {out}", flush=True)


if __name__ == "__main__":
    main()
