"""LoRA fine-tune PhoWhisper on labelled train audio (final turn + history user turns), then merge and save.

python train/asr_finetune.py --root data --base models/asr --out models/asr_ft --holdout cache/holdout_all.txt
Holdout ids are excluded from training and used to report WER before/after on their final-turn audio.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from asr_eval import edit_distance, norm  # noqa: E402
from config import seed_everything  # noqa: E402

SR = 16000


def read_audio(path):
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert sr == SR, f"{path}: expected {SR} Hz, got {sr}"
    return audio


def collect(base, rows, skip_ids):
    """(wav path, text) for the final turn and every user history turn of non-held-out samples."""
    clips = []
    for r in rows:
        if r["id"] in skip_ids:
            continue
        clips.append((base / r["audio"], r["audio_text"]))
        for h in r.get("history") or []:
            if h.get("role") == "user" and h.get("audio") and h.get("audio_text"):
                clips.append((base / h["audio"], h["audio_text"]))
    return clips


class Clips(Dataset):
    def __init__(self, clips):
        self.clips = clips

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        path, text = self.clips[i]
        return read_audio(path), text


def make_collate(fe, tok):
    def collate(batch):
        feats = fe([a for a, _ in batch], sampling_rate=SR, return_tensors="pt").input_features
        ids = [tok(t).input_ids for _, t in batch]  # <sot><vi><transcribe><notimestamps> text <eot>
        m = max(len(x) for x in ids) - 1
        dec_in = torch.full((len(ids), m), tok.eos_token_id)
        labels = torch.full((len(ids), m), -100)
        for k, x in enumerate(ids):
            dec_in[k, : len(x) - 1] = torch.tensor(x[:-1])
            labels[k, : len(x) - 1] = torch.tensor(x[1:])
            labels[k, :3] = -100  # language / task / notimestamps are forced at inference, not learned
        return feats, dec_in, labels

    return collate


@torch.no_grad()
def wer(model, fe, tok, clips, device, bs=16):
    model.eval()
    err = n = 0
    for i in range(0, len(clips), bs):
        chunk = clips[i : i + bs]
        feats = fe([read_audio(p) for p, _ in chunk], sampling_rate=SR, return_tensors="pt").input_features
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(feats.to(device), language="vi", task="transcribe", num_beams=1,
                                 do_sample=False, max_new_tokens=200)
        for hyp, (_, ref) in zip(tok.batch_decode(out, skip_special_tokens=True), chunk):
            r, h = norm(ref), norm(hyp)
            err += edit_distance(r, h)
            n += len(r)
    return err / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--base", default="models/asr")
    ap.add_argument("--out", default="models/asr_ft")
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--eval-n", type=int, default=150)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only the first N training clips")
    args = ap.parse_args()

    seed_everything()
    device = "cuda"
    data = Path(args.root) / args.split
    rows = [json.loads(l) for l in open(data / f"{args.split}.jsonl", encoding="utf-8")]
    hold = set(open(args.holdout).read().split())
    train = collect(data, rows, hold)
    if args.limit:
        train = train[: args.limit]
    val = [(data / r["audio"], r["audio_text"]) for r in rows if r["id"] in hold]
    random.Random(0).shuffle(val)
    val = val[: args.eval_n]
    print(f"train clips: {len(train)}  eval clips: {len(val)}", flush=True)

    fe = WhisperFeatureExtractor.from_pretrained(args.base)
    tok = WhisperTokenizer.from_pretrained(args.base)
    tok.set_prefix_tokens(language="vietnamese", task="transcribe", predict_timestamps=False)
    model = WhisperForConditionalGeneration.from_pretrained(args.base, dtype=torch.bfloat16).to(device)
    model.config.use_cache = False

    print(f"WER before: {wer(model, fe, tok, val, device):.3f}", flush=True)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]))
    model.print_trainable_parameters()

    loader = DataLoader(Clips(train), batch_size=args.batch, shuffle=True, num_workers=4,
                        collate_fn=make_collate(fe, tok), generator=torch.Generator().manual_seed(42), drop_last=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    total = len(loader) * args.epochs // args.accum
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + torch.cos(torch.tensor((s - warm) / max(1, total - warm) * 3.14159265)).item()))

    step = 0
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        run = 0.0
        for i, (feats, dec_in, labels) in enumerate(loader, 1):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_features=feats.to(device), decoder_input_ids=dec_in.to(device)).logits
            loss = F.cross_entropy(logits.float().flatten(0, 1), labels.to(device).flatten(), ignore_index=-100)
            (loss / args.accum).backward()
            run += loss.item()
            if i % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    el = time.time() - t0
                    print(f"epoch {ep} step {step}/{total} loss {run / (20 * args.accum):.4f}  "
                          f"elapsed {el / 60:.1f}m  {el / step:.1f}s/step  eta {el / step * (total - step) / 60:.1f}m", flush=True)
                    run = 0.0

    model = model.merge_and_unload()
    model.config.use_cache = True
    print(f"WER after: {wer(model, fe, tok, val, device):.3f}", flush=True)
    model.to(torch.float16).save_pretrained(args.out)
    fe.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
