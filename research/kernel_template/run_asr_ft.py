"""Kaggle kernel: LoRA fine-tune PhoWhisper-large on the competition audio, then compare dev WER.

PARAMS is written by the harness. Trains on train samples outside PARAMS["exclude_ids"] (dev and holdout),
reports dev WER for the base model and the tuned model, and saves the adapter to /kaggle/working/out/asr_lora.
The merged model is written to /tmp only (used for the dev WER); rebuild it from the adapter when needed.
"""
import gc
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import zipfile
from pathlib import Path

PARAMS = __PARAMS__

CODE_IN = "__CODE_IN__"
DATA_ROOT = "__DATA_ROOT__"
REPO = "/tmp/repo"
MODELS = "/tmp/models"
OUT_DIR = Path("/kaggle/working/out")


def unpack_code():
    os.makedirs(REPO, exist_ok=True)
    for name in os.listdir(CODE_IN):
        src = os.path.join(CODE_IN, name)
        if name.endswith(".zip"):
            with zipfile.ZipFile(src) as z:
                z.extractall(os.path.join(REPO, name[:-4]))
        elif os.path.isdir(src):
            shutil.copytree(src, os.path.join(REPO, name), dirs_exist_ok=True)
        elif os.path.isfile(src):
            shutil.copy(src, os.path.join(REPO, name))


def install_and_download():
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.46,<5", "accelerate>=0.34",
                    "soundfile>=0.12", "peft>=0.13", "huggingface_hub"], check=True)
    from huggingface_hub import snapshot_download
    snapshot_download("vinai/PhoWhisper-large", local_dir=f"{MODELS}/asr")


def resolve_data_root():
    global DATA_ROOT
    if os.path.exists(os.path.join(DATA_ROOT, "train", "train_tools_catalog.json")):
        print("data root:", DATA_ROOT, flush=True)
        return
    for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
        if os.path.basename(dirpath) == "train" and "train_tools_catalog.json" in filenames:
            DATA_ROOT = os.path.dirname(dirpath)
            print("data root resolved to:", DATA_ROOT, flush=True)
            return
    raise FileNotFoundError("train/train_tools_catalog.json not found under /kaggle/input")


def last_text(x):
    if isinstance(x, list):
        return x[-1] if x else ""
    return x or ""


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


def wer(refs, hyps):
    errs = sum(edit_distance(norm(r), norm(h)) for r, h in zip(refs, hyps))
    return errs / max(1, sum(len(norm(r)) for r in refs))


def load_audio(path):
    import soundfile as sf
    import torch
    import torchaudio
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = torchaudio.functional.resample(torch.from_numpy(audio), sr, 16000).numpy()
    return audio


def transcribe_all(model, proc, items, split_dir, batch=8):
    import torch
    model.eval()
    hyps = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        feats = proc.feature_extractor([load_audio(split_dir / it["audio"]) for it in chunk],
                                       sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out = model.generate(feats.to("cuda", torch.float16), language="vi", task="transcribe",
                                 do_sample=False, num_beams=1, max_new_tokens=200)
        hyps += [t.strip() for t in proc.batch_decode(out, skip_special_tokens=True)]
        print(f"transcribed {len(hyps)}/{len(items)}", flush=True)
    return hyps


def main():
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("stage: start", PARAMS["out_name"], flush=True)
    unpack_code()
    install_and_download()
    resolve_data_root()
    print("stage: ready", flush=True)

    # Import only after pip has settled the transformers version; importing earlier mixes two versions.
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    split_dir = Path(DATA_ROOT) / "train"
    by_id = {}
    with open(split_dir / "train.jsonl", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                by_id[r["id"]] = r
    excluded = set(PARAMS["exclude_ids"])
    dev = [by_id[i] for i in PARAMS["ids"]]
    train = [r for i, r in sorted(by_id.items()) if i not in excluded]
    if PARAMS.get("history_clips"):  # user history turns carry their own audio and gold text
        train += [h for r in train[:] for h in r.get("history") or []
                  if h.get("role") == "user" and h.get("audio") and h.get("audio_text")]
    print(f"asr-ft: {len(train)} train clips, {len(dev)} dev clips", flush=True)

    cfg = PARAMS["asr_ft"]
    proc = WhisperProcessor.from_pretrained(f"{MODELS}/asr")
    proc.tokenizer.set_prefix_tokens(language="vi", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(f"{MODELS}/asr", torch_dtype=torch.float16).to("cuda")

    refs = [last_text(it["audio_text"]) for it in dev]
    base_hyps = transcribe_all(model, proc, dev, split_dir)
    wer_base = wer(refs, base_hyps)
    print(f"dev WER base: {wer_base:.4f}", flush=True)

    model.config.use_cache = False
    # Non-reentrant checkpointing: the encoder input (log-mel features) needs no grad, and the reentrant
    # variant would then give the encoder LoRA layers no gradient at all.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        r=cfg["r"], lora_alpha=2 * cfg["r"], lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # fp32 LoRA weights under the fp16 grad scaler
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()
    bs, accum, epochs = cfg["batch"], cfg["accum"], cfg["epochs"]
    steps_total = epochs * (len(train) // (bs * accum))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, int(0.05 * steps_total))) *
        0.5 * (1 + __import__("math").cos(__import__("math").pi * min(1.0, s / max(1, steps_total)))))
    rng = random.Random(cfg["seed"])
    tok = proc.tokenizer
    step = 0
    t_train = time.time()
    model.train()
    for ep in range(epochs):
        order = train[:]
        rng.shuffle(order)
        for b in range(0, len(order) - bs + 1, bs):
            chunk = order[b:b + bs]
            feats = proc.feature_extractor([load_audio(split_dir / it["audio"]) for it in chunk],
                                           sampling_rate=16000, return_tensors="pt").input_features
            ids = [tok(last_text(it["audio_text"])).input_ids for it in chunk]
            m = max(len(x) for x in ids)
            labels = torch.full((len(ids), m), -100)
            for k, x in enumerate(ids):
                labels[k, :len(x)] = torch.tensor(x)
            dec_in = labels[:, :-1].clone()
            dec_in[dec_in == -100] = tok.pad_token_id
            tgt = labels[:, 1:]
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(input_features=feats.to("cuda", torch.float16),
                            decoder_input_ids=dec_in.to("cuda"))
            loss = torch.nn.functional.cross_entropy(
                out.logits.float().reshape(-1, out.logits.size(-1)), tgt.reshape(-1).to("cuda"),
                ignore_index=-100)
            scaler.scale(loss / accum).backward()
            if ((b // bs) + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                print(f"epoch {ep} step {step}/{steps_total} loss {loss.item():.4f} "
                      f"elapsed {round(time.time() - t_train)}s", flush=True)
    print("stage: asr training done", flush=True)

    model.save_pretrained(OUT_DIR / "asr_lora")
    model = model.merge_and_unload().half().eval()
    model.config.use_cache = True
    gc.collect()
    ft_hyps = transcribe_all(model, proc, dev, split_dir)
    wer_ft = wer(refs, ft_hyps)
    print(f"dev WER base {wer_base:.4f} -> tuned {wer_ft:.4f}", flush=True)
    with open(OUT_DIR / f"{PARAMS['out_name']}.json", "w", encoding="utf-8") as f:
        json.dump({"wer_base": wer_base, "wer_tuned": wer_ft, "train_clips": len(train),
                   "seconds": round(time.time() - t_start, 1),
                   "dev": [{"id": it["id"], "ref": r, "base": b, "tuned": t}
                           for it, r, b, t in zip(dev, refs, base_hyps, ft_hyps)]},
                  f, ensure_ascii=False)
    print(f"stage: done in {round(time.time() - t_start, 1)}s", flush=True)


if __name__ == "__main__":
    main()
