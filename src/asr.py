import json
import os
from pathlib import Path

from config import ASR_MODEL


class ASR:
    def __init__(self, model_name=ASR_MODEL, cache_path=None):
        self.cache = {}
        if cache_path and Path(cache_path).exists():
            with open(cache_path, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    self.cache[r["id"]] = r["text"]
        self.model_name = model_name
        self.pipe = None
        self.prompt = open(os.environ["ASR_PROMPT_FILE"], encoding="utf-8").read().strip() if os.environ.get("ASR_PROMPT_FILE") else None

    def _load(self):
        import torch
        from transformers import pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_name,
            torch_dtype=torch.float16,
            device=0 if torch.cuda.is_available() else -1,
            chunk_length_s=30,
        )

    def transcribe(self, wav_path, sample_id=None):
        if sample_id is not None and sample_id in self.cache:
            return self.cache[sample_id]
        if self.pipe is None:
            self._load()
        import soundfile as sf
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        kwargs = {"do_sample": False, "num_beams": 1, "language": "vi", "task": "transcribe"}
        if self.prompt:  # domain words as the previous context nudge Whisper towards them
            kwargs["prompt_ids"] = self.pipe.tokenizer.get_prompt_ids(self.prompt, return_tensors="pt").to(self.pipe.device)
        out = self.pipe({"raw": audio, "sampling_rate": sr}, generate_kwargs=kwargs)
        text = out["text"].strip()
        return text[len(self.prompt):].strip() if self.prompt and text.startswith(self.prompt) else text
