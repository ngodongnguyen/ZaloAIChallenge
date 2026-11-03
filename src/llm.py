import os
import sys
import time

from chat import needs_manual, prompt_text
from config import LLM_MODEL, MAX_NEW_TOKENS

PREFILL = os.environ.get("LLM_PREFILL", "")  # text the assistant turn starts with (e.g. "{"), to force JSON from models not tuned for it


def json_closed(text):
    """True once the first top-level {...} object in text is complete (string- and escape-aware)."""
    depth, in_str, esc, started = 0, False, False, False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth, started = depth + 1, True
        elif ch == "}":
            depth -= 1
            if started and depth == 0:
                return True
    return False


class LLM:
    def __init__(self, model_path=LLM_MODEL):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_path)
        if torch.cuda.is_available():
            dev = "cuda"
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16  # T4 has no bf16
        elif torch.backends.mps.is_available():
            dev, dtype = "mps", torch.float16
        else:
            dev, dtype = "cpu", torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(dev).eval()

    def generate(self, messages, temperature=None):
        torch = self.torch
        if needs_manual(self.tok):  # Qwen3.5: same rendering as in training (see chat.py)
            ids = self.tok(prompt_text(messages), return_tensors="pt", add_special_tokens=False).input_ids
        else:
            tpl = getattr(self.tok, "chat_template", None) or ""
            kw = {"enable_thinking": False} if "enable_thinking" in tpl else {}
            text = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **kw) + PREFILL
            ids = self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids
        ids = ids.to(self.model.device)
        t0 = time.time()
        from transformers import StoppingCriteria, StoppingCriteriaList

        start, tok = ids.shape[1], self.tok

        class JsonDone(StoppingCriteria):  # the loop only uses the first JSON object; stop when it is complete
            def __call__(self, input_ids, scores, **kwargs):
                done = json_closed(PREFILL + tok.decode(input_ids[0, start:], skip_special_tokens=True))
                return torch.full((input_ids.shape[0],), done, dtype=torch.bool, device=input_ids.device)

        with torch.no_grad():
            out = self.model.generate(
                ids,
                max_new_tokens=MAX_NEW_TOKENS,
                **({"do_sample": True, "temperature": temperature, "top_p": 0.95} if temperature else {"do_sample": False}),
                pad_token_id=self.tok.eos_token_id,
                stopping_criteria=StoppingCriteriaList([JsonDone()]),
            )
        if os.environ.get("LLM_TIMING"):  # per-step prompt/new tokens and seconds, to find what makes a sample slow
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print(f"[llm] prompt {ids.shape[1]} tok, new {out.shape[1] - ids.shape[1]} tok, {time.time() - t0:.1f}s", file=sys.stderr, flush=True)
        return (PREFILL + self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)).strip()
