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
        if os.environ.get("LLM_4BIT") == "1" and dev == "cuda":  # probe/teach with models too big for bf16 (e.g. Qwen3-14B on 16 GB)
            from transformers import BitsAndBytesConfig
            q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=q, device_map={"": 0}).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(dev).eval()

    def _run(self, ids, max_new, stop, temperature=None):
        """Generate up to max_new tokens after ids, stopping once stop(new_text) is true; returns the new text."""
        torch = self.torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        start, tok = ids.shape[1], self.tok

        class Stop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                done = stop(tok.decode(input_ids[0, start:], skip_special_tokens=False))
                return torch.full((input_ids.shape[0],), done, dtype=torch.bool, device=input_ids.device)

        with torch.no_grad():
            out = self.model.generate(
                ids, max_new_tokens=max_new,
                **({"do_sample": True, "temperature": temperature, "top_p": 0.95} if temperature else {"do_sample": False}),
                pad_token_id=tok.eos_token_id, stopping_criteria=StoppingCriteriaList([Stop()]),
            )
        return tok.decode(out[0, start:], skip_special_tokens=False)

    def generate_thinking(self, messages, temperature=None):
        """LLM_THINK=1 (probe/teacher only): let a thinking model (Qwen3) reason first, capped at THINK_TOKENS;
        if the cap is hit the reasoning is closed by force. Only the JSON after </think> is returned."""
        budget = int(os.environ.get("THINK_TOKENS", 1024))
        text = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=True)
        ids = self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.model.device)
        t0 = time.time()
        after = lambda s: s.split("</think>", 1)[1] if "</think>" in s else None
        out = self._run(ids, budget + MAX_NEW_TOKENS, lambda s: after(s) is not None and json_closed(after(s)), temperature)
        if "</think>" not in out:  # budget spent while still thinking: close it and ask for the action
            out = out + "\n</think>\n\n"
            ids2 = self.tok(text + out, return_tensors="pt", add_special_tokens=False).input_ids.to(self.model.device)
            out += self._run(ids2, MAX_NEW_TOKENS, json_closed, temperature)
        if os.environ.get("LLM_TIMING"):
            print(f"[llm-think] prompt {ids.shape[1]} tok, out {len(out)} chars, {time.time() - t0:.1f}s", file=sys.stderr, flush=True)
        return after(out).replace("<|im_end|>", "").strip()

    def generate(self, messages, temperature=None):
        torch = self.torch
        if temperature is None and float(os.environ.get("LLM_TEMPERATURE", 0)) > 0:  # sampled runs for consistency filtering
            temperature = float(os.environ["LLM_TEMPERATURE"])
        if os.environ.get("LLM_THINK") == "1":  # official template with thinking on (not the no-think manual rendering)
            return self.generate_thinking(messages, temperature)
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
