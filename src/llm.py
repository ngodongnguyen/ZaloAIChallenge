from chat import needs_manual, prompt_text
from config import LLM_MODEL, MAX_NEW_TOKENS


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

    def generate(self, messages):
        torch = self.torch
        if needs_manual(self.tok):  # Qwen3.5: same rendering as in training (see chat.py)
            ids = self.tok(prompt_text(messages), return_tensors="pt", add_special_tokens=False).input_ids
        else:
            ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
            if hasattr(ids, "input_ids"):
                ids = ids.input_ids
        ids = ids.to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tok.eos_token_id,
            )
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
