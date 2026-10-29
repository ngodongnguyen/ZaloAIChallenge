from config import LLM_MODEL, MAX_NEW_TOKENS


def no_gqa_sdpa():
    """transformers >= 4.5x passes enable_gqa=True to sdpa when there is no mask; on GPUs without flash attention
    (T4) that falls back to the math kernel and builds the full heads x seq x seq matrix (OOM on Qwen3-4B).
    Repeating k/v instead keeps the memory-efficient kernel. No-op on transformers versions without this switch."""
    try:
        from transformers.integrations import sdpa_attention
    except ImportError:
        return
    if hasattr(sdpa_attention, "use_gqa_in_sdpa"):
        sdpa_attention.use_gqa_in_sdpa = lambda *args, **kwargs: False


class LLM:
    def __init__(self, model_path=LLM_MODEL):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        no_gqa_sdpa()
        self.tok = AutoTokenizer.from_pretrained(model_path)
        if torch.cuda.is_available():
            dev = "cuda"
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16  # T4 has no bf16
        elif torch.backends.mps.is_available():
            dev, dtype = "mps", torch.float16
        else:
            dev, dtype = "cpu", torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, attn_implementation="sdpa").to(dev).eval()

    def generate(self, messages):
        torch = self.torch
        ids = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
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
