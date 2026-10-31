"""Prompt rendering for chat models whose template treats <think> blocks by position (Qwen3.5).

Qwen3.5's template emits "<think>\\n\\n</think>\\n\\n" only for assistant turns after the last user message, while the generation
prompt with enable_thinking=False always ends with it. In our loop every tool result is a user message, so training text and
inference prompts disagree and the label offsets (computed from template prefixes) shift. Here every assistant turn is
rendered the same way in training and inference: header + empty think block, then the answer.
"""
THINK = "<think>\n\n</think>\n\n"


def needs_manual(tok):
    tpl = getattr(tok, "chat_template", None) or ""
    return "enable_thinking" in tpl and "<think>" in tpl


def segments(messages, generation_prompt=False):
    """[(text, supervised)]: only the answer + <|im_end|> of an assistant turn is supervised."""
    segs = []
    for m in messages:
        if m["role"] == "assistant":
            segs.append((f"<|im_start|>assistant\n{THINK}", False))
            segs.append((f"{m['content']}<|im_end|>\n", True))
        else:
            segs.append((f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n", False))
    if generation_prompt:
        segs.append((f"<|im_start|>assistant\n{THINK}", False))
    return segs


def prompt_text(messages):
    return "".join(t for t, _ in segments(messages, generation_prompt=True))


def encode_with_labels(tok, messages):
    ids, labels = [], []
    for text, supervised in segments(messages):
        t = tok(text, add_special_tokens=False)["input_ids"]
        ids += t
        labels += t if supervised else [-100] * len(t)
    return ids, labels
