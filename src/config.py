import os
import random

SEED = 42
ASR_MODEL = os.environ.get("ASR_MODEL", "vinai/PhoWhisper-large")
LLM_MODEL = os.environ.get("LLM_MODEL", "models/lifemate-llm")  # merged SFT model
BASE_LLM = os.environ.get("BASE_LLM", "Qwen/Qwen2.5-3B-Instruct")

MAX_STEPS = 8
MAX_NEW_TOKENS = 320  # room for the one-line thought before the action
TOOL_RESULT_CHARS = 1500
HISTORY_RESULT_CHARS = int(os.environ.get("HISTORY_RESULT_CHARS", 1500))  # was 500: cut off the options (e.g. flight ids) the gold call picks from; set 500 to run models trained before E13
TIME_BUDGET_S = 26.0  # hard limit per sample is 30s


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass
