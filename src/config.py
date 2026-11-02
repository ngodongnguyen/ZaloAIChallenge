import os
import random

SEED = 42
ASR_MODEL = os.environ.get("ASR_MODEL", "vinai/PhoWhisper-large")
LLM_MODEL = os.environ.get("LLM_MODEL", "models/lifemate-llm")  # merged SFT model
BASE_LLM = os.environ.get("BASE_LLM", "Qwen/Qwen3-4B-Instruct-2507")

MAX_STEPS = 7  # gold has at most 6 calls + the final done
MAX_NEW_TOKENS = 192  # the longest gold call is 304 chars (~130 tokens); a non-stopping model burned 320
TOOL_RESULT_CHARS = 1500
HISTORY_RESULT_CHARS = int(os.environ.get("HISTORY_RESULT_CHARS", 1500))  # was 500: cut off the options (e.g. flight ids) the gold call picks from; set 500 to run models trained before E13
MONEY_HINT = os.environ.get("MONEY_HINT", "1") == "1"  # E14: amounts restated in millions; models trained before E14 need MONEY_HINT=0
COND_HINT = os.environ.get("COND_HINT", "0") == "1"  # E18: code compares a spoken price threshold with the returned price; must match between training and inference
RESULT_DIGEST = os.environ.get("RESULT_DIGEST", "0") == "1"  # E15: per-floor device summary in tool results; must match between training and inference
TIME_BUDGET_S = float(os.environ.get("TIME_BUDGET_S", 26.0))  # hard limit per sample is 30s on the A30; local runs on a slower GPU raise it


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
