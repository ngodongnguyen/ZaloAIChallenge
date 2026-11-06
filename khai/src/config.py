import os
import random

SEED = 42
ASR_MODEL = os.environ.get("ASR_MODEL", "vinai/PhoWhisper-large")
LLM_MODEL = os.environ.get("LLM_MODEL", "models/lifemate-llm")  # merged SFT model
BASE_LLM = os.environ.get("BASE_LLM", "Qwen/Qwen3-4B-Instruct-2507")

MAX_STEPS = 8
MAX_NEW_TOKENS = 320  # room for the one-line thought before the action
TOOL_RESULT_CHARS = 1500
HISTORY_RESULT_CHARS = int(os.environ.get("HISTORY_RESULT_CHARS", 1500))  # was 500: cut off the options (e.g. flight ids) the gold call picks from; set 500 to run models trained before E13
MONEY_HINT = os.environ.get("MONEY_HINT", "0") == "1"  # E14: amounts restated in millions; off for the E16 models we submit (r018/r021/...), which were trained without it
RESULT_DIGEST = os.environ.get("RESULT_DIGEST", "0") == "1"  # E15: per-floor device summary in tool results; must match between training and inference
DATE_TABLE = os.environ.get("DATE_TABLE", "0") == "1"  # weekday calendar around current_time in the context; must match between training and inference
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
