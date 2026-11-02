"""Experiment configs: load, validate, and hash.

A config is a JSON object. `hypothesis` and `parent` are required metadata and do not change the
config hash. Every other key is an input that affects results, so it is part of the hash.
Unknown keys are rejected so a typo cannot silently fall back to a default.
"""
import hashlib
import json
from pathlib import Path

ALLOWED_KEYS = {
    "hypothesis", "parent",          # metadata, excluded from config_hash
    "llm_dtype",                     # "bf16" | "fp16"
    "llm_device",                    # "1gpu" | "2gpu"
    "n_samples",                     # int, number of dev samples
    "split",                         # name of a split file under splits/
    "max_steps",                     # int, agent loop steps
    "time_budget_s",                 # float, agent time budget per sample
    "prompt_variant",                # str, id of a prompt template
    "sft",                           # dict or null: n_samples, epochs, lr, max_len, seed (LoRA SFT before eval)
}
META_KEYS = {"hypothesis", "parent"}
ALLOWED_DTYPES = {"bf16", "fp16"}
ALLOWED_DEVICES = {"1gpu", "2gpu"}


class ConfigError(ValueError):
    pass


def load_config(path):
    """Read a config file. Strips a UTF-8 BOM if present (files written by some editors have one)."""
    with open(path, encoding="utf-8-sig") as f:
        return validate_config(json.load(f))


def validate_config(cfg):
    if not isinstance(cfg, dict):
        raise ConfigError("config must be a JSON object")
    unknown = set(cfg) - ALLOWED_KEYS
    if unknown:
        raise ConfigError(f"unknown keys: {sorted(unknown)}")
    hyp = cfg.get("hypothesis")
    if not isinstance(hyp, str) or len(hyp.strip()) < 10:
        raise ConfigError("hypothesis is required and must be at least 10 characters")
    if not isinstance(cfg.get("parent"), (str, type(None))):
        raise ConfigError("parent must be a string run id or null")
    if "parent" not in cfg:
        raise ConfigError("parent is required (use null for a root config)")
    if "llm_dtype" in cfg and cfg["llm_dtype"] not in ALLOWED_DTYPES:
        raise ConfigError(f"llm_dtype must be one of {sorted(ALLOWED_DTYPES)}")
    if "llm_device" in cfg and cfg["llm_device"] not in ALLOWED_DEVICES:
        raise ConfigError(f"llm_device must be one of {sorted(ALLOWED_DEVICES)}")
    if "n_samples" in cfg and (not isinstance(cfg["n_samples"], int) or cfg["n_samples"] <= 0):
        raise ConfigError("n_samples must be a positive integer")
    return cfg


def config_hash(cfg):
    """Stable hash over result-affecting inputs only (metadata excluded, keys sorted)."""
    payload = {k: v for k, v in sorted(cfg.items()) if k not in META_KEYS}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def code_hash(files):
    """Hash of the code files that decide results. `files` maps a label to a path (bytes are hashed)."""
    h = hashlib.sha256()
    for label in sorted(files):
        h.update(label.encode("utf-8"))
        h.update(Path(files[label]).read_bytes())
    return h.hexdigest()[:16]
