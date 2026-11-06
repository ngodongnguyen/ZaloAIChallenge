"""Settings shared by the SFT baseline and the research loop, so both runs use the same recipe."""

# max_len 6000 keeps about 95% of sampled trajectories (median 3.7k tokens, p90 5.5k). Longer ones are dropped.
SFT_CONFIG = {"n_samples": 250, "epochs": 1, "lr": 2e-4, "max_len": 6000, "seed": 42,
              "use_asr_text": True, "asr_frac": 0.7}
DEV_DTYPE = "fp16"
DEV_DEVICE = "1gpu"


def eval_params(split, out_name):
    """Kernel parameters for one run. Dev is evaluated; dev and holdout ids are excluded from SFT."""
    return {
        "ids": split["dev"],
        "exclude_ids": sorted(set(split["dev"]) | set(split["holdout"])),
        "llm_dtype": DEV_DTYPE,
        "llm_device": DEV_DEVICE,
        "sft": SFT_CONFIG,
        "out_name": out_name,
    }
