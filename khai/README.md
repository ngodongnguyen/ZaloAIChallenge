# khai

Khai's branch work, kept in one folder so it does not touch the main pipeline.

- `src/`, `train/`, `scripts/`, `Dockerfile`, `requirements.txt`: the pipeline as it ran for our best leaderboard
  submissions (Qwen3-4B-Instruct-2507 + LoRA, PhoWhisper-large + LoRA). `train/dpo.py` is step-wise DPO (did not
  beat SFT). `scripts/build_models.py` builds the merged models for the Docker image.
- `research/`: evaluation harness (scorer copy, splits, paired bootstrap), Kaggle kernel templates, experiment
  loop scripts, synthetic data for unseen apps, tests.
  - `research/loop/make_soup.py`: averages several LoRA adapters into one adapter (model soup).
    Soup of two SFT seeds/data variants scored 52.88 on the public LB vs 51.64 for the best single model.
  - `research/splits/`: Nguyen's 287-id holdout and the 187 unseen-app ids used for all local comparisons.
- `submission/`: public test submissions with their LB scores in the file names (8.23 to 51.64).

Paths in the research scripts point to the local machine (`C:/Project/...`); adjust before running.
