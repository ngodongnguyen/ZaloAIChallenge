# Submissions (public test)

| File | Date | Leaderboard | Model |
|---|---|---|---|
| 2026-10-27_baseline-qwen2.5-3b_8.23.jsonl | 2026-10-27 | 8.23 | PhoWhisper-large + Qwen2.5-3B-Instruct (no fine-tune), fp16 |
| 2026-10-28_sft-qwen2.5-3b_15.80.jsonl | 2026-10-28 | 15.80 | PhoWhisper-large + Qwen2.5-3B-Instruct, LoRA SFT on 1177 train trajectories (70% ASR text), fp16 |
| 2026-10-29_v3-qwen2.5-3b_35.94.jsonl | 2026-10-29 | 35.94 | v3: Nguyên's prompt v2 + agent fixes; PhoWhisper-large LoRA fine-tuned on train audio; Qwen2.5-3B-Instruct LoRA SFT 2 epochs on 1418 train trajectories (gold text, all train except a 127-id dev split), 2× T4 data parallel, fp16 |
| 2026-10-30_v3-qwen3-4b-histasr_46.94.jsonl | 2026-10-30 | 46.94 | v3 recipe on Qwen3-4B-Instruct-2507 (LoRA SFT 2 epochs, transformers 4.57.6); history user turns transcribed by ASR (test history has audio only); sdpa without enable_gqa to avoid T4 OOM |
| 2026-10-29_v3-qwen2.5-3b-histasr_38.11.jsonl | 2026-10-29 | 38.11 | 35.94 recipe; history user turns transcribed by ASR |
| 2026-10-31_r016-v4-qwen3-4b-promptv1-tokenloss_45.91.jsonl | 2026-10-31 | 45.91 | Qwen3-4B, prompt v1, token-weighted loss, history ASR |
| 2026-11-01_r017-e16-histasr_49.71.jsonl | 2026-11-01 | 49.71 | E16 recipe rerun on Kaggle (paraphrases ×3, 1 epoch, max-len 6144, Nguyên's holdout excluded), fp16 2× T4, ASR LoRA a002, history ASR |
| 2026-11-01_r018-e16-final-histasr_51.64.jsonl | 2026-11-01 | **51.64** | E16 recipe trained on all of train (3452 trajectories) |
| 2026-11-01_r019-e16-rename-histasr_48.40.jsonl | 2026-11-01 | 48.40 | r017 + tool/app renaming on half of the paraphrases (no gain) |
| 2026-11-01_r020-e16-final-seed7-histasr_50.21.jsonl | 2026-11-01 | 50.21 | r018 with seed 7 only: one seed moves LB by ~1.4, treat gaps under 2 as noise |
| 2026-11-02_r021-e16-synth-histasr_51.45.jsonl | 2026-11-02 | 51.45 | r017 + 672 synthetic scenarios on 30 invented apps (research/synth); 0.589 vs 0.538 on the 187 unseen-app samples |
| 2026-11-02_u004-r021-asr-hist_50.65.jsonl | 2026-11-02 | 50.65 | r021 LLM unchanged, ASR LoRA retrained with history clips (dev WER 0.110 vs 0.134): no LB gain |

Known issue in the 15.80 file: 20 samples hit CUDA OOM on a Kaggle T4 (ASR and LLM on one GPU), so 12 rows have no tool calls.
The 35.94 file transcribes every clip first and frees ASR before the LLM loads, so no sample failed.
Train the v3 LLM on two GPUs with `torchrun --nproc_per_node 2 train/sft.py --data train/sft.jsonl --base models/base-llm --out models/lifemate-llm --epochs 2 --max-len 7500 --fp16`.

From 49.71 on, files come from the Kaggle harness in `research/` (kernel_template/run_eval.py), not from src/infer.py.
