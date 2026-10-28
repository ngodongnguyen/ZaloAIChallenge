# Submissions (public test)

| File | Date | Leaderboard | Model |
|---|---|---|---|
| 2026-10-27_baseline-qwen2.5-3b_8.23.jsonl | 2026-10-27 | 8.23 | PhoWhisper-large + Qwen2.5-3B-Instruct (no fine-tune), fp16 |
| 2026-10-28_sft-qwen2.5-3b_15.80.jsonl | 2026-10-28 | 15.80 | PhoWhisper-large + Qwen2.5-3B-Instruct, LoRA SFT on 1177 train trajectories (70% ASR text), fp16 |

Known issue in the 15.80 file: 20 samples hit CUDA OOM on a Kaggle T4 (ASR and LLM on one GPU), so 12 rows have no tool calls.
