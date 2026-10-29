# Submissions (public test)

| File | Date | Leaderboard | Model |
|---|---|---|---|
| 2026-10-27_baseline-qwen2.5-3b_8.23.jsonl | 2026-10-27 | 8.23 | PhoWhisper-large + Qwen2.5-3B-Instruct (no fine-tune), fp16 |
| 2026-10-28_sft-qwen2.5-3b_15.80.jsonl | 2026-10-28 | 15.80 | PhoWhisper-large + Qwen2.5-3B-Instruct, LoRA SFT on 1177 train trajectories (70% ASR text), fp16 |
| 2026-10-29_v3-qwen2.5-3b_35.94.jsonl | 2026-10-29 | 35.94 | v3: Nguyên's prompt v2 + agent fixes; PhoWhisper-large LoRA fine-tuned on train audio; Qwen2.5-3B-Instruct LoRA SFT 2 epochs on 1418 train trajectories (gold text, all train except a 127-id dev split), 2× T4 data parallel, fp16 |
| 2026-10-30_v3-qwen3-4b-histasr_46.94.jsonl | 2026-10-30 | 46.94 | v3 recipe on Qwen3-4B-Instruct-2507 (LoRA SFT 2 epochs, transformers 4.57.6); history user turns transcribed by ASR (test history has audio only); sdpa without enable_gqa to avoid T4 OOM |

Known issue in the 15.80 file: 20 samples hit CUDA OOM on a Kaggle T4 (ASR and LLM on one GPU), so 12 rows have no tool calls.
The 35.94 file transcribes every clip first and frees ASR before the LLM loads, so no sample failed.
Train the v3 LLM on two GPUs with `torchrun --nproc_per_node 2 train/sft.py --data train/sft.jsonl --base models/base-llm --out models/lifemate-llm --epochs 2 --max-len 7500 --fp16`.
