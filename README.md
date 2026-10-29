# ZaloAIChallenge – LifeMate

Voice (Vietnamese) → ASR → LLM agent → tool calls on `zaic_tools` → `submission.jsonl`.

Models: PhoWhisper-large (~1.5B) + Qwen2.5-3B-Instruct (LoRA SFT, merged) ≈ 4.5B < 6B total.

## Workflow
```bash
pip install -r requirements.txt
bash scripts/download_models.sh                      # once, online

# 1. (optional) real ASR text for train, to make the LLM robust to ASR errors
python src/transcribe_split.py --root . --split train --out cache/train_asr.jsonl

# 2. build trajectories + fine-tune
python train/build_sft.py --root . --split train --asr-cache cache/train_asr.jsonl --out train/sft.jsonl
python train/sft.py --data train/sft.jsonl --base models/base-llm --out models/lifemate-llm  # 2 GPUs: torchrun --nproc_per_node 2 train/sft.py ...

# 3. infer + local eval (use --holdout in sft.py to evaluate on unseen train ids)
python src/infer.py --root . --split public_test --out submission.jsonl
python src/infer.py --root . --split train --limit 100 --out cache/train_pred.jsonl
python src/eval.py --pred cache/train_pred.jsonl --root . --split train

# 4. Docker (offline)
docker build -t lifemate .
docker run --gpus all --network none -v /path/to/private_test:/data lifemate
```

## Notes / assumptions (verify once data is available)
- Format of `history` and `audio_text` in the jsonl is assumed (list or string); see `train/build_sft.py:last_text`.
- Agent never reads `*_environment/`; one `ToolEnv` per sample; `env.validate` is used before `env.call` so schema-invalid calls are never logged.
- `src/eval.py` approximates the official score (formula was an image); it does not check `state_hash`.
- Decoding is greedy, seeds fixed (`src/config.py`).
