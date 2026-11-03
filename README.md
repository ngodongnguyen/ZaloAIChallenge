# ZaloAIChallenge – LifeMate

Voice (Vietnamese) → ASR → LLM agent → tool calls on `zaic_tools` → `submission.jsonl`.

Models: PhoWhisper-large (~1.5B) + Qwen2.5-3B-Instruct (LoRA SFT, merged) ≈ 4.5B < 6B total.

## Workflow
```bash
pip install -r requirements.txt
bash scripts/download_models.sh                      # once, online

# 1. (optional) real ASR text for train, to make the LLM robust to ASR errors
python src/transcribe_split.py --root data --split train --out cache/train_asr.jsonl

# 2. build trajectories + fine-tune
python train/build_sft.py --root data --split train --asr-cache cache/train_asr.jsonl --out train/sft.jsonl
python train/sft.py --data train/sft.jsonl --base models/base-llm --out models/lifemate-llm

# 3. infer + local eval (use --holdout in sft.py to evaluate on unseen train ids)
python src/infer.py --root data --split public_test --out submission.jsonl
python src/infer.py --root data --split train --limit 100 --out cache/train_pred.jsonl
python src/eval.py --pred cache/train_pred.jsonl --root data --split train

# 4. Docker (offline)
docker build -t lifemate .
docker run --gpus all --network none -v /path/to/private_test:/data lifemate
```

## Notes / assumptions (verify once data is available)
- `history` has `user` turns (`audio`, plus `audio_text` in train only) and `assistant` turns (`text`, `tool_calls`, `tool_results`). At inference, history user audio is transcribed by ASR too (`src/infer.py`).
- Agent never reads `*_environment/`; one `ToolEnv` per sample; `env.validate` is used before `env.call` so schema-invalid calls are never logged.
- `src/eval.py` approximates the official score (formula was an image); it does not check `state_hash`.
- Decoding is greedy, seeds fixed (`src/config.py`).
