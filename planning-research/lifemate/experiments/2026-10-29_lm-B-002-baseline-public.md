---
run_id: lm-B-002-baseline-public
track: lifemate
lane: B
status: failed
parent_run: lm-B-000-local-dryrun
change_vs_parent: chạy thật trên Kaggle (GPU) thay vì dry-run local
hypothesis: notebook agent baseline cho submission public đầu tiên
account: acc3
agent: user (chạy), claude-code (phân tích log)
started: 2026-10-29 02:45 +07:00
finished: 2026-10-29 02:53 +07:00
notebook: notebook-kaggle/lifemate/agent/lifemate_agent.ipynb
kaggle_version: scriptVersionId 363281470 (kernel lifemate-agent)
output_link: https://www.kaggle.com/code/<owner>/lifemate-agent/log?scriptVersionId=363281470
config_hash: none (worker không chạy tới bước ghi summary)
dataset_version: lifemate v3.2.0
model: vinai/PhoWhisper-medium + Qwen/Qwen3-4B-Instruct-2507
precision: fp16
backend: vllm 0.32.0 (cài thẳng vào môi trường gốc)
hardware: Kaggle 2x Tesla T4 15GB, 4 CPU, 32GB RAM, Python 3.13
gpu_hours: ~0.2
split: public_test
subset: all
public_score: none
tags: [baseline, public, env-failure]
---

# lm-B-002-baseline-public

## Mục tiêu
Submission public đầu tiên của LifeMate.

## Kết quả
Không có submission: 0/634 mẫu hoàn tất (`merge_report.json`: pending 634).

## Sự cố vận hành
1. Cell cài đặt chạy `pip install vllm` vào môi trường gốc → vLLM 0.32.0 kéo torch 2.13.0+cu130; torchaudio có sẵn vẫn build cho CUDA 12.8.
2. Worker ASR: `from transformers import pipeline` → `RuntimeError: Detected that PyTorch and TorchAudio were compiled with different CUDA versions` → ModuleNotFoundError `pipeline`; cả 2 shard exit 1 sau 0.2 phút → 0/1933 transcript.
3. Worker agent dừng đúng thiết kế: "296/338 samples have no transcript … run ASR first" (không bịa transcript / hash).

## Kết luận và bước tiếp
- Sửa: vLLM cài vào venv riêng `/tmp/venv_vllm` (worker LLM chạy bằng python của venv; ASR và fallback transformers giữ môi trường gốc). Đã sửa trong `lifemate_agent.ipynb`, `datamate_helper_agent.ipynb` và notebook `zalo2026-wheels`.
- Run tiếp: `lm-B-003-baseline-public` với notebook đã sửa (nên chạy `zalo2026-wheels` trước để có SMOKE vLLM trên T4).
- Đã thêm vào `insights.md`: có.
