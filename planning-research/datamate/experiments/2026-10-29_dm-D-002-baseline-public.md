---
run_id: dm-D-002-baseline-public
track: datamate
lane: D
status: failed
parent_run: dm-D-000-local-dryrun
change_vs_parent: chạy thật trên Kaggle (GPU) thay vì dry-run local
hypothesis: notebook helper-agent baseline cho submission public đầu tiên
account: acc3
agent: user (chạy), claude-code (phân tích log)
started: 2026-10-29 02:45 +07:00
finished: 2026-10-29 02:55 +07:00
notebook: notebook-kaggle/datamate/helper-agent/datamate_helper_agent.ipynb
kaggle_version: scriptVersionId 363281455 (kernel datamate-helper-agent)
output_link: https://www.kaggle.com/code/<owner>/datamate-helper-agent/log?scriptVersionId=363281455
config_hash: none (worker chết lúc nạp model)
dataset_version: datamate (train 3000 / public_test 1000)
model: Qwen/Qwen3-4B-Instruct-2507
precision: fp16
backend: vllm 0.32.0 (cài thẳng vào môi trường gốc) → fallback transformers 5.16 cũng hỏng
hardware: Kaggle 2x Tesla T4 15GB, Python 3.13
gpu_hours: ~0.3
split: public_test
subset: all
public_score: none
tags: [baseline, public, env-failure]
---

# dm-D-002-baseline-public

## Mục tiêu
Submission public đầu tiên của DataMate.

## Kết quả
Profiler (CPU) chạy xong, nhưng cả 2 worker chết lúc nạp model (~576 s): không câu nào được trả lời.

## Sự cố vận hành
1. `pip install vllm` vào môi trường gốc → torch 2.13+cu130, torchaudio có sẵn vẫn cu128.
2. vLLM khởi động lỗi `Detected that PyTorch and TorchAudio were compiled with different CUDA versions` → `[llm] vLLM unavailable … using transformers`.
3. Fallback transformers cũng chết cùng lý do: `ModuleNotFoundError: Could not import module 'Qwen3ForCausalLM'` (import torchaudio trong chuỗi import của transformers).
Cùng nguyên nhân với `lm-B-002-baseline-public` (LifeMate).

## Kết luận và bước tiếp
- Sửa: vLLM vào venv riêng `/tmp/venv_vllm`, worker LLM chạy bằng python của venv (kèm pandas/pyarrow/openpyxl/calamine cho executor); môi trường gốc không đổi nên fallback transformers dùng được.
- Run tiếp: `dm-D-003-baseline-public` với notebook đã sửa.
- Đã thêm vào `insights.md`: có.
