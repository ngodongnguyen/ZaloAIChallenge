# KERNEL THỰC NGHIỆM — TRACK 1: LIFEMATE
## ZALO AI CHALLENGE 2026

Thư mục này chứa mã nguồn thực nghiệm và cấu hình triển khai Kaggle Kernel cho bảng đấu **Track 1: LifeMate** (Agentic Voice-to-Action).

---

## 1. THÀNH PHẦN FILE

- **`kernel-metadata.json`**: Tệp cấu hình đẩy kernel lên Kaggle thông qua Kaggle API.
  - Slug: `<owner>/zalo2026-lifemate-baseline`
  - Accelerator: GPU (NVIDIA T4 / P100)
  - Dataset đính kèm: `thanhkhil/zaloai-lifemate-dataset`
- **`baseline_pipeline.py`**: Pipeline suy luận hai chế độ:
  - `--mode mock`: Chạy dry-run kiểm thử toàn vẹn logic, xử lý out-of-scope, định dạng jsonl và mã băm `state_hash`.
  - `--mode real`: Chạy suy luận ASR (`whisper-large-v3-turbo`) + LLM (`Qwen2.5-3B-Instruct`) + gọi tool qua `zaic_tools.ToolEnv`.

---

## 2. HƯỚNG DẪN THỰC THI

### 2.1. Chạy thử nghiệm cục bộ (Mock Mode)
```bash
python notebook-kaggle/lifemate/baseline_pipeline.py \
  --input path/to/sample.jsonl \
  --output submission.jsonl \
  --mode mock
```

### 2.2. Đẩy lên Kaggle chạy bằng GPU
```bash
# Đẩy code lên Kaggle
python notebook-kaggle/scripts/kaggle_sync.py push-kernel --track lifemate

# Theo dõi tiến độ
python notebook-kaggle/scripts/kaggle_sync.py status-kernel --track lifemate

# Kéo file submission.jsonl về
python notebook-kaggle/scripts/kaggle_sync.py pull-output --track lifemate
```

### 2.3. Kiểm tra tính hợp lệ trước khi submit
```bash
python -m zaic_tools --split public_test --root . --validate notebook-kaggle/lifemate/output/submission.jsonl
```
