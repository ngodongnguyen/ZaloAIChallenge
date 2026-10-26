# KERNEL THỰC NGHIỆM — TRACK 2: DATAMATE
## ZALO AI CHALLENGE 2026

Thư mục này chứa mã nguồn thực nghiệm và cấu hình triển khai Kaggle Kernel cho bảng đấu **Track 2: DataMate** (Tabular Data Analytics).

---

## 1. THÀNH PHẦN FILE

- **`kernel-metadata.json`**: Tệp cấu hình đẩy kernel lên Kaggle thông qua Kaggle API.
  - Slug: `<owner>/zalo2026-datamate-baseline`
  - Accelerator: GPU (NVIDIA T4 / P100)
  - Dataset đính kèm: `thanhkhil/zaloai-datamate-dataset`
- **`baseline_pipeline.py`**: Pipeline suy luận hai chế độ:
  - `--mode mock`: Chạy dry-run kiểm thử logic xử lý 9 kiểu dữ liệu, bắt lỗi an toàn và tạo tệp `submission.jsonl`.
  - `--mode real`: Chạy suy luận Code Generation với mô hình họ Qwen2.5-Coder ($\le$ 15B), thực thi sandbox Python và định dạng đầu ra.

---

## 2. HƯỚNG DẪN THỰC THI

### 2.1. Chạy thử nghiệm cục bộ (Mock Mode)
```bash
python notebook-kaggle/datamate/baseline_pipeline.py \
  --input path/to/questions.jsonl \
  --output submission.jsonl \
  --mode mock
```

### 2.2. Đẩy lên Kaggle chạy bằng GPU
```bash
# Đẩy code lên Kaggle
python notebook-kaggle/scripts/kaggle_sync.py push-kernel --track datamate

# Theo dõi tiến độ
python notebook-kaggle/scripts/kaggle_sync.py status-kernel --track datamate

# Kéo file submission.jsonl về
python notebook-kaggle/scripts/kaggle_sync.py pull-output --track datamate
```

### 2.3. Quy tắc "Silence is Free" (Chiến lược bảo toàn điểm)
- Khi code phân tích thành công và tự tin $\rightarrow$ Xuất đáp án (+1 điểm).
- Khi thiếu dữ liệu $\rightarrow$ Xuất `{"type": "unanswerable"}` (+1 điểm).
- Khi phát sinh Exception hoặc kết quả không chắc chắn $\rightarrow$ Kích hoạt `{"type": "abstain"}` (0 điểm), tránh tối đa án phạt -1 điểm.
