---
run_id: <track>-<lane>-<NNN>-<mô-tả-ngắn>     # vd lm-B-001-baseline-public, dm-D-002-qwen3-14b-4bit
track: lifemate | datamate
lane: A | B | C | D | 0 | sft
status: planned | running | done | failed | abandoned
parent_run: <run_id gốc để so, hoặc none>
change_vs_parent: <MỘT thay đổi chính so với parent>
hypothesis: <kỳ vọng gì, vì sao>
account: acc<N>                                 # bí danh trong pool, KHÔNG ghi username (bảng map ở AGENTS.local.md)
agent: <ai chạy/ghi: user | claude-code | codex | agy | dsh>
started: YYYY-MM-DD HH:MM +07:00
finished: YYYY-MM-DD HH:MM +07:00
notebook: notebook-kaggle/<track>/<folder>/<file>.ipynb
kaggle_version: <số version notebook trên Kaggle>
output_link: <link output version Kaggle, nếu có>
config_hash: <run_hash in summary.shard*.json>
dataset_version: <vd lifemate v3.2.0 build 2026-10-24>
model: <HF id @ revision>
precision: fp16 | int8 | 4bit-nf4 | …
backend: vllm | transformers
hardware: <vd Kaggle 2x T4 16GB>
gpu_hours: <số giờ GPU tiêu tốn>
split: train | public_test
subset: <all | dev_seen | dev_tune | final_holdout | …>
public_score: <điểm LB hoặc none>
tags: [baseline, asr, prompt, sft, …]
---

# <run_id>

## Mục tiêu
<1–3 dòng: hỏi câu gì, tại sao chạy run này>

## Cấu hình
<các tham số khác default trong CONFIG của notebook; dán khối CONFIG nếu ngắn>

## Kết quả
| Metric | Giá trị | So với parent |
|---|---|---|
| Score / PA local | | |
| Hash OK (LifeMate) / coverage (DataMate) | | |
| WER / CER (nếu có ASR) | | |
| Thời gian / mẫu, tổng thời gian | | |
| Peak VRAM | | |

Chi tiết theo nhóm (dán từ `merge_report.json`: `by_group` / `by_shop` / `by_gold_type`):

## Phân tích lỗi
<top nhóm lỗi + số lượng + 2–3 ví dụ id; LifeMate: ASR thực thể, write/hash, read thừa/thiếu, missing-info, nhãn rỗng, lịch sử, app mới, timeout; DataMate: đọc dữ liệu/join, lọc đơn, doanh thu, thời gian, xếp hạng/đơn vị, nợ, kho/giá vốn, luật ngành mới, hiểu câu hỏi, format, unanswerable>

## Sự cố vận hành
<lỗi cài đặt, OOM, vLLM fallback, hết quota, timeout session, cách đã xử lý>

## Kết luận và bước tiếp
- Giữ / bỏ thay đổi này vì …
- Run tiếp theo: …
- Đã thêm vào `insights.md`: có / không
