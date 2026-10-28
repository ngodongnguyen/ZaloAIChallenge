---
run_id: dm-D-000-local-dryrun
track: datamate
lane: D
status: done
parent_run: none
change_vs_parent: notebook helper-agent baseline đầu tiên
hypothesis: formatter/scorer đúng thì nhãn train format lại tự chấm được 100%; profiler đọc được mọi shop không lỗi
account: local (máy dev, không dùng Kaggle)
agent: claude-code
started: 2026-10-29 01:20 +07:00
finished: 2026-10-29 03:30 +07:00
notebook: notebook-kaggle/datamate/helper-agent/datamate_helper_agent.ipynb
kaggle_version: none
output_link: none
config_hash: none
dataset_version: datamate (train 3000 câu / 30 shop, public_test 1000 câu / 14 shop), tải 2026-10-29
model: MockLLM (không model thật)
precision: none
backend: mock
hardware: CPU local
gpu_hours: 0
split: train + public_test
subset: toàn bộ (profiler, scorer); public limit 6 (chạy thử notebook)
public_score: none
tags: [baseline, dry-run, profiler, scorer-check]
---

# dm-D-000-local-dryrun

## Mục tiêu
Kiểm kỹ thuật trước khi tốn GPU: profiler đọc mọi shop, scorer/formatter khớp luật 9 kiểu, executor chạy code an toàn, resume không trùng/mất id, notebook chạy trọn.

## Kết quả
| Metric | Giá trị |
|---|---|
| Format lại + tự chấm 3000 nhãn train | 3000/3000 đúng |
| Profiler 44 shop (train + public) | 0 lỗi, ~5.5 phút CPU tổng (shop chậm nhất ~18 s) |
| Mốc Tết từ `calendar.csv` | 2025: 2025-01-22..2025-02-07; 2026: 2026-02-10..2026-02-26 |
| Notebook raw/helper mô phỏng (LLM giả, 6 câu public) | chạy hết mọi cell, `submission.jsonl` hợp lệ (6 dòng abstain) |
| Prompt helper (rulebook trích + card shop + câu hỏi) | ~25K ký tự tối đa (~10K token) + 1536 token output — vừa 16K |

## Sự cố / lỗi đã sửa trong lúc chạy
- Profiler chậm (biz_105: 153 s) → parse theo giá trị unique, regex biên dịch sẵn, engine `calamine` (cùng kết quả với openpyxl) → 14 s.
- Khoảng ngày shop bị kéo bởi cột ngày sinh / hạn dùng (1966, 2029) và bị `None` vì khớp chuỗi con "han" trong "tháng"/"hàng" → lọc theo từ nguyên.
- ffill nhầm cột mặt hàng ở dev_002 do dòng footer "Tổng cộng" không có mã đơn → bỏ dòng tổng, nhóm đơn theo dải mã liên tiếp.
- Ngày dạng số serial Excel (45778) và khoá đơn tên `HĐ` không được nhận → đã thêm.

## Kết luận và bước tiếp
- Chưa có PA thật. Run tiếp: `dm-D-001-baseline-public` (Qwen3-4B-Instruct-2507, 2x T4) để có submission đầu, rồi `split=train, subset=dev_tune` để đo PA theo shop / kiểu đáp án.
- Rủi ro chưa kiểm: vLLM trên T4 (fallback transformers chậm hơn nhiều); thời gian 1000 câu trong 10.5 h.
