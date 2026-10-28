---
run_id: lm-B-000-local-dryrun
track: lifemate
lane: B
status: done
parent_run: none
change_vs_parent: notebook agent baseline đầu tiên
hypothesis: vòng lặp agent + ToolEnv + scorer + merge đúng thì agent "oracle" phát lại nhãn gold đạt 100 điểm
account: local (máy dev, không dùng Kaggle)
agent: claude-code
started: 2026-10-29 03:00 +07:00
finished: 2026-10-29 03:30 +07:00
notebook: notebook-kaggle/lifemate/agent/lifemate_agent.ipynb
kaggle_version: none
output_link: none
config_hash: none
dataset_version: lifemate v3.2.0 (build 2026-10-24, train 1545 / public_test 634)
model: MockLLM (không model thật)
precision: none
backend: mock
hardware: CPU local
gpu_hours: 0
split: train
subset: 60 mẫu đầu (oracle), 200 mẫu đầu (always-done), dev_seen limit 12 (chạy thử notebook)
public_score: none
tags: [baseline, dry-run, scorer-check]
---

# lm-B-000-local-dryrun

## Mục tiêu
Kiểm kỹ thuật trước khi tốn GPU: notebook chạy trọn, mỗi mẫu đúng một ToolEnv, scorer local khớp luật, submission đúng thứ tự id.

## Kết quả
| Metric | Giá trị |
|---|---|
| Oracle (phát lại label.tool_calls) trên 60 mẫu train | 100.00 điểm, hash OK 100%, 0 lỗi schema |
| Agent luôn trả `{"done": true}` trên 200 mẫu | điểm = đúng tỉ lệ mẫu nhãn rỗng (kiểm hash gate + luật \|G\|=0) |
| Notebook mô phỏng local (mode diagnostic, LLM giả) | chạy hết mọi cell, ghi `merge_report.json`, trace |
| Kích thước prompt trên public (không tính lịch sử ASR dài) | p50 8.8K ký tự, p95 13.5K, max 16.1K (~6–7K token) — vừa context 16K |

## Sự cố vận hành
- Oracle ban đầu 98.33: test tự gây (hai mẫu cùng câu nói nhưng khác ngữ cảnh). Không phải lỗi pipeline.
- `__init__.py` rỗng làm cell `%%writefile` không có thân → thêm docstring.

## Kết luận và bước tiếp
- Chưa có số đo chất lượng thật. Run tiếp: `lm-B-002-baseline-public` (đã chạy, failed — xem card) (Qwen3-4B-Instruct-2507 + PhoWhisper-medium, 2x T4) để có submission đầu, song song `diagnostic` trên `dev_seen`.
- Rủi ro chưa kiểm: vLLM trên T4 (có fallback transformers, chậm hơn); thời gian ASR + agent cho 634 mẫu.
