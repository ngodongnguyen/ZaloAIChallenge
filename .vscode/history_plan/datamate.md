# DataMate — Tracking

File duy nhất của track: bảng run (cách làm → kết quả) + những gì đã biết. Quy ước: [README](README.md).

## Runs

Mỗi lần chạy (kể cả hỏng) một dòng, mới nhất ở cuối. Account chỉ ghi bí danh `acc<N>`.

| run_id | Ngày | Account | Status | Thay đổi chính | Kết quả | Kết luận / bước tiếp |
|---|---|---|---|---|---|---|
| dm-D-000-local-dryrun | 2026-10-29 | local | done | Notebook helper-agent đầu tiên (profiler + formatter + scorer), LLM giả | Scorer khớp 3000/3000 nhãn train; profiler 44/44 shop | Scorer/formatter local tin được |
| dm-D-001-baseline-dev | 2026-10-29 | — | planned | Qwen3-4B thật, helper-agent trên 6 shop dev_tune cố định | chưa chạy | — |
| dm-D-002-baseline-public | 2026-10-29 | acc3 | failed | Qwen3-4B, `pip install vllm` vào môi trường gốc | Worker chết lúc nạp model: vLLM thay torch → cả vLLM lẫn transformers lỗi | Ngõ cụt: vLLM phải ở venv riêng |
| dm-C-003-raw-dev | 2026-10-29 | — | planned | Raw code-agent vs helper trên cùng 6 shop dev_tune | chưa chạy | — |
| dm-D-004-baseline-public | 2026-10-29 | acc3 | running | vLLM venv riêng, public helper (scriptVersionId 363296233) | Profiler 14/14 shop, 163 bảng; worker chưa có output | — |

## Nộp Public LB (tối đa 5 lần/ngày/track)

| Ngày | Lần | run_id | Account | Điểm LB | Ghi chú |
|---|---|---|---|---|---|

## Đã xác nhận

Mỗi mục một dòng: ngày · nguồn · nội dung.

- 2026-10-29 · HF metadata · Qwen3-4B-Instruct-2507: 4.022.468.096 tham số, Apache-2.0, revision `cdbee75f17c01a7cc42f958dc650907174af0554` (2025-09-17). [Nguồn](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- 2026-10-29 · regression CPU local · Landlock ABI 8 trên máy dev chặn read qua symlink ra ngoài, write dữ liệu nguồn và import helper trong raw; đọc CSV/Excel/SQLite/parquet được. Kaggle chưa probe. Nguồn: `notebook-kaggle/verify_notebooks.py`.
- 2026-10-29 · build-wheels (acc3, version 363285931) · vLLM 0.32.0 trong venv riêng chạy được trên 2x T4 (driver 580.178.04); môi trường gốc giữ torch 2.11.0+cu128.
- 2026-10-29 · dm-D-002 · `pip install vllm` vào môi trường Kaggle (Python 3.13) thay torch 2.13+cu130 → vLLM lẫn transformers không import được → vLLM phải ở venv riêng.
- 2026-10-29 · dm-D-000 · Formatter + scorer local khớp 3000/3000 nhãn train (round-trip).
- 2026-10-29 · dm-D-000 · Mốc Tết theo `calendar.csv`: 2025 = 2025-01-22..2025-02-07, 2026 = 2026-02-10..2026-02-26 (cột thật là `lunar_leap`, không phải `is_leap_month` như docs).
- 2026-10-29 · dm-D-000 · Dạng ô gặp thật: tiền `227,262,100` · `1,090,000.0000` · `1.000.000đ` · `-210.000đ` · `16 076` · `750k` · `1tr090` · `12tr460` · `1,45tr`; ngày `1 thg 2 2025` · `01/02/2025` · `1/5` (không năm → năm của sheet) · serial Excel `45778`; số lượng `3 m`, `1 chai`; trạng thái tiếng Anh `completed`.
- 2026-10-29 · dm-D-000 · Có shop ghi thông tin đơn lặp mọi dòng (dev_002), có shop chỉ ghi dòng đầu (dev_013); có dòng footer "Tổng cộng"; khoá đơn có thể tên `HĐ`, `Số CT`.
- 2026-10-29 · dm-D-000 · Mọi CSV đều UTF-8 (có BOM); `calamine` đọc xlsx nhanh ~10x openpyxl, kết quả giống hệt.
- 2026-10-29 · rulebook · Rulebook public thêm ngành (mẹ & bé, sửa xe, điện thoại) và luật chung mới so với train → luôn đọc `rulebook.md` của split lúc inference.

## Ngõ cụt

- 2026-10-29 · dm-D-002 · Cài vLLM thẳng vào môi trường Kaggle bằng pip: hỏng cả vLLM lẫn fallback transformers.

## Nhóm lỗi thường gặp

| Nhóm | Run | Số câu | Ghi chú / cách xử lý |
|---|---|---|---|

## Chưa kiểm / ý tưởng chờ thử

- Scorer (giả định): `type` dự đoán phải trùng `type` nhãn; percent so sau làm tròn 2 chữ số; cột table khớp theo tên rồi mới theo vị trí.
- GPU chờ CPU: chạy code batch k song song với lượt sinh batch k+1, `exec_workers=2` sau khi đo RAM, nạp bảng shop một lần. Làm sau baseline, đo `s_per_item` trước.
- Model lớn hơn trong trần 15B: Qwen3-14B 4-bit, Qwen3.5-9B; thinking có giới hạn token cho câu khó.
- Ánh xạ bảng chuẩn orders/order_lines theo shop (sau phân tích lỗi đợt 1).
