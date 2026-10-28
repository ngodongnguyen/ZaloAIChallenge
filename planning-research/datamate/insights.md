# DataMate — Insights (kiến thức đúc kết)

Chỉ ghi điều đã kiểm chứng hoặc đã thử, kèm nguồn (run_id / file / ngày). Đọc file này trước khi đề xuất thí nghiệm mới để không lặp lại ngõ cụt. Mỗi mục một dòng, mới nhất lên đầu trong từng phần.

## Đã xác nhận
- 2026-10-29 · build-wheels (Kaggle, acc3, version 363285931) · vLLM **0.32.0** trong venv riêng chạy được trên Kaggle 2x T4 (driver 580.178.04): Qwen3-0.6B fp16 sinh OK. Môi trường gốc giữ torch 2.11.0+cu128 / torchaudio 2.11.0, `transformers.pipeline` vẫn import được. Đã ghim `vllm==0.32.0` trong notebook.
- 2026-10-29 · dm-D-002 · `pip install vllm` vào môi trường Kaggle (image 2026-10, Python 3.13) thay torch bằng 2.13+cu130 → torchaudio cu128 lệch → cả vLLM lẫn transformers (`Qwen3ForCausalLM`) không import được → **vLLM phải ở venv riêng** (giống LifeMate lm-B-002).
- 2026-10-29 · dm-D-000 · Formatter + scorer local khớp 3000/3000 nhãn train (round-trip).
- 2026-10-29 · dm-D-000 · Mốc Tết theo `calendar.csv`: 2025 = 2025-01-22..2025-02-07, 2026 = 2026-02-10..2026-02-26 (cột thật là `lunar_leap`, không phải `is_leap_month` như docs).
- 2026-10-29 · dm-D-000 · Dạng ô gặp thật: tiền `227,262,100` · `1,090,000.0000` · `1.000.000đ` · `-210.000đ` · `16 076` · `750k` · `1tr090` · `12tr460` · `1,45tr`; ngày `1 thg 2 2025` · `01/02/2025` · `1/5` (không năm → năm của sheet) · số serial Excel `45778`; số lượng `3 m`, `1 chai`; trạng thái tiếng Anh `completed`.
- 2026-10-29 · dm-D-000 · Có shop ghi thông tin đơn lặp lại mọi dòng (dev_002), có shop chỉ ghi dòng đầu (dev_013); có dòng footer "Tổng cộng"; khoá đơn có thể tên `HĐ`, `Số CT`.
- 2026-10-29 · dm-D-000 · Mọi CSV đều UTF-8 (có BOM); `calamine` đọc xlsx nhanh ~10x openpyxl, kết quả giống hệt.
- 2026-10-29 · rulebook · Rulebook public thêm ngành (mẹ & bé, sửa xe, điện thoại) và luật chung mới so với train → luôn đọc `rulebook.md` của split lúc inference.

## Đã thử, không hiệu quả / ngõ cụt
- 2026-10-29 · dm-D-002 · Cài vLLM thẳng vào môi trường Kaggle bằng pip: hỏng cả vLLM lẫn fallback transformers.

## Nhóm lỗi thường gặp
| Nhóm | Run phát hiện | Số câu | Ghi chú / cách xử lý |
|---|---|---|---|

## Giả định chưa kiểm (cần xác nhận với BTC hoặc LB)
- Scorer: `type` dự đoán phải trùng `type` nhãn; percent so sau làm tròn 2 chữ số; cột table khớp theo tên rồi mới theo vị trí.

## Câu hỏi mở / ý tưởng chờ thử
- Raw code-agent (đọc file thô) vs helper-agent trên cùng `dev_tune` để chọn kiến trúc.
- Model lớn hơn trong trần 15B: Qwen3-14B 4-bit, Qwen3.5-9B; thinking có giới hạn token cho câu khó.
- Ánh xạ bảng chuẩn orders/order_lines theo shop (sau khi có phân tích lỗi đợt 1).
