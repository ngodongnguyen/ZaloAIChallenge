# LifeMate — Insights (kiến thức đúc kết)

Chỉ ghi điều đã kiểm chứng hoặc đã thử, kèm nguồn (run_id / file / ngày). Đọc file này trước khi đề xuất thí nghiệm mới để không lặp lại ngõ cụt. Mỗi mục một dòng, mới nhất lên đầu trong từng phần.

## Đã xác nhận
- 2026-10-29 · build-wheels (Kaggle, acc3, version 363285931) · vLLM **0.32.0** trong venv riêng chạy được trên Kaggle 2x T4 (driver 580.178.04): Qwen3-0.6B fp16 sinh OK. Môi trường gốc giữ torch 2.11.0+cu128 / torchaudio 2.11.0, `transformers.pipeline` vẫn import được. Đã ghim `vllm==0.32.0` trong notebook.
- 2026-10-29 · lm-B-002 · Image Kaggle (2026-10): Python 3.13, transformers 5.16, 2x T4 15GB, 4 CPU, 32GB RAM. `pip install vllm` (0.32.0) kéo torch 2.13+cu130 thay bản có sẵn, torchaudio có sẵn vẫn cu128 → `from transformers import pipeline` lỗi → **vLLM phải cài vào venv riêng**.
- 2026-10-29 · lm-B-000 · Phát lại `label.tool_calls` qua `ToolEnv` của split cho hash khớp 100% và 0 lỗi schema → scorer local (hash gate + F1) dùng được để đo dev.
- 2026-10-29 · lm-B-000 · `installed_apps` trong `<split>.jsonl` chỉ có danh sách tool; state nằm trong `<split>_environment/<id>/state.json` (agent không được đọc) — mọi đọc dữ liệu đi qua tool read.
- 2026-10-29 · dataset README · `env.validate` chỉ chặn lỗi schema; ID không tồn tại / không đủ tiền vẫn trả `ok: true`, không đổi state nhưng **vẫn bị ghi** vào `tool_calls` (tính U).
- 2026-10-29 · zaic_tools · `LIVE_APPS` / `FREE_TEXT_KEYS` khác nhau theo split → luôn import `zaic_tools` của đúng split đang chạy.
- 2026-10-29 · lm-B-000 · Prompt agent trên public (không tính lịch sử ASR dài) p50 8.8K ký tự, max 16.1K (~6–7K token) — vừa context 16K.

## Đã thử, không hiệu quả / ngõ cụt
- 2026-10-29 · lm-B-002 · Cài vLLM thẳng vào môi trường Kaggle bằng pip: phá ASR (xem trên).

## Nhóm lỗi thường gặp
| Nhóm | Run phát hiện | Số mẫu | Ghi chú / cách xử lý |
|---|---|---|---|

## Giả định chưa kiểm (cần xác nhận với BTC hoặc LB)
- Matcher: tham số optional chỉ có ở một bên làm call không khớp; số so tuyệt đối; chuỗi không fuzzy so sau lower + gộp khoảng trắng.

## Câu hỏi mở / ý tưởng chờ thử
- ASR: gipformer-68M (mạnh phương ngữ, theo model card) vs PhoWhisper-medium — cần đo WER + recall tên riêng trên dev train.
- Sửa thực thể sau ASR cho tên trong danh bạ (`contacts.search.query` không fuzzy).
- SFT/LoRA trên trajectory thật của `train_fit`; synthetic cho 11 app chỉ có ở public bằng simulator.
