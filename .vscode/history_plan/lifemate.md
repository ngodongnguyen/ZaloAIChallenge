# LifeMate — Tracking

File duy nhất của track: bảng run (cách làm → kết quả) + những gì đã biết. Quy ước: [README](README.md).

## Runs

Mỗi lần chạy (kể cả hỏng) một dòng, mới nhất ở cuối. Account chỉ ghi bí danh `acc<N>`.

| run_id | Ngày | Account | Status | Thay đổi chính | Kết quả | Kết luận / bước tiếp |
|---|---|---|---|---|---|---|
| lm-B-000-local-dryrun | 2026-10-29 | local | done | Notebook agent đầu tiên, LLM giả (oracle) | Oracle 100.00 điểm, hash 100% trên 60 mẫu train | Scorer local (hash gate + F1) dùng được để đo dev |
| lm-B-001-baseline-paired-dev | 2026-10-29 | — | planned | Model thật; transcript sạch vs PhoWhisper trên cùng dev_seen | chưa chạy | — |
| lm-B-002-baseline-public | 2026-10-29 | acc3 | failed | PhoWhisper-medium + Qwen3-4B, `pip install vllm` vào môi trường gốc | ASR 0/1933: vLLM thay torch → torchaudio lệch CUDA | Ngõ cụt: vLLM phải ở venv riêng |
| lm-A-003-asr-backend-dev | 2026-10-29 | — | planned | So PhoWhisper vs gipformer trên cùng audio dev_seen | chưa chạy | — |
| lm-B-004-baseline-public | 2026-10-29 | acc3 | done | vLLM vào venv riêng | 634/634, 0 lỗi schema; budget wall-clock cắt agent sau 1 call (440 `budget`, 0 mẫu >1 call) | Đo budget theo `llm_s` mỗi mẫu |
| lm-B-005-budgetfix-public | 2026-10-29 | acc6 | running | Budget theo `llm_s`; ASR cache của lm-B-004 | — | — |
| lm-B-006-budgetfix-dev | 2026-10-29 | acc4 | running | E2E dev_seen + dev_unseen (417), chấm local | — | — |
| lm-sft-007-lora-gold-vs-asr | 2026-10-29 | acc5 | running | LoRA SFT trên train_fit, gold vs ASR transcript | — | So với lm-B-006 |
| lm-B-008-qwen35-4b-dev | 2026-10-29 | acc7 | running | LLM Qwen3.5-4B zero-shot | — | So với lm-B-006 |
| lm-A-009-gipformer-dev | 2026-10-29 | acc8 | running | ASR gipformer-68M thay PhoWhisper | — | So với lm-B-006 |

## Nộp Public LB (tối đa 5 lần/ngày/track)

| Ngày | Lần | run_id | Account | Điểm LB | Ghi chú |
|---|---|---|---|---|---|

## Đã xác nhận

Mỗi mục một dòng: ngày · nguồn · nội dung.

- 2026-10-29 · HF metadata · Qwen3-4B-Instruct-2507: 4.022.468.096 tham số, Apache-2.0, revision `cdbee75f17c01a7cc42f958dc650907174af0554` (2025-09-17). [Nguồn](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- 2026-10-29 · HF metadata · PhoWhisper-medium BSD-3-Clause, revision `55a7e3eb6c906de891f8f06a107754427dd3be79` (2024-11-12). Gipformer 68M/MIT, revision `318322b009200f8e77a3751fa64638e5cc85a7dc` (2026-09-17), ONNX encoder/decoder/joiner + tokens. Chưa có số chất lượng trên dev. [PhoWhisper](https://huggingface.co/vinai/PhoWhisper-medium), [gipformer](https://huggingface.co/g-group-ai-lab/gipformer-68M-rnnt)
- 2026-10-29 · lm-B-004 · Budget mỗi mẫu không được đo bằng wall-clock khi chạy batch: một bước batch 32 trên T4 mất 100–220 s → đo bằng `llm_s` của mẫu.
- 2026-10-29 · local · Split seed 2026: train_fit 974 / dev_seen 154 / final_holdout 154 / dev_unseen 263 (apps fitness, flights, music, tv). Oracle 99.76: 2/1545 nhãn train gọi lặp một call giống hệt (vd `train_0922`) → luật dừng `repeat` mất các mẫu này; vẫn giữ luật.
- 2026-10-29 · local · SFT train_fit: 3034 example (một/bước + done), prompt p50 10.1K ký tự, p95 14.8K, max 19.4K.
- 2026-10-29 · zaic_tools README · Validator CLI chạy trong thư mục split với `--root ..` (`cd public_test && python -m zaic_tools --split public_test --root .. --validate f`).
- 2026-10-29 · acc3 · Quota GPU tuần 30h đã hết sau các run 2026-10-28/29.
- 2026-10-29 · build-wheels (acc3, version 363285931) · vLLM 0.32.0 trong venv riêng chạy được trên 2x T4 (driver 580.178.04); môi trường gốc giữ torch 2.11.0+cu128, `transformers.pipeline` vẫn import được.
- 2026-10-29 · lm-B-002 · Image Kaggle 2026-10: Python 3.13, transformers 5.16, 2x T4 15GB, 4 CPU, 32GB RAM. `pip install vllm` kéo torch 2.13+cu130 → vLLM phải ở venv riêng.
- 2026-10-29 · lm-B-000 · Phát lại `label.tool_calls` qua `ToolEnv` cho hash khớp 100%, 0 lỗi schema.
- 2026-10-29 · lm-B-000 · `installed_apps` chỉ có danh sách tool; state nằm trong `state.json` (agent không được đọc) → mọi đọc dữ liệu qua tool read.
- 2026-10-29 · dataset README · `env.validate` chỉ chặn lỗi schema; ID không tồn tại / không đủ tiền vẫn `ok: true`, không đổi state nhưng vẫn bị ghi vào `tool_calls` (tính U).
- 2026-10-29 · zaic_tools · `LIVE_APPS` / `FREE_TEXT_KEYS` khác nhau theo split → luôn import `zaic_tools` của đúng split.
- 2026-10-29 · lm-B-000 · Prompt agent trên public p50 8.8K ký tự, max 16.1K (~6–7K token), vừa context 16K.

## Ngõ cụt

- 2026-10-29 · lm-B-002 · Cài vLLM thẳng vào môi trường Kaggle bằng pip: phá ASR.

## Nhóm lỗi thường gặp

| Nhóm | Run | Số mẫu | Ghi chú / cách xử lý |
|---|---|---|---|

## Chưa kiểm / ý tưởng chờ thử

- Matcher (giả định): tham số optional chỉ có ở một bên làm call không khớp; số so tuyệt đối; chuỗi so sau lower + gộp khoảng trắng.
- GPU rảnh khi batch co lại: nạp mẫu mới ngay khi một mẫu xong (continuous refill). Làm sau baseline, đo `s_per_sample` trước.
- Sửa thực thể sau ASR cho tên trong danh bạ (`contacts.search.query` không fuzzy).
- Synthetic cho 11 app chỉ có ở public bằng simulator.
