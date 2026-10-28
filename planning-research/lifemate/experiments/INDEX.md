# LifeMate — Sổ thực nghiệm

Mỗi run một file `YYYY-MM-DD_<run_id>.md` trong thư mục này, theo mẫu [`../../_RUN_TEMPLATE.md`](../../_RUN_TEMPLATE.md). Có thể đăng ký trước với status `planned`; **ngay khi bắt đầu** đổi sang `running`, cập nhật khi xong. `planned` chưa tiêu GPU và chưa có kết quả. Kiến thức rút ra ghi vào [`../insights.md`](../insights.md).

## Tốt nhất hiện tại

| Hạng mục | run_id | Giá trị |
|---|---|---|
| Public LB | — | chưa nộp |
| Local dev (diagnostic) | — | — |
| Local dev (e2e) | — | — |
| ASR WER dev | — | — |

## Danh sách run

| run_id | Ngày | Account | Status | Parent | Thay đổi chính | Kết quả chính | Public | Card |
|---|---|---|---|---|---|---|---|---|
| lm-B-000-local-dryrun | 2026-10-29 | local (không Kaggle) | done | none | notebook agent đầu tiên, LLM giả | oracle 100.00 điểm, hash 100% (60 mẫu train) | — | [card](2026-10-29_lm-B-000-local-dryrun.md) |
| lm-B-001-baseline-paired-dev | 2026-10-29 (lập kế hoạch) | chưa chọn | planned | lm-B-000-local-dryrun | model thật; paired clean/ASR cùng dev_seen | chưa chạy; chưa có score | — | [card](2026-10-29_lm-B-001-baseline-paired-dev.md) |
| lm-B-002-baseline-public | 2026-10-29 | acc3 | failed | lm-B-000 | lần chạy Kaggle đầu (public, PhoWhisper + Qwen3-4B) | ASR 0/1933: `pip install vllm` thay torch → torchaudio lệch CUDA → transformers.pipeline lỗi; không có submission | — | [card](2026-10-29_lm-B-002-baseline-public.md) |

## Lần nộp Public Leaderboard (tối đa 5 lần/ngày/track)

| Ngày | Lần trong ngày | run_id | Account nộp | File | Điểm LB | Ghi chú |
|---|---|---|---|---|---|---|

## Quota GPU đã dùng theo account

Ghi số giờ GPU mỗi run để chia quota; username thật chỉ nằm trong `AGENTS.local.md`.

| Tuần (bắt đầu) | Account | Run | GPU-hour |
|---|---|---|---|
