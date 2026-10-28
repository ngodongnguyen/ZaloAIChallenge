# DataMate — Sổ thực nghiệm

Mỗi run một file `YYYY-MM-DD_<run_id>.md` trong thư mục này, theo mẫu [`../../_RUN_TEMPLATE.md`](../../_RUN_TEMPLATE.md). Có thể đăng ký trước với status `planned`; **ngay khi bắt đầu** đổi sang `running`, cập nhật khi xong. `planned` chưa tiêu GPU và chưa có kết quả. Kiến thức rút ra ghi vào [`../insights.md`](../insights.md).

## Tốt nhất hiện tại

| Hạng mục | run_id | Giá trị |
|---|---|---|
| Public LB | — | chưa nộp |
| Local PA dev_tune | — | — |
| Coverage (tỉ lệ không abstain) | — | — |

## Danh sách run

| run_id | Ngày | Account | Status | Parent | Thay đổi chính | Kết quả chính | Public | Card |
|---|---|---|---|---|---|---|---|---|
| dm-D-000-local-dryrun | 2026-10-29 | local (không Kaggle) | done | none | notebook helper-agent đầu tiên, LLM giả | scorer 3000/3000 nhãn train; profiler 44/44 shop | — | [card](2026-10-29_dm-D-000-local-dryrun.md) |
| dm-D-001-baseline-dev | 2026-10-29 (lập kế hoạch) | chưa chọn | planned | dm-D-000-local-dryrun | Qwen3-4B thật, helper giữ nguyên trên dev_tune cố định | chưa chạy; chưa có PA | — | [card](2026-10-29_dm-D-001-baseline-dev.md) |
| dm-D-002-baseline-public | 2026-10-29 | acc3 | failed | dm-D-000 | lần chạy Kaggle đầu (public, Qwen3-4B) | profiler xong; worker chết lúc nạp model: `pip install vllm` thay torch → torchaudio lệch CUDA → cả vLLM lẫn transformers lỗi; không có submission | — | [card](2026-10-29_dm-D-002-baseline-public.md) |

## Lần nộp Public Leaderboard (tối đa 5 lần/ngày/track)

| Ngày | Lần trong ngày | run_id | Account nộp | File | Điểm LB | Ghi chú |
|---|---|---|---|---|---|---|

## Quota GPU đã dùng theo account

Ghi số giờ GPU mỗi run để chia quota; username thật chỉ nằm trong `AGENTS.local.md`.

| Tuần (bắt đầu) | Account | Run | GPU-hour |
|---|---|---|---|
