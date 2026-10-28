# LifeMate — notebook baseline

Kế hoạch: [`planning-research/lifemate/planning.md`](../../planning-research/lifemate/planning.md).

`agent/lifemate_agent.ipynb`: ASR (PhoWhisper-medium) → agent Qwen3-4B-Instruct-2507 gọi tool qua `zaic_tools.ToolEnv` (mỗi mẫu một env, validate trước khi call) → `submission.jsonl`.

## Chạy để có file nộp Public Test

1. Kaggle → New Notebook → File → Import Notebook → `agent/lifemate_agent.ipynb`.
2. Add Input: dataset `thanhkhil/zaloai-lifemate-dataset`. Settings: **GPU T4 x2**, Internet **On**.
3. **Save Version → Save & Run All**. Xong tải `/kaggle/working/submission.jsonl` ở tab Output → nộp Public Leaderboard.

`CONFIG["mode"]`: `public` (mặc định) · `diagnostic` (train, transcript sạch; đặt `split="train"`, `subset="dev_seen"`) · `e2e` (train, transcript ASR).
Hết giờ giữa chừng: chạy version mới, attach output cũ làm input, điền đường dẫn vào `resume_from_dirs` (cùng config). Mẫu chưa xong không bao giờ bị điền hash giả.

Code nằm trong các cell `%%writefile` (thư viện `lmlib` ghi vào `/kaggle/working/lib`); sửa logic thì sửa cell đó.
