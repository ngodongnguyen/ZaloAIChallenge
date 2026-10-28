# DataMate — notebook baseline

Kế hoạch: [`planning-research/datamate/planning.md`](../../planning-research/datamate/planning.md).

`helper-agent/datamate_helper_agent.ipynb`: làm sạch dữ liệu mọi shop (CPU) → Qwen3-4B-Instruct-2507 viết code pandas trên bảng sạch + helper → chạy code trong process riêng → format 9 kiểu đáp án → `submission.jsonl`.

## Chạy để có file nộp Public Test

1. Kaggle → New Notebook → File → Import Notebook → `helper-agent/datamate_helper_agent.ipynb`.
2. Add Input: dataset `thanhkhil/zaloai-datamate-dataset`. Settings: **GPU T4 x2**, Internet **On**.
3. **Save Version → Save & Run All**. Xong tải `/kaggle/working/submission.jsonl` ở tab Output → nộp Public Leaderboard.

Mặc định `split="public_test"`, `finalize=True` (câu chưa kịp xử lý ghi `abstain`, 0 điểm). Đo trên train: `split="train"`, `subset="dev_tune"` → `merge_report.json` có PA theo shop / kiểu đáp án.
Hết giờ giữa chừng: chạy version mới, attach output cũ làm input, điền đường dẫn vào `resume_from_dirs` (cùng config).

Code nằm trong các cell `%%writefile` (thư viện `dmlib` ghi vào `/kaggle/working/lib`); sửa logic thì sửa cell đó.
