# DataMate — Codex trực tiếp tính đáp án

Codex đọc câu hỏi, rulebook và các bảng gốc của Public Test, viết phép tính riêng cho từng nhóm câu hỏi rồi thực thi bằng Python trên CPU. Không tải/chạy model local, không gọi model API trong các script, không dùng lại submission baseline. Các script tái chạy phép tính Codex đã lập trong phiên làm việc; chúng không phải agent tổng quát dùng cho Private Test.

Đầu ra để nộp: **`submission.jsonl`** trong thư mục này. `report.json` ghi kiểm tra cấu trúc và phân bổ kiểu đáp án; `evidence_*.json` ghi dấu vết từng câu. Điểm chính xác phải lấy từ leaderboard vì Public Test không có nhãn.

Tái chạy từ root repo:

```bash
uv venv /tmp/datamate-codex-venv
uv pip install --python /tmp/datamate-codex-venv/bin/python -r notebook-kaggle/datamate/LLM-API-Agents/codex/requirements.txt
/tmp/datamate-codex-venv/bin/python notebook-kaggle/datamate/LLM-API-Agents/codex/run.py
```

Dữ liệu mặc định: `dataset/datamate/public_test/`, chỉ đọc. Có thể đặt biến `DATAMATE_DATA` trỏ tới bản sao của đúng split này. `run.py` kiểm SHA-256 câu hỏi/rulebook trước khi chạy để tránh áp phép tính theo thứ tự câu hỏi lên một phiên bản khác. Cache đọc Excel chỉ nằm trong `/tmp/datamate-codex-cache/`.

Chỉ ghép và kiểm tra lại các kết quả đã tính:

```bash
/tmp/datamate-codex-venv/bin/python notebook-kaggle/datamate/LLM-API-Agents/codex/run.py --merge-only
```

Phân công code: `solve_root.py` + `root_data.py` (dev_001–004, 009–010), `solve_pharmacy_pos.py` (005–006), `solve_retail_baby.py` (007–008, 013), `solve_phone_repair.py` (011–012, 014); `common.py` xử lý định dạng số/ngày và đọc Excel. `audit_root.py` đối chiếu doanh thu từng đơn và số dư công nợ với số tổng phát hành trong dữ liệu. Các đối chiếu này không phải phép đo độ chính xác trên nhãn Public Test.

Các câu thiếu thông tin trong dữ liệu được ghi `unanswerable`; câu chưa đủ chắc chắn sau rà soát được ghi `abstain`. File cuối không tự động nộp lên portal.
