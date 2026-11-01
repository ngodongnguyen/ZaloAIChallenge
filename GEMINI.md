# Quy tắc workspace cho Antigravity / agy

Đọc và tuân thủ toàn bộ [AGENTS.md](AGENTS.md) trước khi làm việc trong repo; đây là nguồn quy tắc chung. Khi dùng Kaggle, đọc `AGENTS.local.md` nếu tồn tại, không đưa credentials/username thật vào file tracked.

Thư viện Kaggle: tuân thủ mục thư viện trong `AGENTS.md`. Chỉ thêm gói nặng / cài lâu vào wheels; gói nhỏ, nhẹ (như `wrapt`) cài trực tiếp trong notebook, không yêu cầu rebuild wheels. vLLM luôn ở venv riêng; gói worker cần phải cài vào đúng venv.

Tracking: mỗi track một file `.vscode/history_plan/<track>.md` (bảng run + kết quả + điều đã biết). Mỗi run Kaggle một dòng. Không tạo file plan, run card hay snapshot; không tạo lại `planning-research/`.

Input dataset: mọi notebook phải có khối `metadata.kaggle` (dataset cuộc thi của track, GPU T4, Internet) theo mục "Input dataset trong notebook" trong `AGENTS.md`, vì user Import Notebook thủ công.

Dataset local: bản tải sẵn của 2 track ở `dataset/lifemate/train+public_test/` và `dataset/datamate/{train,public_test}/` (chỉ đọc, gitignored). Đọc trực tiếp khi cần xem format / EDA, không tải lại; notebook Kaggle vẫn dùng `/kaggle/input/...`. Chi tiết: mục `dataset/` trong `AGENTS.md`.
