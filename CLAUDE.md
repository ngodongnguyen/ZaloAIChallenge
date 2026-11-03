@AGENTS.md
@AGENTS.local.md

Hướng dẫn Add Input Kaggle: luôn gửi **tên input + URL đầy đủ trên cùng một dòng**, mỗi input một dòng, để user copy; dùng owner thật trong hội thoại. Theo mục "Cách gửi input Kaggle cho user" trong `AGENTS.md`.

Notebook Kaggle: ưu tiên **gắn sẵn mọi input được hỗ trợ vào `metadata.kaggle.dataSources` trong `.ipynb`**, pin đúng version và kiểm quyền truy cập, kèm GPU/Internet, để user import rồi Run All. Không chỉ sửa `kernel-metadata.json`; chỉ hướng dẫn Add Input thủ công cho phần không thể gắn sẵn, kèm tên + link và lý do.

Thư viện Kaggle: tuân thủ mục thư viện trong `AGENTS.md`. Chỉ thêm gói nặng / cài lâu vào wheels; gói nhỏ, nhẹ (như `wrapt`) cài trực tiếp trong notebook, không yêu cầu rebuild wheels. vLLM luôn ở venv riêng; gói worker cần phải cài vào đúng venv.

Tracking: mỗi track một file `.vscode/history_plan/<track>.md` (bảng run + kết quả + điều đã biết). Mỗi run Kaggle một dòng. Không tạo file plan, run card hay snapshot; không tạo lại `planning-research/`.

Input dataset: mọi notebook phải có khối `metadata.kaggle` (dataset cuộc thi của track, GPU T4, Internet) theo mục "Input dataset trong notebook" trong `AGENTS.md`, vì user Import Notebook thủ công.

Dataset local: bản tải sẵn của 2 track ở `dataset/lifemate/train+public_test/` và `dataset/datamate/{train,public_test}/` (chỉ đọc, gitignored). Đọc trực tiếp khi cần xem format / EDA, không tải lại; notebook Kaggle vẫn dùng `/kaggle/input/...`. Chi tiết: mục `dataset/` trong `AGENTS.md`.

Kaggle push: mọi notebook/kernel và dataset đẩy hay tạo trên Kaggle (kể cả `push --no-run`) chỉ thuộc tài khoản mặc định `acc3` (username trong `AGENTS.local.md`); không tạo trên tài khoản khác. Hết quota GPU thì báo user, user tự share thủ công cho tài khoản còn GPU. Theo mục "Rules khi dùng Kaggle" trong `AGENTS.md`.
