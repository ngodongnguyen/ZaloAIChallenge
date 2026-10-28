# ZALO AI CHALLENGE 2026 — MASTER DOCUMENTATION HUB
## Thư mục Tài liệu Kỹ thuật Chuẩn & Nguồn Sự Thật Duy Nhất (Single Source of Truth)

Thư mục `.vscode/docs/` chứa toàn bộ tài liệu hướng dẫn kỹ thuật, quy chế thi đấu, đặc tả chi tiết 2 bảng đấu (LifeMate & DataMate), công thức chấm điểm, bảng so khớp dữ liệu, các cạm bẫy cần tránh và mẫu prompt chuẩn hóa cho mọi AI Agent và lập trình viên làm việc trong dự án.

> ⚠️ **LƯU Ý QUAN TRỌNG CHO AI AGENTS**:
> - Mọi thông tin trong thư mục này đã được xác thực 100% với hệ thống máy chủ `challenge.zalo.ai`, Thể lệ chính thức của VNG/Zalo và bộ dữ liệu thực tế phát hành trên Kaggle.
> - Khi giải quyết bài toán, tối ưu prompt, kiểm tra logic nghiệp vụ hay định dạng nộp bài, **bắt buộc đọc kỹ tài liệu trong thư mục này, tuyệt đối không tự suy đoán**.

---

## 1. MỤC LỤC TÀI LIỆU (DOCUMENTATION INDEX)

| Tên tệp | Nội dung chính | Đối tượng sử dụng |
| :--- | :--- | :--- |
| **[`COMPETITION_OVERVIEW_2026.md`](COMPETITION_OVERVIEW_2026.md)** | Tổng quan mùa giải 2026, cơ cấu giải thưởng 12.000 USD (Quy tắc Top 2), dòng thời gian thi đấu chi tiết, quy chế nộp bài Docker offline, hạ tầng phần cứng máy chấm và liên kết cổng thi đấu. | Toàn bộ Agents & Developers |
| **[`AGENT_CHEATSHEET.md`](AGENT_CHEATSHEET.md)** | Bảng ma trận so sánh nhanh 2 track, checklist 12 cạm bẫy chết người (Fatal Pitfalls), các lệnh dòng lệnh (CLI) hữu ích cho `zaic_tools`, Kaggle CLI và Docker. | Tra cứu nhanh khi code / debug |
| **[`TRACK1_LIFEMATE_SPECIFICATION.md`](TRACK1_LIFEMATE_SPECIFICATION.md)** | Đặc tả kỹ thuật toàn diện cho Track 1: LifeMate (Voice-to-Action Agent). Cấu trúc dữ liệu audio + context, catalog công cụ OpenAPI, các cờ schema `x-ref`, `x-one-of`, `x-match: fuzzy`, thư viện giả lập `zaic_tools`, thuật toán băm `state_hash`, công thức metric F1, ngân sách mô hình (< 6B total, LLM < 5B) và phần cứng A30. | Agents phụ trách Track 1 |
| **[`TRACK2_DATAMATE_SPECIFICATION.md`](TRACK2_DATAMATE_SPECIFICATION.md)** | Đặc tả kỹ thuật toàn diện cho Track 2: DataMate (Tabular Data Agent). Cấu trúc dữ liệu CSV/Excel/SQLite, toàn bộ 10 điều khoản Rulebook chính thức (doanh thu, thời gian `as_of_date`, xếp hạng, công nợ, giá vốn cấm FIFO, ngành F&B/Dược/VLXD, xử lý `unanswerable`), bảng đặc tả 9 kiểu đầu ra, công thức Penalized Accuracy (PA), ngân sách mô hình ($\le 15\text{B}$) và phần cứng 2× RTX 5090. | Agents phụ trách Track 2 |
| **[`OFFICIAL_METRICS_AND_MATCHING_RULES.md`](OFFICIAL_METRICS_AND_MATCHING_RULES.md)** | Phiên âm và giải mã toán học chi tiết từ 3 hình ảnh gốc do Ban tổ chức công bố: công thức F1 và cổng chặn State Hash (LifeMate), bảng quy tắc so khớp 9 kiểu đầu ra (DataMate) và công thức Penalized Accuracy (DataMate). | Đánh giá, scoring & validation |
| **[`PROMPT_TEMPLATES_AND_EXAMPLES.md`](PROMPT_TEMPLATES_AND_EXAMPLES.md)** | Khung Prompt hệ thống (System Prompt), ReAct trajectory mẫu cho LifeMate, System Prompt chuyên dụng cho Code Interpreter Agent (DataMate), kèm các đoạn mã Python mẫu (hàm format an toàn 9 kiểu, tính doanh thu chuẩn rulebook, xếp hạng bán chạy, tra cứu lịch âm Tết). | Triển khai Prompting & Modeling |
| **[`zalo-ai-challenge-regulations-vi.pdf`](zalo-ai-challenge-regulations-vi.pdf)** | Bản thể lệ cuộc thi chính thức bằng tiếng Việt ban hành bởi Zalo AI / VNG Corporation (Văn bản pháp lý gốc có hiệu lực cao nhất). | Tham chiếu pháp lý & điều lệ gốc |
| **[`zalo-ai-challenge-regulations-en.pdf`](zalo-ai-challenge-regulations-en.pdf)** | Official Competition Regulations in English issued by Zalo AI / VNG Corporation. | English reference regulations |

---

## 2. HÌNH ẢNH CÔNG THỨC GỐC TỪ BAN TỔ CHỨC (`.vscode/extracted_images/`)

Các hình ảnh trích xuất trực tiếp từ cổng thông tin thi đấu để đối chiếu:
- **`lifemate_formula_0.png`**: Ảnh công thức tính điểm $F_1$ kèm cổng chặn trạng thái `state_hash` của Track 1.
- **`datamate_formula_0.png`**: Ảnh bảng quy tắc so khớp chuẩn xác (Matching Rules) cho 9 kiểu dữ liệu đầu ra của Track 2.
- **`datamate_formula_1.png`**: Ảnh công thức tính điểm Penalized Accuracy (PA) của Track 2.

---

## 3. TỔNG HỢP LIÊN KẾT CHÍNH THỨC (CANONICAL LINKS)

### 3.1. Cổng Thông tin Cuộc thi
* **Trang chủ Zalo AI Challenge**: [https://challenge.zalo.ai](https://challenge.zalo.ai)
* **Track 1: LifeMate Portal**: [https://challenge.zalo.ai/portal/lifemate](https://challenge.zalo.ai/portal/lifemate)
* **Track 2: DataMate Portal**: [https://challenge.zalo.ai/portal/datamate](https://challenge.zalo.ai/portal/datamate)
* **Quy chế cuộc thi**: [https://challenge.zalo.ai/regulations](https://challenge.zalo.ai/regulations)
* **Quản lý đội thi**: [https://challenge.zalo.ai/manage-team](https://challenge.zalo.ai/manage-team)
* **Email hỗ trợ kỹ thuật của BTC**: `aichallenge@zalo.me`

### 3.2. Bộ Dữ liệu Chính thức trên Kaggle
* **Track 1 (LifeMate Dataset)**: [https://www.kaggle.com/datasets/thanhkhil/zaloai-lifemate-dataset](https://www.kaggle.com/datasets/thanhkhil/zaloai-lifemate-dataset)
* **Track 2 (DataMate Dataset)**: [https://www.kaggle.com/datasets/thanhkhil/zaloai-datamate-dataset](https://www.kaggle.com/datasets/thanhkhil/zaloai-datamate-dataset)

---

## 4. BẢNG TỔNG KẾT THÔNG SỐ KỸ THUẬT HAI TRACK

```
┌─────────────────────────────────┬────────────────────────────────────────┬────────────────────────────────────────┐
│ TIÊU CHÍ KỸ THUẬT               │ TRACK 1: LIFEMATE                      │ TRACK 2: DATAMATE                      │
├─────────────────────────────────┼────────────────────────────────────────┼────────────────────────────────────────┤
│ Bản chất bài toán               │ Voice-to-Action Agent (Agentic AI)     │ Tabular Data Analysis Agent            │
│ Ngữ cảnh & Lĩnh vực             │ Smart Home, Ride, Bank, Email, Food... │ Bán lẻ, Cà phê, Nhà thuốc, VLXD...     │
│ Đầu vào (Input)                 │ Audio .wav + Context + Tools Catalog   │ questions.jsonl + calendar.csv + Excel │
│ Đầu ra (Output)                 │ submission.jsonl (tool_calls + hash)   │ submission.jsonl (id, type, answer)    │
│ Giới hạn kích thước Model       │ Toàn hệ thống < 6B, riêng LLM < 5B     │ Tổng toàn bộ hệ thống ≤ 15B (MoE total)│
│ Mốc phát hành weights cho phép  │ Công khai trước ngày 26/10/2026        │ Công khai trước ngày 01/10/2026        │
│ Giới hạn context length         │ Tối đa 16K (16,384 tokens)             │ Không giới hạn cứng (tùy bộ nhớ GPU)   │
│ Metric đánh giá chính           │ F1-Score với Cổng chặn State Hash      │ Penalized Accuracy (PA: +1 / 0 / -1)   │
│ Điều kiện nhận 0 điểm           │ state_hash không khớp hoặc M = 0       │ Chọn {"type": "abstain"} hoặc bỏ trống │
│ Hạ tầng máy chấm của BTC        │ 1× NVIDIA A30 (24GB VRAM), 64GB RAM    │ 2× NVIDIA RTX 5090 (64GB VRAM), 512GB  │
│ Yêu cầu phần mềm máy chấm       │ Tương thích CUDA 12.1 – 12.4           │ CUDA ≥ 12.8 (Kiến trúc Blackwell)      │
│ Thời gian tối đa                │ ≤ 30 giây / mẫu                        │ Tối đa 6 giờ cho toàn bộ Private Test  │
│ Độ tất định (Reproducibility)   │ Greedy decoding (temp=0), 5 lần TB     │ |PA(uploaded) - PA(rerun)| ≤ 0.03       │
│ Giải thưởng mỗi track           │ Quán quân: $3,500 | Á quân: $2,500     │ Quán quân: $3,500 | Á quân: $2,500     │
└─────────────────────────────────┴────────────────────────────────────────┴────────────────────────────────────────┘
```

---

## 5. HƯỚNG DẪN DÀNH CHO AI AGENTS TRONG REPOSITORY NÀY

1. **Tuân thủ phân vùng thư mục**:
   - `.vscode/docs/`: Thư mục tài liệu (chỉ đọc, tham chiếu thông tin).
   - `planning-research/<track>/`: Nơi viết tài liệu phân tích, kế hoạch thực nghiệm, phân tích lỗi (chỉ chứa file Markdown).
   - `notebook-kaggle/<track>/`: Nơi triển khai code chạy được trên Kaggle (`baseline_pipeline.py`, `kernel-metadata.json`).
2. **Kaggle CLI Pool**:
   - Sử dụng lệnh `kaggle-acc run <user> kaggle ...` để chạy lệnh dưới tài khoản phụ mà không ảnh hưởng tới các Agent khác.
   - Không commit `kernel-metadata.json` hay token nhạy cảm vào Git.
3. **Chiến lược an toàn khi nộp bài**:
   - **Track 1**: Luôn gọi `env.validate()` trước `env.call()`. Không bao giờ bịa ID khi tham số có cờ `x-ref`.
   - **Track 2**: Khi không chắc chắn hoặc code crash $\rightarrow$ xuất `{"type": "abstain"}` để bảo toàn 0 điểm, tránh bị trừ -1 điểm. Tuân thủ tuyệt đối quy tắc cấm tính FIFO giá vốn và đơn trả 1 phần bỏ cả đơn khỏi doanh thu.
