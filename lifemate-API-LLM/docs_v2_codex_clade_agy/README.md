# ZALO AI CHALLENGE 2026 — MASTER DOCUMENTATION HUB (V2)
## Thư mục Tài liệu Kỹ thuật Tối ưu cho Codex, Claude & Antigravity (AGY)
### Phiên bản cập nhật: Mô hình tự do — Tập trung tối đa vào hiệu năng và kết quả

Thư mục `docs_v2_codex_clade_agy/` chứa toàn bộ tài liệu kỹ thuật, đặc tả 2 bảng đấu (LifeMate & DataMate), công thức chấm điểm, bảng so khớp dữ liệu, các cạm bẫy thực thi cần tránh và mẫu prompt chuẩn hóa cho mọi AI Agent (Codex, Claude, Antigravity) và kỹ sư phát triển.

> 🚀 **LƯU Ý CỐT LÕI CHO AI AGENTS**:
> - **Chính sách mô hình**: **TỰ DO HOÀN TOÀN**. Không bị giới hạn kích thước mô hình, số lượng tham số hay độ dài context. Mục tiêu duy nhất là lựa chọn/kết hợp những mô hình mạnh nhất (Open-source, SOTA, Commercial APIs, Fine-tuned) để đạt kết quả cao nhất trên Leaderboard.
> - **Quy tắc thực thi**: Mọi logic nghiệp vụ (10 điều khoản Rulebook của DataMate, cổng chặn State Hash và ràng buộc Schema `zaic_tools` của LifeMate) vẫn phải tuân thủ nghiêm ngặt 100%.

---

## 1. MỤC LỤC TÀI LIỆU (DOCUMENTATION INDEX)

| Tên tệp | Nội dung chính | Đối tượng sử dụng |
| :--- | :--- | :--- |
| **[`COMPETITION_OVERVIEW_2026.md`](COMPETITION_OVERVIEW_2026.md)** | Tổng quan mùa giải 2026, cơ cấu giải thưởng 12.000 USD (Quy tắc Top 2), dòng thời gian thi đấu, quy chế nộp bài Docker offline, hạ tầng phần cứng máy chấm và liên kết cổng thi đấu. | Toàn bộ Agents & Developers |
| **[`AGENT_CHEATSHEET.md`](AGENT_CHEATSHEET.md)** | Bảng ma trận so sánh nhanh 2 track, checklist các cạm bẫy chết người (Fatal Pitfalls), các lệnh dòng lệnh (CLI) hữu ích cho `zaic_tools`, Kaggle CLI và Docker. | Tra cứu nhanh khi code / debug |
| **[`TRACK1_LIFEMATE_SPECIFICATION.md`](TRACK1_LIFEMATE_SPECIFICATION.md)** | Đặc tả kỹ thuật toàn diện cho Track 1: LifeMate (Voice-to-Action Agent). Cấu trúc dữ liệu audio + context, catalog công cụ OpenAPI, các cờ schema `x-ref`, `x-one-of`, `x-match: fuzzy`, thư viện giả lập `zaic_tools`, thuật toán băm `state_hash`, công thức metric F1. | Agents phụ trách Track 1 |
| **[`TRACK2_DATAMATE_SPECIFICATION.md`](TRACK2_DATAMATE_SPECIFICATION.md)** | Đặc tả kỹ thuật toàn diện cho Track 2: DataMate (Tabular Data Agent). Cấu trúc dữ liệu CSV/Excel/SQLite, toàn bộ 10 điều khoản Rulebook chính thức (doanh thu, thời gian `as_of_date`, xếp hạng, công nợ, giá vốn cấm FIFO, ngành F&B/Dược/VLXD, xử lý `unanswerable`), bảng đặc tả 9 kiểu đầu ra, công thức Penalized Accuracy (PA). | Agents phụ trách Track 2 |
| **[`OFFICIAL_METRICS_AND_MATCHING_RULES.md`](OFFICIAL_METRICS_AND_MATCHING_RULES.md)** | Phiên âm và giải mã toán học chi tiết từ 3 hình ảnh gốc do Ban tổ chức công bố: công thức F1 và cổng chặn State Hash (LifeMate), bảng quy tắc so khớp 9 kiểu đầu ra (DataMate) và công thức Penalized Accuracy (DataMate). | Đánh giá, scoring & validation |
| **[`PROMPT_TEMPLATES_AND_EXAMPLES.md`](PROMPT_TEMPLATES_AND_EXAMPLES.md)** | Khung Prompt hệ thống (System Prompt), ReAct trajectory mẫu cho LifeMate, System Prompt chuyên dụng cho Code Interpreter Agent (DataMate), kèm các đoạn mã Python mẫu (hàm format an toàn 9 kiểu, tính doanh thu chuẩn rulebook, xếp hạng bán chạy, tra cứu lịch âm Tết). | Triển khai Prompting & Modeling |

---

## 2. HÌNH ẢNH CÔNG THỨC GỐC TỪ BAN TỔ CHỨC ([`extracted_images/`](extracted_images/))

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
│ Chính sách Model                │ Tự do hoàn toàn mọi mô hình            │ Tự do hoàn toàn mọi mô hình            │
│ Metric đánh giá chính           │ F1-Score với Cổng chặn State Hash      │ Penalized Accuracy (PA: +1 / 0 / -1)   │
│ Điều kiện nhận 0 điểm           │ state_hash không khớp hoặc M = 0       │ Chọn {"type": "abstain"} hoặc bỏ trống │
│ Hạ tầng máy chấm của BTC        │ 1× NVIDIA A30 (24GB VRAM), 64GB RAM    │ 2× NVIDIA RTX 5090 (64GB VRAM), 512GB  │
│ Thời gian tối đa                │ ≤ 30 giây / mẫu                        │ Tối đa 6 giờ cho toàn bộ Private Test  │
│ Độ tất định (Reproducibility)   │ Greedy decoding (temp=0), 5 lần TB     │ |PA(uploaded) - PA(rerun)| ≤ 0.03       │
│ Giải thưởng mỗi track           │ Quán quân: $3,500 | Á quân: $2,500     │ Quán quân: $3,500 | Á quân: $2,500     │
└─────────────────────────────────┴────────────────────────────────────────┴────────────────────────────────────────┘
```

---

## 5. HƯỚNG DẪN DÀNH CHO AI AGENTS (CODEX, CLAUDE, AGY)

1. **Tập trung vào hiệu năng & độ chính xác**:
   - Tự do kết hợp các LLM và ASR mạnh nhất hiện nay (Gemini, Claude, GPT, Qwen2.5, Llama-3, Whisper-large-v3, PhoWhisper...).
2. **Chiến lược an toàn khi nộp bài**:
   - **Track 1**: Luôn gọi `env.validate()` trước `env.call()`. Không bao giờ bịa ID khi tham số có cờ `x-ref`. Khớp chính xác `state_hash`.
   - **Track 2**: Khi không chắc chắn hoặc code crash $\rightarrow$ xuất `{"type": "abstain"}` để bảo toàn 0 điểm, tránh bị trừ -1 điểm. Tuân thủ tuyệt đối quy tắc cấm tính FIFO giá vốn và đơn trả 1 phần bỏ cả đơn khỏi doanh thu.
