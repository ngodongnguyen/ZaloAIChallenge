# CẨM NANG TRA CỨU NHANH CHO AI AGENTS & SUBAGENTS (CHEATSHEET)
## ZALO AI CHALLENGE 2026 — LIFEMATE & DATAMATE

Tài liệu này được tối ưu hóa riêng cho các AI Subagents và Developers trong quá trình lập trình, kiểm thử và đóng gói nghiệm thu.

---

## 1. BẢNG TRA CỨU NHANH (QUICK REFERENCE MATRIX)

| Tiêu chí | Track 1: LifeMate | Track 2: DataMate |
| :--- | :--- | :--- |
| **Bản chất** | Voice-to-Action Agent (Audio $\rightarrow$ Tool Calls) | Tabular Data Agent (Messy Excel/CSV $\rightarrow$ Answer) |
| **Tên file nộp** | `submission.jsonl` | `submission.jsonl` |
| **Cấu trúc dòng** | `{"id": "...", "tool_calls": [...], "state_hash": "..."}` | `{"id": "...", "type": "...", "answer": ...}` |
| **Giới hạn tham số (Model Budget)** | Toàn hệ thống < 6B, riêng LLM < 5B (context $\le$ 16K) | Toàn hệ thống $\le$ 15B (tính toàn bộ tham số MoE) |
| **Mốc phát hành mô hình cho phép** | Công khai trước ngày 26/10/2026 | Công khai trước ngày 01/10/2026 |
| **Metric đánh giá** | $F_1 = \frac{2PR}{P+R}$ với **Cổng chặn State Hash** | Penalized Accuracy: $\text{PA} = \frac{1}{N}\sum s_i$ ($s_i \in \{+1, 0, -1\}$) |
| **Điều kiện bị 0 điểm** | $\text{hash}_{\text{sys}} \neq \text{hash}_{\text{gold}}$ hoặc $M = 0$ | Bỏ trống / Chọn `{"type": "abstain"}` |
| **Phần cứng máy chấm** | 1× NVIDIA A30 (24GB VRAM), 16 vCPUs, 64GB RAM | 2× NVIDIA RTX 5090 (64GB total VRAM), 512GB RAM (CUDA $\ge$ 12.8) |
| **Thời gian tối đa** | $\le 30$ giây / mẫu | Tối đa 6 giờ cho toàn bộ Private Test set |
| **Độ tất định** | Bắt buộc chạy lại khớp kết quả | $\|\Delta \text{PA}\| \le 0.03$, bắt buộc **Greedy Decoding (temp=0)** |

---

## 2. CHECKLIST TRÁNH 12 CẠM BẪY CHẾT NGƯỜI (FATAL PITFALLS)

### Nhóm Bẫy Track 1 (LifeMate):
- [ ] **Bẫy 1: Nghiêm cấm vi phạm giới hạn kích thước Model**: Toàn bộ hệ thống phải < 6B tham số, riêng LLM phải < 5B tham số và context length $\le 16\text{K}$ tokens. Vi phạm trần tham số sẽ bị truất quyền thi đấu.
- [ ] **Bẫy 2: Không đọc trực tiếp `state.json`**: Mọi thao tác phải đi qua `ToolEnv(split, id, root)`.
- [ ] **Bẫy 3: Luôn `env.validate()` trước khi `env.call()`**: Lỗi schema trong `env.call()` sẽ bị ghi nhận vĩnh viễn vào `tool_calls` nộp về.
- [ ] **Bẫy 4: Tuân thủ `x-ref`**: Không bao giờ bịa ID thiết bị/người dùng; phải gọi tool `read/search` để lấy ID trước.
- [ ] **Bẫy 5: Nhận diện thiếu thông tin**: Khi người dùng thiếu dữ liệu bắt buộc $\rightarrow$ gọi `sys.ask_missing_info` với `tool_name` và `params`.
- [ ] **Bẫy 6: Xử lý Out-of-Scope**: Không app nào xử lý được $\rightarrow$ trả về `tool_calls: []` kèm hash của state ban đầu.

### Nhóm Bẫy Track 2 (DataMate):
- [ ] **Bẫy 7: TUYỆT ĐỐI CẤM TÍNH FIFO GIÁ VỐN**: Giá vốn dòng bán = $Số\_lượng \times Giá\_vốn\_ghi\_trên\_dòng$. Cấm tính FIFO, cấm tự tính từ phiếu nhập.
- [ ] **Bẫy 8: Không nhầm lẫn `unanswerable` và `abstain`**:
  - Dữ liệu thiếu $\rightarrow$ nộp `{"type": "unanswerable", "reason": "..."}` $\implies$ **Được +1 điểm**.
  - Code lỗi / nghi ngờ $\rightarrow$ nộp `{"type": "abstain"}` $\implies$ **Được 0 điểm**.
- [ ] **Bẫy 9: Đơn trả 1 phần $\rightarrow$ Bỏ CẢ ĐƠN khỏi doanh thu**: Kể cả đơn chỉ trả 1 món, toàn bộ đơn đó không được tính vào doanh thu.
- [ ] **Bẫy 10: Đơn đang giao $\rightarrow$ VẪN TÍNH DOANH THU**: Trạng thái `đang giao` vẫn được tính vào doanh thu bình thường.
- [ ] **Bẫy 11: Bán chạy nhất $\rightarrow$ Xếp theo ĐƠN VỊ CƠ BẢN**: Xếp theo `sum(quantity_base)`, không xếp theo doanh thu tiền hay số lần xuất hiện.
- [ ] **Bẫy 12: Format cực chuẩn 9 kiểu dữ liệu**:
  - `percent`: Con số phần trăm thực tế (12.4% ghi là `12.4`, **cấm** `0.124`, **cấm** `%`).
  - `table`: Phải đúng JSON `{"columns": [...], "rows": [[...], ...]}`.
  - `list`: Dù 1 phần tử vẫn phải là mảng `["..."]`.
  - `number`: Tiền tệ là số nguyên VND thuần túy, cấm chữ "đ" hay dấu phân cách.

---

## 3. CÁC LỆNH DÒNG LỆNH HỮU ÍCH (CLI CHEATSHEET)

### 3.1. Thao tác với `zaic_tools` (LifeMate)
```bash
# Liệt kê app và tools trong catalog của split
python -m zaic_tools --split public_test --root . --apps

# Kiểm tra schema của toàn bộ file nộp bài trước khi submit
python -m zaic_tools --split public_test --root . --validate submission.jsonl

# Chạy thử 1 tool trên 1 sample cụ thể
python -m zaic_tools --split public_test --root . --id public_0001 --tool smart_home.get_status --args '{}'
```

### 3.2. Quản lý Pool 9 Tài khoản Kaggle
```bash
# Xem danh sách và trạng thái 9 tài khoản
kaggle-acc list

# Xem tài khoản hiện tại
kaggle-acc current

# Chuyển đổi tài khoản (theo tên hoặc số thứ tự từ 1 đến 9)
kaggle-switch <user>
kaggle-switch 4

# Chạy lệnh trực tiếp dưới tài khoản phụ
kaggle-acc run <user> kaggle datasets download -d thanhkhil/zaloai-lifemate-dataset
```

### 3.3. Đóng gói và kiểm thử Docker Offline
```bash
# Build image
docker build -t zalo2026-solution .

# Chạy test giả lập hoàn toàn OFFLINE giống máy chấm BTC
docker run --gpus all --network none -v /path/to/test_data:/data zalo2026-solution
```
