# ZALO AI CHALLENGE 2026 — TỔNG QUAN CHÍNH THỨC & QUY CHẾ THI ĐẤU
### Phiên bản tối ưu cho Codex, Claude & Antigravity (AGY) — Mô hình tự do, tập trung tối đa kết quả

Tài liệu này là nguồn thông tin chính thức, cung cấp toàn bộ quy chế thi đấu, tổng quan 2 bảng đấu, cơ chế chấm điểm và quy chuẩn đóng gói nghiệm thu.

---

## 1. THÔNG TIN CHUNG & CỔNG THI ĐẤU CHÍNH THỨC

* **Đơn vị tổ chức**: VNG Corporation / Zalo.
* **Thời gian mùa giải**: 26/10/2026 – 12/12/2026.
* **Tổng giải thưởng**: **12.000 USD** tiền mặt:
  * **Quán quân (First Prize)**: 3.500 USD / track.
  * **Á quân (Second Prize)**: 2.500 USD / track.
  * **Quy tắc Top-2 Exclusive**: **Chỉ có Top 2 đội dẫn đầu mỗi track** được vào vòng Chung kết và nhận giải. Đội đứng thứ 3 trở đi hoàn toàn không có giải thưởng.

### Hệ thống liên kết chính thức (Canonical URLs)
* **Trang chủ cuộc thi**: [https://challenge.zalo.ai](https://challenge.zalo.ai)
* **Cổng Track 1 (LifeMate)**: [https://challenge.zalo.ai/portal/lifemate](https://challenge.zalo.ai/portal/lifemate)
* **Cổng Track 2 (DataMate)**: [https://challenge.zalo.ai/portal/datamate](https://challenge.zalo.ai/portal/datamate)
* **Quy chế cuộc thi (Regulations)**: [https://challenge.zalo.ai/regulations](https://challenge.zalo.ai/regulations)
* **Thể lệ bản tiếng Việt (PDF)**: [https://dl-challenge.zalo.ai/zalo-ai-challenge-regulations-vi.pdf](https://dl-challenge.zalo.ai/zalo-ai-challenge-regulations-vi.pdf)
* **Quản lý đội thi**: [https://challenge.zalo.ai/manage-team](https://challenge.zalo.ai/manage-team)

---

## 2. DÒNG THỜI GIAN THI ĐẤU (OFFICIAL TIMELINE)

| Mốc thời gian | Sự kiện | Chi tiết kỹ thuật |
| :--- | :--- | :--- |
| **26/10/2026 (10:00 GMT+7)** | **Chính thức mở đề & Dev Kit** | Phát hành dữ liệu Train & Public Test, mở cổng nộp Public Leaderboard (tối đa 5 lần/ngày). |
| **16/11/2026 (23:59 GMT+7)** | **HẠN CHÓT NỘP DOCKER** | Đóng băng giải pháp. Nộp toàn bộ source code (train + infer), weights và Docker package. |
| **20/11/2026 (09:00 GMT+7)** | **Mở Private Test** | Ban tổ chức phát hành tập dữ liệu kiểm thử ẩn (Private Test). |
| **23/11/2026 (23:59 GMT+7)** | **Hạn nộp kết quả Private Test** | Nộp file kết quả dự đoán cuối cùng chạy từ Private Test. |
| **02/12/2026** | **Công bố Finalists** | Công bố Top 2 đội tuyển xuất sắc nhất mỗi track bước vào Chung kết. |
| **12/12/2026** | **Gala Chung kết & Trao giải** | Thuyết trình giải pháp và vinh danh quán quân. |

---

## 3. TÓM TẮT HAI BẢNG ĐẤU (TRACKS)

### Track 1: LifeMate — *Agentic AI That Turns Your Voice Into Action*
* **Bản chất**: Trợ lý ảo AI tự chủ nhận lệnh bằng giọng nói tiếng Việt (đa phương ngữ Bắc - Trung - Nam, có chêm xen tiếng Anh), lập kế hoạch nhiều bước và thực thi tool call trên môi trường ứng dụng giả lập (`zaic_tools`).
* **Đầu vào**: Audio giọng nói (`.wav`), ngữ cảnh người dùng (`context`: vị trí, thời gian, hồ sơ cá nhân), danh sách app đã cài (`installed_apps`), lịch sử hội thoại (`history`).
* **Đầu ra**: File `submission.jsonl` chứa chuỗi `tool_calls` và mã băm trạng thái cuối `state_hash`.
* **Lựa chọn mô hình**: **Tự do hoàn toàn**. Được phép sử dụng mọi mô hình (ASR + LLM/Agentic) tối ưu nhất để đạt kết quả cao nhất.
* **Độ trễ**: Thời gian suy luận $\le 30$ giây / mẫu.

### Track 2: DataMate — *Talking To Your Data Through Agentic AI*
* **Bản chất**: Trợ lý AI phân tích dữ liệu bảng thực tế (CSV, Excel nhiều sheet, SQLite) của hộ kinh doanh/tiểu thương Việt Nam để trả lời các câu hỏi bằng tiếng Việt về doanh thu, công nợ, kho bãi, nhân viên, khách hàng.
* **Đầu vào**: Câu hỏi bằng tiếng Việt, mốc thời gian `as_of_date` ("Hôm nay"), dữ liệu bảng của doanh nghiệp (`data/<business_id>/`).
* **Đầu ra**: File `submission.jsonl` gồm `id`, `type`, `answer` tuân thủ 9 format chuẩn (`number`, `percent`, `string`, `date`, `boolean`, `list`, `table`, `unanswerable`, `abstain`).
* **Lựa chọn mô hình**: **Tự do hoàn toàn**. Được phép sử dụng mọi mô hình (LLM / Code Interpreter / Agent) tối ưu nhất để đạt điểm Penalized Accuracy cao nhất.
* **Chấm điểm**: **Penalized Accuracy (PA)**: Đúng **+1**, Bỏ trống/Abstain **0**, Sai **-1**.

---

## 4. QUY CHẾ SUBMISSION & ĐÓNG GÓI DOCKER

1. **Vòng Public Test (Hàng ngày: 26/10 – 16/11)**:
   - Mỗi đội được nộp tối đa **5 lần / ngày / track** (hạn mức reset vào cuối ngày).
   - Nộp file `submission.jsonl` trực tiếp trên web portal để tính điểm Public Leaderboard.
2. **Vòng Thẩm định Docker (Bắt buộc để xét giải — Hạn chót 23:59 ngày 16/11/2026)**:
   - Các đội thi đóng băng giải pháp và nộp gói Docker hoàn chỉnh bao gồm mã nguồn và mô hình qua form "Solution Submission".
   - **Cơ chế gắn ổ đĩa khi chấm thi**:
     - Thư mục dữ liệu test được gắn vào container tại đường dẫn: `/data` (chế độ chỉ đọc - read only).
     - File kết quả suy luận bắt buộc xuất ra tại: `/result/submission.jsonl`.
   - **Hạ tầng máy chấm thi của BTC**:
     - **Track 1 (LifeMate)**: 1× NVIDIA A30 GPU (24 GB VRAM), 16-core Intel Xeon Gold 6442Y, 64 GB RAM. Giới hạn tối đa **30 giây / mẫu**.
     - **Track 2 (DataMate)**: 2× NVIDIA RTX 5090 (tổng 64 GB VRAM), 512 GB RAM. Giới hạn toàn bộ private test trong vòng **6 giờ**.
   - **Quy tắc Tái lập Tất định (Reproducibility)**:
     - Cố định mọi seed ngẫu nhiên (Python, NumPy, PyTorch), dùng giải mã tất định (Greedy decoding, `temperature = 0`).
     - Track 1: Kết quả chạy lại từ Docker phải khớp chuỗi `tool_calls` và `state_hash`.
     - Track 2: Đạt độ lệch chuẩn $\|\Delta \text{PA}\| \le 0.03$.

---

## 5. DỮ LIỆU THI ĐẤU & QUẢN LÝ TÀI KHOẢN KAGGLE

### 5.1. Bộ dữ liệu chính thức
* **LifeMate Dataset**: [https://www.kaggle.com/datasets/thanhkhil/zaloai-lifemate-dataset](https://www.kaggle.com/datasets/thanhkhil/zaloai-lifemate-dataset)
* **DataMate Dataset**: [https://www.kaggle.com/datasets/thanhkhil/zaloai-datamate-dataset](https://www.kaggle.com/datasets/thanhkhil/zaloai-datamate-dataset)

### 5.2. Quản lý Pool 9 Tài khoản Kaggle
Hệ thống đã tích hợp sẵn công cụ quản lý 9 tài khoản Kaggle toàn cầu:
* Lệnh xem danh sách tài khoản: `kaggle-acc list`
* Lệnh kiểm tra tài khoản đang active: `kaggle-acc current`
* Lệnh chuyển đổi tài khoản: `kaggle-switch <username_or_number>`
* Lệnh chạy trực tiếp dưới tài khoản cụ thể: `kaggle-acc run <username> kaggle <args...>`
