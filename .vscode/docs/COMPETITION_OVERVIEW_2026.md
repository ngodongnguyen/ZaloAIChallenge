# ZALO AI CHALLENGE 2026 — TỔNG QUAN CHÍNH THỨC & QUY CHẾ THI ĐẤU

Tài liệu này là nguồn thông tin chính thức (Single Source of Truth), đã được đối chiếu và xác thực 100% với mã nguồn hệ thống thi đấu `challenge.zalo.ai`, Thể lệ chính thức của VNG/Zalo và dữ liệu thực tế.

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
* **Thể lệ bản tiếng Anh (PDF)**: [https://dl-challenge.zalo.ai/zalo-ai-challenge-regulations-en.pdf](https://dl-challenge.zalo.ai/zalo-ai-challenge-regulations-en.pdf)
* **Quản lý đội thi**: [https://challenge.zalo.ai/manage-team](https://challenge.zalo.ai/manage-team)

*(Lưu ý: Các liên kết cũ như `/portal/aero-eyes` và `/portal/road-buddy` là tàn dư của mùa giải 2024, hệ thống hiện tự động chuyển hướng về `/portal/lifemate` và `/portal/datamate`).*

---

## 2. DÒNG THỜI GIAN THI ĐẤU (OFFICIAL TIMELINE)

| Mốc thời gian | Sự kiện | Chi tiết kỹ thuật |
| :--- | :--- | :--- |
| **26/10/2026 (10:00 GMT+7)** | **Chính thức mở đề & Dev Kit** | Phát hành dữ liệu Train & Public Test, mở cổng nộp Public Leaderboard (tối đa 5 lần/ngày). |
| **16/11/2026 (23:59 GMT+7)** | **HẠN CHÓT NỘP DOCKER (21 NGÀY)** | Đóng băng giải pháp. Nộp toàn bộ source code (train + infer), weights và Docker package. |
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
* **Giới hạn tham số cứng**: **Toàn bộ hệ thống (ASR + LLM) < 6B parameters**, riêng **LLM < 5B parameters**.
* **Độ trễ**: Thời gian suy luận $\le 30$ giây / mẫu.

### Track 2: DataMate — *Talking To Your Data Through Agentic AI*
* **Bản chất**: Trợ lý AI phân tích dữ liệu bảng thực tế (CSV, Excel nhiều sheet, SQLite) của hộ kinh doanh/tiểu thương Việt Nam để trả lời các câu hỏi bằng tiếng Việt về doanh thu, công nợ, kho bãi, nhân viên, khách hàng.
* **Đầu vào**: Câu hỏi bằng tiếng Việt, mốc thời gian `as_of_date` ("Hôm nay"), dữ liệu bảng của doanh nghiệp (`data/<business_id>/`).
* **Đầu ra**: File `submission.jsonl` gồm `id`, `type`, `answer` tuân thủ 9 format chuẩn (`number`, `percent`, `string`, `date`, `boolean`, `list`, `table`, `unanswerable`, `abstain`).
* **Giới hạn tham số**: Toàn bộ hệ thống $\le 15\text{B}$ parameters.
* **Chấm điểm**: **Penalized Accuracy (PA)**: Đúng **+1**, Bỏ trống/Abstain **0**, Sai **-1**.

---

## 4. QUY CHẾ SUBMISSION & ĐÓNG GÓI DOCKER

1. **Vòng Public Test (Hàng ngày: 26/10 – 16/11)**:
   - Mỗi đội được nộp tối đa **5 lần / ngày / track** (hạn mức reset vào cuối ngày).
   - Nộp file `submission.jsonl` trực tiếp trên web portal để tính điểm Public Leaderboard. Điểm số dùng để tham khảo trong quá trình phát triển.
2. **Vòng Thẩm định Docker (Bắt buộc để xét giải — Hạn chót 23:59 ngày 16/11/2026)**:
   - Các đội thi đóng băng giải pháp và nộp gói Docker hoàn chỉnh bao gồm: mã nguồn huấn luyện (training code), mã nguồn suy luận (inference code), mô hình (weights) và tài liệu hướng dẫn qua form "Solution Submission" trên cổng thi đấu.
   - Do file Docker dung lượng lớn, đội thi cung cấp **mã kiểm tra tính toàn vẹn (checksum)** và **đường dẫn Google Drive công khai** để BTC tải về.
   - Mỗi đội được nộp tối đa **2 lần**; BTC chỉ chấm bài nộp cuối cùng.
   - **Môi trường OFFLINE tuyệt đối (`--network none`)**:
     - Mọi trọng số (weights), tokenizers, dependencies Python phải được đóng gói sẵn trong Docker image.
     - Nghiêm cấm mọi lệnh tải động qua mạng (HuggingFace Hub, PyPI, GitHub). Bất kỳ lệnh gọi mạng nào sẽ lập tức làm sập container và nhận điểm 0.
   - **Cơ chế gắn ổ đĩa khi chấm thi**:
     - Thư mục dữ liệu test được gắn vào container tại đường dẫn: `/data` (chế độ chỉ đọc - read only).
     - File kết quả suy luận bắt buộc xuất ra tại: `/result/submission.jsonl`.
   - **Hạ tầng máy chấm thi của BTC (Hardware Specs per Track)**:
     - **Track 1 (LifeMate)**:
       - GPU: **1× NVIDIA A30 GPU** (24 GB VRAM).
       - CPU: **16-core Intel Xeon Gold 6442Y**.
       - RAM: **64 GB**.
       - Giới hạn thời gian: Tối đa **30 giây / mẫu**. Mẫu quá thời gian hoặc crash tính 0 điểm.
     - **Track 2 (DataMate)**:
       - GPU: **2× NVIDIA RTX 5090** (32 GB VRAM mỗi card, tổng cộng 64 GB VRAM).
       - RAM: **512 GB**.
       - Giới hạn thời gian: Toàn bộ tập private test phải chạy xong trong vòng **6 giờ** (tính cả thời gian load model).
       - **Yêu cầu GPU Blackwell**: Bắt buộc dùng **CUDA ≥ 12.8** và bản build PyTorch / vLLM tương thích kiến trúc Blackwell.
   - **Quy tắc Tái lập Tất định (Reproducibility) & Xử lý Hòa điểm (Tie-breaking)**:
     - **Track 1 (LifeMate)**:
       - Cố định mọi seed ngẫu nhiên (Python, NumPy, PyTorch), dùng giải mã tất định (Greedy decoding, temperature = 0).
       - Nếu kết quả chạy lại sai khác với file đã nộp, BTC sẽ chạy Docker image 5 lần và lấy **điểm số trung bình** làm điểm chính thức.
       - *Tie-breaking*: Nếu bằng điểm, BTC chấm lại theo công thức F1 bỏ qua điều kiện `state_hash`. Nếu vẫn hòa, đội có thời gian xử lý trung bình mỗi mẫu thấp hơn sẽ xếp trên.
     - **Track 2 (DataMate)**:
       - Điểm số chạy lại Docker của BTC trên tập private test là điểm số chung cuộc chính thức.
       - Bài thi được coi là tái lập thành công nếu:
         $$|\text{PA}_{\text{uploaded}} - \text{PA}_{\text{rerun}}| \le 0.03$$
       - Nếu sai lệch vượt quá 0.03 mà không thể giải trình hợp lý, bài thi sẽ bị hủy kết quả.
       - Nếu container quá thời gian 6 giờ hoặc dừng sớm, chỉ những câu đã ghi vào `/result/submission.jsonl` được chấm, các câu còn lại tính là `abstain` (0 điểm).
       - *Tie-breaking*: Ưu tiên đội có tổng thời gian suy luận thấp hơn trong lần chạy lại của BTC, sau đó xét theo thời điểm nộp bài sớm hơn.
3. **Vòng Private Test (20/11 – 23/11)**:
   - 09:00 ngày 20/11: BTC phát hành tập Private Test.
   - Các đội chạy giải pháp đã đóng băng và nộp kết quả trước 23:59 ngày 23/11/2026.
   - BTC chạy lại Docker để xác minh, rà soát mã nguồn chống gian lận và công bố Top 2 đội mỗi track vào ngày 02/12/2026.

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
* Danh sách tài khoản: xem file local `AGENTS.local.md` (gitignored, không commit).
