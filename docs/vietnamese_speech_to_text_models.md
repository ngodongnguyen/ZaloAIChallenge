# Đánh Giá & So Sánh Các Mô Hình Speech-to-Text (STT / ASR) Tiếng Việt Tốt Nhất Hiện Nay

Báo cáo tổng hợp, phân tích định lượng (Word Error Rate - WER) và định tính giữa các mô hình **Mã nguồn mở (Open Source)** và giải pháp **Thương mại (Commercial)** phục vụ nhận dạng tiếng nói tiếng Việt.

---

## 1. Bảng Tổng Hợp So Sánh Nhanh

| Nhóm | Mô hình / Dịch vụ | Nhà phát triển | Tham số | WER (Thấp nhất/Chuẩn) | License / Triển khai | Phù hợp nhất cho |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Open Source** ⭐ | **PhoWhisper-large** | VinAI Research | 1.55B | **4.67%** (VIVOS), 8.14% (CMV) | BSD-3-Clause (Self-host GPU) | Độ chính xác tiếng Việt cao nhất, nhận diện giọng 3 miền |
| **Open Source** | **VietASR** (Zipformer) | SJTU & Tencent | 68M | **7.68%** (GigaSpeech2), 8.31% (Avg) | Apache 2.0 (CPU / Streaming) | Hệ thống Streaming độ trễ thấp, tài nguyên máy chủ tiết kiệm |
| **Open Source** | **Zipformer-30M-RNNT** | Cộng đồng (Sherpa-ONNX)| 30M | **7.56%** (GigaSpeech2), 8.10% (VLSP25)| Open-source (Edge / Onnx) | Chạy On-device (Mobile, Raspberry Pi, CPU nhẹ) |
| **Open Source** | **Whisper large-v3 / turbo**| OpenAI | 1.55B / 809M | 8.59% (FLEURS) | MIT (Self-host) | Code-switching (Anh - Việt xen lẫn), dịch thuật |
| **Commercial (VN)** ⭐| **Viettel AI STT** | Viettel AI | Proprietary | **~96% Acc** (WER 4-8%) | Cloud API / On-premise | Tổng đài Call Center 8kHz, giọng địa phương miền Trung khó |
| **Commercial (VN)** | **VinBigData ViVoice** | VinBigData | Proprietary | **WER < 6%** (LLM Rescoring) | Enterprise Cloud / On-premise | Trợ lý ảo (ViVi ô tô), văn bản hành chính, pháp lý, y tế |
| **Commercial (VN)** | **FPT.AI Speech to Text**| FPT Smart Cloud | Proprietary | ~92% - 96% Acc | SaaS API / Enterprise | Bóc băng ghi âm tài chính, ngân hàng, CSKH |
| **Commercial (VN)** | **Zalo AI (Kiki Engine)**| Zalo (VNG) | Proprietary | Tối ưu môi trường xe hơi | Tích hợp hệ sinh thái Kiki | Trợ lý ảo xe hơi, xử lý tiếng ồn tạp âm di động |
| **Commercial (Global)**| **Google Cloud Chirp 2** | Google Cloud (USM) | Multi-Billion | **8.38%** (FLEURS) | Pay-as-you-go Cloud API | Đa ngôn ngữ chuẩn mực, hạ tầng toàn cầu |
| **Commercial (Global)**| **Speechmatics Enhanced**| Speechmatics | Proprietary | **7.14%** (FLEURS) | Cloud API / On-prem container | Độ chính xác âm học quốc tế, giữ dấu câu chuẩn xác |
| **Commercial (Global)**| **Deepgram (Nova-2 / 3)** | Deepgram | Proprietary | ~11.36% (FLEURS) | Streaming WebSocket API | Voice AI Agent thời gian thực (latency < 300ms, chi phí rẻ) |

---

## 2. Phân Tích Chi Tiết Nhóm Open-Source

### 1. PhoWhisper (VinAI Research) - Lựa chọn số 1 về độ chính xác
- **Công bố khoa học:** ICLR 2024 Tiny Papers ([arXiv:2406.02555](https://arxiv.org/abs/2406.02555)).
- **Kho mô hình:** Hugging Face `vinai/PhoWhisper-large` (cùng các biến thể: `tiny`, `base`, `small`, `medium`).
- **Dữ liệu huấn luyện:** 844 giờ audio tiếng Việt gán nhãn chuẩn từ 26.000 người nói đại diện cho 63 tỉnh thành Việt Nam kết hợp bộ tăng cường tạp âm (audiomentations).
- **Kết quả Benchmark WER:**
  - VIVOS: **4.67%** (Large), 4.97% (Medium), 6.33% (Small).
  - Common Voice 14 (Vi): **8.14%** (Large), 8.27% (Medium).
  - VLSP 2020 Task-1 (Môi trường sạch): **13.75%** (Large).
  - VLSP 2020 Task-2 (Môi trường nhiễu/hội thoại): **26.68%** (Large).
- **Đánh giá:** Mô hình mã nguồn mở hiểu ngữ âm tiếng Việt và dấu thanh điệu chuẩn xác nhất, giấy phép BSD-3-Clause thương mại thân thiện.

### 2. VietASR (Zipformer RNN-T) - Lựa chọn số 1 về hiệu năng & Streaming
- **Công bố khoa học:** [arXiv:2505.21527](https://arxiv.org/abs/2505.21527) (SJTU & Tencent AI Lab).
- **Kiến trúc:** Causal/Non-causal Zipformer chỉ **68M parameters** (nhẹ hơn Whisper 20 lần).
- **Điểm đột phá:** Pre-train trên 70.000 giờ YouTube tiếng Việt và fine-tune bằng 50 giờ dữ liệu gán nhãn thông qua cơ chế ASR-biased HuBERT codebook.
- **Kết quả Benchmark WER:**
  - GigaSpeech 2 (YouTube đời thực): **7.68%** (vượt trội hơn Whisper-large-v3 đạt 17.94%).
  - Common Voice: **11.46%** (Whisper-large-v3 đạt 13.74%).
  - FLEURS: **10.96%**.
  - Trung bình (Avg WER): **8.31%**.
  - Streaming Causal: WER **10.13%** với độ trễ chỉ 320ms - 640ms.
- **Đánh giá:** Tối ưu vượt trội cho các hệ thống cần triển khai trên CPU hoặc card GPU phổ thông mà vẫn đạt hiệu quả ngang ngửa model tỷ tham số.

---

## 3. Phân Tích Chi Tiết Nhóm Thương Mại (Commercial APIs)

### 1. Viettel AI Speech to Text
- **Đặc trưng cốt lõi:** Được tối ưu riêng cho mạng viễn thông (băng thông hẹp 8kHz) và tổng đài Contact Center.
- **Thế mạnh:** Xử lý tốt nhất trên thị trường đối với các giọng địa phương đặc thù (Bắc Trung Bộ: Nghệ An, Hà Tĩnh, Quảng Bình; miền Tây Nam Bộ).
- **Triển khai:** Hỗ trợ Cloud SaaS API, Private Cloud và On-premise cho doanh nghiệp yêu cầu bảo mật dữ liệu tuyệt đối theo tiêu chuẩn ngân hàng/quân đội.

### 2. VinBigData (ViVoice / VinBase)
- **Đặc trưng cốt lõi:** Huấn luyện trên hơn 30.000 giờ dữ liệu giọng nói tiếng Việt thực tế. Lõi ASR cho trợ lý ảo xe điện VinFast (ViVi).
- **Công nghệ nổi bật:** Tích hợp mô hình ngôn ngữ lớn (LLM Rescoring - Interspeech) giúp giảm 30% WER đối với thuật ngữ chuyên sâu, tên địa danh, từ ngữ pháp lý và hành chính.

### 3. FPT.AI Speech to Text
- **Đặc trưng cốt lõi:** Nền tảng Voicebot doanh nghiệp quy mô lớn nhất Việt Nam, bóc băng tự động cho hàng triệu cuộc gọi CSKH mỗi tháng tại các ngân hàng lớn (TPBank, Vietcombank, MB,...).

### 4. Dịch vụ Quốc tế (Global APIs)
- **Google Cloud Speech-to-Text (Chirp 2):** Dựa trên Universal Speech Model (USM), WER trên FLEURS ~8.38%, ổn định tuyệt đối cho các hệ thống đa quốc gia.
- **Deepgram (Nova-2 / Nova-3):** Dẫn đầu thế giới về độ trễ siêu thấp (< 300ms) và chi phí API cực rẻ, là lựa chọn số 1 cho các ứng dụng Voice Agent đàm thoại hai chiều thời gian thực (duplex conversational AI).
- **Speechmatics:** Dẫn đầu benchmark âm học tiếng Việt (WER 7.14% trên FLEURS), xử lý dấu câu và định dạng số rất tự nhiên.

---

## 4. Khuyến Nghị Lựa Chọn Theo Nhu Cầu Cụ Thể

1. **Cần tự triển khai On-Premise / Không lộ dữ liệu ra ngoài / Miễn phí bản quyền:**
   👉 Chọn **PhoWhisper-large** (nếu có GPU >= 8GB VRAM) hoặc **PhoWhisper-small** (GPU 4GB hoặc CPU).
2. **Cần xử lý Streaming thời gian thực trên Server nội bộ / CPU tiết kiệm chi phí:**
   👉 Chọn **VietASR (Zipformer 68M)** hoặc **Zipformer-30M-RNNT (Sherpa-ONNX)**.
3. **Làm tổng đài CSKH, Voicebot viễn thông, nhận diện giọng địa phương 3 miền phức tạp:**
   👉 Chọn **Viettel AI** hoặc **VinBigData**.
4. **Xây dựng Voice AI Agent trò chuyện phản hồi tức thì (< 300ms) qua WebSocket:**
   👉 Chọn **Deepgram (Nova)** hoặc **Google Chirp 2 Streaming**.
5. **Dữ liệu có nhiều từ ngữ xen lẫn tiếng Anh (Code-switching):**
   👉 Chọn **OpenAI Whisper (large-v3 / turbo)**.
