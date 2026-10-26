# HƯỚNG DẪN QUẢN LÝ KẾ HOẠCH & NGHIÊN CỨU (PLANNING & RESEARCH)
## ZALO AI CHALLENGE 2026 — LIFEMATE & DATAMATE

Thư mục này là trung tâm theo dõi chiến lược kỹ thuật, nhật ký thực nghiệm và phân tích lỗi cho hai bảng đấu:
1. **[`lifemate/`](<lifemate/>)**: Track 1 — Trợ lý ảo AI tiếp nhận giọng nói điều khiển thiết bị (Voice-to-Action).
2. **[`datamate/`](<datamate/>)**: Track 2 — Trợ lý AI phân tích dữ liệu bảng kinh doanh (Tabular Data Analytics).

---

## Cấu trúc chuẩn cho mỗi Track

Mỗi bảng đấu bao gồm 4 tài liệu chuẩn hóa:
- **`strategy.md`**: Bản mô tả kiến trúc mô hình, lựa chọn ASR / LLM / Code Interpreter, prompt engineering và pipeline thiết kế.
- **`experiment_log.md`**: Bảng nhật ký ghi lại chi tiết từng lượt chạy thử nghiệm (Run ID, Model, Tham số, Điểm số Validation / Leaderboard).
- **`error_analysis.md`**: Phân tích chuyên sâu các trường hợp dự đoán sai, vi phạm schema hoặc bị phạt điểm.
- **`submission_checklist.md`**: Danh mục kiểm tra điều kiện cần và đủ trước khi nộp file `submission.jsonl` lên cổng thi hoặc đóng gói Docker Offline.

---

## Quy trình làm việc của AI Agents (Standard Operating Procedure)

1. **Trước khi lập trình / Thử nghiệm mô hình mới**:
   - Đọc kỹ `strategy.md` của track tương ứng để nắm rõ kiến trúc đang triển khai.
   - Cập nhật mục tiêu và giả thuyết thực nghiệm vào `strategy.md`.
2. **Sau khi chạy xong một đợt suy luận (Inference Run)**:
   - Thêm một dòng mới vào bảng nhật ký `experiment_log.md`.
   - Ghi nhận đầy đủ: Thời gian, Run ID, Mô hình, Kỹ thuật áp dụng, Điểm Metric (F1 cho LifeMate, Penalized Accuracy cho DataMate).
3. **Phân tích lỗi (Error Triage)**:
   - Trích xuất các mẫu bị điểm 0 (LifeMate do lệch `state_hash`) hoặc bị trừ -1 (DataMate).
   - Ghi lại nguyên nhân gốc rễ và đề xuất giải pháp vào `error_analysis.md`.
4. **Nộp bài (Submission)**:
   - Chạy toàn bộ checklist trong `submission_checklist.md` trước khi upload file lên web hoặc nộp Docker image.
