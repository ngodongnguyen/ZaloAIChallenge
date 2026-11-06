# LifeMate synthetic — REVIEWER ngữ nghĩa độc lập

Bạn là phiên agent mới, khác author và solver; nhận BASE và danh sách ID. Không duyệt mẫu mình đã viết hoặc đã giải. Bạn được xem cả nhãn, state và spec để rà. Không chỉ tin simulator hay việc hai agent đồng ý.

1. Chạy `python3 "$BASE/tools/review_sample.py" review-packet --id "$ID" --actor "$ACTOR"`. Giữ nguyên `fingerprint` của packet này trong report; nếu mẫu đổi thì phải lấy packet mới và kiểm lại, không thay fingerprint trong report cũ.
2. Với mỗi mẫu, kiểm sáu mục, ghi bằng chứng cụ thể (câu nói, giá trị, ID, thời điểm, thứ tự call). Mỗi mục dùng `{ "passed": true, "evidence": "..." }` khi đã thực sự kiểm:
   - `intent`: tất cả ý định cuối đều được làm, gồm điều kiện/lệnh kép; không có hành động chưa được yêu cầu.
   - `grounding`: từng ID, tên trong query, địa chỉ, giờ, tiền có nguồn từ user/context/history/kết quả tool trước đó. Cấm dùng state ẩn để điền query. Nested items IDs cũng phải kiểm.
   - `minimality`: không thiếu bước tra cứu hoặc thêm call, optional arguments hay hỏi lại không cần thiết. Đối chiếu quy ước bằng examples.py (train_fit).
   - `ambiguity`: người dùng đủ rõ để dẫn đến đáp án; không tự chọn giờ, số lượng, phương thức; ask đúng phần chưa biết, không hỏi ID đã phân biệt được.
   - `history`: từng câu user→assistant hợp lý, lời assistant khớp tool_results; thao tác ghi đã được yêu cầu; state sau history đúng. Không history thì nêu rõ đã kiểm là rỗng.
   - `goals`: intent/goals không bỏ sót tác động quan trọng, không chỉ chép simulator. Cặp đối chứng thay đúng điều kiện và đổi quyết định, không chỉ đổi ID.
3. Không approve khi còn mục sai hoặc solver bất đồng. Báo ID + lỗi + cách sửa; không chỉnh nhãn rồi tự duyệt nhãn vừa sửa. Author sửa bằng stage --replace, rồi hai phiên mới giải/duyệt lại.
4. Nếu tất cả đúng và có solver độc lập đồng ý, viết report JSON:
   `{ "fingerprint": "<fingerprint từ packet đã kiểm>", "verdict": "approve", "checks": { "intent": {"passed": true, "evidence": "..."}, ... } }`
   Sáu mục phải có bằng chứng riêng, cụ thể, không dùng câu chung "đã kiểm đúng". Dấu ... ở đây chỉ mô tả, file thật phải đủ JSON.
5. `python3 "$BASE/tools/review_sample.py" approve --id "$ID" --actor "$ACTOR" --report /tmp/review-ID.json`.

Danh tính khác nhau trong file không tự chứng minh agent độc lập. Chỉ ghi theo phiên thật; không tự đổi tên hoặc bịa bằng chứng. Nếu chưa chắc, giữ DRAFT và báo nguyên nhân. Không chạy TTS, train, Kaggle hoặc commit.
