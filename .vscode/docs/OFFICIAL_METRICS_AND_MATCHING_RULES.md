# TỔNG HỢP CÔNG THỨC CHẤM ĐIỂM & QUY TẮC SO KHỚP CHÍNH THỨC
## ĐỐI CHIẾU TRỰC TIẾP TỪ HÌNH ẢNH GỐC CỦA BAN TỔ CHỨC

Thư mục chứa ảnh gốc: [`.vscode/extracted_images/`](<.vscode/extracted_images/>) (hoặc [`extracted_images/`](<extracted_images/>))

Tài liệu này giải mã, phiên âm và phân tích chi tiết từng ký hiệu, điều kiện biên và quy tắc so khớp từ **3 hình ảnh công thức chính thức của Ban tổ chức Zalo AI Challenge 2026**:
1. [`lifemate_formula_0.png`](<.vscode/extracted_images/lifemate_formula_0.png>): Công thức tính điểm F1 và Cổng chặn State Hash của Track 1 (LifeMate).
2. [`datamate_formula_0.png`](<.vscode/extracted_images/datamate_formula_0.png>): Bảng đặc tả 9 kiểu dữ liệu đầu ra và toàn bộ quy tắc so khớp (Matching Rules) của Track 2 (DataMate).
3. [`datamate_formula_1.png`](<.vscode/extracted_images/datamate_formula_1.png>): Công thức tính điểm Penalized Accuracy (PA) của Track 2 (DataMate).

---

## PHẦN 1: TRACK 1 (LIFEMATE) — CÔNG THỨC CHẤM ĐIỂM F1 & CỔNG CHẶN STATE HASH
**Ảnh gốc**: [`.vscode/extracted_images/lifemate_formula_0.png`](<.vscode/extracted_images/lifemate_formula_0.png>)

![LifeMate Official Metric Formula](<.vscode/extracted_images/lifemate_formula_0.png>)

### 1.1. Phiên âm toán học nguyên gốc
$$\text{Score} = \begin{cases} 
0, & \text{if } \text{hash}_{\text{sys}} \neq \text{hash}_{\text{gold}} \text{ or } M = 0 \\
\frac{2 \cdot P \cdot R}{P + R}, & \text{otherwise}
\end{cases}$$

$$P = \frac{M}{M + U}, \quad R = \frac{M}{|G|}$$

$$\text{Final score} = \frac{100}{N} \sum_{i=1}^N \text{Score}_i$$

### 1.2. Giải nghĩa chi tiết từng biến số
* **$N$**: Tổng số lượng mẫu trong tập kiểm thử (test set). Thang điểm cuối cùng được chuẩn hóa trên thang **100 điểm**.
* **$\text{hash}_{\text{sys}}$**: Chuỗi SHA256 trạng thái môi trường sau khi hệ thống dự đoán thực thi chuỗi `tool_calls` qua `zaic_tools` (lấy từ `env.get_state_hash()`).
* **$\text{hash}_{\text{gold}}$**: Chuỗi SHA256 trạng thái môi trường chuẩn của Ban tổ chức.
* **$M$ (Matched)**: Số lượng lời gọi công cụ dự đoán **khớp hoàn toàn** với lời gọi công cụ trong nhãn chuẩn:
  - Khớp tên công cụ (`name`).
  - Khớp đầy đủ các tham số bắt buộc và giá trị (`arguments`).
  - Đối với tham số văn bản tự do có gắn cờ `x-match: fuzzy`, chỉ cần Token-F1 > 0.6.
* **$U$ (Unmatched / Unwanted)**: Số lượng lời gọi công cụ do hệ thống sinh ra nhưng **thừa, sai tên, hoặc sai tham số** so với nhãn chuẩn:
  $$U = |\text{sub}| - M$$
* **$|G|$ (Gold)**: Tổng số lượng lời gọi công cụ trong nhãn chuẩn.
* **$P$ (Precision)**: Độ chuẩn xác của các công cụ được gọi: $P = \frac{M}{M + U}$.
* **$R$ (Recall)**: Độ bao phủ so với danh sách công cụ chuẩn: $R = \frac{M}{|G|}$.

### 1.3. Cơ chế Hard-Gate & Các trường hợp biên (Edge Cases)
1. ⚠️ **State-Hash Hard Gate (Cổng chặn sinh tử)**:
   - Nếu $\text{hash}_{\text{sys}} \neq \text{hash}_{\text{gold}} \implies \text{Score} = 0$.
   - **Ý nghĩa thực chiến**: Bất kể mô hình dự đoán đúng bao nhiêu công cụ ($M > 0$), nếu trạng thái môi trường cuối cùng lệch dù chỉ 1 trường dữ liệu (ví dụ bật nhầm đèn, chuyển sai số tiền, sai ID thiết bị) $\implies$ Toàn bộ mẫu đó bị **0 điểm**!
2. **Trường hợp ngoài phạm vi (Out-of-Scope, $|G| = 0$)**:
   - Khi nhãn chuẩn không cần gọi công cụ nào ($|G| = 0$):
     - Nếu hệ thống nộp `tool_calls: []` và giữ nguyên trạng thái ban đầu ($\text{hash}_{\text{sys}} == \text{hash}_{\text{gold}}$) $\implies \text{Score} = 1.0$.
     - Nếu hệ thống gọi bất kỳ công cụ nào $\implies U > 0 \implies \text{Score} = 0.0$.
3. **Trường hợp $M = 0$**:
   - Nếu hệ thống không khớp được bất kỳ công cụ nào $\implies \text{Score} = 0$.

---

## PHẦN 2: TRACK 2 (DATAMATE) — BẢNG ĐẶC TẢ 9 KIỂU DỮ LIỆU & QUY TẮC SO KHỚP
**Ảnh gốc**: [`.vscode/extracted_images/datamate_formula_0.png`](<.vscode/extracted_images/datamate_formula_0.png>)

![DataMate Types & Matching Rules Table](<.vscode/extracted_images/datamate_formula_0.png>)

### 2.1. Bản dịch và sao chép nguyên văn bảng quy tắc (Matching Rules Table)

| Kiểu (`type`) | Nội dung đáp án (`Answer content`) | Ví dụ mẫu (`Example`) | Quy tắc so khớp chính thức (`Matching rule`) |
| :--- | :--- | :--- | :--- |
| **`number`** | Giá trị số (Số tiền VND, số đếm, số lượng).<br>*(A numeric value - VND amounts, counts, quantities)* | `48250000` | **So khớp chính xác (Exact match)**. Giá trị tham chiếu là **số nguyên** đối với số tiền VND, số lượng đếm; hoặc **làm tròn 2 chữ số thập phân** đối với các số khác.<br>*(Reference values are integers (VND amounts, counts, quantities) or rounded to 2 decimals).* |
| **`percent`** | Giá trị phần trăm, tính theo đơn vị phần trăm.<br>*(A percentage value, in percent)* | `-12.45` | **So khớp chính xác sau khi làm tròn 2 chữ số thập phân; dấu âm/dương phải tuyệt đối chính xác**.<br>*(Exact match after rounding to 2 decimal places; the sign must be correct).*<br>⚠️ *Lưu ý: 12.45% ghi là `12.45`, giảm 12.45% ghi là `-12.45`. CẤM ghi `0.1245`, CẤM ký tự `%`.* |
| **`string`** | Tên (Sản phẩm, khách hàng, chi nhánh, thứ trong tuần, ...).<br>*(A name - product, customer, branch, weekday, ...)* | `"Sting dâu 330ml"` | Cả đáp án dự đoán và từng dạng chấp nhận được của nhãn chuẩn đều được chuẩn hóa rồi so sánh bằng nhau tuyệt đối.<br>**Quy trình chuẩn hóa**: (1) Unicode NFC; (2) Chuyển chữ thường (lowercase); (3) Mọi ký tự không phải chữ cái tiếng Việt/số/khoảng trắng được thay bằng dấu cách; (4) Xóa khoảng trắng đầu/cuối và thu gọn chuỗi khoảng trắng liền nhau về 1 dấu cách duy nhất. **Dấu tiếng Việt được giữ nguyên**.<br>Mỗi nhãn chuẩn chấp nhận mọi cách viết của tên xuất hiện trong dữ liệu doanh nghiệp đó, cộng với **mã đối tượng** (sau chuẩn hóa, ví dụ: `sting dâu 330ml`, `sp0042`). |
| **`date`** | Ngày tháng theo định dạng chuẩn ISO.<br>*(A date in ISO format)* | `"2026-08-15"`<br>`"2025-09"` | **So khớp chính xác (Exact match)**. Định dạng ngày: `yyyy-mm-dd`; câu hỏi về tháng: `yyyy-mm` (ví dụ `"2025-09"`). |
| **`boolean`** | Giá trị đúng/sai nguyên thủy: `true` / `false`.<br>*(true / false)* | `true` | **So khớp chính xác (Exact match)**. Bắt buộc là boolean JSON thuần (`true` hoặc `false`), không dùng chuỗi `"true"` hay số `1`. |
| **`list`** | Danh sách có thứ tự các tên hoặc số, theo thứ tự câu hỏi yêu cầu (hoặc thứ tự mặc định của rulebook).<br>*(An ordered list of names or numbers...)* | `["Cà phê sữa", "Bạc xỉu", "Trà đào"]` | **Luôn có thứ tự và so khớp từng vị trí (position by position)**. Thứ tự lấy theo yêu cầu câu hỏi hoặc theo mặc định của Rulebook (xem Mục Xếp hạng). Từng phần tử bên trong được so khớp theo quy tắc `string` hoặc `number`.<br>⚠️ *Kể cả danh sách chỉ có 1 phần tử vẫn bắt buộc phải là mảng `["..."]`.* |
| **`table`** | Bảng nhỏ: Tên các cột và các dòng.<br>*(A small table: column names and rows)* | `{"columns": [...], "rows": [[...], ...]}` | Các cột được so khớp theo tên cột đã chuẩn hóa; nếu tên không khớp thì so khớp theo thứ tự vị trí. **Các dòng được so khớp theo tập hợp (set)**; ô văn bản so theo quy tắc `string`, ô số so theo quy tắc `number`. |
| **`unanswerable`** | Dữ liệu không chứa thông tin cần thiết để trả lời.<br>*(The data does not contain what is needed to answer)* | `{"type": "unanswerable", "reason": "..."}` | **Đúng khi và chỉ khi nhãn chuẩn là unanswerable**. Đoán đúng nhãn này được **+1 điểm**! |
| **`abstain`** | Hệ thống chủ động từ chối trả lời.<br>*(The agent chooses not to answer)* | `{"type": "abstain"}` | **Luôn được tính điểm là abstain (0 điểm), bất kể nhãn chuẩn là gì**. |

---

## PHẦN 3: TRACK 2 (DATAMATE) — CÔNG THỨC PENALIZED ACCURACY (PA)
**Ảnh gốc**: [`.vscode/extracted_images/datamate_formula_1.png`](<.vscode/extracted_images/datamate_formula_1.png>)

![DataMate Penalized Accuracy Formula](<.vscode/extracted_images/datamate_formula_1.png>)

### 3.1. Phiên âm toán học nguyên gốc
$$\text{PA} = \frac{1}{N} \sum_{i=1}^N s_i, \quad s_i = \begin{cases} 
1, & \text{câu trả lời } i \text{ ĐÚNG (correct)} \\ 
0, & \text{câu trả lời } i \text{ là } \textbf{abstain} \\ 
-1, & \text{câu trả lời } i \text{ SAI (wrong)} 
\end{cases}$$

### 3.2. Phân tích ma trận điểm thưởng / phạt
* Nếu Agent trả lời một câu hỏi:
  - Nếu câu trả lời thỏa mãn `Matching rule` ở bảng trên: nhận **+1.0 điểm**.
  - Nếu câu trả lời sai giá trị hoặc sai format: nhận **-1.0 điểm**.
* Nếu Agent xuất `{"type": "abstain"}` (hoặc không nộp ID đó): nhận **0.0 điểm** (không được thưởng và không bị phạt).
* **Quy tắc vàng chiến lược**:
  - Khi code chạy thành công, dữ liệu đầy đủ $\rightarrow$ Nộp kết quả (kỳ vọng $+1$).
  - Khi dữ liệu shop thiếu, không có cột, nằm ngoài thời gian $\rightarrow$ Nộp `{"type": "unanswerable"}` (kỳ vọng $+1$).
  - Khi code Python bị lỗi runtime, timeout, hoặc kết quả nghi ngờ mâu thuẫn $\rightarrow$ Nộp `{"type": "abstain"}` để bảo toàn **0 điểm**, tránh mất 1 điểm (-1).

---

## PHẦN 4: LIÊN KẾT NHANH TỚI CÁC TÀI LIỆU LIÊN QUAN

* Chi tiết tích hợp Track 1: Xem [TRACK1_LIFEMATE_SPECIFICATION.md:120](<.vscode/docs/TRACK1_LIFEMATE_SPECIFICATION.md#L120>).
* Chi tiết tích hợp Track 2: Xem [TRACK2_DATAMATE_SPECIFICATION.md:15](<.vscode/docs/TRACK2_DATAMATE_SPECIFICATION.md#L15>).
* Hàm Python định dạng an toàn: Xem [PROMPT_TEMPLATES_AND_EXAMPLES.md:120](<.vscode/docs/PROMPT_TEMPLATES_AND_EXAMPLES.md#L120>).
