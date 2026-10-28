# TRACK 2: DATAMATE — ĐẶC TẢ KỸ THUẬT & TOÀN BỘ 10 ĐIỀU KHOẢN RULEBOOK CHÍNH THỨC

Tài liệu này là cẩm nang đặc tả kỹ thuật, logic kinh doanh nghiệp vụ và quy chế chấm điểm chuẩn xác 100% cho **Track 2: DataMate** tại Zalo AI Challenge 2026, được trích xuất trực tiếp từ Rulebook chính thức và công thức metric của Ban tổ chức.

---

## 1. TỔNG QUAN BÀI TOÁN & CƠ CHẾ ĐIỂM SỐ (PENALIZED ACCURACY)

* **Bản chất**: Xây dựng AI Agent có khả năng trả lời các câu hỏi phân tích dữ liệu kinh doanh bằng tiếng Việt dựa trên dữ liệu bảng phức tạp, lộn xộn (CSV, Excel đa sheet, SQLite) xuất từ phần mềm bán hàng của các hộ kinh doanh/tiểu thương Việt Nam.
* **Quy tắc trích xuất dữ liệu**:
  - Mọi câu trả lời bắt buộc phải được tính toán từ **dữ liệu phát hành** và file `calendar.csv`.
  - Tuyệt đối không dùng tri thức bên ngoài. Nếu dữ liệu không có thông tin thì xem là `unanswerable`, không tự suy đoán.
* **Công thức chấm điểm Penalized Accuracy (PA)** ([`datamate_formula_1.png`](<.vscode/extracted_images/datamate_formula_1.png>)):

![DataMate Penalized Accuracy](<.vscode/extracted_images/datamate_formula_1.png>)

  $$\text{PA} = \frac{1}{N} \sum_{i=1}^N s_i, \quad s_i = \begin{cases} 
  +1, & \text{câu trả lời } i \text{ ĐÚNG} \\ 
  0, & \text{câu trả lời } i \text{ là } \textbf{abstain} \\ 
  -1, & \text{câu trả lời } i \text{ SAI} 
  \end{cases}$$

### Phân tích kỳ vọng toán học & Chiến lược "Silence is Free"
* Kỳ vọng điểm số khi chọn Trả lời với độ tin cậy $p$:
  $$E[s] = p \cdot (+1) + (1 - p) \cdot (-1) = 2p - 1$$
* Ngưỡng hòa vốn lý thuyết: $2p - 1 > 0 \iff p > 0.5$ (50%).
* **Ngưỡng an toàn thực chiến**: Do mô hình LLM thường bị thừa tự tin (overconfident), Agent chỉ nên xuất đáp án khi độ tin cậy thực nghiệm **$p \ge 0.70$** (hoặc code thực thi thành công không lỗi). Nếu code crash hoặc dữ liệu mâu thuẫn $\implies$ Kích hoạt ngay **`{"type": "abstain"}`** để nhận **0 điểm**, bảo toàn điểm số trước án phạt trừ 1 điểm (-1).

---

## 2. CẤU TRÚC DỮ LIỆU ĐẦU VÀO (INPUT SPECIFICATION) & BỘ DỮ LIỆU KAGGLE

Bộ dữ liệu chính thức được lưu trữ tại Kaggle: [thanhkhil/zaloai-datamate-dataset](https://www.kaggle.com/datasets/thanhkhil/zaloai-datamate-dataset).

### 2.1. Cấu trúc thư mục
```text
<split>/ (train, public_test, private_test)
├── questions.jsonl                 # Danh sách câu hỏi cần trả lời
├── answers.jsonl                   # (Chỉ có ở train): Nhãn chuẩn và alias chấp nhận
├── calendar.csv                    # Bảng đối chiếu dương lịch - âm lịch
└── data/                           # Dữ liệu bảng của từng doanh nghiệp
    ├── <business_id>/              # (Train: biz_101..biz_130; Public Test: dev_001..dev_014)
    │   ├── README.md               # Mô tả ngành nghề, quy ước, bảng quy đổi đơn vị
    │   ├── *.xlsx / *.csv / *.sqlite # Các sổ bán hàng, nhập hàng, công nợ, danh mục
```

### 2.2. Chi tiết từng tệp dữ liệu
1. **`questions.jsonl`**:
   - Mỗi dòng là 1 đối tượng JSON đại diện cho 1 câu hỏi:
     ```json
     {"id": "q_06de33fa21cc", "business_id": "biz_118", "question": "sp bán dc nhiều nhất tuần rồi là gì", "as_of_date": "2026-03-04"}
     ```
   - `id`: Mã định danh câu hỏi (dùng để xuất vào `submission.jsonl`).
   - `business_id`: Thư mục dữ liệu tương ứng trong `data/<business_id>/`.
   - `question`: Câu hỏi bằng tiếng Việt tự nhiên của chủ shop / nhân viên.
   - `as_of_date`: Mốc thời gian "Hôm nay" (`YYYY-MM-DD`). Mọi dữ liệu sau ngày này xem như chưa xảy ra!
2. **`answers.jsonl`** (Tập Train):
   - Chứa nhãn đúng và danh sách các cách biểu diễn được chấp nhận (`alias`):
     ```json
     {"id": "q_06de33fa21cc", "type": "string", "answer": "Xi măng Hà Tiên PCB40 bao 50kg", "alias": ["SP0001"]}
     ```
3. **`calendar.csv`**:
   - Gồm các cột: `date` (YYYY-MM-DD), `lunar_day`, `lunar_month`, `lunar_year`, `is_leap_month`.
   - Bắt buộc dùng để tra cứu các mốc thời gian âm lịch (như Mùa Tết: từ 23 Chạp năm trước đến mùng 10 Giêng năm nay).
4. **`data/<business_id>/README.md`**:
   - Chứa thông tin cực kỳ quan trọng:
     - Giới hạn ngày dữ liệu phát hành (`Dữ liệu có đến hết YYYY-MM-DD`).
     - Bảng quy đổi đơn vị (ví dụ: `1 thùng = 24 lon`, `1 lốc = 6 lon`, `1 bao = 25 kg`).
     - Ý nghĩa các cột và trạng thái của từng file Excel/CSV.

---

## 3. TOÀN BỘ 10 ĐIỀU KHOẢN RULEBOOK CHÍNH THỨC CỦA BAN TỔ CHỨC

Mọi đáp án chuẩn của Ban tổ chức đều được tính toán nghiêm ngặt theo 10 điều khoản dưới đây:

### Mục 1: Đơn Hàng và Doanh Thu
1. **Đơn tính vào doanh thu**: Chỉ tính các đơn **không huỷ** và **không trả hàng**.
   * ⚠️ **Quy tắc Trả hàng một phần**: Đơn có trả hàng, **kể cả chỉ trả một phần**, bị coi là đơn trả hàng $\implies$ **BỎ TOÀN BỘ CẢ ĐƠN KHỎI DOANH THU**.
   * ⚠️ **Đơn đang giao**: Đơn có trạng thái `đang giao` **VẪN ĐƯỢC TÍNH VÀO DOANH THU**.
2. **Bán được bao nhiêu tiền / Thu về bao nhiêu tiền**: Là **doanh thu** theo mục này, không phải số tiền khách thực tế đã thanh toán.
3. **Kỳ của đơn**: Xác định theo **ngày đặt hàng** (`order_date`), không theo ngày giao hàng hay ngày thu tiền.
4. **Đơn bị huỷ**: Là đơn có trạng thái `huỷ`. Đơn trả hàng **không** phải là đơn huỷ.
5. **Số đơn**: Chỉ đếm đơn không huỷ, không trả (kể cả khi đếm số đơn có 1 mặt hàng hoặc số đơn của 1 khách).
6. **Doanh thu một đơn**: Bằng **tổng thành tiền các dòng hàng** (đã trừ giảm giá/chiết khấu trên từng dòng hàng).
   * ⚠️ **KHÔNG trừ giảm giá trên cả đơn** (voucher khuyến mãi tổng tiền, voucher shop).
   * ⚠️ **KHÔNG cộng thuế VAT**.
   * ⚠️ **KHÔNG bao gồm phí giao hàng (shipping fee)**.
7. **Hình thức thanh toán**: Xác định theo hình thức ghi trên đơn (không theo lần trả nợ về sau). Đơn ghi nợ không tính vào tiền mặt/chuyển khoản. **Đơn COD là hình thức riêng**, không tính vào tiền mặt.
8. **Nhóm khách hàng**: Đơn không ghi tên khách (khách vãng lai) **không thuộc nhóm nào** (kể cả nhóm `lẻ`). Tỉ trọng doanh thu của nhóm chia cho doanh thu **cả shop** cùng kỳ (mẫu số có tính cả đơn khách vãng lai).
9. **Tỉ lệ đơn huỷ trong kỳ**: Bằng:
   $$\text{Tỉ lệ huỷ} = \frac{\text{Số đơn huỷ trong kỳ}}{\text{Tổng mọi đơn đặt trong kỳ (cả hoàn thành, huỷ, trả hàng, đang giao)}} \times 100\%$$
   *(Làm tròn 2 chữ số thập phân)*.
10. **Phần trăm tăng trưởng / thay đổi**: So với kỳ liền trước; nếu giảm ra **số âm**.
11. **Chênh lệch / Tăng giảm (số tiền, số lượng)**: Bằng $Kỳ\_hỏi - Kỳ\_so$ (nếu ít hơn ra số âm).

---

### Mục 2: Thời Gian & Mốc Tham Chiếu `as_of_date`
1. **Hôm nay**: Là giá trị trong trường `as_of_date` của câu hỏi (`questions.jsonl`). Mọi bản ghi có ngày sau `as_of_date` coi như **chưa xảy ra** (ngoại trừ trạng thái huỷ/trả hàng ghi trên đơn vẫn được ghi nhận).
2. **Đến hết ngày X / Tính đến hôm nay**: Hỏi về số còn lại (tồn kho, công nợ) $\implies$ là tình trạng lũy kế **từ đầu dữ liệu đến hết ngày đó**; không phải chỉ trong tháng này.
3. **Từ ngày X đến ngày Y**: Bao gồm cả ngày X và ngày Y.
4. **N ngày gần nhất**: $N$ ngày liên tiếp nhau **kể cả ngày hôm nay** (`as_of_date`), ngày kết thúc là hôm nay.
5. **Quá N, hơn N, trên N, dưới N**: $X > N$ hoặc $X < N$ (**không bao gồm N**). **Từ N trở lên**: $X \ge N$ (bao gồm cả N).
6. **Tuần**: Bắt đầu từ **thứ Hai**.
   * *Tuần này*: Từ thứ Hai của tuần chứa `as_of_date` đến hết `as_of_date`.
   * *Tuần trước / Tuần rồi*: Trọn vẹn từ thứ Hai đến Chủ Nhật của tuần liền trước.
7. **Tháng này**: Từ ngày 01 của tháng đến **hết hôm nay** (`as_of_date`).
   * ⚠️ **So với tháng trước**: Lấy **cùng số ngày** tương ứng của tháng trước (từ ngày 01 đến ngày trùng số với `as_of_date`). Nếu tháng trước không có ngày đó thì lấy đến hết tháng trước.
8. **Từ đầu năm tới nay**: Từ ngày 01/01 của năm nay đến hết hôm nay.
9. **Tháng M, Quý Q không ghi năm**: Lấy của **năm nay** nếu chưa vượt quá hôm nay; nếu đã vượt quá hôm nay thì lấy của **năm trước**.
10. **Dịp Tết năm Y / Mùa Tết**: Bắt đầu từ ngày dương của **23 tháng Chạp (năm âm Y-1)** đến hết ngày dương của **mùng 10 tháng Giêng (năm âm Y)**. So với trước Tết = lấy cùng số ngày ngay trước 23 tháng Chạp. Bắt buộc tra cứu lịch âm trong file `calendar.csv`.
11. **Ca làm việc**:
    * Ca sáng: `06:00 – 13:59`.
    * Ca chiều: `14:00 – 21:59`.
    * So sánh ca chiều với ca sáng: Mốc so sánh là ca sáng.

---

### Mục 3: Xếp Hạng (Ranking)
1. **Khái niệm mặt hàng**: Sản phẩm/hàng hóa. Dòng **Phí phục vụ** không phải là mặt hàng (không xếp hạng, không đếm, nhưng vẫn tính vào doanh thu đơn).
2. **Bán chạy nhất / Hot nhất / Bán nhiều nhất**:
   * ⚠️ Bắt buộc xếp theo **TỔNG SỐ LƯỢNG ĐÃ ĐỔI RA ĐƠN VỊ CƠ BẢN** (`sum(quantity_base)`).
   * **KHÔNG** xếp theo tổng doanh thu tiền, **KHÔNG** xếp theo số dòng hóa đơn.
   * Loại bỏ các đơn huỷ và đơn trả hàng.
3. **Mặt hàng sinh lời nhất / Lời nhất**: Xếp theo **Tổng lãi gộp** (Doanh thu trừ Giá vốn).
4. **Nhân viên giỏi nhất**: Xếp theo **Tổng doanh thu mang lại**, không xếp theo số lượng đơn tạo.
5. **AOV (Giá trị trung bình 1 đơn)**: Bằng Tổng doanh thu chia Tổng số đơn hợp lệ, làm tròn về đồng (nửa đồng làm tròn lên).
6. **Quy tắc Tie-break (Bằng điểm nhau)**: Nếu hai đối tượng có cùng chỉ số, ưu tiên xếp theo **mã tăng dần** (nếu không có mã thì theo tên bảng chữ cái).

---

### Mục 4: Công Nợ & Khoản Phải Thu
1. **Đơn tính nợ**: Đơn không huỷ; **đơn trả hàng VẪN TÍNH NỢ nếu khách chưa thanh toán**.
2. **Nợ của 1 đơn**: Phần chưa thanh toán của **tổng đơn**:
   * ⚠️ **CÓ TRỪ giảm giá trên cả đơn (voucher shop)**.
   * ⚠️ **CÓ BAO GỒM phí giao hàng**.
3. **COD và Sàn TMĐT**: **KHÔNG PHẢI LÀ KHÁCH NỢ** (kể cả khi tiền chưa về tài khoản).
4. **Tuổi nợ**: Tính từ **ngày đặt hàng** (đặt hôm nay = 0 ngày, đặt hôm qua = 1 ngày), không tính từ hạn thanh toán.
5. **Nợ phát sinh thêm vs Trả nợ**:
   * *Nợ thêm*: Phần chưa trả ngay trong ngày mua hàng.
   * *Trả nợ*: Các khoản thanh toán trả sau phát sinh trong kỳ cho các đơn mua trước đó.
6. **Khách lẻ**: Vẫn cộng dồn vào tổng nợ của shop, nhưng không có dòng riêng trong bảng công nợ theo khách hàng.

---

### Mục 5: Kho, Nhập Hàng & Giá Vốn (CẤM FIFO)
1. **Tiền nhập hàng**: Tổng giá trị các phiếu nhập có ngày trong kỳ (bỏ phiếu huỷ). Nhập đầu kỳ coi như phiếu nhập.
2. **Giá nhập bình quân tháng**: Bình quân gia quyền theo số lượng nhập, không phải trung bình cộng đơn giá.
3. **Tồn kho theo đơn vị lớn**: Trả về danh sách gồm 2 số: `[số đơn vị lớn nguyên, số đơn vị cơ bản lẻ]`.
4. **Giá trị tồn kho**: Bằng:
   $$\text{Giá trị tồn} = \text{Số lượng tồn hết hôm nay} \times \text{Giá vốn 1 đơn vị cơ bản trong danh mục}$$
5. **Mặt hàng ế**: Mặt hàng đã từng bán trước kỳ nhưng trong suốt kỳ hỏi có số lượng bán bằng 0.
6. **Giá vốn và Lãi gộp (QUY TẮC CỐT LÕI - CẤM FIFO)**:
   * ⚠️ **TUYỆT ĐỐI KHÔNG TÍNH THEO FIFO, KHÔNG THEO GIÁ NHẬP LẦN CUỐI**.
   * Giá vốn của từng dòng bán bằng:
     $$\text{Giá vốn dòng} = \text{Số lượng bán} \times \text{Giá vốn ghi trên chính dòng bán đó}$$
   * **TUYỆT ĐỐI KHÔNG TỰ Ý TÍNH LẠI GIÁ VỐN TỪ PHIẾU NHẬP HAY TỪ LÔ**.
   * Lãi gộp = Doanh thu − Giá vốn.

---

### Mục 6: Quy Tắc Theo Ngành Đặc Thù
1. **Ngành F&B / Cà phê**:
   * Size M và Size L nhận diện bằng hậu tố kết thúc bằng `(M)` hoặc `(L)` (không phân biệt chữ hoa/thường).
2. **Ngành Dược / Nhà thuốc**:
   * Thuốc kê đơn: Phân loại theo cột trong danh mục thuốc; tỉ lệ thuốc kê đơn tính theo doanh thu.
   * Thuốc hết hạn trong N ngày: Tính tồn kho từ hôm nay $D$ đến $D + N$, tiền vốn tính theo giá nhập của chính lô đó.
   * Thuốc phối hợp nhiều hoạt chất là **một loại thuốc riêng**, không tách thành từng hoạt chất lẻ.
   * Lô gần hết hạn nhất: Là lô còn tồn chưa hết hạn tại hôm nay có hạn sử dụng sớm nhất.
3. **Ngành Vật liệu xây dựng (VLXD)**:
   * Đơn giao nhiều đợt: Số đợt giao trong tháng = số đợt có ngày giao trong tháng; mỗi đợt của mỗi đơn tính là 1 đợt (kể cả đợt của đơn sau đó bị trả hàng).

---

### Mục 7: Xử Lý Câu Không Trả Lời Được (`unanswerable`)
1. **Khi nào là `unanswerable`**: Khi dữ liệu phát hành không đủ căn cứ để tính toán. Đáp án chuẩn là:
   ```json
   {"type": "unanswerable", "reason": "Khoảng thời gian nằm ngoài dữ liệu"}
   ```
   * ⚠️ Trả lời đúng nhãn này được **CỘNG +1 ĐIỂM**!
2. **Phân biệt Ngoài khoảng vs Chồng lấn một phần**:
   * Nếu khoảng thời gian hỏi nằm **hoàn toàn ngoài khoảng dữ liệu** của shop $\implies$ `unanswerable` (không phải bằng 0).
   * Nếu khoảng thời gian hỏi **có một phần nằm trong dữ liệu** $\implies$ **Vẫn tính toán bình thường trên phần có dữ liệu**, không được coi là unanswerable!
3. **Trường `reason`**: Là chuỗi văn bản tùy chọn giải thích lý do, không chấm tự động.

---

### Mục 8: Quy Ước Kỹ Thuật Đọc Dữ Liệu File
1. **Kế thừa ô trống (Merged Cells / Forward Fill)**: Các ô bị để trống ở các dòng sau của cùng một đơn hàng sẽ kế thừa giá trị ở dòng đầu tiên của đơn đó (tên khách hàng, ghi chú, trạng thái đơn áp dụng cho mọi dòng).
2. **Bốn trạng thái đơn duy nhất**: `hoàn thành`, `huỷ`, `trả hàng`, `đang giao`.
3. **Xử lý dòng trùng lặp (Duplicate Lines)**:
   * Hai dòng của **cùng một đơn hàng** giống hệt nhau ở mọi ô $\implies$ Là lỗi xuất file, **chỉ tính 1 lần**.
   * Hai dòng giống hệt nhau của **hai đơn hàng khác nhau** $\implies$ Là hai lần bán thật, **tính cả hai**.
4. **Ngày không ghi năm**: Lấy theo năm ghi trên tên của Sheet chứa dòng đó.

---

### Mục 9: Đặc Tả 9 Kiểu Dữ Liệu Đầu Ra (Output Formats)
**Ảnh bảng quy tắc gốc từ BTC**: [`datamate_formula_0.png`](<.vscode/extracted_images/datamate_formula_0.png>)

![DataMate 9 Types Matching Rules](<.vscode/extracted_images/datamate_formula_0.png>)

Mỗi dòng trong `submission.jsonl` có format: `{"id": "...", "type": "...", "answer": ...}`.
Mỗi kiểu chỉ có **đúng 1 format duy nhất; sai format là bị phạt -1 điểm**:

| `type` | Quy cách định dạng JSON | Ví dụ HỢP LỆ (+1) | Ví dụ SAI BỊ PHẠT (-1) |
| :--- | :--- | :--- | :--- |
| **`number`** | JSON number thuần, không đơn vị, không phân cách hàng nghìn. Tiền là số nguyên VND. Số khác làm tròn 2 chữ số. | `48250000`<br>`18.6` | `"48.250.000"`, `"48250000đ"`, `"48.25M"`, `48250000.0` |
| **`percent`** | JSON number là **con số phần trăm** (12.4% ghi là `12.4`). Giữ dấu âm nếu giảm. Làm tròn 2 số thập phân. | `12.4`<br>`-8.25` | `"12.4%"`, `0.124`, `"-8.25%"` |
| **`string`** | Chuỗi Unicode NFC, chữ thường (lowercase), loại bỏ ký tự lạ thành dấu cách, giữ nguyên dấu tiếng Việt. Chấp nhận mã hàng (`sp0042`). | `"sting dâu 330ml"`<br>`"sp0042"` | `"Sting dầu 330ml"` (sai dấu), `["sting dâu"]` |
| **`date`** | Chuỗi ISO: `"YYYY-MM-DD"`. Nếu câu hỏi theo tháng: `"YYYY-MM"`. | `"2026-08-15"`<br>`"2025-09"` | `"15/08/2026"`, `"tháng 9/2025"` |
| **`boolean`** | Giá trị JSON boolean nguyên thủy: `true` hoặc `false`. | `true`<br>`false` | `"true"`, `"false"`, `1`, `0`, `"có"` |
| **`list`** | Mảng JSON `[...]`, có thứ tự từng vị trí. **Một phần tử vẫn bắt buộc phải là mảng**. Sắp xếp theo câu hỏi hoặc tie-break mã tăng dần. | `["cà phê sữa"]`<br>`[3, 10]` | `"cà phê sữa"` (quên đóng ngoặc mảng), `"3 thùng 10 chai"` |
| **`table`** | JSON Object: `{"columns": [...], "rows": [[...], ...]}`. Cột đầu là tên, cột sau là chỉ số. Ô số là JSON number. | `{"columns": ["khach_hang", "no"], "rows": [["Chị Lan", 3200000]]}` | Chuỗi Markdown table, hoặc ô số ghi string `"3.200.000đ"` |
| **`unanswerable`** | `{"type": "unanswerable", "reason": "..."}` | `{"type": "unanswerable", "reason": "không có dữ liệu"}` | Trả về `0` hoặc xuất nhãn `abstain` |
| **`abstain`** | `{"type": "abstain"}` (hoặc không xuất dòng id đó). Luôn nhận 0 điểm. | `{"type": "abstain"}` | `None`, `null`, `""` |

---

### Mục 10: Từ Ngữ & Quy Đổi Đơn Vị
* Mỗi mặt hàng có một **đơn vị cơ bản** (đơn vị nhỏ nhất shop quản lý: lon, chai, gói, kg, cái).
* Đơn vị lớn (thùng, lốc, bao, hộp) quy đổi theo bảng quy đổi hoặc `README.md` của từng shop (ví dụ: `1 thùng = 24 lon`, `1 lốc = 6 lon`).
* Mặt hàng không có dòng quy đổi nào thì mặc định đơn vị trên dòng là đơn vị cơ bản (hệ số 1).

---

## 4. QUY ĐỊNH KỸ THUẬT & GIỚI HẠN CỦA BAN TỔ CHỨC (OFFICIAL TECHNICAL REQUIREMENTS)

Mọi giải pháp tham gia Track 2 (DataMate) bắt buộc phải tuân thủ nghiêm ngặt toàn bộ các yêu cầu kỹ thuật do Ban tổ chức ban hành:

1. **Giới hạn số lượng tham số (Model Parameter Budget)**:
   - Tổng số tham số của **tất cả các mô hình được sử dụng khi suy luận** (Language models, embedding models, rerankers, classifiers) — bất kể được nạp cùng lúc hay lần lượt — **không được vượt quá 15 tỷ (≤ 15B parameters)**.
   - Đối với mô hình Mixture-of-Experts (MoE), tính **tổng toàn bộ tham số** của mô hình (không chỉ tính riêng tham số kích hoạt / active parameters).
   - Cho phép sử dụng mọi hình thức lượng hóa (quantization: 4-bit, 8-bit, AWQ, GPTQ...).
   - Các mô hình được sử dụng bắt buộc phải được khai báo rõ ràng trong file `README` của giải pháp.

2. **Dữ liệu và mô hình cho phép (Allowed Models & Resources)**:
   - Được phép sử dụng bất kỳ mô hình nào có trọng số (weights) được phát hành công khai trước ngày **01/10/2026** với giấy phép (license) cho phép sử dụng.
   - Được phép tự do tinh chỉnh (fine-tune) mô hình trước cuộc thi.
   - Không được phép can thiệp thủ công (human-in-the-loop) hay sử dụng bất kỳ dịch vụ/mô hình thương mại bên ngoài nào tại thời điểm suy luận.

3. **Cơ chế thực thi mã nguồn (Code Execution)**:
   - Agent được phép tự sinh và thực thi mã nguồn (code) bên trong container; điều này được khuyến khích nhưng không bắt buộc.
   - Thư mục dữ liệu đầu vào `/data` được gắn kết ở chế độ **chỉ đọc (read-only)**; mọi tệp tạm (temporary files) phải được ghi ra các thư mục khác (như `/tmp`).

4. **Hạ tầng máy chấm thi của Ban tổ chức**:
   - **GPU**: **2× NVIDIA RTX 5090** (32 GB VRAM mỗi card, tổng cộng **64 GB VRAM**).
   - **RAM**: **512 GB**.
   - **Phần mềm**: Kiến trúc Blackwell yêu cầu **CUDA ≥ 12.8**.

5. **Giới hạn thời gian (Time Limit)**:
   - Toàn bộ tập dữ liệu Private Test phải được xử lý hoàn tất trong vòng **tối đa 6 giờ** (wall-clock time) trên máy chấm thi của Ban tổ chức, tính cả thời gian nạp mô hình vào bộ nhớ.
   - Nếu quá 6 giờ hoặc container dừng sớm, chỉ những câu hỏi đã kịp ghi kết quả vào `/result/submission.jsonl` được đưa vào chấm điểm; những câu hỏi còn lại tự động tính là `abstain` (nhận 0 điểm).

6. **Môi trường suy luận Offline Tuyệt đối (`--network none`)**:
   - Docker container được chạy với cờ `--network none` (hoàn toàn không có mạng internet). Mọi weights, thư viện Python, dependencies phải đóng gói sẵn trong image.

7. **Quy chuẩn tái lập tất định (Reproducibility)**:
   - Bắt buộc cố định random seed và sử dụng giải mã tất định (greedy decoding, `temperature = 0`).
   - Kết quả chạy lại từ Docker của Ban tổ chức phải đạt tiêu chuẩn:
     $$|\text{PA}_{\text{uploaded}} - \text{PA}_{\text{rerun}}| \le 0.03$$

8. **Quy tắc phân định thứ hạng khi hòa điểm (Tie-breaking)**:
   - Ưu tiên đội có **tổng thời gian suy luận thấp hơn** trong lần chạy lại của Ban tổ chức.
   - Nếu vẫn tiếp tục hòa điểm, ưu tiên đội nộp bài sớm hơn trên hệ thống.

---

## 5. ĐỊNH DẠNG TỆP NỘP BÀI (SUBMISSION FORMAT)

Tệp nộp bài có tên bắt buộc là **`submission.jsonl`**, mỗi dòng là một JSON object độc lập gồm 3 trường:
```json
{"id": "test_0001", "type": "number", "answer": 1500000}
{"id": "test_0002", "type": "percent", "answer": 15.5}
{"id": "test_0003", "type": "string", "answer": "Cà phê sữa đá"}
{"id": "test_0004", "type": "date", "answer": "2026-05-15"}
{"id": "test_0005", "type": "boolean", "answer": true}
{"id": "test_0006", "type": "list", "answer": ["SP01", "SP02"]}
{"id": "test_0007", "type": "table", "answer": {"columns": ["san_pham", "so_luong"], "rows": [["Cà phê", 100], ["Trà đào", 50]]}}
{"id": "test_0008", "type": "unanswerable", "reason": "Dữ liệu không ghi nhận doanh thu chi nhánh B"}
{"id": "test_0009", "type": "abstain"}
```

* Đường dẫn xuất kết quả trong Docker container: `/result/submission.jsonl`.
