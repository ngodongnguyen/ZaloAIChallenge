# TRACK 1: LIFEMATE — ĐẶC TẢ KỸ THUẬT & HƯỚNG DẪN THỰC THI TOÀN DIỆN

Tài liệu này tổng hợp toàn bộ đặc tả kỹ thuật, schema công cụ, môi trường giả lập `zaic_tools`, công thức chấm điểm chính thức và kiến trúc mô hình chuẩn xác cho **Track 1: LifeMate** tại Zalo AI Challenge 2026.

---

## 1. BẢN CHẤT BÀI TOÁN & QUY TẮC CỐT LÕI

* **Mục tiêu**: Xây dựng hệ thống Agentic AI tiếp nhận khẩu lệnh âm thanh tiếng Việt, phân tích ý định trong ngữ cảnh cụ thể, lập kế hoạch và tương tác với môi trường giả lập ứng dụng thông qua chuỗi lời gọi công cụ (tool calls).
* **Triết lý đánh giá**: *"The agent is judged on what it does, not what it says"* — Mô hình được đánh giá dựa trên **hành động thực tế và sự thay đổi trạng thái môi trường**, không phải văn bản hội thoại thông thường.
* **Quy tắc vàng (Golden Rules)**:
  1. **Không bao giờ đọc trực tiếp `state.json`**: Mọi tương tác đọc/ghi dữ liệu bắt buộc phải thông qua instance `ToolEnv`.
  2. **Một phiên làm việc duy nhất**: Mỗi mẫu kiểm thử chỉ được khởi tạo đúng **1 phiên `ToolEnv` duy nhất**. Mọi cuộc gọi (kể cả cuộc gọi lỗi hay thừa) đều được ghi nhận vĩnh viễn vào danh sách nộp.
  3. **Kiểm tra trước khi gọi**: Luôn dùng `env.validate(name, args)` để kiểm tra tính hợp lệ trước khi thực thi `env.call(name, args)` để tránh bị ghi nhận lỗi schema vào bài thi.
  4. **State-Hash Hard Gate**: Trạng thái môi trường cuối cùng bắt buộc phải khớp mã băm chuẩn (`hash_sys == hash_gold`), nếu sai lệch sẽ bị **0 ĐIỂM TUYỆT ĐỐI**.

---

## 2. ĐỊNH DẠNG DỮ LIỆU ĐẦU VÀO (INPUT SPECIFICATION)

Dữ liệu được tổ chức theo từng split (`train`, `public_test`, `private_test`):

```text
<split>/
├── <split>.jsonl                         # Danh sách mẫu
├── <split>_tools_catalog.json             # Schema toàn bộ công cụ của split
├── <split>_environment/<id>/state.json    # Dữ liệu app ban đầu của từng mẫu
├── audio/<id>/<id>.wav                   # File âm thanh khẩu lệnh lượt cuối
│   ├── <id>_u1.wav, <id>_u2.wav...       # Âm thanh các lượt user trong lịch sử
└── zaic_tools/                           # Thư viện mô phỏng môi trường
```

### 2.1. Cấu trúc từng dòng trong `<split>.jsonl`
Mỗi dòng là một đối tượng JSON đại diện cho 1 mẫu:
* **`id`**: Mã định danh mẫu (`train_0001`, `public_0001`).
* **`audio`**: Đường dẫn tương đối đến file âm thanh yêu cầu cuối của người dùng (`audio/<id>/<id>.wav`).
* **`context`**: Bối cảnh người dùng tại thời điểm gửi yêu cầu:
  * `current_time`, `timezone`: Thời gian hiện tại theo ISO 8601 (ví dụ `2026-11-28T07:10:00+07:00`). Dùng để giải quyết các mốc tương đối như *"mai"*, *"tối nay"*, *"tuần sau"*.
  * `location`: Tọa độ hiện tại `{lat, lng}`.
  * `user_profile`: Tên, địa chỉ nhà riêng, nơi làm việc, thành phố hiện tại.
  * `installed_apps`: Danh sách các ứng dụng người dùng đã cài đặt kèm công cụ của từng app (`{app_id: {"tools": [...]}}`). **Mô hình chỉ được phép gọi các công cụ thuộc danh sách này!**
* **`history`**: Lịch sử hội thoại nhiều lượt trước đó (có thể rỗng):
  * Lượt `user`: Chứa `audio` trỏ tới file wav tương ứng (`audio/<id>/<id>_u1.wav`). **LƯU Ý QUAN TRỌNG: Trong `public_test`, lượt user KHÔNG CÓ `audio_text`, hệ thống phải tự ASR các file này nếu muốn hiểu bối cảnh lịch sử!**
  * Lượt `assistant`: `text`, `tool_calls` và `tool_results`. Các thao tác này đã được thực thi và cập nhật sẵn vào file `state.json` ban đầu.
* **Riêng tập `train` có thêm nhãn:**
  * `audio_text`: Văn bản bóc băng chính xác của audio yêu cầu cuối.
  * `label.tool_calls`: Danh sách công cụ đúng cần gọi theo thứ tự.
  * `label.tool_results`: Kết quả trả về của từng lời gọi công cụ.
  * `label.final_state`: Trạng thái của các ứng dụng bị thay đổi sau khi chạy xong `tool_calls`.

---

## 3. DANH MỤC CÔNG CỤ (TOOLS CATALOG) & CÁC EXTENSION SCHEMA

File `<split>_tools_catalog.json` định nghĩa chuẩn OpenAPI cho **46 ứng dụng và hơn 140 công cụ**:

### 3.1. Phân loại công cụ (`kind`)
1. **`read`**: Chỉ đọc thông tin (tìm danh bạ, kiểm tra thời tiết, xem lịch hẹn, tra cứu số dư). Không làm thay đổi trạng thái môi trường.
2. **`write`**: Thao tác ghi làm thay đổi trạng thái môi trường (bật/tắt đèn, tạo lịch hẹn, chuyển tiền, đặt vé máy bay, gửi tin nhắn).
3. **`meta`**: Công cụ hệ thống đặc biệt: **`sys.ask_missing_info`**. Luôn khả dụng đối với mọi mẫu kể cả khi không nằm trong `installed_apps`.

### 3.2. Ba thuộc tính Schema mở rộng cốt lõi (Bắt buộc phải nắm)
1. **`x-ref` (Tham chiếu khóa ngoại - Foreign Key)**:
   * Có tới **68 tham số** trong catalog được gắn cờ `x-ref` (ví dụ: `contact_id`, `event_id`, `device_id`, `room_id`).
   * **Nguyên tắc**: Agent **tuyệt đối không được bịa ID**. Khi người dùng nói *"Bật đèn phòng khách"*, Agent bắt buộc phải gọi tool đọc (`smart_home.get_status`) để lấy danh sách thiết bị và trích xuất đúng ID (`light_living_room`), sau đó mới truyền vào tool ghi (`smart_home.light_set`).
2. **`x-one-of` (Ràng buộc nhóm tham số lựa chọn)**:
   * Một số công cụ quy định phải cung cấp **đúng một trong các nhóm tham số** (ví dụ `sms.send`: bắt buộc phải có `contact_id` HOẶC `phone_number`; `music.play`: chọn `track_id` HOẶC `playlist_id`).
   * Nếu thiếu cả hai hoặc truyền sai quy cách, cuộc gọi sẽ bị từ chối do vi phạm schema.
3. **`x-match: fuzzy` (Văn bản tự do)**:
   * Có **40 tham số** được đánh dấu `x-match: fuzzy` (tiêu đề sự kiện, nội dung tin nhắn, ghi chú, địa chỉ giao hàng).
   * Khi chấm điểm, tham số này được so khớp theo **Token-F1 > 0.6** (sau khi chuẩn hóa Unicode NFC và lowercased).
   * Các trường này tự động được loại trừ khỏi `state_hash`.

### 3.3. Xử lý các tình huống đặc biệt
* **Trường hợp thiếu thông tin (Missing Information)**:
  * Khi người dùng ra lệnh nhưng thiếu thông tin bắt buộc không thể tự suy luận (ví dụ: *"Nhắn tin cho Nam"* nhưng trong danh bạ có nhiều người tên Nam hoặc chưa có nội dung tin nhắn), Agent không được đoán mò mà phải gọi tool:
    ```json
    {"name": "sys.ask_missing_info", "arguments": {"tool_name": "sms.send", "params": ["content"]}}
    ```
  * Sau khi gọi `sys.ask_missing_info`, Agent kết thúc phiên ngay lập tức.
* **Trường hợp ngoài phạm vi (Out-of-Scope / Unsupported)**:
  * Khi thiết bị chưa cài đặt ứng dụng xử lý yêu cầu hoặc yêu cầu không liên quan đến hệ thống công cụ, Agent **không gọi bất kỳ tool nào** và xuất ngay `{"done": true}`.
  * Khi đó: `tool_calls` nộp về là `[]` và `state_hash` là mã băm của trạng thái ban đầu chưa thay đổi.

---

## 4. MÔI TRƯỜNG GIẢ LẬP `zaic_tools` & MÃ BĂM TRẠNG THÁI `state_hash`

### 4.1. Cách thức hoạt động của `ToolEnv`
Thư viện `zaic_tools` nạp dữ liệu từ `<split>_environment/<id>/state.json` và catalog vào bộ nhớ:

```python
from zaic_tools import ToolEnv

env = ToolEnv(split="public_test", question_id=item["id"], root=".")

# 1. Chốt chặn an toàn: Kiểm tra hợp lệ trước khi gọi
check = env.validate("smart_home.light_set", {"device_id": "light_01", "power": "on"})
if not check.get("ok"):
    # Xử lý sửa lỗi hoặc hỏi lại, KHÔNG gọi env.call() để tránh ô nhiễm log bài thi
    pass

# 2. Thực thi công cụ: Cập nhật state trong RAM và lưu lịch sử call
result = env.call("smart_home.light_set", {"device_id": "light_01", "power": "on"})

# 3. Lấy dữ liệu để xuất file nộp bài
tool_calls = env.get_tool_calls()
state_hash = env.get_state_hash()
```

### 4.2. Cơ chế tính toán Canonical State & `state_hash`
Hàm `state_hash(state)` trong `zaic_tools/utils.py` là mã SHA256 của chuỗi JSON `canon(state)`:
* **Các thành phần bị loại bỏ trước khi băm**:
  1. Trường kỹ thuật `_meta`.
  2. Các ứng dụng thông tin chỉ đọc `LIVE_APPS = {"weather", "gold_price", "fuel_price"}`.
  3. Tất cả các trường văn bản tự do `FREE_TEXT_KEYS = {"address", "body", "content", "dropoff", "label", "location", "name", "note", "pickup", "subject", "text", "title"}`.
* **Chuẩn hóa dữ liệu**: Khoảng trắng được thu gọn, đơn vị đo ghép liền (`700 cm` $\rightarrow$ `700cm`), danh sách giá trị vô hướng được sắp xếp tăng dần.

---

## 5. CÔNG THỨC CHẤM ĐIỂM CHÍNH THỨC (METRIC FORMULA)

Công thức toán học từ ảnh công bố chính thức của Ban tổ chức ([`lifemate_formula_0.png`](<.vscode/extracted_images/lifemate_formula_0.png>)):

![LifeMate Metric Formula](<.vscode/extracted_images/lifemate_formula_0.png>)

$$\text{Score} = \begin{cases} 
0, & \text{nếu } \text{hash}_{\text{sys}} \neq \text{hash}_{\text{gold}} \text{ hoặc } M = 0 \\
\frac{2 \cdot P \cdot R}{P + R}, & \text{ngược lại}
\end{cases}$$

Trong đó:
$$P = \frac{M}{M + U}, \quad R = \frac{M}{|G|}$$

$$\text{Final score} = \frac{100}{N} \sum_{i=1}^N \text{Score}_i$$

* **Ý nghĩa các biến số**:
  * $N$: Tổng số mẫu trong tập test.
  * $\text{hash}_{\text{sys}}$, $\text{hash}_{\text{gold}}$: Mã băm trạng thái của hệ thống dự đoán và nhãn chuẩn.
  * $M$ (Matched): Số lời gọi công cụ khớp hoàn toàn cả tên và tham số.
  * $U$ (Unmatched / Unwanted): Số lời gọi công cụ thừa, sai tên hoặc sai tham số do mô hình sinh ra.
  * $|G|$: Tổng số công cụ trong nhãn chuẩn.
* **Xử lý Out-of-Scope ($|G| = 0$)**: Mẫu không có công cụ nào cần gọi sẽ đạt $\text{Score} = 1.0$ nếu mô hình không gọi tool nào; đạt $\text{Score} = 0.0$ nếu mô hình gọi bất kỳ tool nào.
* **State-Hash Hard Gate**: Nếu $\text{hash}_{\text{sys}} \neq \text{hash}_{\text{gold}}$, điểm số của mẫu lập tức bằng **0**, bất kể F1 tool call cao bao nhiêu!

---

## 6. KIẾN TRÚC MÔ HÌNH KHUYẾN NGHỊ (MODEL SELECTION)

Theo quy chế: **Tổng hệ thống < 6B, riêng LLM < 5B**:

| Thành phần | Lựa chọn tối ưu | Số tham số | Ưu điểm & Đánh giá phần cứng |
| :--- | :--- | :---: | :--- |
| **ASR Module** | `openai/whisper-large-v3-turbo`<br>*(hoặc `vinai/PhoWhisper-medium`)* | 809M<br>(769M) | Tốc độ nhanh gấp 4 lần PhoWhisper-large (chỉ 4 decoder layers), tiêu thụ chỉ ~1.8GB VRAM. Giúp toàn pipeline không bao giờ bị timeout 30s. |
| **LLM Module** | `Qwen/Qwen2.5-3B-Instruct` | 3.09B | Tối ưu nhất phân khúc < 5B. Kiến trúc GQA nhẹ, hiểu tiếng Việt xuất sắc, sinh JSON chuẩn. |
| **Tổng hệ thống** | **ASR + LLM** | **~3.90B** | **Hoàn toàn hợp lệ (< 6B)**, VRAM khi chạy chung chỉ ~10.5GB, chạy cực kỳ an toàn trên GPU 16GB (Kaggle T4) lẫn GPU 24GB (A30/3090). |

---

## 7. ĐỊNH DẠNG TỆP NỘP BÀI & LỆNH KIỂM TRA CLI

Tệp nộp bài có tên bắt buộc là **`submission.jsonl`**, mỗi dòng gồm:
```json
{"id": "public_0001", "tool_calls": [{"name": "alarm.create", "arguments": {"time": "2026-11-11T05:30:00+07:00"}}], "state_hash": "461fcd40625c9968434b56ee113c0953f332f0c1189a3d385575e1dabec263f3"}
{"id": "public_0002", "tool_calls": [], "state_hash": "0c9a2ec0fff5f8137d1f5e256c0caabca26af73ff9f8619c1efbf7f321b07f98"}
{"id": "public_0003", "tool_calls": [{"name": "sys.ask_missing_info", "arguments": {"tool_name": "music.add_to_playlist", "params": ["playlist_id", "track_id"]}}], "state_hash": "23f1c7d2419741b689aaa6bf48a180d3d250730740bafdfe9e69a745fbb4bf7a"}
```

### Lệnh kiểm tra tính hợp lệ Schema trước khi nộp bài:
```bash
python -m zaic_tools --split public_test --root . --validate submission.jsonl
```
Lệnh này sẽ quét toàn bộ file `submission.jsonl`, kiểm tra schema từng tool call đối chiếu với catalog và `installed_apps` của từng mẫu, thông báo chính xác số dòng lỗi nếu có.
