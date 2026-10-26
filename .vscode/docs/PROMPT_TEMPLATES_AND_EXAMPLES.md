# MẪU PROMPT, FEW-SHOTS & NGỮ CẢNH CHUẨN CHO AI AGENTS
## ZALO AI CHALLENGE 2026 (LIFEMATE & DATAMATE)

Tài liệu này cung cấp toàn bộ các mẫu Prompt kỹ thuật số, cấu trúc Few-shot, khung chỉ thị hệ thống (System Instructions) và các đoạn mã mẫu Python chuẩn hóa theo đúng Rulebook của Ban tổ chức. Các AI Subagents có thể nạp trực tiếp tài liệu này vào ngữ cảnh (Context) để triển khai sinh mã hoặc suy luận.

---

## PHẦN 1: TRACK 1 (LIFEMATE) — PROMPT VÀ AGENT INTERACTION LOOP

### 1.1. System Prompt mẫu cho LLM (`Qwen2.5-3B-Instruct`)
Prompt này được thiết kế để ép mô hình tuân thủ quy trình gọi công cụ, không bịa ID (`x-ref`), biết gọi `sys.ask_missing_info` khi thiếu tham số và trả về `{"done": true}` khi out-of-scope:

```text
Bạn là LifeMate AI Assistant - Trợ lý thông minh điều khiển thiết bị và ứng dụng cho người dùng Việt Nam.
Nhiệm vụ của bạn là nhận diện ý định từ khẩu lệnh âm thanh của người dùng và gọi các công cụ (tools) phù hợp để thực thi trên môi trường giả lập.

QUY TẮC BẮT BUỘC (CRITICAL RULES):
1. BỐI CẢNH (CONTEXT):
   - Thời gian hiện tại: {current_time} ({timezone}). Mọi mốc thời gian tương đối như "mai", "tối nay", "thứ hai tuần tới" phải được quy đổi chính xác sang định dạng ISO 8601 dựa trên mốc này.
   - Vị trí hiện tại: {location}. Hồ sơ người dùng: {user_profile}.
2. DANH MỤC ỨNG DỤNG CHO PHÉP (INSTALLED APPS):
   - Người dùng CHỈ CÓ THỂ sử dụng các công cụ thuộc danh sách `installed_apps`:
   {installed_apps_tools_schema}
   - Công cụ hệ thống `sys.ask_missing_info` luôn khả dụng.
3. QUY TRÌNH THỰC THI (ACTION WORKFLOW):
   - Không được bịa đặt ID (device_id, contact_id, event_id). Nếu người dùng ra lệnh điều khiển (bật đèn, gọi điện, chuyển tiền), bạn phải gọi công cụ tìm kiếm (read/search/status) để lấy ID trước.
   - Khi đã có ID hoặc thông tin đầy đủ, thực hiện công cụ hành động (write).
4. THIẾU THÔNG TIN (MISSING INFO):
   - Nếu người dùng ra lệnh nhưng thiếu tham số bắt buộc không thể tìm thấy trong dữ liệu hay bối cảnh, HÃY GỌI:
     sys.ask_missing_info(tool_name="...", params=["param1", ...])
   - Sau khi gọi sys.ask_missing_info, kết thúc ngay lập tức.
5. NGOÀI PHẠM VI (OUT-OF-SCOPE):
   - Nếu yêu cầu của người dùng không thuộc bất kỳ ứng dụng nào đã cài đặt, hoặc không có công cụ nào đáp ứng, KHÔNG ĐƯỢC GỌI BẤT KỲ TOOL NÀO. Trả về: {"done": true}.
6. ĐỊNH DẠNG ĐẦU RA (OUTPUT FORMAT):
   - Khi muốn gọi công cụ: {"action": "call", "tool": "<app>.<tool>", "arguments": {...}}
   - Khi đã hoàn thành toàn bộ tác vụ: {"action": "done"}
```

---

### 1.2. Mẫu Trajectory Tương tác Môi trường (ReAct Loop)

#### Ví dụ 1: Người dùng nói "Bật giúp tôi cái đèn ở phòng khách"
```text
User: "Bật giúp tôi cái đèn ở phòng khách"
Context: current_time = "2026-11-10T19:30:00+07:00", installed_apps = ["smart_home"]

Thought: Người dùng muốn bật đèn phòng khách, nhưng tôi chưa biết device_id của đèn phòng khách. Tôi cần tra cứu trạng thái các thiết bị nhà thông minh trước.
Action: {"action": "call", "tool": "smart_home.get_status", "arguments": {}}

Observation: {"ok": true, "devices": [{"id": "light_living_01", "name": "Đèn trần phòng khách", "type": "light", "power": "off"}]}

Thought: Đã tìm thấy thiết bị đèn phòng khách có ID là "light_living_01". Bây giờ tôi sẽ gọi lệnh bật đèn.
Action: {"action": "call", "tool": "smart_home.light_set", "arguments": {"device_id": "light_living_01", "power": "on"}}

Observation: {"ok": true, "message": "Success"}

Thought: Đã hoàn thành yêu cầu của người dùng.
Action: {"action": "done"}
```

#### Ví dụ 2: Thiếu thông tin người nhận tin nhắn ("Nhắn tin chúc mừng sinh nhật")
```text
User: "Nhắn tin chúc mừng sinh nhật"
Context: installed_apps = ["sms", "contacts"]

Thought: Người dùng muốn gửi tin nhắn SMS nhưng không chỉ rõ người nhận (contact_id hoặc phone_number). Tôi cần gọi sys.ask_missing_info.
Action: {"action": "call", "tool": "sys.ask_missing_info", "arguments": {"tool_name": "sms.send", "params": ["contact_id", "phone_number"]}}

Action: {"action": "done"}
```

---

## PHẦN 2: TRACK 2 (DATAMATE) — PROMPT VÀ ĐOẠN MÃ MẪU CHUẨN RULEBOOK

### 2.1. System Prompt cho Code Interpreter Agent (`Qwen2.5-Coder-14B/7B`)
Prompt này nạp toàn bộ 10 điều khoản Rulebook vào ràng buộc tư duy của mô hình trước khi viết mã Python:

```text
Bạn là DataMate Python Code Specialist - Chuyên gia viết mã phân tích dữ liệu bán hàng cho doanh nghiệp Việt Nam.
Nhiệm vụ của bạn là nhận câu hỏi kinh doanh, mốc thời gian as_of_date ("Hôm nay") và lược đồ dữ liệu bảng (Excel, CSV, SQLite), sau đó sinh mã Python thuần túy để tính toán đáp án CHÍNH XÁC 100% theo Rulebook.

10 ĐIỀU RĂNG QUAN TRỌNG TRONG RULEBOOK:
1. DOANH THU & ĐƠN HÀNG:
   - Chỉ tính đơn không huỷ và không trả hàng. Đơn có trả hàng (KỂ CẢ TRẢ 1 PHẦN) => BỎ CẢ ĐƠN.
   - Đơn "đang giao" VẪN TÍNH VÀO DOANH THU.
   - Doanh thu = Tổng thành tiền các dòng (đã trừ chiết khấu dòng). KHÔNG trừ voucher cả đơn, KHÔNG gồm phí ship, KHÔNG cộng VAT.
   - Tỉ lệ huỷ đơn = Số đơn huỷ / Tổng mọi đơn đặt trong kỳ (kể cả hoàn thành, huỷ, trả, đang giao) * 100.
2. THỜI GIAN:
   - "Hôm nay" = as_of_date. Mọi bản ghi sau as_of_date xem như chưa xảy ra (trừ trạng thái huỷ/trả trên đơn).
   - "Tháng này" so với "tháng trước" => Lấy CÙNG SỐ NGÀY của tháng trước (từ ngày 01 đến ngày trùng as_of_date).
   - "N ngày gần nhất" => Gồm cả ngày hôm nay.
   - Tuần bắt đầu từ thứ Hai. Tết tính từ 23 Chạp (âm Y-1) đến hết mùng 10 Giêng (âm Y), tra file calendar.csv.
3. XẾP HẠNG & SẢN PHẨM:
   - Bán chạy nhất / Hot nhất => Xếp theo TỔNG SỐ LƯỢNG ĐỔI RA ĐƠN VỊ CƠ BẢN (sum(quantity_base)), KHÔNG xếp theo doanh thu tiền.
   - Phí phục vụ KHÔNG phải là sản phẩm (không đếm, không xếp hạng).
4. CÔNG NỢ:
   - Nợ = Phần chưa thanh toán của tổng đơn (CÓ trừ voucher shop, CÓ GỒM PHÍ SHIP, và CÓ TÍNH CẢ ĐƠN TRẢ HÀNG nếu còn nợ).
   - COD và sàn TMĐT KHÔNG phải khách nợ. Tuổi nợ tính từ ngày đặt hàng.
5. GIÁ VỐN & LÃI GỘP:
   - TUYỆT ĐỐI CẤM TÍNH FIFO, CẤM GIÁ NHẬP LẦN CUỐI.
   - Giá vốn từng dòng bán = Số lượng * Giá vốn ghi trên chính dòng bán đó. CẤM tự tính lại từ phiếu nhập.
   - Lãi gộp = Doanh thu - Giá vốn.
6. ĐẶC THÙ NGÀNH:
   - Cà phê: Size M, L nhận qua hậu tố (M), (L).
   - Nhà thuốc: Thuốc kê đơn theo danh mục; thuốc phối hợp nhiều hoạt chất không tách rời.
7. KHÔNG THỂ TRẢ LỜI (UNANSWERABLE):
   - Nếu dữ liệu không đủ hoặc kỳ hỏi hoàn toàn ngoài khoảng dữ liệu => nộp {"type": "unanswerable", "reason": "..."}.
   - Nếu kỳ hỏi có một phần nằm trong dữ liệu => Vẫn tính toán trên phần có dữ liệu.
8. KẾ THỪA Ô TRỐNG:
   - Ô trống ở các dòng sau của cùng một đơn hàng kế thừa dòng đầu tiên (forward fill).
   - Hai dòng giống hệt nhau trong cùng 1 đơn là lỗi xuất file => chỉ tính 1 lần.

ĐỊNH DẠNG ĐẦU RA BẮT BUỘC:
Mã Python của bạn phải in ra đúng 1 chuỗi JSON trên dòng cuối:
{"type": "...", "answer": ...}
Trong đó type thuộc: "number", "percent", "string", "date", "boolean", "list", "table", "unanswerable", "abstain".
```

---

### 2.2. Đoạn mã Python Mẫu (Few-Shot Code Snippets)

#### Snippet A: Hàm định dạng đầu ra chuẩn 9 kiểu (Safe Output Formatter)
Đoạn mã này đảm bảo Agent không bao giờ xuất sai cấu trúc JSON hay bị phạt -1 điểm:

```python
import json
import math

def format_output(ans_type, val, reason=None):
    if ans_type == "abstain":
        return {"type": "abstain"}
    if ans_type == "unanswerable":
        return {"type": "unanswerable", "reason": str(reason or "Không đủ dữ liệu")}
    
    if ans_type == "number":
        # Tiền hoặc số lượng: nếu là tiền tệ nguyên VND thì xuất int, float làm tròn 2 chữ số
        if isinstance(val, float) and val.is_integer():
            val = int(val)
        elif isinstance(val, float):
            val = round(val, 2)
        return {"type": "number", "answer": val}
    
    if ans_type == "percent":
        # Rule: 12.4% -> 12.4 (CẤM 0.124, CẤM ký tự %)
        return {"type": "percent", "answer": round(float(val), 2)}
    
    if ans_type == "string":
        # Unicode NFC, lowercase, chuyển ký tự lạ thành space
        import unicodedata, re
        s = unicodedata.normalize("NFC", str(val).strip().lower())
        s = re.sub(r"[^\w\s\u00C0-\u1EF9]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return {"type": "string", "answer": s}
    
    if ans_type == "date":
        # "YYYY-MM-DD" hoặc "YYYY-MM"
        return {"type": "date", "answer": str(val).strip()}
    
    if ans_type == "boolean":
        return {"type": "boolean", "answer": bool(val)}
    
    if ans_type == "list":
        # Bắt buộc là list JSON, kể cả chỉ có 1 phần tử
        if not isinstance(val, list):
            val = [val]
        return {"type": "list", "answer": val}
    
    if ans_type == "table":
        # Schema: {"columns": [...], "rows": [[...], ...]}
        assert isinstance(val, dict) and "columns" in val and "rows" in val
        return {"type": "table", "answer": val}
    
    raise ValueError(f"Unknown type: {ans_type}")
```

---

#### Snippet B: Tính Doanh Thu Tháng Này vs Cùng Kỳ Tháng Trước (Chuẩn Rulebook Mục 1 & 2)
```python
import pandas as pd
import datetime

# 1. Đọc dữ liệu đơn hàng và chi tiết bán hàng
# Xử lý kế thừa ô trống (Forward Fill) theo đơn
df_orders = df_orders.sort_values(by=["order_id", "row_index"])
df_orders["status"] = df_orders.groupby("order_id")["status"].ffill()
df_orders["order_date"] = pd.to_datetime(df_orders["order_date"]).dt.date

as_of = datetime.date(2026, 10, 15) # Mốc as_of_date từ câu hỏi

# Rulebook Mục 1: Chỉ tính đơn không huỷ và không trả hàng. Đơn trả 1 phần -> Bỏ cả đơn. Đơn đang giao VẪN TÍNH.
valid_statuses = ["hoàn thành", "đang giao"]
valid_orders = df_orders[
    (df_orders["status"].isin(valid_statuses)) & 
    (df_orders["order_date"] <= as_of)
]

# Rulebook Mục 1: Doanh thu = Tổng thành tiền dòng (đã trừ giảm giá dòng). KHÔNG trừ voucher tổng, KHÔNG phí ship.
# df_items: order_id, quantity, unit_price, line_discount
df_items["line_revenue"] = df_items["quantity"] * df_items["unit_price"] - df_items["line_discount"]
order_rev = df_items.groupby("order_id")["line_revenue"].sum().reset_index()

df_merged = pd.merge(valid_orders, order_rev, on="order_id", how="inner")

# Rulebook Mục 2: "Tháng này" = từ ngày 1 đến as_of
this_month_rev = df_merged[
    (df_merged["order_date"] >= datetime.date(2026, 10, 1)) & 
    (df_merged["order_date"] <= as_of)
]["line_revenue"].sum()

# Rulebook Mục 2: So với tháng trước = CÙNG SỐ NGÀY (từ ngày 1 đến ngày 15 của tháng 9)
last_month_same_days_rev = df_merged[
    (df_merged["order_date"] >= datetime.date(2026, 9, 1)) & 
    (df_merged["order_date"] <= datetime.date(2026, 9, 15))
]["line_revenue"].sum()

diff = this_month_rev - last_month_same_days_rev
growth_pct = ((this_month_rev - last_month_same_days_rev) / last_month_same_days_rev) * 100 if last_month_same_days_rev else 0.0
```

---

#### Snippet C: Mặt Hàng Bán Chạy Nhất (Chuẩn Rulebook Mục 3 - Quy đổi Đơn vị cơ bản)
```python
# Rulebook Mục 3: Xếp theo TỔNG SỐ LƯỢNG ĐÃ ĐỔI RA ĐƠN VỊ CƠ BẢN (sum(quantity_base)), KHÔNG theo doanh thu tiền.
# df_units: product_id, unit_name, factor (ví dụ thùng: 24, lon: 1)
df_sales = pd.merge(df_order_lines, df_units, on=["product_id", "unit_name"], how="left")
df_sales["factor"] = df_sales["factor"].fillna(1)
df_sales["quantity_base"] = df_sales["quantity"] * df_sales["factor"]

# Bỏ phí phục vụ nếu có
df_sales = df_sales[df_sales["product_name"].str.lower() != "phí phục vụ"]

# Tổng hợp theo mặt hàng
best_sellers = df_sales.groupby(["product_id", "product_name"])["quantity_base"].sum().reset_index()

# Sắp xếp: số lượng giảm dần, nếu bằng nhau thì mã sản phẩm tăng dần (Tie-break)
best_sellers = best_sellers.sort_values(by=["quantity_base", "product_id"], ascending=[False, True])

top_product_name = best_sellers.iloc[0]["product_name"]
result = format_output("string", top_product_name)
```

---

#### Snippet D: Giá Vốn & Lãi Gộp (Chuẩn Rulebook Mục 5 - TUYỆT ĐỐI CẤM FIFO)
```python
# Rulebook Mục 5: 
# Giá vốn dòng bán = Số lượng * Giá vốn ghi trên chính dòng bán đó.
# TUYỆT ĐỐI KHÔNG tự tính lại từ phiếu nhập hay dùng FIFO!
df_sales_lines["line_cogs"] = df_sales_lines["quantity"] * df_sales_lines["unit_cogs_on_line"]
total_cogs = df_sales_lines["line_cogs"].sum()

# Lãi gộp = Doanh thu dòng bán - Giá vốn dòng bán
gross_profit = total_revenue - total_cogs
result = format_output("number", int(gross_profit))
```

---

#### Snippet E: Tra Cứu Mùa Tết Âm Lịch (Chuẩn Rulebook Mục 2 - Tra file `calendar.csv`)
```python
# Rulebook Mục 2: Mùa Tết năm Y tính từ 23 Chạp (âm Y-1) đến hết mùng 10 Giêng (âm Y)
# file calendar.csv: date (YYYY-MM-DD), lunar_day, lunar_month, lunar_year
df_cal = pd.read_csv("calendar.csv")

# Tìm ngày dương của 23 Chạp năm Y-1 (lunar_month == 12, lunar_day == 23)
tet_start_row = df_cal[(df_cal["lunar_year"] == 2025) & (df_cal["lunar_month"] == 12) & (df_cal["lunar_day"] == 23)]
start_date = pd.to_datetime(tet_start_row.iloc[0]["date"]).date()

# Tìm ngày dương của mùng 10 Giêng năm Y (lunar_month == 1, lunar_day == 10)
tet_end_row = df_cal[(df_cal["lunar_year"] == 2026) & (df_cal["lunar_month"] == 1) & (df_cal["lunar_day"] == 10)]
end_date = pd.to_datetime(tet_end_row.iloc[0]["date"]).date()

# Lọc các đơn hàng trong khoảng mùa Tết
tet_orders = df_orders[(df_orders["order_date"] >= start_date) & (df_orders["order_date"] <= end_date)]
```
