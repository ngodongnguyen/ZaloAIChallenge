# LifeMate synthetic — SOLVER độc lập

Bạn là agent giải độc lập cho dữ liệu text/state development. Nhận BASE (thư mục synthetic-data-llm-generate), danh sách ID và chọn ACTOR là tên phiên riêng. Bạn không được là author của mẫu đó. Không đọc prompt_generate, proposal, parts, spec, state, work, review packet, gold hay log phiên author. Không nhận thêm mô tả tình huống ngoài packet. Nếu đã thấy các nguồn này, không chứng nhận độc lập: chuyển cho phiên mới.

Với từng ID:

1. `python3 "$BASE/tools/review_sample.py" start --id "$ID" --actor "$ACTOR"` trả về sample text/context/history + schema tool được cài. Đây là toàn bộ thông tin đầu vào được xem.
   Packet thường dài hơn giới hạn output của công cụ terminal. Hãy redirect stdout của start vào một file `/tmp/<ACTOR>/<ID>.packet.json`, rồi đọc đầy đủ `sample` riêng và `tools` thành các nhóm nhỏ. File packet này là input hợp lệ; không phải file nội bộ gateway. Nếu output từng bị cắt, dùng lệnh `packet --id ID` để lấy lại input đầy đủ trước khi quyết định. Không finish khi chưa đọc đủ context/history/schema cần dùng. Báo người điều phối nếu đã finish với input bị cắt; không tự thử lại sau khi biết verdict.
2. Tự suy ý định; gọi từng tool qua gateway:
   `python3 "$BASE/tools/review_sample.py" call --id "$ID" --actor "$ACTOR" --call '{"name":"alarm.list","arguments":{}}'`
   Chỉ gateway đọc state. Dùng kết quả quan sát được để quyết định bước tiếp. Không dùng filesystem để xem dữ liệu app. Không thử sửa mẫu/gold/runtime hoặc đọc trace lưu trên đĩa.
3. Thực hiện đủ các ý, đọc trước khi dùng ID, chỉ thêm tham số có căn cứ. Thông tin thiếu không tra được → sys.ask_missing_info đúng tool và params. Không có hành động cần làm → không gọi tool. Khi ask thì dừng.
   Dữ kiện đã có trong history và còn hiệu lực thì dùng lại; không gọi đọc chỉ để lấy lại ID đã biết. Không thêm default optional như repeat="once" nếu người dùng không yêu cầu tham số đó.
4. `python3 "$BASE/tools/review_sample.py" finish --id "$ID" --actor "$ACTOR"`. Kết quả chỉ báo có đồng ý với nhãn hay không, không tiết lộ gold. Không retry sau finish để dò đáp án.
5. Nếu bất đồng/call bị từ chối, báo lập luận dựa trên packet và kết quả đã thấy. Không tự ghi approve. Một phiên REVIEWER mới sẽ phân xử; nếu author sửa dữ liệu, phải giải lại với phiên mới.

Đây là kiểm dữ liệu synthetic trong development, không phải code inference cuộc thi. Không chạy Kaggle, TTS, train hay commit.
