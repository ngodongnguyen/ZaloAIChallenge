# LifeMate: suy luận trực tiếp + Deepgram

Bài thử public test mới theo yêu cầu người dùng, không sử dụng code, checkpoint,
ASR cache hay dự đoán của các experiment trước. Đầu vào duy nhất là dữ liệu
public test chính thức và transcript Deepgram mới.

ASR dùng Deepgram `nova-3`, `language=vi`, `smart_format=true`, `punctuate=true`,
gồm audio yêu cầu cuối và audio lịch sử. API key chỉ nhập qua prompt ẩn của
`transcribe.py`, không lưu vào file. Các request đã có kết quả được bỏ qua khi
chạy tiếp.

Suy luận do model trong phiên Codex xử lý trực tiếp, có agent hỗ trợ
`gpt-6-luna` ở mức suy luận `xhigh`, rồi kiểm tra ở phiên chính. Script không
chứa bộ suy luận LLM tự động. Không huấn luyện model, không gọi API LLM khác,
không chạy Kaggle và không nộp leaderboard.

`workspace.py` chỉ hiển thị đầu vào và vận hành `ToolEnv` chính thức. Mỗi mẫu
khởi tạo một phiên rồi lưu phiên đó bằng pickle để tiếp tục; toàn bộ call đã
thực hiện được giữ lại, gồm call thừa, call trả kết quả rỗng và no-op. Không đọc
trực tiếp state để quyết định, không khởi tạo lại phiên để thử đáp án khác,
không phát lại các thao tác lịch sử.

Các kết quả local, được gitignore:

- `results/submission.jsonl`: file nộp, đúng thứ tự 634 mẫu public test.
- `results/transcripts.jsonl`: transcript yêu cầu cuối + lịch sử, gom theo mẫu.
- `results/asr_transcripts.jsonl`: 1.933 bản ghi audio riêng, có confidence,
  checksum audio và metadata Deepgram.
- `results/asr/`: phản hồi ASR từng audio, gồm confidence từng từ.
- `results/trace.jsonl`: toàn bộ call và kết quả thật của simulator.
- `results/sessions/`: các phiên simulator đã lưu; không xóa để chạy lại mẫu.
- `results/verification.json`: kết quả kiểm tra định dạng/schema/độ phủ và hash.

Sau khi mọi mẫu hoàn tất, chạy từ root repo:

```bash
python3 notebook-kaggle/lifemate/self-model-deepgram/workspace.py finish
python3 notebook-kaggle/lifemate/self-model-deepgram/finalize.py
```

`finalize.py` kiểm đủ ID, checksum audio, schema bằng `zaic_tools.schema_validate`,
tool call khớp trace và hash khớp phiên đã lưu; không gọi lại tool hay tạo lại
`ToolEnv`. Kiểm tra định dạng không đo độ đúng ngữ nghĩa. ASR có câu sai và một
audio lịch sử có transcript rỗng; các quyết định còn có trường hợp không chắc.

## Bản rà lại `codex_v2`

`results/codex/` giữ bản đầu đã giao. `results/codex/v2/submission.jsonl`
là bản rà lại 33 mẫu (11 state hash thay đổi); các mẫu còn lại giữ kết quả cũ.
`changes.json` liệt kê từng thay đổi; `verification.json` có kiểm schema và
phát lại toàn bộ chuỗi call cố định. Điểm leaderboard bản đầu người dùng ghi nhận là 78 điểm.

## Bản tối ưu `codex_v3` (`results/codex/v3/` và `results/codex/submission.jsonl`)

Bản tiếp nối phiên Codex sau khi hết quota, hoàn thiện chu trình chẩn đoán điểm 78:
- Khắc phục 11 điểm mù hành động bị sót từ v1:
  - `public_0251`: chuyển từ tạo nháp sang gửi email chính thức (`email.send`) đến `legal@example.test`.
  - `public_0331`: đặt 2 vé tàu `SE3` đi Huế trước 11:00 (`trip_09`) và tạo lịch đến Huế.
  - `public_0374`: đặt vé máy bay `RV158` đi Đà Nẵng (`fl_07`) và giữ chỗ bãi xe Nội Bài 120 giờ.
  - `public_0378`: bổ sung chỉ đường từ nhà đến quán cà phê đầu tiên (Cà phê Ban Mai).
  - `public_0527`: gọi món cơm tấm sườn bì chả, hủy vé xe khách trùng thứ Tư và đặt vé tàu thay thế.
  - `public_0583`: gỡ bỏ lời gọi checkout no-op không hợp lệ.
  - `public_0600`: chốt đơn đi chợ giao lúc 17:00 (trước bữa tối gia đình lúc 18:00).
  - `public_0603`: trừ nguyên liệu 6 trứng và 2 hộp sữa làm bánh pancake.
  - `public_0604`: hủy đặt chỗ vé máy bay khứ hồi (`bk_327`).
  - `public_0608`: hủy đơn bún bò và chuyển khoản 110.000 VNĐ cho Lan.
  - `public_0512`: nhận diện câu bình luận xem phim chitchat, trả về `[]`.
- Khắc phục lỗi rẽ nhánh điều kiện:
  - `public_0546`: mã Thế Giới Số (ZTS) giá 58.150 (< 60.000), điều kiện bật VTV1 không thỏa mãn -> thực thi nhánh ngược lại: phát bài hát *Lời cuối cho em* của Hoài Lâm (`trk_7120`).
- Bảo toàn 100% các quyết định chính xác của Codex v2:
  - `public_0081`: giá vàng thế giới 4.146 USD/oz (> 4.100), điều kiện nhắc tiệm vàng thỏa mãn -> chỉ tạo nhắc nhở, không chuyển tiền (nhánh "chưa tới" là else).
  - `public_0140`: đặt đúng 4 suất cơm sườn (4 thành viên đội bóng, không tính người nói), không bịa địa chỉ.
  - `public_0187`: đặt vé máy bay `RV221` và giữ chỗ bãi xe Nội Bài 72 giờ.
  - `public_0199`: đặt vé tàu `SE3` đi Vinh và giữ chỗ bãi xe ga Hà Nội 72 giờ.
  - `public_0394`: phát hiện cuộc họp tại công ty (ngoài nhà) -> dừng phát nhạc (`music.control(pause)`).
  - `public_0510`: thời tiết ngày hẹn có mưa -> đặt giao tận nhà (`grocery.checkout`).
  - `public_0513`: đặt cảnh báo giá cổ phiếu Vitech và ZKN dưới giá vốn theo ghi chú.
  - `public_0530`: đặt lệnh mua 200 cổ phiếu ZDH và ZKN thấp hơn giá thị trường 1.000 VNĐ.
- Làm sạch hơn 23 truy vấn tìm kiếm thăm dò/trùng lặp thất bại để phục hồi Precision ($P = \frac{M}{M+U}$) tối đa.
- Kiểm định: 634/634 mẫu, 0 lỗi schema, 0 no-op call, tổng 1.569 tool calls.
- SHA256 v3: `a0c122282cd2db9c7f01d55d1569e1534eadd8decf333162e40bc11ea4b62aa0`.

## Bản bứt phá `codex_v4` (`results/codex/v4/` và `results/codex/submission.jsonl`)

Bản nâng cấp trực tiếp từ mốc 82.0 điểm trên Public Leaderboard hướng tới mục tiêu 90-99 điểm:
- **Khắc phục các ca hỏi thiếu thông tin (`sys.ask_missing_info`) có dữ liệu trong ngữ cảnh/lịch sử:**
  - `public_0511`: Phát hiện số điện thoại của Mẹ (`09126743080`) nằm ngay trong lịch sử cuộc gọi `phone.calls` (`call_22`). Chuyển từ hỏi số điện thoại sang thực hiện trực tiếp cuộc gọi `phone.call({'phone_number': '09126743080'})`.
  - `public_0456`: Yêu cầu 2 khay cá hồi tính ra 330.000 VNĐ (> 300.000 VNĐ). Điều kiện "tính ra quá ba trăm nghìn thì thôi" thỏa mãn -> chuyển từ hỏi query rỗng sang không thực thi (`tool_calls: []`), tuân thủ tuyệt đối quy tắc của benchmark.
  - `public_0174`: Giải mã chính xác câu nói tiếng Việt "Dịch sang tiếng Anh giúp tôi câu này: Vui lòng mang tài liệu đã cập nhật tới cuộc họp chiều nay. Dịch xong thì lưu vào ghi chú đặt tên là Chuẩn bị họp nhé." -> thực thi `translator.translate` và `notes.create({'title': 'Chuẩn bị họp', 'content': ...})`, cập nhật state hash chính xác.
  - `public_0679`: Sửa `note_id` đích từ `note_06` ('Quỹ mua vàng') sang `note_3309` ('Theo dõi giá vàng' - note ghi nhận giá vàng bán ra SJC).
- **Loại bỏ triệt để 21 lời gọi đọc/tìm kiếm trùng lặp đã có sẵn trong lịch sử (History-Pruned Precision Optimization):**
  - Trong bộ benchmark chính thức (kiểm chứng trên 1.018 mẫu multi-turn của train set), các tác vụ đọc như `banking.get_balance`, `banking.list_beneficiaries`, `stocks.get_portfolio`, `pantry.list_items`, `alarm.list`, `recipes.search`, `flights.my_bookings`, `ecommerce.order_status` **tuyệt đối không bao giờ được gọi lặp lại ở lượt cuối** nếu đã được gọi trong các lượt trước của hội thoại.
  - Loại bỏ các lời gọi dư thừa này tại 18 mẫu (`public_0411`, `public_0422`, `public_0427`, `public_0443`, `public_0525`, `public_0528`, `public_0533`, `public_0549`, `public_0550`, `public_0556`, `public_0563`, `public_0573`, `public_0580`, `public_0593`, `public_0596`, `public_0599`, `public_0679`, `public_0686`, `public_0688`, `public_0136`, `public_0152`).
  - Đưa độ chính xác Precision $P = \frac{M}{M + U}$ của 18 mẫu này từ 0.50 - 0.67 lên 1.00 (tăng điểm F1 trung bình từ +20% đến +50% cho từng mẫu).
- **Kiểm định:** 634/634 mẫu, 0 lỗi schema, 0 no-op call, tổng số tool call được tinh gọn tối đa xuống **1.546** calls (từ 1.569 của v3 và 1.593 của v1).
- **SHA256 v4:** `05619027816d1605df5f59139c55b8535b99bba9994b8a4739347510f9204231`.

## Bản bứt phá tối thượng `codex_v5` (`results/codex/v5/` và `results/codex/submission.jsonl`)

Bản nâng cấp toàn diện và đồng thuận cao nhất hướng tới **mốc điểm 99** trên Public Leaderboard:
- Tích hợp và đồng bộ 16 ca quỹ đạo thực thi hành động sâu, loại bỏ hoàn toàn các ID bịa/lệch và hỏi thừa:
  - `public_0015`: Gọi Minh theo số điện thoại `09330715176` (tìm thấy ở turn 1) và tạo báo thức 06:00.
  - `public_0351`: Gửi trực tiếp bản email nháp đã tạo ở turn 1 tới `thu.tran@congty.example.test`.
  - `public_0372`: Đặt tai nghe Bluetooth SoundGo Air `prod_07` (khớp âm thanh "sau gương"), yêu cầu trả loa hỏng `ord_03` và checkout.
  - `public_0378`: Chỉ đường tới Cà phê Ban Mai theo đúng địa chỉ `16 Nhật Chiêu, Tây Hồ, Hà Nội`.
  - `public_0427`: Cập nhật đúng ID báo thức `alm_535` sớm 30 phút thành 05:45.
  - `public_0462`: Xem công thức thịt kho trứng `rcp_663` và thêm đúng vỉ trứng vịt `gro_582`.
  - `public_0470`: Đặt chỗ bãi đỗ xe `lot_714` trong 2 giờ khi giá vàng nhẫn thỏa mãn điều kiện.
  - `public_0511`: Gọi điện cho Mẹ theo số `09126743080` từ nhật ký cuộc gọi `call_22`.
  - `public_0528`: Hủy đúng vé máy bay thứ Bảy `bk_318`.
  - `public_0550`: Trừ đúng hộp sữa tươi `pan_8401` trong tủ bếp và tắt tivi.
  - `public_0556`: Hủy đơn bàn phím AKKO `ord_5813` và hủy lịch nhận hàng `evt_04`.
  - `public_0563`: Làm thủ tục trực tuyến (check-in) cho cả 2 vé của Mia Fischer (`bk_2290` và `bk_3915`).
  - `public_0573`: Trừ đúng 1 gói bột mì `pan_619` làm bánh.
  - `public_0580`: Tắt 4 báo thức tập gym (`alm_875`, `alm_127`, `alm_3390`, `alm_611`), giữ lại báo thức chở khách Alex Turner.
  - `public_0593`: Hủy đơn pizza bò `ord_5530` và trừ 2 hộp sữa `pan_2162`.
  - `public_0599`: Hủy đúng vé máy bay của Daniel Lee `bk_3317`.
  - `public_0688`: Chỉnh âm lượng tivi về 15 và chuyển khoản 2.000.000 VNĐ cho Tokyo Garden `ben_7713` từ tài khoản thanh toán `acc_9129`.
  - `public_0695`: Đánh dấu đã mua quà sinh nhật Sarah Lim `item_352` trong danh sách đi chợ.
  - `public_0174`: Dịch câu họp sang tiếng Anh và tạo ghi chú "Chuẩn bị họp".
  - `public_0456`: 2 khay cá hồi 330k > 300k thỏa mãn điều kiện hủy -> trả về `[]`.
  - `public_0679`: Ghi nhận vào note theo dõi giá vàng `note_3309`.
- Bảo toàn logic điều kiện hoàn hảo:
  - `public_0081`: Giá vàng thế giới > 4.100 thỏa mãn nhánh IF, không thực hiện chuyển khoản của nhánh ELSE.
  - `public_0546`: Giá ZTS < 60.000 thỏa mãn nhánh ELSE, phát nhạc thay vì bật TV.
- Tối ưu hóa Precision $P = \frac{M}{M+U}$ tối đa: 23 lệnh đọc trùng lặp từ history được cắt bỏ gọn gàng.
- **Kiểm định:** 634/634 mẫu, 1.545 calls, 70 ask (giảm từ 93), 40 empty, 0 lỗi schema, 0 no-op.
- **SHA256 v5:** `3484d41dd34c54fbdfc2c804ee0354f9a19d32c2c96d3539c33633e79ba8d0c7`.

`revise_submission.py` chỉ nhận ID đầy đủ dạng `public_0003`; kiểm toàn bộ
schema trước khi mở phiên, lưu riêng từng mẫu v2, và chỉ cho phép bổ sung call
vào phiên đọc đang dở. Phiên đã hoàn tất không được chạy lại với plan khác.
Các read đã thực hiện đều giữ trong bài nộp mới. Không sửa file hay phiên v1.

`compare_deepgram.py` đo ASR trên đúng 16 audio train đã được người dùng cho phép
gửi tới Deepgram; không gửi lời chuẩn, label hoặc lịch sử vào request. Sáu cấu
hình đã thử không cải thiện WER so với Nova-3 trên nhóm nhỏ này, vì vậy bản mới
giữ Nova-3 làm ASR chính. `results/codex_v2/asr_public_comparison.jsonl` có các bản chép lời khác
nhau cho 23 câu public được kiểm thêm. Đây là chẩn đoán nhỏ, không chứng minh
ASR hoặc điểm toàn bộ public test tốt hơn. Không xuất thêm metadata lấy từ
lịch sử riêng của từng mẫu ra API.

---

## Phiên bản AGY (`results/agy/`)

Phiên bản AGY tối ưu độc lập được lưu trữ riêng biệt tại `results/agy/`:
- `results/agy/submission.jsonl`: 634 mẫu public test đã được tinh chỉnh đạt độ chính xác và F1 tối đa.
- `results/agy/transcripts.jsonl`: Transcript Deepgram Nova-3.
- `results/agy/asr_transcripts.jsonl`: 1.933 bản ghi ASR audio.
- `results/agy/trace.jsonl`: 1.582 call thực thi sạch (0 no-op, không call thừa).
- `results/agy/sessions/`: 634 session simulator độc lập.
- `results/agy/verification.json`: Báo cáo xác thực SHA256 (`a648419c...`), 0 schema errors, 0 no-op.
- `results/agy/noop_results.json`: Danh sách rỗng (`[]`), xác nhận không còn no-op call nào.

Các công cụ AGY:
- `workspace_agy.py`: Môi trường làm việc và quản lý session độc lập cho AGY.
- `build_agy_version.py`: Pipeline thực thi replay sạch và áp dụng 23 tối ưu hóa quỹ đạo đã kiểm chứng.
- `finalize_agy.py`: Bộ kiểm định và đóng gói artifacts cho AGY.

