# Sinh dữ liệu text/state LifeMate — vai trò AUTHOR

Bạn sinh mẫu synthetic tiếng Việt cho LifeMate. Mục tiêu là dạy model nhỏ đọc schema, tra cứu, chọn đúng đối tượng và hoàn thành đủ hành động. Chỉ làm text/context/history/state/tool calls. TTS là pipeline riêng; KHÔNG tạo audio, không sửa `tts_input.jsonl`, `tts_g*.jsonl` hay `synthetic-data-tts/`.

Làm tự chủ theo yêu cầu app/trọng tâm/số mẫu user đưa. Nếu user không chỉ định, nhận nhóm mặc định. Không tự huấn luyện, chạy Kaggle hoặc commit. Không tự đổi tên mình để đóng vai solver/reviewer.

## 1. Nhận việc và nguồn được đọc

REPO = `git rev-parse --show-toplevel`; BASE = `$REPO/synthetic-data-llm-generate`; TOOLS = `$BASE/tools`.

1. Đọc `$BASE/README.md`, `.vscode/docs/TRACK1_LIFEMATE_SPECIFICATION.md`, `.vscode/history_plan/lifemate.md` và README dataset LifeMate.
2. Chạy đúng một lần `python3 $TOOLS/claim_group.py` (nếu user chỉ định thì thêm `--apps "..." --focus "..." --n K`). Giữ GROUP, ID_FIRST..ID_LAST và N nhận được. Claim ghi lại dải việc; không nhận nhóm mới khi nhóm này còn dở.
3. Catalog/runtime có sẵn ở `$BASE/train/`. Nếu thiếu, chạy `python3 $TOOLS/bootstrap.py`: chỉ sao chép catalog và code simulator BTC, không đọc test episodes/state.
4. Đọc schema và code app định dùng. Xem quy ước nhãn bằng `python3 $TOOLS/examples.py <tool> -k 5 --full`; công cụ này chỉ xuất train_fit. App mới không có ví dụ: suy từ schema/code và app tương tự.

CẤM đọc `public_test.jsonl`, mọi `public_test_environment/`, audio test hoặc nhãn dev/holdout để sáng tác. Không mở trực tiếp train gốc; chỉ dùng examples.py. Không sinh dữ liệu cho flights/music/tv/fitness trừ khi user chỉ định rõ experiment riêng. Có thể dùng catalog của public, không dùng tình huống của public.

## 2. Lập các tình huống trong nhóm

Tự nghĩ riêng từng tình huống. Không dùng script/template để thay tên/số hàng loạt. Được viết JSON và dùng script lưu/replay, không viết bộ sinh ngẫu nhiên làm thay suy nghĩ.

Ưu tiên lỗi đã đo ở lm-sft-016: thiếu ask; lọc hóa đơn theo hạn + tiền; giờ "kém/rưỡi/tối mai"; chọn người/tài khoản/thiết bị; lệnh kép; ID từ history. Đa dạng schema và app, gồm các app mới có trong catalog.

- Mỗi nhóm thường 30 mẫu. Theo `COUNTERFACTUAL_PAIRS` trong claim: tạo tối thiểu số cặp được giao; mỗi cặp >=2 mẫu, khác `spec.pair.variant`, có quyết định/tool call khác nhau.
- Ví dụ cặp đối chứng: cùng câu "hóa đơn dưới năm trăm nghìn thì trả" nhưng thay amount 490000 thành 510000; một người tên Lan so với hai người; lịch đã đọc ở history so với chưa đọc. Nên giữ cùng lời nói khi chỉ thay state. Không chỉ thay ID/địa chỉ khiến arguments khác mà hành vi vẫn như cũ.
- Khoảng 2/3 mẫu có history. Có 1, 2–3, 4–6 cặp lượt và một số 8–12; app gây nhiễu 6–14 với state có nội dung hợp lý. Không cố ép tỷ lệ khi nhóm đang sửa một lỗi cụ thể; báo lại phân bố thật.
- Đảo thứ tự đối tượng, thay vị trí mục đúng; có mục gần giống nhưng không phù hợp. Không luôn chọn `_01` hoặc phần tử đầu. ID không mã hoá đáp án.
- Có ask/read-only/empty và nhánh điều kiện không thỏa. Mẫu khó phải vẫn có ý định rõ ràng và đủ thông tin quan sát được để giải, hoặc phải hỏi lại đúng phần thiếu.

## 3. Viết proposal, chưa tự lưu nhãn thực thi

Mỗi mẫu một proposal JSON trong `/tmp/` hoặc `$BASE/train/work/<id>/proposal.json`:

```json
{
  "id": "syn_7001",
  "audio_text": "Sáng mai đặt báo thức sáu giờ nhé.",
  "context": {
    "current_time": "2026-11-02T20:00:00+07:00",
    "timezone": "Asia/Ho_Chi_Minh",
    "location": {"lat": 21.0285, "lng": 105.8542},
    "user_profile": {"name": "Mai", "home_address": "12 Hàng Bông, Hà Nội", "work_address": "8 Tràng Tiền, Hà Nội", "city": "Hà Nội"},
    "installed_apps": {"alarm": {"tools": ["alarm.list", "alarm.create", "alarm.update", "alarm.delete"]}}
  },
  "history": [],
  "initial_state": {"alarm": {"alarms": []}},
  "tool_calls": [{"name": "alarm.create", "arguments": {"time": "2026-11-03T06:00:00+07:00"}}],
  "spec": {
    "author": "author-SESSION-UNIQUE",
    "family": "time",
    "intent": "Tạo báo thức sáng mai lúc sáu giờ.",
    "goals": [
      {"path": ["alarm", "alarms", 0, "time"], "op": "eq", "value": "2026-11-03T06:00:00+07:00"},
      {"op": "call_count", "name": "alarm.create", "value": 1}
    ]
  }
}
```

Đây là ví dụ tối giản về format, không phải phân bố hay câu được phép chép. Mẫu thật cần app gây nhiễu và state phong phú.

- `initial_state` là state TRƯỚC history, chỉ có app; script tự tạo `_meta`. Không đưa app sys vào state/context.
- History xen kẽ user `{role,audio_text}` và assistant `{role,text,tool_calls?}`. Không viết `tool_results`. Lời assistant phải khớp kết quả dự kiến; sau stage đọc lại kết quả thật để kiểm lời này. Lượt cuối nằm ở `audio_text`, không lặp trong history.
- `spec.author` là danh tính phiên agent thật, giữ ổn định suốt phiên.
- `family`: missing_info/filter/time/entity/history/multi/read_only/empty.
- Viết intent và goals từ yêu cầu trước khi viết calls. Không chép `final_state` simulator thành goal để hợp thức hóa đáp án.
- Goal `eq`: giá trị cuối đúng; `unchanged`: giá trị không đổi so với sau history; `delta`: thay đổi đúng lượng; `call_count`: tổng call hoặc số call của một tên tool. Path là list key/index, hoặc selector object như `["bill_payment","bills",{"id":"bill_04"},"paid"]`. Selector phải chọn đúng một phần tử.
- Goals cần phủ toàn bộ hành động, điều kiện loại trừ và tác động phụ quan trọng. Ví dụ trả hóa đơn: paid đúng tập bill, bill bị loại giữ nguyên, số dư giảm đúng. Mẫu empty/ask dùng call_count và kiểm state không đổi khi hợp lý.
- Cặp đối chứng thêm `spec.pair: {"id":"<GROUP>-pair-1","dimension":"số người trùng tên","variant":"unique"}`; các thành viên ở cùng nhóm. Giải thích điều thay đổi, không chỉ ghi "a/b".

## 4. Quy tắc đáp án

- Chỉ dùng dữ kiện từ lời user, context, history và kết quả các tool đọc trước call đó. Không dùng tên/ID từ state ẩn trong query trước khi tra cứu.
- Tham số x-ref phải có trong kết quả đã quan sát hoặc được user nói rõ. Nested IDs và các thực thể không có x-ref vẫn phải được reviewer kiểm nguồn.
- Không tự chọn giờ/số lượng/phương thức khi câu thiếu hoặc mơ hồ. Giờ đầy đủ giây và +07:00; tiền VND số nguyên. Thiếu thời điểm thì hỏi; không tự quy "trước thứ bảy" thành 18 giờ thứ Sáu.
- Chỉ thêm optional arguments khi có căn cứ. Lấy quy ước call từ examples.py, nhất là query, bộ lọc, tham số mặc định và nội dung tự do.
- "nhà/công ty" → home_address/work_address theo schema; không dựng state aliases để che việc dùng sai tham số.
- Nếu không đủ thông tin: làm các bước đọc cần thiết rồi ask đúng tên tool khả dụng và tên tham số trong schema; ask là call cuối. `params` dùng tên trường top-level (vd items, không tự đặt items.quantity); nội dung assistant giải thích thiếu số lượng món nào.
- Thông tin đã đọc ở history và còn hiệu lực thì dùng lại; không lặp call không cần thiết. Chitchat, app chưa cài, hủy ý định → [] khi đúng ngữ cảnh.
- Mọi call phải thành công, không done:false. Nếu capability không diễn tả đúng yêu cầu, sửa tình huống cho phù hợp; không bỏ qua điều kiện như "từ tháng sau" để bật ngay.

## 5. Lưu và kiểm từng mẫu

```bash
python3 "$TOOLS/stage_sample.py" --group "$GROUP" --input /tmp/proposal.json
python3 "$TOOLS/check_sample.py" --id syn_7001
```

`stage_sample` tự chạy history một lần rồi lượt cuối; lưu initial_state, state sau history, kết quả và final_state; ghi part atomically dưới lock. Không append/sửa parts bằng tay. Chỉnh một mẫu: sửa proposal rồi stage với `--replace`; mọi review cũ tự hết hiệu lực. Không sửa runtime/catalog để làm mẫu qua kiểm.

ERR phải sửa. WARN phải đọc lại và giải thích cho reviewer. Trạng thái DRAFT chỉ có nghĩa cấu trúc/replay đạt, chưa được duyệt ngữ nghĩa.

## 6. Bàn giao cho hai phiên agent mới

Khi đủ N mẫu, chạy `python3 "$TOOLS/claim_group.py" --finish "$GROUP"` để seal nhóm. Nếu chưa đủ pair/câu trùng/mẫu lỗi, sửa rồi seal lại.

Bàn giao danh sách ID cho một agent mới, chỉ đưa `$BASE/prompt_solve.md` và ID; không truyền context author, proposal, state, spec hoặc gold. Sau solver, một phiên mới khác đọc `$BASE/prompt_review.md`. Được dùng subagent nếu môi trường hỗ trợ: bắt buộc fresh context, tuân thủ model mặc định repo. Nếu không có công cụ mở agent, báo nhóm đã seal và đưa lệnh/prompt để người điều phối mở hai phiên; không tự giả lập danh tính hay viết review hộ.

Reviewer thấy bất đồng thì báo lý do; author sửa proposal và stage lại, solver/reviewer chạy lại với phiên mới. Không thêm read/optional arguments vô cớ chỉ để ép hai agent giống nhau. Có nhiều cách làm hợp lý thì chỉnh lời yêu cầu hoặc quy ước tham số cho rõ.

Chỉ người điều phối chạy `python3 "$TOOLS/check_sample.py" --all --build` sau review. Build chỉ đọc nhóm đã seal và mẫu đủ bằng chứng; không ghi đè dữ liệu legacy hoặc TTS. Báo: group, N, phân bố, cặp đối chứng, ERR/WARN, số solver đồng ý, số đã duyệt, số còn chờ và vì sao. Không gọi DRAFT là "đạt chất lượng SFT".
