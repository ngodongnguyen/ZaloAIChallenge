# Sinh dữ liệu LifeMate bằng agent

Pipeline **text + context + history + state + tool calls** phục vụ SFT. TTS được làm riêng: các công cụ ở đây không cần WAV, không tạo audio, không sửa `tts_input.jsonl`/`tts_g*.jsonl` hay thư mục `synthetic-data-tts/`.

Mỗi mẫu đi qua **AUTHOR → SOLVER ở phiên mới → REVIEWER ở phiên mới → release**. Simulator kiểm cấu trúc/thực thi; solver và reviewer kiểm khả năng giải từ thông tin quan sát được và đúng ý người dùng. Hai agent đồng ý vẫn có thể cùng sai; reviewer phải kiểm bằng chứng cụ thể. Công cụ kiểm danh tính phiên khác nhau và snapshot dữ liệu, không thể tự chứng minh người chạy đã thực sự tách context.

## Bắt đầu

Từ root repo:

```bash
python3 synthetic-data-llm-generate/tools/bootstrap.py
python3 synthetic-data-llm-generate/tools/claim_group.py
```

Thông thường user chỉ cần nói **“sinh dữ liệu LifeMate”**: agent đọc [prompt_generate.md](prompt_generate.md), tự claim và làm đúng nhóm được giao. Có thể chỉ định `--apps "banking, bill_payment" --focus "lọc theo hạn và tiền" --n 30`. Mỗi nhóm có dải ID riêng, claim nguyên tử, số lượng 1..999. Không tạo nhiều claim cho cùng một phiên đang làm dở.

`bootstrap` chỉ lấy catalog và Python runtime BTC từ bản local LifeMate. Không đọc/sao chép câu hỏi, audio hay state test. Không ghi vào dataset. Nếu runtime đang có khác bản BTC thì dừng để tránh làm sai simulator.

## Các bước cho một nhóm

1. **AUTHOR** làm theo [prompt_generate.md](prompt_generate.md). Viết từng proposal JSON với initial_state trước history, câu nói/context/history, calls dự kiến và spec: tác giả, nhóm lỗi, ý định, điều kiện thành công. Không tự viết tool_results/final_state.
2. Lưu mỗi mẫu bằng `stage_sample.py --group g7 --input /tmp/proposal.json`. Script chạy simulator cho history rồi lượt cuối, kiểm ID đã quan sát/schema/goals và ghi atomically. Sửa bằng cùng lệnh thêm `--replace`; không sửa parts trực tiếp.
3. `check_sample.py --group g7`: cấu trúc/replay hợp lệ mới là **DRAFT**, chưa đủ để dùng train. ERR phải sửa; WARN là việc reviewer cần cân nhắc.
4. `claim_group.py --finish g7`: seal khi đủ N, không trùng câu, đủ cặp đối chứng có hành vi khác nhau. Nhóm còn ghi dở không được build đọc.
5. Một **SOLVER mới** nhận [prompt_solve.md](prompt_solve.md) + ID. Chỉ xem packet (text/context/history/schema), gọi tool qua gateway; không xem proposal/state/spec/gold hay dùng context tác giả. `finish` ghi nhận có đồng ý với nhãn không, không cho xem nhãn để dò lại.
6. Một **REVIEWER mới khác** nhận [prompt_review.md](prompt_review.md) + ID. Kiểm sáu mặt: ý định, nguồn dữ kiện, đủ/không thừa call, mơ hồ/thiếu thông tin, history, goals. Report phải ghi `fingerprint` từ packet đã kiểm và bằng chứng cụ thể rồi mới approve. Nếu bất đồng, sửa proposal và chạy lại bằng phiên solver/reviewer mới.
7. Người điều phối build sau khi review:

```bash
python3 synthetic-data-llm-generate/tools/check_sample.py --all --build
python3 synthetic-data-llm-generate/tools/check_sample.py --verified-path
```

Lệnh cuối in **đường dẫn JSONL của release đã duyệt**. Đó là file dùng cho experiment mới. Dữ liệu vẫn có layout từng dòng như train gốc; `audio` là đường dẫn dự kiến cho pipeline TTS riêng.

## Release và chống ghép nhầm

- `train/parts/gN.jsonl`: nguồn draft; `gN.claim.json`: phạm vi công việc; `gN.complete.json`: seal nội dung.
- `train/work/<id>/initial_state.json`, `spec.json`: dữ kiện trước history và yêu cầu kiểm độc lập.
- `train/train_environment/<id>/state.json`: state sau history, trước yêu cầu cuối.
- `train/work/<id>/solvers/`: lịch sử gateway; `review.json`: kết quả duyệt có bằng chứng.
- `train/releases/<digest>/train.jsonl` và `manifest.json`: snapshot đã duyệt. Manifest chứa checksum, bằng chứng, phân bố và mục bị bỏ qua.
- `train/verified.json`: con trỏ cập nhật nguyên tử sau khi release ghi đủ. Release dữ liệu đã xuất không phụ thuộc vào việc draft tiếp tục được sửa.

Chỉ mẫu thuộc nhóm seal, không lỗi, có solver đồng ý và reviewer khác author/solver được nhận. Sửa câu nói, calls, history, state, spec, catalog hoặc runtime sẽ làm review cũ hết hiệu lực. Build có lỗi không thay con trỏ release trước; 0 mẫu đủ điều kiện cũng không xuất bản rỗng. Pending review bị bỏ qua và ghi trong manifest; các thành viên cặp đối chứng phải cùng đủ điều kiện thì mới xuất bản cả cặp.

Đối chiếu solver chuẩn hóa chuỗi Unicode NFC, chữ hoa/thường và khoảng trắng, đồng thời kiểm hash state cuối. Tên/thứ tự call và tập tham số vẫn phải khớp; thêm optional default như `repeat="once"` không tự được coi là đồng ý. Không sửa nhãn chỉ để ép khớp một câu trả lời của solver.

Packet có thể dài hơn giới hạn hiển thị terminal: solver phải lưu stdout của `start` vào `/tmp` rồi đọc đủ sample/history và schema thành phần nhỏ. Nếu đã finish khi input bị cắt, người điều phối loại **bằng chứng** đó (giữ nguyên trace) và giao phiên mới:

```bash
python3 synthetic-data-llm-generate/tools/review_sample.py exclude \
  --id syn_7001 --actor '<solver-session>' \
  --reason 'Input packet was truncated; a fresh independent full-input pass is required.'
```

`recheck` chỉ dành cho người điều phối tính lại verdict trên trace đã lưu sau khi sửa checker (vd xử lý hoa/thường), không gọi LLM lại hay tạo câu trả lời mới. Verdict cũ được giữ trong `agreement_history`; không được recheck trace bị loại hoặc mẫu đã thay đổi. Không dùng cơ chế này để hợp thức hóa bất đồng ngữ nghĩa.

Các app đang giữ lại cho dev_unseen (fitness/flights/music/tv) không được có trong lời gọi label/history của release sạch. Nếu cần experiment huấn luyện các app này, dùng `--include-holdout-apps`; xuất con trỏ riêng `verified_with_holdout.json`, không thay release sạch. Không gọi kết quả trên dev_unseen đó là khả năng tổng quát sang app chưa học.

## Dữ liệu cũ g1–g6

**174 mẫu trong `train/train.jsonl` là bản ghép legacy chưa được chứng nhận theo quy trình mới.** Giữ nguyên để không làm mất công việc cũ và không ảnh hưởng TTS đang build riêng; không dùng file này làm “release đã duyệt”. Sáu mẫu đã loại vẫn nằm trong `rejected/`.

Các mẫu này chưa có spec/goals và initial_state; checker mới sẽ báo thiếu thay vì cho qua bằng WARN. Không tự thêm review giả hoặc chép final_state làm goal để hợp thức hóa.

Để sửa từng mẫu:

```bash
# Tự viết spec với author/family/intent/goals độc lập.
python3 synthetic-data-llm-generate/tools/prepare_legacy.py \
  --id syn_1016 --spec /tmp/spec.json --output /tmp/proposal.json
# Sửa câu nói/calls/state và goals trong proposal; dùng group g1 của mẫu cũ.
python3 synthetic-data-llm-generate/tools/stage_sample.py \
  --group g1 --input /tmp/proposal.json --replace
```

Nếu history có ghi state thì `prepare_legacy.py` bắt buộc `--initial /tmp/initial.json`; không tự suy ngược state cũ. Người sửa phải dựng lại một state trước history hợp lý rồi reviewer kiểm lại toàn bộ tình huống; không tuyên bố đó là state gốc đã lưu. Mẫu chỉ có đọc/meta hoặc không history có thể lấy state hiện tại làm điểm xuất phát.

Sau khi sửa đủ cả nhóm, seal và đưa qua hai phiên độc lập. Chưa sửa đủ thì nhóm vẫn ở draft, không xuất bản một phần nhóm lỗi.

## Sinh dữ liệu để cải thiện performance

Dùng lỗi lm-sft-016 để chọn tình huống mới: hỏi thiếu thông tin, lọc hóa đơn theo hạn + số tiền, giờ tiếng Việt, chọn đối tượng giữa nhiều kết quả, lệnh kép và tham chiếu history. Không sao chép tình huống dev; giữ final_holdout kín.

Mỗi nhóm thường 30 mẫu, tối thiểu 5 cặp đối chứng: cùng yêu cầu nhưng đổi điều kiện thì quyết định phải đổi. Khoảng 2/3 có history; phủ 1, 2–3, 4–6 cặp và một số dài hơn. Dùng 6–14 app, đối tượng gây nhiễu, đảo vị trí đáp án. Đây là hướng dẫn cho dữ liệu đa dạng, không phải ép mọi nhóm chuyên biệt giống hệt phân bố gốc. Claim/seal kiểm số lượng và cặp; reviewer kiểm độ khó và tính hợp lý.

Nên hoàn thành một đợt 300–500 mẫu đã duyệt, rồi so SFT có/không synthetic trên cùng cấu hình và ngân sách. Khi đó mới quyết định tăng nhóm dữ liệu nào. Pipeline này chưa chứng minh tăng điểm LifeMate.

## Công cụ và kiểm tra local

| Công cụ | Công dụng |
|---|---|
| `tools/examples.py` | Ví dụ quy ước nhãn từ train_fit, giấu dev/holdout |
| `tools/stage_sample.py` | Tạo kết quả và state bằng simulator, lưu draft |
| `tools/claim_group.py` | Claim/finish nhóm, tránh trùng dải ID |
| `tools/review_sample.py` | Packet mù, gateway tool, finish và approve |
| `tools/check_sample.py` | Kiểm draft, build và lấy đường dẫn release |
| `tools/prepare_legacy.py` | Xuất mẫu cũ thành proposal để sửa có chủ đích |
| `tools/bootstrap.py` | Chuẩn bị catalog/runtime BTC |

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s synthetic-data-llm-generate/tools -p 'test_*.py' -v
```

Tests dùng simulator thật, fixture/artefact ở `/tmp`, không sinh dữ liệu train hoặc gọi API. `SYN_OUT` đổi thư mục output cho sandbox/test; `LM_TRAIN_DIR` đổi nguồn train local cho examples/dedup. Không thêm dependency nặng. Không sửa dataset, không gọi Kaggle, không commit tự động.
