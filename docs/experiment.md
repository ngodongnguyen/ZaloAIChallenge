# Experiment log

Nhật ký thí nghiệm theo thứ tự thời gian: mục tiêu, cấu hình, lệnh chạy, kết luận. Số liệu tổng hợp nằm ở `result.md`.
Quy ước: mỗi thí nghiệm mới thêm một mục `E<n>` ở cuối, ghi **trước khi chạy** (mục tiêu, giả thuyết) và cập nhật **sau khi chạy** (kết quả, kết luận).
Mục "Trạng thái": `done` / `running` / `planned` / `aborted`.

---

## E1 - Baseline zero-shot
- Trạng thái: done
- Mục tiêu: lấy điểm gốc, không train.
- Cấu hình: PhoWhisper-large (ASR) + Qwen2.5-3B-Instruct zero-shot, vòng lặp ReAct trong `src/agent.py`.
- Kết quả: offline 40 mẫu 0.196; điểm web lần đầu 7/100.
- Kết luận: zero-shot không đủ, cần SFT.

## E2 - SFT Qwen2.5-1.5B (Kaggle T4)
- Trạng thái: done
- Mục tiêu: học định dạng gọi tool từ các trajectory có nhãn.
- Cấu hình: LoRA r=32/alpha=64, fp16, 2 epoch, `--max-len 6144`, loss chỉ trên lượt assistant, holdout app-level (`train/split.py`: 6 app + 100 mẫu seen).
- Chạy: `kaggle/train_kernel.py`. Loss chuẩn hoá theo `num_items_in_batch` (sửa lỗi loss bị nhân 16).
- Kết quả: holdout (transcript gold) seen 0.5504 / unseen 0.3335; điểm web 17.85.
- Kết luận: chênh lớn giữa holdout và web, nghi ngờ đầu vào ASR.
- Thất bại trước đó: 3B QLoRA OOM / ~250 s/bước, nên chuyển sang 1.5B.

## E3 - Chẩn đoán dự đoán rỗng
- Trạng thái: done
- Mục tiêu: hiểu vì sao 137/634 (22%) dự đoán là `[]` (kỳ vọng ~7%).
- Cấu hình: kernel `kaggle_diag/diag_kernel.py` ghi transcript, app cài, output LLM đầu tiên cho 137 mẫu rỗng + 25 mẫu có gọi.
- Kết quả: 116/137 là model trả `{"done": true}` ngay lượt đầu; transcript ASR bị nhiễu nặng (sai chính tả, số viết bằng chữ), trong khi `audio_text` gold sạch và dùng chữ số. Độ dài prompt không phải nguyên nhân.
- Kết luận: ASR là nút thắt chính; LLM bỏ cuộc khi đầu vào vô nghĩa.

## E4 - SFT Qwen3-4B-Instruct-2507 (RTX 4060 Ti 16GB)
- Trạng thái: done
- Mục tiêu: LLM lớn hơn (tổng tham số ≈5.6B < 6B).
- Cấu hình: `--epochs 2 --max-len 6144`, 1241 mẫu train, 156 bước (batch hiệu dụng 16).
  - QLoRA 4-bit: ước tính >6 giờ, bỏ.
  - LoRA bf16 (không `--qlora`): VRAM ~14GB, **3h46**.
- Chạy: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train/sft.py --data train/sft.jsonl --base Qwen/Qwen3-4B-Instruct-2507 --out models/lifemate-llm --epochs 2 --max-len 6144 --holdout cache/holdout_all.txt`
- Kết quả: loss 0.45 -> ~0.02. Merge lần đầu hỏng vì đầy đĩa; merge lại từ adapter bằng PeftModel.
- Eval với ASR gốc: holdout seen mean_score 0.4057 (tool-call 0.5503, hash 54/100, out-of-scope 7/10); điểm web 26.61.

## E5 - So sánh ASR theo WER
- Trạng thái: done
- Mục tiêu: chọn ASR tốt nhất trước khi fine-tune.
- Cấu hình: `src/asr_eval.py`, 150 clip đầu của train, chuẩn hoá chữ thường + bỏ dấu câu, không đổi số thành chữ.
- Chạy: `python src/asr_eval.py --root data --limit 150 --models models/asr openai/whisper-large-v3-turbo`
- Kết quả: PhoWhisper-large WER 0.275 / CER 0.205; Whisper-large-v3-turbo WER 0.395 / CER 0.315.
- Kết luận: giữ PhoWhisper, đổi sang turbo không lợi.

## E6 - Fine-tune PhoWhisper bằng LoRA
- Trạng thái: done
- Mục tiêu: học kiểu viết gold (chữ số, tên riêng, từ vựng app) và giọng ba miền.
- Cấu hình: `train/asr_finetune.py`; 4134 clip (câu cuối + các lượt user trong history, bỏ id holdout), LoRA r=32 trên q/k/v/out_proj/fc1/fc2, lr 2e-4, batch 8 x accum 2, 3 epoch = 774 bước, bf16, cosine + 5% warmup.
- Chạy: `python train/asr_finetune.py --root data --base models/asr --out models/asr_ft --holdout cache/holdout_all.txt`
- Smoke test (64 clip, 1 epoch, 4 bước): WER 0.222 -> 0.537 (tệ đi). Nghi do lr lớn trong run quá ngắn; không phải bằng chứng lỗi code, run thật không bị.
- Kết quả: loss 1.82 (bước 20) -> ~0.34 (epoch 0) -> ~0.11 (epoch 1) -> ~0.04 (epoch 2, bắt đầu nhớ dữ liệu). WER holdout 150 clip: **0.235 -> 0.098**.
- Kết luận: cải thiện lớn, model lưu tại `models/asr_ft`.

## E7 - 4B + ASR đã fine-tune
- Trạng thái: done
- Mục tiêu: đo hiệu quả đầu-cuối của E6.
- Chạy: `ASR_MODEL=models/asr_ft LLM_MODEL=models/lifemate-llm python src/infer.py ...` trên holdout seen (100 mẫu) và public_test (634).
- Kết quả: holdout seen mean_score **0.6715** (tool-call 0.7474, hash 79/100, out-of-scope 7/10); điểm web **46.69**.
- Ghi chú: log có cảnh báo CUDA OOM lúc nạp model nhưng đủ 100/100 dòng; `nvidia-smi` cho thấy GPU rảnh.
- Kết luận: ASR đúng là nút thắt. Còn khoảng cách ~20 điểm giữa holdout seen (0.67) và web (0.47), giả thuyết: 11 app chỉ có ở test.
- Phân tích 10 mẫu tệ nhất trên holdout seen (`src/analyze.py`, chưa có phần đếm theo loại lỗi), các kiểu lỗi quan sát được:
  - Chọn sai một trong nhiều lựa chọn: `flight_id` (fl_04/fl_02 thay vì fl_05), `from_account_id` (acc_01 thay vì acc_02), `source` (bank thay vì ewallet). Có thể do model suy luận tiêu chí ("rẻ nhất", "sớm nhất") hoặc do ASR nghe sai.
  - Số lượng/mặt hàng sai (`quantity` 4 + thêm món thay vì 3), gọi thừa một lần `bill_payment.pay`.
  - Tham số dạng chuỗi bị rút gọn hoặc mở rộng (`dropoff`, `pickup`); các tham số này khớp chính xác, không fuzzy.
  - `title` sai hẳn ("Khoa học" so với "Sơ cứu cộng đồng cơ bản"), nghi do ASR hoặc đọc sai history.
  - Gold rỗng nhưng model gọi `weather.get_forecast` (spurious); `sys.ask_missing_info` sai `tool_name` (flights.book thay vì flights.search).
  - So transcript `asr_ft` với `audio_text` của 8 id sai (0212, 0529, 0732, 0514, 0361, 0333, 0080, 0378):
    - Giống hệt gold nhưng vẫn sai: 0361, 0333 (lỗi LLM: chọn tài khoản / nguồn tiền).
    - Gần đúng, lỗi vẫn do LLM: 0514 (địa chỉ rút gọn), 0529 (chọn sai chuyến trong history).
    - Lỗi do ASR: 0080 ("khoan trả, để đó đã" -> "vay trả, để nó rẻ" làm model trả thừa hoá đơn), 0732 ("khoá học" -> "Khoa học").
    - Hỗn hợp: 0212 (ASR mất "ai" và thêm "rang"; model cũng đếm sai số người).
    - 0378: gold rỗng dù câu có điều kiện mưa, chưa hiểu quy ước nhãn.
  - Kết luận: ASR không còn là nguyên nhân chính; lỗi LLM tập trung ở suy luận trên kết quả tool (so số dư, chọn theo tiêu chí, đếm, sao chép địa chỉ đầy đủ).
  - Ý tưởng cần thử (planned, chưa làm): cho model bước "thought" ngắn trước khi gọi tool (phải tự sinh rationale); train lại trên toàn bộ dữ liệu (E9).

## E8 - Phân tích lỗi trên holdout (app lạ)
- Trạng thái: done (số liệu ở `result.md`)
- Kết quả tóm tắt (187 mẫu unseen, `asr_ft` + 4B): mean_score 0.5002, hash 112/187. Loại lỗi: ok 35%, wrong_tool 29%, arg_mismatch 22%, extra_calls 10%, empty_pred 4%. Trên seen (100 mẫu): ok 58%, arg_mismatch 22%, wrong_tool 12%.
- Cụm lỗi 1 - rẽ nhánh theo so sánh số trên kết quả tool (gold_price): "giá vàng bán ra trên X triệu thì nhắc đi bán, không thì chuyển 2 triệu". Tool trả `sell` dạng 123600000 VND, ngưỡng nói bằng "124 triệu"; model chọn nhánh sai theo cả hai chiều (train_0224: 131.8M > 131M nhưng model chuyển tiền; train_0272: 123.6M < 124M nhưng model nhắc nhở), tức đoán chứ không so sánh. Ít nhất 7/20 mẫu tệ nhất thuộc cụm này.
- Cụm lỗi 2 - app tự động hoá nhà (`home_automation`, lạ hoàn toàn): model bỏ `create_rule` và dùng `smart_home.light_set` / `reminders.create`, hoặc bỏ tham số `title`. Trong catalog `title` là **optional** nhưng gold truyền ở 16/20 lời gọi; `gold_price.get_price.type` optional nhưng gold truyền ở 33/33. Quy tắc 5 trong `src/prompt.py` ("không truyền tham số tùy chọn mà người dùng không nói đến") mâu thuẫn với hành vi của gold. Toàn bộ dữ liệu: gold truyền 1070/3229 (33%) lần có thể truyền tham số optional.
- Cụm lỗi 3 - sao chép giá trị đầy đủ từ kết quả tool/history (địa chỉ, tên sự kiện), chọn đúng một mục trong nhiều lựa chọn (chuyến bay, tài khoản, nguồn tiền).
- Điểm theo app unseen thấp nhất: package_tracking 0.24 (n=25), smart_home 0.26 (21), home_automation 0.29 (37), banking 0.38 (17); 6 app này bị loại hoàn toàn khỏi train nên điểm thấp là dự kiến.
- Trạng thái cũ: planned
- Mục tiêu: biết phần điểm bị mất nằm ở đâu (wrong_tool / arg_mismatch / empty_pred / spurious / hash_only) và app nào kém.
- Chạy:
  - `ASR_MODEL=models/asr_ft LLM_MODEL=models/lifemate-llm python src/infer.py --root data --split train --ids cache/holdout_unseen.txt --out out/pred_unseen_ft.jsonl`
  - `python src/eval.py --pred out/pred_unseen_ft.jsonl --root data --split train`
  - `python src/analyze.py --pred out/pred_unseen_ft.jsonl --root data --split train --show 20`
- Ghi chú từ bản 1.5B (không phải bản 4B): gold hay truyền `type: "sjc"` cho `gold_price.get_price` còn model bỏ tham số. Nghi quy tắc 5 trong `src/prompt.py` ("không truyền tham số tùy chọn") làm model bỏ cả tham số mà gold luôn truyền. Cần xác nhận bằng dữ liệu bản 4B.

## E9 - Huấn luyện bản cuối trên toàn bộ train
- Trạng thái: planned -> chuẩn bị chạy (prompt v1, sau khi E11 không hiệu quả)
- Mục tiêu: thêm 287 mẫu holdout (gồm 6 app chưa dùng) cho cả ASR và LLM.
- Cấu hình: giữ nguyên E4 (Qwen3-4B, LoRA bf16, 2 epoch, `--max-len 6144`) và E6 (PhoWhisper LoRA, 3 epoch), chỉ bỏ `--holdout`. Đầu ra `models/lifemate-llm-final` và `models/asr_final`.
- Không còn holdout nên không đo được lợi ích; chỉ kiểm tra "không hỏng" (chạy trên seen, điểm sẽ cao hơn vì đã học) và đo bằng điểm web public_test.
- Chi phí ước tính: LLM ~4.7 giờ (194 bước), ASR ~2.3 giờ; tuần tự trên một GPU.
- Tiến độ: LLM cuối XONG (`models/lifemate-llm-final`): 1525/1545 mẫu (20 mẫu vượt max-len bị loại), 192 bước, 4h37, loss 0.47 -> ~0.03, train_loss 0.0719. ASR cuối XONG (`models/asr_final`): 5036 clip, 3 epoch, 943 bước, 151 phút (9.7 s/bước), loss 1.92 -> ~0.04; không có clip holdout nên không đo WER. Kiểm tra không hỏng: seen 100 (đã học) 0.8966, hash 92/100, OOS 8/10, tối đa 13.2 s/mẫu. Điểm web public_test: **50.21** (E7: 46.69, +3.52). Ghi chú cũ: chạy public_test bằng `asr_final` + `lifemate-llm-final` và nộp để lấy điểm web.
- Lưu ý đĩa: mỗi model ~7.6GB (LLM) / ~3GB (ASR); cần giải phóng trước khi chạy.

## E10 - Augmentation để khái quát sang app mới
- Trạng thái: done - KHÔNG hiệu quả (unseen 0.4757 so với E8 0.5002; hash 105/187 so với 112/187). Không dùng cho bản cuối.
- Kết quả unseen 187 (`asr_ft`): mean_score 0.4757, tool-call 0.6340, hash 105/187; ok 34%, wrong_tool 32%, arg_mismatch 25%, extra_calls 7%, empty_pred 2% (E8: ok 35%, wrong_tool 29%, arg_mismatch 22%, extra_calls 10%). Chênh -0.025 nằm trong nhiễu (~±0.04), tức là không có lợi ích đo được; wrong_tool nhích lên.
- Phân tích: 11/20 mẫu tệ nhất là cùng một mẫu câu của `gold_price` ("nếu giá vàng bán ra trên X triệu thì nhắc ..., chưa tới thì chuyển 2 triệu"): model gọi đúng tool đọc nhưng chọn sai nhánh vì so sánh sai số lớn (vd. 131.8 triệu so với ngưỡng 131). Đổi tên không động tới lỗi lập luận này. Trong train chỉ có 33 mẫu gold_price (20 mẫu có điều kiện), 1.3% dữ liệu. Các cụm lỗi còn lại (hỏi lại thừa, thiếu `title`) giống E8.
- Kết luận: tên app/tool không phải nút thắt chính; lỗi nằm ở lập luận trên kết quả tool và chọn tool cho app lạ. Giữ bản E9 (50.21).
- Kết quả train: 1239 mẫu (E4: 1241), 156 bước, 3h48, loss 0.40 -> ~0.03, train_loss 0.0743 (E4: 0.0775, E11: 0.0735).
- Mục tiêu: nâng điểm app chưa thấy (E8 unseen 0.5002 so với seen 0.6715). Public test có 11 app mới, nên đây là chỗ mất điểm lớn nhất.
- Ý tưởng: khi tạo dữ liệu SFT, đổi tên app và tool trong cùng một ánh xạ cho catalog, ngữ cảnh, history, gold call, kết quả; model buộc phải đọc mô tả thay vì học thuộc tên. Lúc suy luận vẫn dùng tên thật.
- Cài đặt: `train/augment.py` + cờ `--aug-frac`, `--seed` trong `train/build_sft.py`.
  - Tên app: 40% đặt tên mờ (`svc_xxx`), 40% thêm tiền tố/hậu tố (`my_ride`, `ewallet_x`), 20% giữ nguyên.
  - Tên tool: từng từ trong phần sau dấu chấm đổi sang từ đồng nghĩa với xác suất 0.7 (`get_status` -> `read_state`, `list` -> `all`, `search` -> `find`).
  - Xáo thứ tự tool trong catalog của mẫu được đổi tên.
  - Các chỗ đổi: tên đầy đủ `app.tool` ở mọi nơi (kể cả mô tả "lấy từ ..."), `x-ref` (chỉ phần app), `installed_apps`, `final_state`, trường `id`/`app` của catalog. Không đổi từ đứng riêng trong kết quả tool (vd. `"type": "phone"`) hay văn bản tự do; lỗi này đã gặp và sửa trong lúc làm.
  - Chưa đổi: tên tham số (ví dụ `device_id`) vì liên quan `required` và `x-one-of`; để vòng sau nếu E10 có tác dụng.
- Kiểm tra: tạo 1545 trajectory, 792 mẫu đổi tên (`--aug-frac 0.5`), mọi tool trong lời gọi đều nằm trong danh sách tool của prompt, không còn `svc_` đứng riêng trong kết quả.
- Quy mô giữ nguyên E4 (1545 trajectory, nửa số đó đổi tên) nên thời gian train ~3h50, so sánh công bằng với E7/E8.
- Chạy (trên server): `python train/build_sft.py --root data --out train/sft_aug.jsonl --aug-frac 0.5` rồi `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train/sft.py --data train/sft_aug.jsonl --base Qwen/Qwen3-4B-Instruct-2507 --out models/lifemate-llm-aug --epochs 2 --max-len 6144 --holdout cache/holdout_all.txt`
- Đánh giá: unseen 187 và seen 100 với `asr_ft` (so E8 0.5002 và E7 0.6715); tiêu chí giữ: unseen tăng rõ hơn nhiễu (~±0.04), seen không tụt.
- Rủi ro: nếu tên mờ làm mất tín hiệu ngữ nghĩa hữu ích thì seen có thể giảm; nếu không có lợi thì giữ bản E9 (50.21).

## E11 - Prompt v2 (đo tác động của chỉnh sửa prompt trên cùng tập holdout)
- Trạng thái: done - KHÔNG hiệu quả, đã hoàn nguyên `src/prompt.py` về v1
- Kết quả đánh giá: seen 0.6308 (E7: 0.6715), unseen 0.5027 (E8: 0.5002), hash 111/187 (E8: 112/187); gold_price 0.58 (0.55), home_automation 0.31 (0.29). Cụm lỗi rẽ nhánh theo số và thiếu `title` vẫn còn; v2 còn hỏi lại thừa (0486, 0520, 0543, 0620). Gộp 287 mẫu: 0.547 so với 0.560.
- Kết luận: hai giả thuyết (a) rule 5 và (b) số lớn khó so sánh không được dữ liệu ủng hộ (chênh trong sai số). Quay về prompt v1 vì đơn giản hơn và không có bằng chứng v2 tốt hơn; chạy bản cuối trên toàn bộ dữ liệu với prompt v1. Lỗi rẽ nhánh theo số cần cách khác (ví dụ augmentation hoặc bước suy luận), chưa thử.
- Kết quả train: 1235 mẫu (6 mẫu bị loại do vượt max-len so với 1241 của E4), 156 bước, 3h51, loss 0.39 -> ~0.03, train_loss trung bình 0.0735 (E4: 0.0775).
- Mục tiêu: kiểm tra hai giả thuyết rút ra từ E8: (a) rule 5 cũ làm model bỏ tham số optional mà gold hay truyền; (b) model không so sánh được số lớn trên kết quả tool (gold_price, số dư, hoá đơn).
- Thay đổi (`src/prompt.py`):
  - Rule 5: "truyền tham số tùy chọn khi người dùng nói đến hoặc suy ra được" (thay cho "không truyền khi người dùng không nói đến").
  - Rule 6 mới: yêu cầu có điều kiện thì gọi tool đọc, so sánh số trong KẾT QUẢ với ngưỡng, chỉ thực hiện đúng MỘT nhánh.
  - `humanize_numbers` trong `truncate()`: số nguyên >= 100000 được gắn bản rút gọn, ví dụ `123600000 (≈123.6 triệu)`, `847000 (≈847 nghìn)`. Chỉ áp dụng cho trường tiền (đã kiểm tra: amount, balance, price, total, buy, sell, value, cost); không đụng tới ID, số điện thoại, thời gian.
- Dữ liệu: `train/build_sft.py` -> `train/sft_v2.jsonl` (1545 trajectory, dài hơn ~1.6% ký tự).
- Giữ nguyên mọi thứ khác so với E4 (Qwen3-4B, LoRA bf16, 2 epoch, `--max-len 6144`, cùng `cache/holdout_all.txt`) để so sánh công bằng. Đầu ra `models/lifemate-llm-v2`.
- So sánh bằng: seen 100 (E7: 0.6715) và unseen 187 (E8: 0.5002), cùng `asr_ft`.
- Lưu ý: model v1 (`models/lifemate-llm`) chỉ chạy đúng với prompt v1; sau khi pull code mới thì không dùng v1 được nữa trừ khi quay lại commit cũ.
- Rủi ro: hai thay đổi nằm trong cùng một lần train nên nếu điểm tăng thì không tách được phần đóng góp của từng thay đổi.

## E12 - Chẩn đoán: LLM với transcript đúng (gold) so với ASR thật
- Trạng thái: chờ chạy trên server
- Mục tiêu: tách phần điểm mất do ASR khỏi phần mất do LLM, để chọn hướng cải tiến cho mục tiêu điểm cao. Mục tiêu của người dùng là >90 điểm web (hiện 50.21); chưa có bằng chứng đạt được, cần dữ liệu trước khi chọn hướng train.
- Cách làm: giống E7/E8 nhưng thay ASR bằng transcript gold qua `--asr-cache cache/asr_gold.jsonl` (id, text), LLM `models/lifemate-llm` (không học holdout), chạy trên unseen 187 và seen 100.
- Mốc so sánh (ASR thật, cùng LLM): seen 0.6715 (E7), unseen 0.5002 (E8).
- Cách đọc: gold gần 0.9 -> ưu tiên ASR và độ chịu nhiễu ASR của LLM; gold còn 0.6–0.7 -> ưu tiên LLM (bước thought, lập luận điều kiện, app lạ).
- Kết quả seen 100 (gold transcript, `lifemate-llm`): mean_score 0.7654, tool-call 0.8257, hash 83/100, OOS 8/10. So với ASR thật (E7): 0.6715, hash 79/100 -> ASR làm mất ~0.094 điểm trên seen; trần của LLM hiện tại với ASR hoàn hảo chỉ ~0.77, nên phần lớn khoảng cách tới 90 là lỗi LLM, không phải ASR.
- Phân loại lỗi với transcript gold (loại hẳn nhiễu ASR): seen ok 70%, arg_mismatch 16%, wrong_tool 9%, extra 3%, spurious 2%; unseen ok 48%, arg_mismatch 21%, wrong_tool 20%, extra_calls 10%, empty 1% (với ASR thật E8: ok 35%). Tức là ASR làm mất ~13 điểm phần trăm "ok" ở unseen, còn 52% mẫu vẫn sai dù ASR đúng.
- Các cụm lỗi LLM thuần (từ 25 mẫu tệ nhất mỗi tập):
  1. Chọn sai ID trong danh sách kết quả tool (chuyến bay fl_04/fl_05, khoá lock_06/07, tài khoản acc_01/02, cảnh báo, thiết bị): cần so sánh/lọc theo điều kiện trong lời nói.
  2. Rẽ nhánh theo số (gold_price: 6/20 mẫu tệ nhất ở unseen).
  3. Hỏi lại thừa (`ask_missing_info`) khi thông tin có trong kết quả tool/ngữ cảnh (scene_id, pickup, subscription_id, bike_id).
  4. Dừng sớm khi yêu cầu có nhiều việc (chỉ làm việc đầu rồi done: 0761, 0781, 0827).
  5. Tính giờ/ngày sai: "khi hết bài hát" (21:04 so với 04:40), ngày tương đối (09-05 so với 09-09), thiếu `title` của `home_automation.create_rule`.
  6. Nội dung văn bản tự do (tin nhắn, tiêu đề) lệch quá ngưỡng token-F1 0.6.
- Kết luận E12: LLM là nút thắt chính. Trần với ASR hoàn hảo khoảng 0.77 (seen); cần cải thiện lập luận chọn ID/điều kiện/tính toán chứ không phải ASR.

## E13 - Thought: model đọc/ghi một dòng suy nghĩ trước mỗi hành động (teacher mở chạy local)
- Trạng thái: code xong, chờ chạy trên server
- Mục tiêu: sửa các lỗi lập luận ở E12 (chọn ID theo điều kiện, so sánh số, giờ/ngày, quên việc, dừng sớm, hỏi lại thừa). Student (Qwen3-4B) viết `{"thought": "...", "name": ..., "arguments": ...}`; thought nêu điều kiện/số liệu/ID rồi mới hành động; ở bước `done` thought tóm tắt các việc đã xong.
- Luật: dùng model mở chạy local trên server (không gửi dữ liệu cuộc thi cho bên thứ ba, Regulations 4.3); annotation thêm vào train hợp lệ (4.2); teacher không dùng lúc suy luận. Đã bỏ phương án API ngoài.
- Teacher: `Qwen/Qwen3-14B` 4-bit (nf4) hoặc `Qwen/Qwen3-8B` nếu thiếu đĩa; teacher thấy lời người dùng, lịch sử, kết quả tool và hành động gold tiếp theo, viết 1-2 câu (<=40 từ) lập luận (rationalization kiểu STaR). `train/gen_thoughts.py`, 5086 bước (1545 mẫu), prompt teacher ~1000 token (bỏ danh sách tool), batch 4; ước tính 2-3 giờ (chưa đo).
- Thay đổi đồng bộ train/suy luận: `call_msg(name, args, thought)` và `done_msg(thought)` trong `src/prompt.py`; `src/agent.py` giữ thought của model trong lịch sử; `MAX_NEW_TOKENS` 256 -> 320; `train/build_sft.py --thoughts`. Không dùng `--thoughts` thì dữ liệu giống hệt trước.
- Rủi ro: (1) thought do teacher viết khi đã biết đáp án có thể nêu lý do mà student không tự suy ra được ở lúc test; (2) suy luận dài hơn khoảng 40 token mỗi bước (~+3-5 s/mẫu); (3) nếu thought sai thì kéo hành động sai theo.
- Đánh giá: train với `--holdout cache/holdout_all.txt`, đo unseen 187 và seen 100 bằng `asr_ft` (E8 0.5002, E7 0.6715) và với transcript gold (E12) để thấy riêng phần lập luận. Tiêu chí giữ: unseen tăng rõ hơn nhiễu (~+0.04).
- Kết quả thử 14B (21 bước, 0.9 phút; ~2 s/bước nên toàn bộ 5086 bước ~3 giờ): các bước sau khi có kết quả tool đúng và cụ thể (`train_0123`: "giá bán 126,7 triệu, chưa đạt 127 triệu"; `train_0333`: "ví 363.756 đ, hoá đơn 147.000 đ, ví đủ"; `train_0786`: bài hát 4 phút kết thúc 21:04). Lỗi thấy: (1) `train_0019` (ngoài phạm vi) lập luận sai vì teacher không biết app đã cài -> đã thêm danh sách app vào prompt teacher, sau khi sửa ra "Không có ứng dụng nào xử lý được yêu cầu đặt phòng khách sạn"; (2) `train_0529`, `train_0786` bước 0: teacher nói "tool không có fl_05/bài hát" vì không thấy tool list/kết quả -> thêm lệnh cấm nói "không tồn tại" và bộ lọc `BAD_THOUGHT` trong `build_sft.py`.
- Phát hiện phụ (không cần thought): `render_history` cắt kết quả tool trong history ở 500 ký tự. Trong train có 957 mẫu có history kèm kết quả tool, 570/3661 kết quả vượt 500 ký tự, và 48 mẫu (~3%) mà ID gold (vd. `fl_05` ở `train_0529`, `c_10`, `sub_05`, `course_04`) chỉ nằm trong phần bị cắt, nên model không thể chọn đúng mà chỉ đoán (khớp lỗi chọn chuyến bay ở E12). Đã đổi giới hạn thành `HISTORY_RESULT_CHARS = 1500` (chỉ 12 kết quả vượt 1500). Phải train lại vì prompt đổi; thought phải sinh lại với history đầy đủ.
- Chạy gộp: `bash scripts/run_e13.sh` (thought -> dữ liệu SFT -> train với holdout -> đánh giá unseen/seen bằng `asr_ft` -> submission public_test bằng `asr_final`); `FINAL=1 bash scripts/run_e13.sh` train trên toàn bộ dữ liệu và xuất submission. Mỗi giai đoạn bỏ qua nếu đã có đầu ra (chạy lại thì tiếp tục). Đầu ra: `models/lifemate-llm-thought[-final]`, `out/submission_thought[-final].jsonl`. Thời gian ước tính ~3 giờ thought + ~5 giờ train + ~1.5 giờ suy luận.
- Tốc độ suy luận với thought (RTX 4060 Ti): mẫu chậm nhất lên 20–25 s (trước đó 13 s), nhiều bước hơn x ~40 token thought mỗi bước. A30 có băng thông bộ nhớ ~3x nên ước tính (chưa đo) ~9-12 s; giới hạn thật là 30 s gồm cả ASR. `TIME_BUDGET_S` (mặc định 26) nay đọc từ biến môi trường; `run_e13.sh` đặt 120 để mẫu chậm không bị cắt khi đo trên máy này. Docker giữ mặc định 26.


### E13 web result (2026-10-30)
- Thought model (teacher Qwen3-14B, holdout-excluded training, asr_final, TIME_BUDGET_S=26 on 4060 Ti): web 39.07, below v1 46.69 and E9 50.21. Thought is not adopted; keep E9 (submission_final.jsonl) as the best.
- Suspects: samples cut by the 26 s budget on the slow GPU, student copying wrong/rationalized thoughts, longer outputs. Holdout numbers still to be recorded.


### E13 holdout + E14 plan (2026-10-30)
- E13 holdout: unseen 0.4159 (hash 96/187), seen 0.5382 (hash 67/100): worse than E8 0.5002 / E7 0.6715. Thought rejected.
- gold_price (n=33, 0.47): threshold branch ("sell > 127 trieu ? reminder : transfer"); model does both branches / wrong branch / stops. Raw 9-digit numbers vs "127 trieu" in speech.
- E14: result_msg appends money_hint ("quy doi: sell=126.6 trieu") in train and inference (src/prompt.py), no thoughts, HISTORY_RESULT_CHARS=1500. Needs retrain; compare unseen with E8 0.5002.


### E15 plan (2026-10-30)
- smart_home (126 train samples, 0.21 unseen) needs filtering a long device list by floor/state. RESULT_DIGEST=1 appends a per-floor one-line summary of devices/locks to tool results (E14 money hint stays on). Run on the second machine, compare with E14 on the same holdout.


### E14 result (2026-10-30)
- E14 (money hint, no thoughts, asr_ft): unseen 0.4844 (hash 111/187, tool-call only 0.6369), seen 0.6798 (hash 84/100). E8 unseen 0.5002 (112/187), E7 seen 0.6715. No gain; gold_price 0.50 (was 0.47), smart_home 0.30. Branch errors on gold_price persist, so the cause is probably ASR on the threshold/direction, not number comparison. Log also shows a CUDA OOM warning (free 131MB) while running ASR + LLM together.


### E16 plan (2026-10-30)
- Hypothesis: prompt/format tweaks (E10, E13, E14) all land near unseen 0.50; the limit is wording diversity of ~1.5k training utterances. E16 rewrites each train utterance (Qwen3-14B 4-bit, local, 2 styles per sample, filtered to keep every number, time/condition word and name) and trains on gold + rewrites with the same labels; MONEY_HINT=0, RESULT_DIGEST=0, HISTORY_RESULT_CHARS=1500, 1 epoch over 3x data. Compare unseen with E8 0.5002.


### E17 plan (2026-10-31)
- The LLM never saw real ASR text in training (E7-E16 used gold transcripts only) but gets ASR text at test time. E17 transcribes the train audio with the BASE PhoWhisper (asr_ft is overfit to train audio), and build_sft --asr-cache adds each ASR transcript as an extra trajectory with the same label. 1 epoch over ~2x data; USE_PARA=1 adds the E16 rewrites. Compare unseen with E8 0.5002 and E16.


### E16 result and E18 plan (2026-10-31)
- E16 (rewrites x2, 1 epoch, gold transcripts, asr_ft eval): unseen 0.5136 (hash 114/187, tool-call only 0.6559), seen 0.7270 (hash 83/100). vs E8 unseen 0.5002 / E7 seen 0.6715: small gain, inside the +-0.04 noise on unseen; seen +0.055. Weighted over 287: 0.588 vs 0.560.
- gold_price (n=33) still 0.52: all template samples are "sell > X ? reminder : transfer" with the price within 1 million of X; the model picks the wrong branch both ways.
- E18: src/condition.py compares the spoken threshold with the returned price in code and appends "(so sanh: ... -> dieu kien DUNG/SAI)" to the tool result (COND_HINT=1, train and inference). Offline check on train: 18/18 template samples give the gold branch, no hint on the other gold_price samples.


### E17+E18 result (2026-10-31)
- run_e17.sh COND=1 USE_PARA=1 (base-ASR transcripts + rewrites + threshold hint): web 49.72 (E16 50.55); holdout unseen 0.5035 (hash 110/187), seen 0.6671 (75/100); gold_price 0.66 (E16 0.52); home_automation 0.23, smart_home 0.26 (both lower than E16).
- BUG: src/condition.py words_to_int gave 104/104/103 for "mot tram hai muoi bon / ba mot / hai mot" (tens lost, mot handling), so the hint was wrong on ASR text (ASR writes numbers in words) in training and at inference; the offline 18/18 check only used gold text (digits). Fixed and unit-tested on 12 spoken forms. The run above is therefore not a fair test of E18; rerun needed (SUFFIX=-v2).
- Also unfixable by rule: ASR mishearing digits (train_0224: 131 heard as 151).


### Model switch research (2026-10-31)
- No verified Vietnamese benchmark ranking of ~4B models found (VMLU/SEA-HELM snapshots list only large or fine-tuned entries). Candidate: Qwen/Qwen3.5-4B (Feb-Mar 2026, Apache-2.0, 201 languages, MMMLU 76.1, hybrid Gated DeltaNet + attention, thinking on by default, vision encoder included: HF shows 5B params, language model 4B). Risks: param limit (strip vision, count LM only), needs latest transformers and fla/causal-conv1d kernels, LoRA target modules differ in DeltaNet layers, speed on the A30 unmeasured. Added enable_thinking=False to llm.py and sft.py (no effect on Qwen3-2507).

- Qwen3.5-4B probe: AutoModelForCausalLM loads Qwen3_5ForCausalLM text-only, 4.206B params (vision 0), total with PhoWhisper ~5.76B < 6B. DeltaNet linear names: in_proj_qkv, in_proj_z, out_proj (added to LoRA target_modules).

- Qwen3.5-4B zero-shot probe (60 unseen, gold transcripts, same prompt): 0.3528 (hash 34/60) vs Qwen3-4B-Instruct-2507 0.2471 (25/60), +0.106. Per-sample time similar (1-23 s on the 4060 Ti) even without fla/causal_conv1d kernels (fallback warning). Next: install fla, train with E16 data: BASE=Qwen/Qwen3.5-4B SUFFIX=-q35 bash scripts/run_e16.sh.

- Docker prep (2026-10-31): Dockerfile now pins MONEY_HINT=0 RESULT_DIGEST=0 COND_HINT=0 (config default MONEY_HINT=1 would mismatch the E16/Qwen3.5 training prompt). Open items: pin transformers/fla/torch to the versions used on the server (Qwen3.5 needs a recent transformers + flash-linear-attention; base image is torch 2.4.1), Triton first-call compile time on the A30, final model paths.


### Qwen3.5-4B first run failed (2026-10-31)
- run_e16.sh BASE=Qwen/Qwen3.5-4B SUFFIX=-q35: holdout unseen 0.0071 (hash 23/187), seen 0.1633; 87 empty + 100 wrong_tool; empties take ~15 s = 320 tokens of non-JSON text.
- Cause: Qwen3.5 chat template emits "<think>\n\n</think>\n\n" only for assistant turns after the last user message (our tool results are user messages), and the generation prompt with enable_thinking=False always ends with it. Training text for intermediate steps had no think block, inference prompts had one, and label offsets from template prefixes were shifted ~6 tokens. Qwen3-2507 has no such logic, so earlier runs were unaffected.
- Fix: src/chat.py renders every assistant turn as header + empty think block + answer for Qwen3.5 (detected from the template), used by train/sft.py and src/llm.py. Checked with a fake tokenizer: supervised span = answers + im_end, inference prompt is a prefix of the training text. Retrain needed (SUFFIX=-q35b).


### Qwen3.5-4B result (2026-11-01)
- run_e16.sh BASE=Qwen/Qwen3.5-4B SUFFIX=-q35b (fixed prompt rendering, E16 data, 1 epoch, holdout-excluded model, asr_final on public_test): web 56 (E16 on Qwen3-4B 50.55, E9 50.21). First clear gain beyond the +-2 noise. Holdout numbers still to be recorded.
- Next: FINAL=1 BASE=Qwen/Qwen3.5-4B SUFFIX=-q35b (all data) on machine 1; machine 2: same recipe + COND_HINT=1 (fixed parser) as a clean test.


### Qwen3.5 final result (2026-11-01)
- Web: final q3.5 (all data, asr_final, loop/JSON-stop fixes) 55.88 vs holdout-model q3.5 56.32: same within noise (+-1.8). 42% of public samples call a tool of the 11 apps absent from train (410/634 have one installed); holdout unseen (6 held-out apps) 0.5821 is the matching dev metric.
- Latency fixed: stop at first complete JSON, MAX_NEW_TOKENS 192, tool called >3 times stops, MAX_STEPS 7: samples that took 20-100 s now 7-11 s on the 4060 Ti.
- Added src/arg_diff.py to rank which argument fields differ most.


### Argument error ranking on Qwen3.5 holdout (2026-11-01)
- src/arg_diff.py (scorer-compatible fuzzy rule): on unseen, 119/187 samples have the gold tool sequence and only 66 of those have all arguments accepted. Systematic: home_automation.create_rule.title missing (7); scattered: reminders time minutes off, delay_minutes 150 vs 60, wrong appliance id (ASR/reasoning).
- Post-process in agent.py: if create_rule has no title, build "<Bat/Tat> <device name> luc <h> gio <part of day>" from smart_home device names; on train gold titles it passes F1>0.6 for 11/15. Expected small gain (<~1 point); no retraining.

- Re-run of holdout unseen with the inference fixes + default create_rule title (same para-q35b model): 0.5970 (tool-call only 0.7318, hash 126/187) vs 0.5821 (0.7178). Slowest sample 14.3 s incl. ASR on the 4060 Ti.


### 3-epoch Qwen3.5 run (2026-11-02)
- EPOCHS=3 BASE=Qwen/Qwen3.5-4B SUFFIX=-q35e3 run_e16.sh: training finished (531 steps, 12 h 46 min, final loss 0.00035, mean train loss 0.021, i.e. close to memorising); the merged save failed with no space left, adapter saved at models/lifemate-llm-para-q35e3_adapter. Merge-only script train/merge_adapter.py added; remove the half-written output dir first (it holds config.json and would make run_e16.sh skip the merge).

## E19: 3 epoch trên Qwen3.5-4B
- Giả thuyết: học thêm (1 → 3 epoch) giúp nhớ định dạng và quy tắc tham số tốt hơn.
- Kết quả web: 58.41 (+2.09 so với 1 epoch). Hướng học nhiều hơn có tác dụng, nên là mặc định từ nay.

## E20: gộp E17 + E16 + E18, 1 epoch trên Qwen3.5
- Giả thuyết: transcript ASR nhiễu giúp LLM chịu lỗi nghe, hint so ngưỡng giúp rẽ nhánh giá vàng.
- Kết quả web 57.0: chỉ +0.7 so với 1 epoch thường (56.32), thấp hơn 3 epoch (58.41). Kết luận: hai thay đổi không đáng kể; số epoch có tác dụng lớn hơn.
- E21 (ASR Whisper prompt): thất bại, WER 0.105.

## E22: dữ liệu tổng hợp cho 11 app chỉ có ở public_test (ecommerce, event_tickets, grocery, habit_tracker, pantry, parking, public_transit, recipes, stocks, train_bus_tickets, translator)
- Động cơ: 410/634 mẫu public cài ít nhất một app lạ; unseen holdout 0.60 so với seen 0.77-0.79.
- Cách làm (`train/synth_unseen.py`): Qwen3-14B cục bộ viết câu nói theo schema + trạng thái app; student (bản 3 epoch) trả lời 3 lần trong môi trường thật (1 greedy, 2 lấy mẫu T=0.7); giữ khi cả 3 lần gọi y hệt nhau, không lỗi, không hỏi lại. Chỉ dùng catalog và trạng thái ban đầu của public_test, không dùng audio/nhãn.
- Lưu ý quy chế: §4.1/4.2 không nói rõ việc dùng schema/môi trường tập test để sinh dữ liệu train; cần mô tả trong phương pháp nộp (§5.2) hoặc hỏi ban tổ chức.
- Đo: holdout unseen chỉ đo gián tiếp (6 app khác); phép đo thật là điểm web.

## E22 kết luận
- Web 56.31, không cải thiện so với E20. Lý do giả định: nhãn do chính model tạo (chỉ lọc đồng thuận) nên không thêm thông tin; đồng thuận ở T=1.0 vẫn không chứng minh đúng.

## E23: thử LLM nền khác (zero-shot probe, 60 mẫu unseen, transcript gold)
- Động cơ: điểm thấp, thử model mới <5B tham số (tổng <6B cùng PhoWhisper). Loại trước: Gemma 4 E4B (8.61B), Kimi nhỏ nhất ~16B, DeepSeek không có model <5B.
- Cách đo: `LLM_MODEL=<id> python src/infer.py ... --ids cache/probe60.txt` rồi `src/eval.py`. Mốc: Qwen3.5-4B 0.3528 (34/60 hash), Qwen3-4B-2507 0.2471 (25/60); sau fine-tune hai model này đạt 0.58 và 0.51 nên probe dự đoán đúng thứ tự.
- Gemma-SEA-LION-v4-4B-VL: 0.0333 (4/60). Hunyuan-4B-Instruct: 0.0167 (7/60). gemma-3-4b-it bị khóa quyền; Ministral-3-3B lỗi import Gemma3nConfig (transformers cũ), không chạy.
- Nghi lỗi pipeline nên xem output thô: SEA-LION nạp đúng (Gemma3ForConditionalGeneration), trả JSON hợp lệ bọc ```json nhưng chọn sai app hoặc bỏ bước đọc; Hunyuan trả văn xuôi trong <answer> thay vì JSON.
- Sửa: `LLM_PREFILL` (ép lượt trả lời bắt đầu bằng "{", tắt enable_thinking nếu template có), và `chat.needs_manual` chỉ áp dụng cho template có <|im_start|>. Probe lại với LLM_PREFILL='{': Hunyuan 0.0722 (tool-call 0.1992, 7/60), SEA-LION 0.0167 (2/60).
- Kết luận: hai model vẫn kém xa Qwen3.5 (0.3528) sau khi sửa định dạng, nên là yếu thật với bài này. Không fine-tune. Qwen3.5-4B vẫn là lựa chọn tốt nhất trong giới hạn <5B.

## E24: bỏ phiếu theo n-best ASR (suy luận, không train lại)
- Động cơ: ASR làm mất 34-42% điểm; thử để Whisper cho K=3 bản nghe (beam search), chạy LLM riêng cho từng bản, chọn trạng thái cuối được nhiều bản đồng ý nhất (hòa: bản xếp hạng cao nhất). `src/infer.py --nbest 3 --nbest-cache ...`, `ASR.nbest` (generate num_beams=3, num_return_sequences=3). Mặc định tắt.
- Kết quả holdout_all 287 (`asr_ft` + `lifemate-llm-para-q35e3`): 0.6644, hash 214/287; mốc một bản nghe 0.6623, hash 214/287. Không lợi ích đo được (nhiễu ±0.04-0.05), chi phí suy luận gấp 2-3 lần.
- Đa dạng: 0/287 mẫu có 3 bản nghe khác nhau về từ hay chữ số, tức beam gần như chỉ cho một câu . Kiểm tra tay nbest3_holdout.jsonl: ba bản giống hệt từng ký tự, tức là LỖI CÀI ĐẶT (generate riêng của Whisper trả beam tốt nhất lặp lại), phép thử này KHÔNG có giá trị. Sửa bằng GenerationMixin.generate với decoder_input_ids tự dựng (beam khác nhau thật, ví dụ 'mô vào' / 'mua vào').
- Đo lại holdout_all 287: 0.6714, hash 218/287 so với mốc 0.6623, 214/287. Từ sai sang đúng 7, từ đúng sang sai 3 (kiểm định dấu p ~0.34, chưa đủ bằng chứng). Chi phí: 18-20 s/mẫu so với 7-9 s (gần gấp đôi), GPU gần hết bộ nhớ khi giữ ASR + LLM với beam 3. Kết luận: tín hiệu dương nhỏ (+0.5-1 điểm kỳ vọng), không nộp trừ khi giới hạn thời gian của ban tổ chức cho phép; để quyết định ở bước Docker.
