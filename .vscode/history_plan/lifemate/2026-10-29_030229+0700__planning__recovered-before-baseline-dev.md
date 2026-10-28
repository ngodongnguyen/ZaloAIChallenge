# LifeMate — Planning triển khai, experiments và Public Test

**Mục tiêu:** nhắm Top 1 bằng ASR đúng thực thể, LLM nhỏ thực thi đúng tool và dừng đúng lúc. Đầu ra đầu tiên là notebook experiments chạy trên Kaggle và tạo được `submission.jsonl` hợp lệ cho Public Test; điểm cao được tối ưu sau khi có kết quả.

**Trạng thái:** đang làm baseline, chưa chạy GPU. Repo chỉ giữ một notebook baseline tại `notebook-kaggle/lifemate/agent/lifemate_agent.ipynb`; notebook ASR benchmark (Làn A) và LoRA (đợt 2) tạm cất, làm lại khi xong baseline. Đây là file kế hoạch duy nhất của track.

### Cấu trúc triển khai tại `notebook-kaggle/lifemate/`:
- `agent/` (`lifemate_agent.ipynb`): baseline end-to-end. Tự ASR mọi audio còn thiếu (PhoWhisper-medium, shard theo ID trên 2 GPU) rồi chạy agent Qwen3-4B-Instruct-2507; 3 mode: `public` (mặc định, toàn bộ ID → `submission.jsonl`, chạy validator `zaic_tools`), `diagnostic` (train, transcript sạch), `e2e` (train, transcript ASR). Ghi bền từng mẫu, resume qua `resume_from_dirs`.
- `kernel-metadata.example.json` với `kernel_type: "notebook"`; `README.md` hướng dẫn chạy.

## 1. Nguồn sự thật và giới hạn

- Đọc `.vscode/docs/TRACK1_LIFEMATE_SPECIFICATION.md`, `OFFICIAL_METRICS_AND_MATCHING_RULES.md`, `COMPETITION_OVERVIEW_2026.md`, quy chế PDF và README/mã `zaic_tools` đúng split. Quy chế PDF ưu tiên khi mâu thuẫn; không suy luật chỉ từ simulator.
- Theo đặc tả hiện có: toàn hệ thống <6B tham số, LLM <5B, context ≤16K; model công khai trước 26/10/2026, license phù hợp. Kiểm lại tổng tham số thật trước chọn checkpoint; quantization không làm giảm con số tính theo luật.
- Chấm trên một A30 24GB, tối đa 30 giây/mẫu; offline, greedy, seed cố định. Mỗi mẫu inference một `ToolEnv`; chỉ truy cập app state qua API được phép, không đọc `state.json` trực tiếp.
- Không dùng input/audio/state public/private test để train hoặc sinh nhãn. Replay gold phục vụ scorer offline, phải tách khỏi inference.
- Theo timeline repo: đóng băng Docker trước 23:59 16/11/2026; private 20–23/11. Public nộp tối đa 5 lần/ngày/track. Không cần Docker hoàn chỉnh để nộp file Public Test.

## 2. Dữ liệu và những điểm quyết định thiết kế

Số liệu kế thừa từ khảo sát trước, implement phải kiểm lại manifest/version: train 1.545 mẫu, 5.036 audio (~6,8 giờ); public 634 mẫu, 1.933 audio. Audio 16 kHz mono, gồm cả lượt user trong lịch sử; lịch sử dài nhất được ghi nhận 12 audio user. Train có transcript sạch, public phải ASR mọi audio cần thiết.

- Khảo sát ghi nhận 46 app train, 36 app public và 11 app chỉ ở public: ecommerce, event_tickets, grocery, habit_tracker, pantry, parking, public_transit, recipes, stocks, train_bus_tickets, translator. Vì vậy prompt phải dựa trên schema, không chỉ học thuộc tên app.
- Gold train có 111 mẫu không gọi tool và 1.182 read call trong tổng 3.541 call. Read cũng được chấm: thừa/thiếu đều mất điểm. `contacts.search.query` nhạy với lỗi tên riêng; cần đo lỗi thực thể, không chỉ WER.
- Lịch sử có `tool_results`; điều kiện đã được trả lời có thể dẫn tới nhãn rỗng. Giữ thông tin này trong prompt, không gọi read lại một cách máy móc.
- `sys.ask_missing_info` dùng khi thiếu thông tin hoặc kết quả mơ hồ và là hành động cuối. Không đoán ID từ trí nhớ.
- Simulator có thể ghi call sai ngữ nghĩa dù trả `ok`; validation chỉ chặn schema. `LIVE_APPS`/`FREE_TEXT_KEYS` có thể khác theo split: import package đúng split.
- Khảo sát trước ghi replay gold khớp state 1545/1545. Điều này kiểm phần hash, chưa chứng minh matcher local tương đương BTC.

## 3. Pipeline và model ban đầu

Audio hiện tại + lịch sử → ASR → transcript; context + lịch sử text/tool/results + schema app đã cài → LLM sinh một call JSON hoặc done → kiểm installed/schema/duplicate → `env.call` → đưa result vào bước tiếp. Dừng khi done, ask, hết bước hoặc hết ngân sách. Xuất trực tiếp `env.get_tool_calls()` và `env.get_state_hash()`; không hash giả hoặc sinh một chuỗi ID phụ thuộc read khi hết giờ.

- Baseline: `vinai/PhoWhisper-medium` + `Qwen/Qwen3-4B-Instruct-2507`.
- ASR so sánh từ research cũ: `g-group-ai-lab/gipformer-68M-rnnt`, tùy quota thêm `Qwen/Qwen3-ASR-0.6B`.
- LLM đối chứng: `Qwen/Qwen3.5-4B`, tắt thinking nếu model hỗ trợ. Model ID là ứng viên kế thừa, chưa là quyết định thắng; xác minh revision, số tham số, cutoff, license và backend T4 trước run. Không cần thêm vòng research dài để bắt đầu baseline.
- Bắt đầu tối đa 8 bước, ≤256 output tokens/bước, tổng input+output ≤16K. Đây là config thí nghiệm; log truncation và kiểm mẫu lịch sử dài. Deadline có ngân sách ASR + generation + tool, giới hạn thời gian còn lại từng bước. Deadline mềm không chứng minh SLA.
- Không gọi tool ngoài installed apps, trừ tool hệ thống được phép. Chuẩn hóa thời gian theo `current_time` và múi giờ +07:00.

## 4. Split và scorer trước tuning

Cố định manifest/seed trước SFT hoặc synthetic. Tách train-fit, dev-seen, dev-unseen theo app và final-holdout. Dev dùng chọn phương án; final chỉ dùng khi chốt shortlist. Với app chưa thấy, loại khỏi train-fit các mẫu cài app đó, không sinh synthetic cho app holdout; kiểm số mẫu còn lại để tránh train quá ít. Nếu không đủ dữ liệu chia nhiều tập, ưu tiên train/dev/final rõ ràng và ghi phép đo unseen nào thực hiện được, không gọi tập đã tuning là final.

ASR fine-tune cũng phải loại audio dev/final; cache transcript không đồng nghĩa train trên chúng. Nhóm near-duplicate hội thoại/template khi chia được nhận diện. Không dùng transcript sạch để công bố end-to-end; chỉ dùng diagnostic để tách lỗi ASR và planner.

Scorer: hash khác gold → 0; hash khớp thì chấm F1 call theo luật, nhãn rỗng đúng → 1. Kiểm call lặp theo multiset, ngưỡng fuzzy Token-F1 >0.6, optional params, normalization, sai hash. Phần chưa xác nhận so chuỗi exact hay `_norm` phải ghi giả định và báo riêng hash/matched calls; không suy normalization matcher từ normalization state hash.

## 5. Experiments: chạy trước, phân tích sau

Chuẩn bị sẵn notebook ASR benchmark/cache, notebook agent eval có chế độ diagnostic/end-to-end/public và notebook LoRA cho vòng sau. Agent implement chỉ kiểm kỹ thuật 20–50 mẫu và resume trước khi chạy batch; không cần điểm đẹp mới được bắt đầu.

| Đợt / làn | Cấu hình và dữ liệu | Kết quả bắt buộc |
|---|---|---|
| 1A / A | So PhoWhisper và gipformer trên cùng manifest audio train/dev; 2 GPU shard audio | Transcript cache, WER, lỗi tên/số/giờ, latency/VRAM |
| 1B / B | Hai LLM baseline/đối chứng trên cùng dev; transcript sạch diagnostic trước, rồi ASR cache baseline của A | Score/hash/calls, trace và lỗi từng ID; diagnostic tách riêng |
| 1C / A+B sau cache | Baseline end-to-end thật trên dev; cấu hình đã cố định chạy public toàn bộ | Dev metrics; public submission hợp lệ, không đợi SFT |
| 2 / B | SFT train trajectories thật so zero-shot | Adapter + full resume state + dev score theo nhóm |
| 3 / A+B | Thêm synthetic tỷ lệ 0/nhỏ/lớn theo lỗi đợt trước | Chất lượng tăng trên dev sạch, không mất hành vi nhãn rỗng |
| 4 / A hoặc B | ASR fine-tune nếu lỗi thực thể là bottleneck; compression/serving nếu chậm | Score end-to-end và chi phí/latency; mỗi run đổi một yếu tố |
| Cuối | Finalist trên holdout, rồi một A30 và Docker offline | Reproducibility, max latency, output hợp lệ |

SFT dùng trajectory replay có tool result thật; so input sạch/noisy theo ngân sách, không mặc định chỉ một loại. Synthetic chỉ dùng catalog/mã app được phép và state tự sinh; kiểm cả đúng ý định, read call tối thiểu và state, không chỉ hash. Không đưa app holdout vào synthetic train. Teacher ngoài chỉ dùng nếu quy chế cho phép; không có API thương mại khi inference. DPO không nằm đường găng, chỉ mở khi SFT và lỗi dev cho thấy cần.

## Vận hành Kaggle: 4 tài khoản, 2× T4, CPU ít

- Tất cả notebook được upload private về tài khoản chính người dùng đã chỉ định (`<owner>` trong tài liệu), **không auto-run**. Người dùng tự share/copy sang 4 tài khoản còn quota và khởi chạy. Không tự đổi tài khoản global. Kiểm quyền riêng cho notebook, dataset/model và checkpoint; share notebook chưa đủ cấp quyền input private.
- Phân công đợt đầu: **A = LifeMate ASR; B = LifeMate agent; C = DataMate raw code-agent; D = DataMate chuẩn hóa + code-agent**. C/D dùng cùng model và tập câu để so kiến trúc. Chuẩn bị profiler/cache CPU dùng chung trước; không để D phải chờ phân tích C. B chạy diagnostic transcript sạch trong lúc đợi cache ASR của A, rồi chạy end-to-end sau khi cache đã xuất version. Sau đợt đầu mới phân tích để chia lại quota.
- T4 có 16 GB/card; hai card không tự gộp thành một vùng VRAM. Dùng FP16 hoặc quantization tương thích; không mặc định BF16. Model vừa một card thì ưu tiên thử hai replica xử lý shard riêng. Chỉ chia model/DDP khi benchmark chứng minh cần và có lợi. Đo peak VRAM từng GPU, CPU/RAM, throughput, thời gian load/save; không giữ GPU bận bằng tác vụ CPU không cần thiết.
- ASR: hai worker GPU, chia audio ID ổn định. LifeMate end-to-end có thể thử ASR GPU0/LLM GPU1; đây không phải phép đo SLA một A30. DataMate model khoảng 14B thử 4-bit một card với context vừa phải nếu đủ KV/runtime memory; nếu không vừa thì thử chia hai card. Quantization không giảm tham số tính theo luật.
- CPU: bắt đầu 1–2 worker toàn session, một luồng BLAS/OpenMP mỗi process, DataLoader workers 0–1; executor DataMate bắt đầu 1, chỉ tăng lên 2 sau đo RAM/CPU. Giới hạn queue. Parse Excel/SQLite, tokenize hoặc decode audio một lần rồi cache; không mỗi GPU nạp và parse lại toàn bộ dữ liệu.
- Cache key chứa input hash, split, model revision, precision/decoding config và phiên bản code liên quan. Dữ liệu inference không chứa nhãn. Cache bảng DataMate thêm rulebook và loader version; giữ dữ liệu gốc và nguồn file/sheet/dòng.

## Giới hạn 12 giờ và tiếp tục phiên sau

1. **Đợt đầu chạy phiên 3–4 giờ**, dừng có kiểm soát và lưu output version; resume sang phiên sau nếu chưa xong. Chỉ kéo dài khi đã kiểm chứng resume. Trần 12 giờ theo yêu cầu người dùng tính cả setup/load; khi chạy dài, dự kiến ngừng nhận việc ở 10h30 và export trước 11h, điều chỉnh sớm hơn theo tốc độ lưu thực tế.
2. Inference ghi kết quả sau từng ID, flush và ghi manifest hoàn tất sau record durable; mỗi worker có shard file riêng. Resume kiểm config/hash, bỏ dòng cuối hỏng, không mất/trùng ID. Lỗi tạm giữ `pending` hoặc `retryable`; chỉ `completed` khi có kết quả cuối. Không ghép kết quả khác cấu hình.
3. Train lưu checkpoint mỗi 15–30 phút hoặc N optimizer steps: adapter/weights cần thiết, optimizer, scheduler, scaler, RNG, global step/epoch, sampler/cursor, manifest và config. Adapter-only là warm-start, không gọi là exact resume. Giữ latest/previous hoàn chỉnh và best; không sao chép base weights vào mọi checkpoint adapter.
4. Ghi file tạm → rename, checksum và complete marker. `/kaggle/working` chưa đảm bảo tồn tại qua session: phải xác nhận output version đã lưu thành công, rồi attach nó hoặc dataset checkpoint private làm input phiên sau. Nếu session chết trước export, phần checkpoint local có thể mất. Muốn bảo vệ giữa phiên phải kiểm chứng nơi lưu bền vững riêng; không giả định Kaggle tự lưu.
5. Test resume trước chạy lớn: inference kiểm tập ID và kết quả; train so 20 steps + resume 20 với 40 liên tục, kiểm step/scheduler/sampler/loss với sai số khai báo. Không chờ tín hiệu kill để checkpoint.

## Hợp đồng bàn giao notebook và kết quả

Notebook có: config → preflight → kiểm input/resume → chạy → checkpoint/export → summary. Config bắt buộc: track, experiment/run ID, parent run, model revision/precision, split manifest, seed, shard index/count, context/output budget, CPU workers/threads, wallclock budget, resume và output path. Dùng `/kaggle/input/...` cho dữ liệu, `/kaggle/working/...` cho output; Docker sau này dùng `/data` và `/result`.

Mỗi run xuất: config/manifest, environment/hardware, predictions theo ID, trace/errors, summary/timing/VRAM, checkpoint manifest nếu có. Không lưu credentials. Sau phiên gom link notebook/output version về bảng kết quả cuối file này; username chỉ dùng bí danh A–D. Một thay đổi chính mỗi experiment, giữ cùng ID và ngân sách so sánh. Chạy kiểm kỹ thuật ngắn rồi thực thi các cấu hình đợt 1, **phân tích chất lượng sau khi có kết quả**. Không yêu cầu baseline đạt điểm cao mới được chạy đợt đầu. Vòng sau chỉ hai ứng viên tốt nhất mỗi nhánh chạy full dev; giữ quota dự phòng lỗi/resume/finalist.

## 6. Output để phân tích và nộp public

Trace dev từng ID: transcript, call/result, validation failures, termination, hash match, matched calls, score, ASR/agent/total latency, peak VRAM. Phân nhóm lỗi: ASR thực thể, write/hash, read thừa/thiếu, missing-info, nhãn rỗng, lịch sử, app mới, thời gian/timeout. Điểm cao chọn sau khi có đợt 1; đo bootstrap/biến thiên theo nhóm khi so chênh lệch nhỏ.

Public mode phải chọn split rõ ràng, ASR cả lịch sử, xử lý toàn bộ ID từ manifest hiện tại (khảo sát là 634), hỗ trợ shard/merge và resume. Mẫu đang chạy dở phải khởi tạo lại môi trường từ đầu khi resume. Nếu không khởi tạo được ToolEnv thì ghi lỗi và sửa, không bịa hash để lấp ID.

- [ ] Một dòng UTF-8 JSONL/ID, không trùng/mất, đúng `id`, `tool_calls`, `state_hash`.
- [ ] Tool calls/hash lấy từ môi trường thực, đúng package/split và validate-before-call.
- [ ] Chạy validator chính thức `zaic_tools` theo README của dataset; xác minh cú pháp trên version đang dùng.
- [ ] Tải `submission.jsonl` từ output Kaggle về để người dùng nộp Public Leaderboard. Notebook upload không phải submission cuộc thi.
- [ ] Public đầu tiên hợp lệ có thể nộp trước tối ưu; ghi điểm/version sau khi có kết quả, không yêu cầu đã cải thiện so với một baseline chưa tồn tại.

## 7. Sau experiments và mốc chốt

Sau đợt 1 mới phân tích để chọn ASR/LLM và nhóm lỗi đáng sửa. Theo lịch repo: baseline 31/10–03/11; SFT/synthetic 04–07/11; shortlist + thử GPU đích 07–10/11; freeze và nghiệm thu Docker 11–14/11; nộp dự phòng 15/11. Lịch là mục tiêu, không đánh dấu đã hoàn thành khi chưa có artifact.

Docker sau này chứa weights/tokenizers/dependencies, chạy network-none, đọc `/data`, ghi `/result/submission.jsonl`, cùng config đóng băng và seed/greedy. Kiểm chạy lại, cả max latency và timeout trên A30 thực; hai T4/A10 không xác nhận SLA A30. Có source train/infer, hướng dẫn, checksum/link nộp theo quy chế. Đây là giai đoạn sau public baseline.

## 8. Kết quả và nguồn tham khảo

Lịch sử run, điểm, account, log nộp Public LB: [`experiments/INDEX.md`](experiments/INDEX.md) (mỗi run một card). Kiến thức đúc kết: [`insights.md`](insights.md).

Nguồn model để xác minh khi implement: [PhoWhisper](https://huggingface.co/vinai/PhoWhisper-medium), [gipformer](https://huggingface.co/g-group-ai-lab/gipformer-68M-rnnt), [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B), [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), [T4 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet.pdf). Model card/benchmark ngoài không thay thế số đo dev của cuộc thi.
