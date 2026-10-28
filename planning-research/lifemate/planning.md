# LifeMate — Planning triển khai, experiments và Public Test

> **Lịch sử plan:** [bản trước đợt baseline-dev](../../.vscode/history_plan/lifemate/2026-10-29_030229+0700__planning__recovered-before-baseline-dev.md) · [bản trước bổ sung quy tắc lịch sử](../../.vscode/history_plan/lifemate/2026-10-29_030229+0700__planning__before-history-policy.md) · [sổ phiên bản](../../.vscode/history_plan/INDEX.md). Phải lưu snapshot trước mỗi đợt sửa tiếp theo.

**Mục tiêu:** nhắm Top 1 bằng ASR đúng thực thể, LLM nhỏ thực thi đúng tool và dừng đúng lúc. Đầu ra đầu tiên là notebook experiments chạy trên Kaggle và tạo được `submission.jsonl` hợp lệ cho Public Test; điểm cao được tối ưu sau khi có kết quả.

**Trạng thái:** đang làm baseline, chưa chạy GPU. Ngày 2026-10-29 người dùng chọn đo baseline thật trên dev trước: `lm-B-001-baseline-paired-dev` (mục 9–10), thay ưu tiên public-first trước đây. Repo chỉ giữ một notebook baseline tại `notebook-kaggle/lifemate/agent/lifemate_agent.ipynb`; notebook ASR benchmark (Làn A) và LoRA (đợt 2) tạm cất, làm lại khi xong baseline. Đây là plan đang hiệu lực của track; các bản trước được giữ trong `.vscode/history_plan/`.

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

## 9. Thiết kế experiment kế tiếp — 2026-10-29

**Phạm vi đã chọn:** một experiment LifeMate và một experiment DataMate, đo model thật từ notebook hiện có. Chỉ lập kế hoạch trong đợt này; chưa sửa notebook, chạy GPU hoặc nộp Public LB.

**Experiment LifeMate:** `lm-B-001-baseline-paired-dev`. Câu hỏi: Qwen3-4B hiện làm tool tốt đến đâu, và thay transcript sạch bằng PhoWhisper làm điểm thay đổi bao nhiêu?

- Dùng notebook `notebook-kaggle/lifemate/agent/lifemate_agent.ipynb`; một lần thực thi Kaggle điều phối hai nhánh diagnostic/e2e tuần tự, hai thư mục output riêng. Một biến nghiên cứu: nguồn transcript. Không thêm model, prompt mới, SFT hoặc sửa thực thể trong run này.
- Cố định `train/dev_seen`, manifest seed 2026, toàn bộ ID của subset; hai nhánh dùng cùng ID, revision LLM, FP16, greedy, seed 0 và ngân sách agent hiện tại. `final_holdout` và `dev_unseen` chưa chạy. Đây là phép đo dev seen; không kết luận khả năng tổng quát app mới.
- Diagnostic dùng transcript sạch của yêu cầu cuối **và mọi lượt user lịch sử**. E2E bắt buộc ASR đủ toàn bộ audio tương ứng, loại `audio_text` khỏi input agent để không fallback về gold. Nhãn chỉ dùng ở scorer sau inference.
- Tái sử dụng ASR chỉ khi khớp split, input fingerprint, model revision, decoding và code fingerprint; không quét mọi cache chỉ theo tên backend. Log đủ expected/completed/missing audio.
- Giữ `max_steps=8`, `max_new_tokens=256`, `max_invalid=2`, `sample_budget_s=30`, `result_chars=1500`. Báo riêng latency agent và ASR; chạy ASR cache trước rồi agent trên 2 T4 chưa chứng minh SLA 30 giây/A30.
- Điểm chính: score hash-gated thang 100 trên **cùng đủ N ID**; thêm hash OK, F1 call trước hash gate, read thừa/thiếu, empty/ask/write/read-only, history/no-history. Đo chênh lệch e2e − diagnostic theo từng ID; không ép chênh lệch phải âm và không gọi đây là quan hệ nhân quả hoàn toàn.
- Không công nhận baseline hoàn tất nếu thiếu ID, transcript lịch sử, lỗi schema, lẫn config/cache hoặc scorer dùng mẫu đã xong làm mẫu số. Run lỗi vẫn giữ card và artifact; chưa có điểm thật thì không điền kết quả dự đoán.
- Phiên đầu 3–4 giờ gồm setup, ASR, cả hai nhánh, merge/export; hết budget thì checkpoint và resume cùng config ở version sau. Không tự kéo thành 10,5 giờ **mỗi pha**. Chỉ công bố điểm cuối khi hai nhánh hoàn tất cùng manifest.

**Các cách đã cân nhắc:** (1) đo baseline thật bằng notebook hiện có — người dùng đã chọn; (2) mở ASR benchmark riêng và raw code-agent DataMate — giữ cho vòng sau vì chưa có điểm model thật làm mốc. Chưa dùng LB để tune cấu hình trong experiment này.

**Đầu ra cần implement:** preset paired-dev, kiểm toàn vẹn manifest/cache, chặn gold fallback, báo điểm đầy đủ và artifact so từng ID. Thiết kế đã được người dùng duyệt trong phiên 2026-10-29; checklist triển khai ở mục 10. Mọi source vẫn nằm trong notebook, kế hoạch/run card ở track tương ứng. Public là lần chạy riêng sau đó với run_id/card mới.

## 10. LifeMate paired-dev — Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` to implement task-by-task. Chỉ dùng subagent-driven-development nếu người dùng chọn cách đó. Các checkbox dưới đây đều chưa thực hiện; không commit/push nếu chưa được yêu cầu.

**Goal:** tạo baseline model thật có hai phép đo clean/ASR trên cùng dev, tái lập và không rò transcript sạch sang nhánh e2e.

**Architecture:** bổ sung preset `paired_dev` vào notebook hiện có; driver lần lượt chạy ASR cache, diagnostic, e2e, rồi so từng ID. Các worker vẫn dùng ToolEnv và agent loop hiện tại; tách output/hash từng nhánh và scorer khỏi input inference.

**Tech Stack:** Python trong Kaggle notebook, PyTorch/Transformers hoặc vLLM đã preflight, PhoWhisper, `zaic_tools`, JSONL. Không thêm model trong đợt này.

**Spec:** thiết kế mục 9; luật ở `.vscode/docs/TRACK1_LIFEMATE_SPECIFICATION.md` và `OFFICIAL_METRICS_AND_MATCHING_RULES.md`; lịch sử [`lm-B-000`](experiments/2026-10-29_lm-B-000-local-dryrun.md).

### Global Constraints

- Theo đặc tả repo: toàn hệ thống <6B, LLM <5B, context ≤16.384 token; mỗi mẫu inference đúng một ToolEnv, validate trước call; state chỉ qua API. Xác minh revision/license/cutoff/tổng tham số trước run, không tự chọn checkpoint khác.
- Model cố định `Qwen/Qwen3-4B-Instruct-2507`, ASR `vinai/PhoWhisper-medium`; resolve rồi lưu commit revision cụ thể trong preflight, chưa có thì dừng trước chạy lớn. Không để `revision=None` trong cấu hình đóng băng.
- `manifest_seed=2026`, `seed=0`, `dtype=half`, thinking off, không quantization. Preflight chọn **một** backend hoạt động rồi ghi rõ `vllm` hoặc `hf`; hai nhánh cùng backend, không silent fallback giữa nhánh/session.
- Chạy 2 T4 theo shard ID ổn định, mỗi GPU một replica, giữ số shard khi resume. Agent batch 32, HF internal batch 8, ASR batch 16; OOM cần đổi cấu hình thì ghi run mới, không trộn output. Không coi tổng VRAM là một card.
- Default mới cho preset: `session_wallclock_s=14400`, `export_reserve_s=900`; tính từ cell setup đầu tiên, bao gồm mọi pha. Worker dùng thời gian còn lại của deadline chung, không mỗi pha có 4 giờ riêng. Giữ public preset cũ riêng biệt.
- Checkpoint/resume cùng input/code/config; chỉ đổi đường dẫn runtime và deadline được phép. Mỗi lần thực thi Kaggle/version đều có card riêng, kể cả continuation (lấy NNN kế tiếp khi chạy); card gốc liên kết các phần, không ghi đè lịch sử phiên trước.

### Review Focus

| Điều kiện dễ làm sai số đo | Kiểm tra thuộc task |
|---|---|
| Thiếu transcript audio lịch sử → fallback về gold | L2: thiếu một history audio phải chặn e2e, cả khi final audio đã đủ |
| Cache cùng tên nhưng khác revision/audio/code | L1–L2: fingerprint lệch bị từ chối, duplicate text mâu thuẫn bị từ chối |
| Chỉ xong một phần dev nhưng báo điểm cuối cao | L3: dùng đủ N, `baseline_complete=false` khi pending > 0 |
| Cùng mẫu nhưng nhánh/backend/manifest khác | L4: so khớp metadata trước khi tính delta, không ghép hai config |
| Session hết giờ sau ASR hoặc giữa hai nhánh | L4: driver dừng có checkpoint và export, resume không chạy lại ID hoàn tất |

### Task L1: Cố định preset và provenance trước inference

**Files:** sửa cell CONFIG/setup và cell nhúng `lmlib/data.py`, `lmlib/common.py`, `lmlib/llm.py`; thêm cell regression assertions trong `lifemate_agent.ipynb`. Không cần tách thư viện ra ngoài notebook.

**Interfaces:** thêm `prepare_dev_run(cfg: dict, rows: list[dict]) -> dict` trong `lmlib/data.py`; trả manifest gồm `ids`, `manifest_sha256`, `input_sha256`, `code_sha256`, `heldout_ids_sha256`. Lưu `manifest.json` và `preflight.json`; `model_revision`, `asr_revision`, `resolved_backend`, package versions là dữ liệu preflight, không là kết luận đã kiểm ở tài liệu này.

- [ ] Viết và chạy assertions trước sửa: cùng input/seed cho cùng ID/hash; dev disjoint final/train_fit; đổi một input byte làm input hash đổi. Expected ban đầu: thiếu `prepare_dev_run`.
- [ ] Thêm `experiment="lm-B-001-baseline-paired-dev"`, `preset="paired_dev"`, `split="train"`, `subset="dev_seen"`, `limit=None`, `asr_inline=True`; đóng băng manifest trước chạy. Input fingerprint chỉ đọc manifest/audio/catalog/package được phép; **không** duyệt/hash/parse `<split>_environment/` hoặc `state.json`. Không dùng `subset="dev_seen+..."` vì `select_rows` hiện không hiểu cú pháp này.
- [ ] Lưu code hash từ nội dung các module nhúng và driver; preflight resolve model revisions, kiểm đủ audio, tools package đúng split, context tokenizer và backend trên 8 ID đầu của `train_fit` trong thư mục smoke riêng, không chấm để tune.
- [ ] Chạy lại assertions; với preflight Kaggle, expected: backend cụ thể, không mock, hai GPU được nhận diện, resolved revision không rỗng. Thiếu tài nguyên → log lỗi, không tự đổi model/precision.

### Task L2: Bảo đảm chỉ nguồn transcript thay đổi giữa hai nhánh

**Files:** cell nhúng `lmlib/asr.py`, `lmlib/run.py`, `lmlib/agent.py` và cell điều phối ASR.

**Interfaces:** thêm `load_verified_transcripts(cache_dir: str, expected: dict, audio_ids: list[str]) -> dict[str, str]` trong `lmlib/asr.py`; thêm `inference_row(row: dict, mode: str) -> dict` trong `lmlib/run.py`. Hàm sau luôn bỏ `label`, e2e còn bỏ mọi `audio_text` (final/history), giữ history assistant tool_calls/tool_results hợp lệ. Diagnostic lấy gold text ngoài input row; không sửa trực tiếp dữ liệu gốc.

- [ ] Viết assertions cho 2 lượt history + final: thiếu một audio thì raise; cache revision/input/code khác thì raise; cùng audio có hai transcript mâu thuẫn thì raise; prompt e2e với sentinel gold không chứa sentinel. Diagnostic phải có đủ cả ba transcript sạch.
- [ ] Thay tìm cache rộng `/kaggle/input/**` bằng một cache version/path đã khai báo và header fingerprint được kiểm. `audio_manifest.json` liệt kê mọi audio của đúng dev; chỉ `completed` đủ bộ và không xung đột mới chuyển sang agent.
- [ ] Sửa driver truyền **list ID thực** cho ASR, không truyền subset list vào `.split('+')` của `run_asr` hiện tại. `asr.evaluate` chỉ chấm đúng audio manifest của run; WER/CER và recall entity heuristic được ghi rõ là diagnostic ASR.
- [ ] Chạy regression cell không GPU bằng transcript fixture; expected tất cả assertion pass. Không dùng backend gold làm cache e2e; nếu cache thiếu thì resume ASR, không điền chuỗi rỗng hoặc gold.

### Task L3: Scorer dùng đủ manifest và chấm prediction thật

**Files:** cell nhúng `lmlib/scorer.py`, `lmlib/run.py` (merge/trace) và regression cell.

**Interfaces:** giữ `sample_score(...)`; mở rộng `merge(cfg, finalize=False) -> dict` với `expected_n`, `completed_n`, `pending_n`, `baseline_complete`, `score_full_manifest`, `score_completed_only`, `hash_ok_rate_full_manifest`, `tool_f1_before_gate`. Hash gold replay chỉ trong scorer offline, tách instance khỏi inference.

- [ ] Viết fixture: 2 ID, 1 hoàn tất đạt 1 và 1 pending → score full=50, completed-only=100, complete=false; đủ 2/2 mới complete. Empty gold/pred đạt 1; hash sai nhưng calls đúng → gated=0 và ungated F1=1; read thừa bị giảm F1.
- [ ] Merge kiểm header hash và tập ID, từ chối ID ngoài manifest hoặc completed duplicate mâu thuẫn; record lỗi schema không được công nhận final. Report partial dùng mẫu số N, không biến pending thành prediction/hash bịa.
- [ ] Thêm `run_hash` vào summary LifeMate (hiện chưa có), log termination, prompt/output tokens, schema/invalid count và peak VRAM theo GPU; trace hiện có chỉ đủ một phần các số đo này.
- [ ] Chạy regression assertions và kiểm `per_id_scores.jsonl` đúng một dòng/ID khi hoàn tất. Báo group có `n`; group vắng ghi n=0/NA, không suy điểm cho nhóm chưa xuất hiện.

### Task L4: Điều phối paired run, so từng ID và kiểm resume

**Files:** cell run/merge hiện có; thêm cell nhúng `lmlib/compare.py` và cell regression assertions.

**Interfaces:** thêm `compare_paired(clean_dir: str, e2e_dir: str, manifest_path: str) -> dict` trong `lmlib/compare.py`; output `paired_report.json`, `paired_per_id.jsonl`. Lưu output dưới `/kaggle/working/lm_baseline/{asr,diagnostic,e2e}`; mỗi nhánh có config/hash riêng và cùng `comparison_id="lm-B-001-baseline-paired-dev"`.

- [ ] Test fixtures có khác ID/backend/revision/agent budget phải bị từ chối; khác đúng transcript source được chấp nhận. Hai điểm theo ID [1,0] và [0,1] → mean delta=0, một giảm/một tăng; không báo “ASR không ảnh hưởng”.
- [ ] Driver chạy ASR → diagnostic → e2e tuần tự, giải phóng process/model giữa các pha; dùng deadline chung từ L1, không chạy hai replica LLM và ASR trên cùng GPU cùng lúc. Từng nhánh ghi shard/trace/summary riêng, không copy chung preds.
- [ ] Báo mean delta e2e−diagnostic và paired bootstrap 2.000 lần seed 2026, resample cùng ID cho cả hai; CI theo ID là mô tả dev, chưa xử lý toàn bộ tương quan template. Báo thêm WER/CER, tỉ lệ đủ audio, lỗi name/number/time heuristic và score theo history.
- [ ] Fixture resume: ngắt sau một batch, append dòng cuối hỏng, resume rồi so tập ID/prediction với chạy liền; expected không mất/trùng, hash đổi bị từ chối. Kiểm trước bước chạy lớn bằng mock; model thật chỉ smoke kỹ thuật. Do `run_hash` đang bỏ qua `num_shards`, thêm kiểm riêng trong manifest checkpoint để đổi số shard khi resume bị từ chối.
- [ ] Timing dùng ASR theo audio và agent theo episode; báo latency phase/model load/preprocess/export riêng. Không cộng thời gian trung bình batch rồi gọi đó là latency A30; phép thử deadline cứng 30 giây/A30 là vòng sau.

### Task L5: Chạy, chốt baseline và bàn giao kết quả

**Files:** [`run card planned`](experiments/2026-10-29_lm-B-001-baseline-paired-dev.md), [`INDEX`](experiments/INDEX.md), [`insights`](insights.md), `notebook-kaggle/lifemate/README.md`, `agent/kernel-metadata.example.json` nếu cần thêm field.

- [ ] Chạy regression cell trong Kaggle sau materialize module, expected `REGRESSION_OK` và không download model cho fixture tests. Local chỉ kiểm cấu trúc notebook/source; không gọi local fixture là baseline model thật.
- [ ] Cập nhật README cách chọn `paired_dev`, attach cache/output, resume và đọc `paired_report.json`; metadata thật gitignored dùng `<owner>` trong example. Theo vận hành hiện có, upload private chưa auto-run; người dùng khởi chạy khi triển khai được yêu cầu.
- [ ] Khi chạy: đổi card planned → running, ghi account alias, version, revisions, input/code/manifest/config hashes. Kết thúc lưu output version và GPU-hours đo thực; run hỏng vẫn ghi lý do. Bản resume/version mới có card và INDEX mới.
- [ ] Chỉ điền “baseline dev e2e” khi cả hai nhánh đủ cùng N, schema_errors=0, audio coverage=100%, không gold fallback và provenance khớp. Điểm thấp vẫn là baseline hợp lệ; không đặt ngưỡng điểm tự suy đoán để trì hoãn báo kết quả.
- [ ] Nếu diagnostic yếu: run kế sửa planner theo lỗi phổ biến; nếu paired delta cho thấy ASR là bottleneck: run kế benchmark/sửa ASR; nếu timeout chiếm nhiều: ưu tiên budget/serving. Đây là nhánh quyết định **sau có số đo**, chưa là thêm experiment được chạy trong đợt này. Public/holdout cần card/config riêng.
