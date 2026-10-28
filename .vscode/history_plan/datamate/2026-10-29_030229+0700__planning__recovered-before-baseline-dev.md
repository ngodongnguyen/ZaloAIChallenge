# DataMate — Planning triển khai, experiments và Public Test

**Mục tiêu:** nhắm Top 1 bằng pipeline đọc dữ liệu đúng, áp dụng đúng rulebook và tính toán có kiểm chứng. Đầu ra đầu tiên là notebook experiments và `submission.jsonl` hợp lệ cho Public Test. Chạy đợt đầu lấy kết quả trước, sau đó mới phân tích để chọn kiến trúc/model và tối ưu.

**Trạng thái:** đang làm baseline, chưa chạy GPU. Repo chỉ giữ một notebook baseline tại `notebook-kaggle/datamate/helper-agent/datamate_helper_agent.ipynb`; notebook profiler riêng (Làn 0) và raw code-agent đối chứng (Làn C) tạm cất, làm lại khi cần so kiến trúc. Đây là file kế hoạch duy nhất của track.

### Cấu trúc triển khai tại `notebook-kaggle/datamate/`:
- `helper-agent/` (`datamate_helper_agent.ipynb`): baseline Làn D. Tự profile + làm sạch mọi shop của split (xlsx nhiều sheet, csv, sqlite; parse tiền/ngày VN, ffill trong đơn, bỏ dòng trùng cùng đơn, `_status`, README cutoff/quy đổi), rồi Qwen3-4B-Instruct-2507 sinh code trên bảng sạch + helper (kỳ thời gian, Tết, lọc đơn, xếp hạng), executor process riêng, sửa 1 lần, lỗi → abstain. Mặc định toàn bộ public_test → `submission.jsonl`; `split="train"`, `subset="dev_tune"` để đo PA.
- `kernel-metadata.example.json` với `kernel_type: "notebook"`; `README.md` hướng dẫn chạy.

## 1. Nguồn sự thật và giới hạn

- Đọc `.vscode/docs/TRACK2_DATAMATE_SPECIFICATION.md`, `OFFICIAL_METRICS_AND_MATCHING_RULES.md`, overview và quy chế PDF. Quy chế PDF ưu tiên khi mâu thuẫn. Luật tính nghiệp vụ lấy từ `rulebook.md` đúng split và README shop; không dùng luật train đóng gói sẵn để thay luật test.
- Theo đặc tả hiện có: tổng model inference ≤15B tham số (MoE tính tổng), model công khai trước 01/10/2026, license phù hợp. Kiểm tổng tham số thật; quantization không làm model >15B hợp lệ.
- PA: đúng +1, abstain 0, sai −1. `unanswerable` là đáp án có thể được +1 hoặc −1, khác abstain.
- Máy chấm hai RTX 5090 32GB/card, RAM 512GB, tối đa 6 giờ toàn tập gồm load model. Offline, greedy/seed; sai lệch PA uploaded/rerun ≤0.03 theo tài liệu repo. Backend phải tương thích Blackwell; T4 không xác nhận chạy được trên 5090.
- Theo timeline repo: Docker đóng băng trước 23:59 16/11/2026; private 20–23/11; public tối đa 5 lần/ngày/track. Public submission chưa cần hoàn tất Docker.

## 2. Dữ liệu và điểm quyết định thiết kế

Số liệu kế thừa từ khảo sát trước, cần xác minh version/manifest: train 3.000 câu/30 shop, public 1.000 câu/14 shop mới. Các kiểu train được ghi nhận: number 1.884, percent 578, list 193, date 145, string 88, unanswerable 70, boolean 23, table 19. Vì vậy lõi doanh thu/đơn/thời gian quan trọng, nhưng phải hỗ trợ đủ kiểu.

- Các họ input: CSV phẳng; Excel nhiều file/sheet; đơn nhiều dòng; sổ cần forward-fill theo quy ước; sheet tháng/năm; SQLite POS; nhiều CSV join mã; dữ liệu trộn. Public có ngành mới như mẹ & bé, sửa xe và điện thoại.
- Rulebook public thêm hoặc đổi luật: ca tiền mặt, trung bình thứ trong tuần, khách trung thành, hàng ưa chuộng theo số đơn, nhà cung cấp theo số phiếu, topping chung ô, trả góp/thu máy cũ… Private có thể tiếp tục thay đổi. Đọc rulebook lúc inference, luôn kèm luật chung và luật phụ thuộc khi retrieval.
- README shop mô tả file/cột, quy đổi đơn vị, mốc cuối dữ liệu, cách ghi trả hàng và tồn. Dùng để tạo mapping, nhưng không coi mapping LLM là đúng nếu chưa đối soát.
- Câu hỏi có viết tắt/không dấu, câu kép hoặc kỳ so sánh; giữ nguyên câu gốc cạnh bản chuẩn hóa để tránh sửa nhầm tên/mốc thời gian. `calendar.csv` đúng split bắt buộc truyền vào phép tính âm lịch.
- Hồ sơ năng lực có ba trạng thái present/confirmed absent/unknown. Chưa thấy cột trong 20 dòng mẫu hoặc loader lỗi không chứng minh dữ liệu không tồn tại.

## 3. Pipeline và model

Profiler đọc README + file/sheet/header + mẫu → giữ bảng raw có nguồn dòng → loader/mapping có policy → bảng chuẩn + đối soát → câu hỏi/as_of_date + luật liên quan → LLM sinh code trên bảng/helper → executor giới hạn thời gian/tài nguyên → formatter → JSONL. Baseline raw code-agent là đối chứng và fallback. Verifier/ensemble là vòng sau khi có dữ liệu chứng minh lợi ích.

Bảng chuẩn dự kiến: orders, order_lines, items, purchases, payments/debts, stock, batches, expenses, shifts; chỉ tạo bảng có căn cứ từ nguồn. Giữ row count, keys, join cardinality, ngày/đơn vị/null, tổng tiền trước/sau khi có số đối chiếu. Không tự xóa duplicate hoặc forward-fill toàn cục. Shop cache complete chỉ khi bảng và checks hoàn tất. Helper phép toán tách khỏi policy rulebook: luật split quyết định doanh thu, đơn hợp lệ, nợ và tie-break.

Model ứng viên kế thừa research, phải kiểm revision/cutoff/license/số tham số và backend trước run:

| Vai trò | Ứng viên | Cách thử |
|---|---|---|
| Baseline nhỏ | Qwen/Qwen3-4B-Instruct-2507 | Một model chung cho hai kiến trúc đợt 1; cấu hình thay được |
| Model lớn đơn | Qwen/Qwen3-14B; tùy quota Qwen/Qwen2.5-Coder-14B-Instruct | Thử khoảng 14B gần trần sau baseline; 4-bit trên T4 nếu vừa |
| Trung gian | Qwen/Qwen3.5-9B | So PA/cost với baseline và 14B khi quota cho phép |
| Ensemble tùy chọn | 9B + 4B hợp lệ theo tổng tham số thực | Chỉ sau đối chứng một model mạnh; không mặc định tốt hơn |

Không bắt buộc dùng hết 15B; ưu tiên model đơn cho đến khi ensemble chứng minh tăng PA đủ bù thời gian. Thinking là ablation có token budget: off, bounded hoặc chỉ câu khó; không khóa off toàn bộ trước khi đo. Không dùng self-confidence làm xác suất đúng. Đổi precision/GPU phải đánh giá lại và hiệu chỉnh abstain.

## 4. Split, scorer và executor

Chia theo shop thành train-fit/dev-tune/final-holdout, cố định manifest/seed trước tuning. Bổ sung phép đo họ format/ngành/luật được giữ lại nếu số shop cho phép; cung cấp luật mới tại inference nhưng không dùng chúng để sửa prompt/helper trước đánh giá. Không tạo mapping từ nhãn shop holdout. Final chỉ dùng khi chốt shortlist. Public không có ground truth local; dùng kiểm vận hành và LB.

Scorer/formatter phải theo rulebook hiện hành và có fixture: number VND/count là số nguyên theo luật; số khác/percent làm tròn đúng hai chữ số theo quy tắc cung cấp, không chọn epsilon tương đối tùy tiện. Nếu cách xử lý điểm giữa chưa được nguồn xác nhận, ghi rõ giả định thay vì khẳng định tương đương scorer BTC. Boolean phải JSON bool, không nhận 1; NaN/Infinity không hợp lệ. String NFC/lowercase/chuẩn hóa dấu câu nhưng giữ dấu tiếng Việt, so alias/mã theo dữ liệu. Date ISO; list so từng vị trí và giữ phần tử trùng; table ghép cột theo quy tắc tên/vị trí và dòng theo set. Test rõ cột chỉ khớp một phần. `unanswerable.reason` là chuỗi; abstain không có answer bắt buộc.

Executor process riêng có timeout, RAM/output limit và phạm vi file cho shop/calendar/helper; không truy cập answers/nhãn hoặc credentials/network. Subprocess timeout chưa là sandbox filesystem. Nhãn chỉ nằm scorer ngoài executor. Baseline một lời giải, tối đa một lần sửa code do lỗi thực thi với ngân sách token/thời gian hữu hạn; log cả hai. Lỗi cuối → abstain, không bịa số 0. Thiếu dữ liệu chỉ unanswerable khi rulebook và dữ liệu xác nhận; unknown thử fallback hoặc abstain.

## 5. Experiments đợt đầu và vòng sau

Chuẩn bị notebook profiler/cache CPU; raw code-agent; chuẩn hóa + helper code-agent; model comparison dùng lại input/output contract. Hai kiến trúc dùng cùng model, manifest và budget để so. Có config public toàn bộ ngay từ baseline; không đợi model lớn/SFT hay phân tích sâu mới xuất submission đầu.

| Đợt / làn | Thay đổi/cấu hình | Đầu ra bắt buộc |
|---|---|---|
| 0 / CPU | Scorer/formatter/profiler, 3–5 shop phủ CSV/Excel/SQLite; thử resume | Checks, fixture results, manifest/cache và logs, không gọi là PA mô hình |
| 1C / C | Một model nhỏ, raw code-agent trên dev manifest cố định | Predictions, PA/coverage, code/rule trace, timing/VRAM |
| 1D / D | Cùng model/budget nhưng có chuẩn hóa/helper; dùng cache CPU chuẩn bị trước | Cùng schema output để so với C sau khi cả hai chạy |
| Public đầu | Một cấu hình baseline đã qua kiểm kỹ thuật; đủ mọi ID public | Submission hợp lệ để nộp, không yêu cầu PA đã cao |
| 2 / C+D | Sửa lỗi loader/helper/câu hỏi theo phân tích đợt 1 | Đối soát và PA theo shop/format/loại câu |
| 3 / D | Model nhỏ so 9B/14B, thinking budget; mỗi run đổi một yếu tố | PA, coverage, latency, lỗi code, GPU-hour và VRAM |
| 4 / C+D | Giữ luật/format mới; verifier có chọn lọc; abstain calibration | ΔPA và chi phí so với baseline tốt nhất |
| Cuối | Precision/backend/GPU đích, holdout và Docker | Tổng thời gian, tính tái lập và output hợp lệ |

Sau đợt 1 mới chọn model/kiến trúc theo kết quả. Không cần sửa hết mọi nhóm lỗi để có submission đầu. Model lớn nên thử đủ sớm 07–10/11 khi pipeline có baseline; không chờ loader hoàn hảo. Hai model đồng thuận trên một bảng sai không là bằng chứng đúng; verifier ưu tiên đường tính/đối soát độc lập. Gate abstain lấy tín hiệu kiểm tra và dev, không dựa vào câu model tự nhận chắc chắn.

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

## 6. Trace, public submission và phân tích sau chạy

Lưu từng câu dev: ID/shop, câu gốc và chuẩn hóa, rule IDs/text version, bảng nguồn/loader revision, code sinh, stdout/error, retry, verifier và lý do abstain; cùng answer/timing. Group lỗi chính: đọc dữ liệu/join, lọc đơn, doanh thu, thời gian, xếp hạng/đơn vị, nợ, kho/giá vốn, luật ngành mới, hiểu câu hỏi, format hoặc unanswerable. Loader lỗi gom theo shop vì có thể ảnh hưởng nhiều câu.

Public chọn split rõ, đọc rulebook/calendar của split đó, mọi shop tự động từ manifest. Dự kiến 1.000 câu nhưng số ID thực của input là chuẩn. Mỗi worker ghi shard riêng; merge kiểm đúng IDs và schema. Pending ở intermediate checkpoint phải còn chạy tiếp; chỉ điền abstain cho câu chưa xử lý khi xuất bản submission cuối chủ động chấp nhận coverage đó.

- [ ] Đủ ID, không trùng; từng dòng UTF-8 JSONL có id/type/answer; unanswerable có reason, abstain đúng schema.
- [ ] Number/percent/date/boolean/list/table đúng kiểu, quy tắc làm tròn/thứ tự; không đơn vị hoặc ký tự % trong số.
- [ ] Validator toàn file; nhãn không lọt vào model/executor. Lỗi từng câu có log và không làm sập toàn pipeline.
- [ ] Lưu `submission.jsonl` trong output Kaggle để người dùng tải về nộp Public Leaderboard; chưa cần Docker hoàn chỉnh.
- [ ] Public baseline hợp lệ có thể nộp ngay; log điểm/version rồi phân tích sau.

## 7. Thời gian máy chấm và lịch chốt

Kaggle 12h là budget phát triển, khác 6h trên máy BTC. Với N thực, tính ngân sách sau load/preprocess/export và safety margin; chọn retry/verifier/token budget theo throughput. Không lấy 1.000 câu làm số private chắc chắn. Cache shop thay vì parse mỗi câu; tránh CPU offload lớn khi CPU/RAM/I/O hạn chế.

Baseline 31/10–03/11; loader/model experiments 04–07/11; thử model/precision/backend trên GPU đích 07–10/11; nghiệm thu Docker hoàn chỉnh 12–14/11; nộp dự phòng 15/11 theo timeline repo. Chỉ việc đã có output mới đánh dấu hoàn tất.

Docker sau public baseline phải có dependencies/weights offline, backend Blackwell tương thích CUDA theo spec, network-none, input read-only `/data`, tạm `/tmp`, kết quả ghi tăng dần `/result/submission.jsonl`. Có code train nếu sử dụng/infer, hướng dẫn, checksum/link theo quy chế. Test load + toàn pipeline dưới 6h với buffer; chạy lại kiểm PA lệch trong giới hạn. Đây là nghiệm thu riêng, không suy từ điểm trên T4.

## 8. Kết quả và nguồn tham khảo

Lịch sử run, điểm, account, log nộp Public LB: [`experiments/INDEX.md`](experiments/INDEX.md) (mỗi run một card). Kiến thức đúc kết: [`insights.md`](insights.md).

Nguồn tham khảo để kiểm chứng: [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B), [Qwen2.5-Coder-14B](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct), [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), [SemEval DataBench overview](https://aclanthology.org/2025.semeval-1.324/), [T4 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet.pdf). Benchmark ngoài không thay điểm cuộc thi; các phát hiện khảo sát cũ cần kiểm trên input version thực tế.
