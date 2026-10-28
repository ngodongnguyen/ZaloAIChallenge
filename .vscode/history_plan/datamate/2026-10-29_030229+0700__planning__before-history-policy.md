# DataMate — Planning triển khai, experiments và Public Test

**Mục tiêu:** nhắm Top 1 bằng pipeline đọc dữ liệu đúng, áp dụng đúng rulebook và tính toán có kiểm chứng. Đầu ra đầu tiên là notebook experiments và `submission.jsonl` hợp lệ cho Public Test. Chạy đợt đầu lấy kết quả trước, sau đó mới phân tích để chọn kiến trúc/model và tối ưu.

**Trạng thái:** đang làm baseline, chưa chạy GPU. Ngày 2026-10-29 người dùng chọn đo baseline thật trên dev trước: `dm-D-001-baseline-dev` (mục 9–10), thay ưu tiên public-first trước đây. Repo chỉ giữ một notebook baseline tại `notebook-kaggle/datamate/helper-agent/datamate_helper_agent.ipynb`; notebook profiler riêng (Làn 0) và raw code-agent đối chứng (Làn C) tạm cất, làm lại khi cần so kiến trúc. Đây là file kế hoạch duy nhất của track.

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

## 9. Thiết kế experiment kế tiếp — 2026-10-29

**Phạm vi đã chọn:** một experiment LifeMate và một experiment DataMate, đo model thật từ notebook hiện có. Chỉ lập kế hoạch trong đợt này; chưa sửa notebook, chạy GPU hoặc nộp Public LB.

**Experiment DataMate:** `dm-D-001-baseline-dev`. Câu hỏi: helper-agent Qwen3-4B hiện có tạo PA dương trên shop dev chưa dùng để tuning không, và điểm mất chủ yếu do loader, code, nghiệp vụ hay format?

- Dùng `notebook-kaggle/datamate/helper-agent/datamate_helper_agent.ipynb`, giữ kiến trúc `helper`, model Qwen3-4B hiện tại. Đây là lần đo baseline model thật đầu tiên; không coi chênh lệch với mock round-trip là cải thiện PA.
- Cố định `train/dev_tune`, manifest seed 2026, 6 shop do `dmlib.split.make_manifest` chọn, lấy mọi câu của các shop đó. Lưu danh sách shop/ID thực và độ phủ format/ngành trước inference; không tự bù từ final khi thiếu nhóm hiếm. `train_fit`/`final_holdout` chưa chạy model.
- Giữ FP16, greedy, seed 0, `max_model_len=16384`, output 1536 token, executor 60 giây/6144 MB, một executor mỗi worker GPU, tối đa một repair, `rules_k=8`, `context_chars=9000`. Có hai worker GPU thì tổng executor đồng thời là 2; không nhầm giới hạn mỗi worker với toàn session.
- Dùng rulebook/calendar/README của train. Profiler cache phải khớp input/rulebook/loader fingerprint; marker `COMPLETE` tự nó chưa đủ. Giữ lỗi loader và nguồn dòng trong trace, không biến lỗi parse thành bằng chứng `unanswerable`.
- Tách nhãn khỏi prompt, cache và không gian file đọc được của executor; regex chặn tên `answers.jsonl` hiện có chưa chứng minh cách ly filesystem. Chuẩn bị kiểm chứng trước run để không đánh giá trên câu trả lời bị rò nhãn.
- Metric chính PA = (correct − wrong)/N, kèm C/W/A, coverage = (C+W)/N, accuracy trên câu có trả lời, theo shop/gold type và nguồn abstain. Báo cả PA của phương án luôn abstain (=0) làm mốc; code chạy thành công chưa chứng minh đáp án đúng.
- Scorer phải chấm chính record được xuất sau validate/format; không chấm record lỗi gốc trong khi submission đã thay nó bằng abstain. Partial report dùng đủ N và tách pending khỏi abstain chủ động; baseline cuối cần pending=0 và đủ mọi ID.
- Phiên đầu 3–4 giờ **toàn notebook**, gồm setup/profile/load/infer/merge/export; checkpoint trước hết giờ, `finalize=False` ở dev để pending còn resume. Không tự đổi helper/prompt/rules hoặc model sau khi xem từng phần điểm; sửa lỗi làm đổi dự đoán cần lineage/run mới.

**Các cách đã cân nhắc:** (1) đo baseline thật trên notebook hiện có — người dùng đã chọn; (2) thêm raw code-agent đối chứng — vòng sau, dùng cùng manifest/model/budget khi helper đã có số đo. Model lớn/thinking/verifier cũng để sau phân tích lỗi này.

**Đầu ra cần implement:** preset dev, kiểm manifest/cache và cách ly nhãn, scorer khớp submission, thống kê PA/lỗi/vận hành. Thiết kế đã được người dùng duyệt trong phiên 2026-10-29; checklist triển khai ở mục 10. Public là lần chạy riêng sau đó với run_id/card mới; không tự nộp LB trong đợt planning.

## 10. DataMate helper baseline — Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` to implement task-by-task. Chỉ dùng subagent-driven-development nếu người dùng chọn cách đó. Các checkbox đều chưa thực hiện; không commit/push nếu chưa được yêu cầu.

**Goal:** đo PA đầu tiên của helper-agent model thật trên dev shop cố định, đồng thời định vị lỗi đáng sửa tiếp theo.

**Architecture:** thêm preset `baseline_dev` cho notebook helper hiện có. Profiler/cache kiểm fingerprint trước inference; executor chỉ đọc dữ liệu được cấp; merge chấm đúng các record đã validate và xuất. Không thay thuật toán chuẩn hóa/luật/prompt để tăng điểm ngay trong run.

**Tech Stack:** Python trong Kaggle notebook, pandas/calamine/parquet/SQLite, Qwen3-4B FP16 qua một backend đã preflight, executor subprocess, scorer JSONL. Linux Landlock là phương án cách ly file được đề xuất ở D2; hỗ trợ kernel Kaggle phải được đo trước chạy GPU.

**Spec:** thiết kế mục 9; `.vscode/docs/TRACK2_DATAMATE_SPECIFICATION.md`, `OFFICIAL_METRICS_AND_MATCHING_RULES.md`; lịch sử [`dm-D-000`](experiments/2026-10-29_dm-D-000-local-dryrun.md).

### Global Constraints

- Theo đặc tả repo: tổng model ≤15B, checkpoint công khai trước 01/10/2026 và license phù hợp; máy đích 2 RTX 5090, tổng ≤6 giờ. Run 2 T4 chỉ là số đo phát triển, không xác nhận GPU đích.
- Giữ `Qwen/Qwen3-4B-Instruct-2507`, resolve commit revision trước chạy; FP16, không quantization, greedy/seed 0, thinking off. Preflight chọn và lưu `vllm` hoặc `hf`, không silent fallback sau đóng băng; không dùng mock để báo PA baseline.
- `split=train`, `subset=dev_tune`, `manifest_seed=2026`, tất cả câu thuộc đúng 6 shop dev. Không dùng nhãn để đổi shop hoặc chọn câu dễ; giữ final_holdout nguyên vẹn. Manifest không chứa answer/alias.
- Agent: max output 1536, context 16384, repair tối đa 1, executor 60 giây/6144 MB, `exec_workers=1` mỗi GPU worker, `rules_k=8`, shop context 9000 ký tự. Main batch 16, HF batch 4; hai worker tương đương tối đa hai executor đồng thời, kiểm RAM thực trước batch lớn.
- `session_wallclock_s=14400`, `export_reserve_s=900`, `finalize=False` cho dev. Deadline chung tính cả setup/profile/load/infer/merge/export; checkpoint từng ID. Resume giữ config/hash/số shard và không trộn cache khác version.
- Mỗi Kaggle execution/version (kể cả resume/failure) có run card mới với NNN kế tiếp khi chạy; phân biệt `done` của một execution với `baseline_complete` của cả manifest. Chỉ Markdown ở planning; không commit dữ liệu/weights/predictions.

### Review Focus

| Điều kiện dễ làm sai số đo | Kiểm tra thuộc task |
|---|---|
| Cache có COMPLETE nhưng input/rulebook/loader đã đổi | D1: attach cache khác fingerprint phải bị từ chối hoặc build mới |
| Code đọc nhãn bằng đường dẫn ghép chuỗi | D2: kiểm file access thực, không chỉ regex tên answers |
| Loader lỗi bị hiểu thành dữ liệu không tồn tại | D2/D4: lỗi loader được phân loại riêng, không tự gán unanswerable |
| Submission thay record lỗi bằng abstain nhưng scorer chấm record cũ | D3: score và checksum gắn đúng record được xuất |
| PA partial che giấu pending hoặc shop nặng chưa chạy | D3/D4: đủ mẫu số N, báo pending và progress từng shop |

### Task D1: Đóng băng dev, preset và cache provenance

**Files:** sửa CONFIG/setup và các cell `dmlib/split.py`, `dmlib/profiler.py`, `dmlib/common.py`, `dmlib/llm.py`; thêm cell regression assertions trong `datamate_helper_agent.ipynb`.

**Interfaces:** thêm `prepare_dev_run(cfg: dict, questions: list[dict]) -> dict` trong `dmlib/split.py`, output `manifest.json` gồm shop/ID từng part và hashes input/code/manifest; thêm `verify_shop_cache(shop_dir: str, cache_dir: str, expected: dict) -> dict` trong `dmlib/profiler.py`. `expected` chứa split, source SHA256, rulebook SHA256 và loader/code SHA256.

- [ ] Viết fixtures: 30 shop → đúng 6 dev, 6 final, 18 fit, không giao nhau; seed/input cố định cho manifest cố định; không thêm câu final khi dev thiếu kiểu table/boolean. Lần đầu fail vì chưa có API mới.
- [ ] Thêm `preset="baseline_dev"`, `experiment="dm-D-001-baseline-dev"`, `mode="helper"`, `limit=None`, freeze ID/shop; lưu bảng n câu theo shop, format nguồn và ngành từ README, chưa đọc gold type để chọn câu.
- [ ] Kiểm cache attach bằng nội dung nguồn/rulebook/code; marker COMPLETE hiện được kiểm khi tự build nhưng nhánh dùng cache attach chỉ kiểm tồn tại. Cả hai đường phải dùng cùng verifier, ghi `cache_manifest.json`; fingerprint mismatch → build thư mục mới, không overwrite cache input read-only.
- [ ] Chuẩn bị smoke manifest 12 ID train_fit (round-robin shop rồi sort ID) và thư mục smoke riêng, không chấm để tune. **Chỉ chạy smoke model ở D4 sau khi guard D2 và scorer D3 qua**, để lỗi cách ly file được phát hiện trước load model. Không thay policy làm sạch vì nhìn thấy lỗi điểm ở dev.
- [ ] Chạy lại regression cell; expected manifest/cache checks pass. Ghi package/GPU/CPU/RAM/source version thực vào `preflight.json`; model ID vẫn chưa là bằng chứng cutoff/license, phải kiểm nguồn checkpoint trước run.

### Task D2: Tách label khỏi executor và kiểm dữ liệu trước khi trả lời

**Files:** cell `dmlib/executor.py`, `dmlib/agent.py`, `dmlib/profiler.py`; thêm cell nhúng `dmlib/fs_guard.py` và regression fixtures.

**Interfaces:** thêm `restrict_executor_files(read_paths: list[str], scratch_dir: str) -> dict` trong `dmlib/fs_guard.py`, trả ABI/policy digest khi cài thành công, raise `RuntimeError` nếu không enforce được. `run_code(...)` giữ contract, thêm metadata policy trong trace; áp policy ở runner trước preload/code, không làm syscall phức tạp trong `preexec_fn` của process spawn từ thread.

- [ ] Viết fixture runner: đọc CSV/shop, calendar, parquet cache hợp lệ phải thành công; đọc label sentinel qua `Path(DATA_DIR).parents[...] / ('ans'+'wers.jsonl')`, đọc shop khác hoặc symlink trỏ ra ngoài phải nhận PermissionError. Không đặt sentinel trong allowlist runtime.
- [ ] Dùng Landlock filesystem allowlist trong child, `no_new_privs`, close inherited FDs ngoài stdio. Cấp read đúng shop/calendar/cache và thư viện Python/runtime cần thiết; scratch riêng được write; không cấp toàn `/kaggle/input`, `/kaggle/working`, home hoặc `/proc`. Không cho phép read/write data gốc. Chọn rights theo ABI thực; unsupported → preflight failed trước load model, không gọi regex là sandbox. Căn cứ API: [Linux kernel Landlock documentation](https://docs.kernel.org/userspace-api/landlock.html); chưa khẳng định Kaggle hỗ trợ.
- [ ] Giữ static guard hiện có như lớp kiểm code; không dùng guard để khẳng định cách ly OS. Kiểm runtime imports/pandas/calamine/parquet thật trong môi trường Kaggle để allowlist không làm mọi câu thành abstain.
- [ ] Prompt/executor chỉ nhận question/as_of/shop data/rulebook, không answers/alias/gold type. Worker inference không load `answers.jsonl`; scorer nạp sau khi worker kết thúc. Các source path và lỗi parser/capability unknown phải còn trong profile/trace.
- [ ] Test không có cột trong sample hoặc lỗi parse không tự thành `unanswerable`: loader incident được báo riêng; code không chạy sau tối đa 1 repair → abstain có reason trong trace. Không thêm heuristic đổi label trong baseline.

### Task D3: PA đúng với file xuất và trạng thái pending

**Files:** cell `dmlib/merge.py`, `dmlib/answer.py`, `dmlib/run.py` và regression assertions.

**Interfaces:** giữ `merge(...)`; thêm `canonical_records(recs: dict, ids: list[str], finalize: bool) -> tuple[dict, dict]` trong `dmlib/merge.py`. Dict đầu là record sau validate/thay lỗi bằng abstain dùng **chung** cho JSONL và scorer; dict thứ hai là invalid/pending metadata. Report thêm `baseline_complete`, `pending`, `abstain_explicit`, `abstain_after_error`, `invalid_replaced`, `prediction_sha256`.

- [ ] Fixture N=4: 2 correct, 1 wrong, 1 abstain → PA=.25, coverage=.75, accuracy_answered=2/3. Luôn abstain → PA=0; tất cả wrong → −1. Pending không biến mất khỏi mẫu số và được báo riêng khỏi abstain đã xử lý.
- [ ] Fixture một prediction bool=1 không hợp lệ bị thay abstain → chấm 0 trên canonical record; không chấm −1 của prediction gốc. `finalize=False` + pending > 0 → không ghi submission cuối, complete=false.
- [ ] Validate đủ 9 kiểu: number/percent hữu hạn, bool thuần, date ISO, list có thứ tự/phần tử trùng, table theo rule names/fallback/set; giữ nhãn unanswerable khác abstain. Round-trip 3000 gold cũ chỉ là fixture formatter, không thay các test sai type/format này.
- [ ] Merge kiểm header hash và tập ID, không nhận completed duplicate mâu thuẫn/ID ngoài manifest. Lưu `per_id_scores.jsonl`, metric C/W/A/N và theo shop/gold type; group thiếu n=0/NA. `type_counts` dựa canonical records.
- [ ] Chạy regression fixtures, expected mọi assertion pass; serialize→load JSONL→score cho cùng metric. Scorer local vẫn báo các giả định matcher còn ở insights, chưa gọi là scorer BTC tương đương tuyệt đối.

### Task D4: Chạy có kiểm soát và phân tích lỗi baseline

**Files:** cell run/merge; `dmlib/run.py`, `dmlib/agent.py`; thêm cell nhúng `dmlib/analysis.py`.

**Interfaces:** thêm `summarize_baseline(out_dir: str, manifest_path: str) -> dict` trong `dmlib/analysis.py`, tạo `baseline_report.json` và `error_groups.jsonl`. Output gốc `/kaggle/working/dm_baseline`; giữ config/manifest/cache fingerprints cùng predictions/traces.

- [ ] Test resume bằng fixture mock: gián đoạn sau một batch và dòng cuối hỏng, chạy tiếp cho cùng tập ID/record; config/code/cache đổi → từ chối. Hai shard không ghi chung một file và merge không phụ thuộc thứ tự file. `run_hash` hiện bỏ qua `num_shards`; thêm kiểm riêng checkpoint manifest để từ chối đổi số shard khi resume.
- [ ] Sau D2/D3 mới chạy smoke D1 bằng model thật: loader có bảng, executor policy hiệu lực, backend/revision cố định, memory đủ. Smoke fail là lỗi kỹ thuật, không là PA baseline; giữ artifact/lý do trước khi chạy toàn dev.
- [ ] Driver dùng deadline chung từ setup, log thời gian profile/load/generation/execute/repair/export; dừng nhận batch sớm đủ thời gian tối đa một batch + repair (ước lượng từ smoke), không đợi hết 4h mới export. Giữ safety reserve 900s tối thiểu.
- [ ] Log per-question prompt/output token, rules thực chọn, file/sheet/row lineage, code/stdout/error cả hai attempt, reason abstain, backend/hash. Poll GPU memory theo process/device để lấy peak; snapshot cuối run hiện có chưa phải peak VRAM.
- [ ] Error report có shop-wide loader failures, runtime/syntax/timeout/context-too-long/truncated output, format, và các sai nghiệp vụ phải rà bằng trace + gold sau inference. Một câu gán một nguyên nhân chính để không cộng trùng, có thể thêm tag phụ; đếm tổng và 2–3 ID ví dụ mỗi nhóm.
- [ ] Baseline complete cần pending=0, đúng tất cả ID, config/provenance hợp lệ, scorer khớp JSONL và fixture cách ly nhãn pass. Coverage thấp hay PA âm vẫn báo trung thực, không sửa prompt/helper ngay trong run để xóa kết quả xấu.

### Task D5: Vận hành, chọn mốc và quyết định experiment vòng sau

**Files:** [`run card planned`](experiments/2026-10-29_dm-D-001-baseline-dev.md), [`INDEX`](experiments/INDEX.md), [`insights`](insights.md), `notebook-kaggle/datamate/README.md`, `helper-agent/kernel-metadata.example.json` nếu có field mới.

- [ ] Chạy regression cell trong Kaggle sau materialize module, expected `REGRESSION_OK`; smoke FS/loader/backend trước batch lớn. Local syntax/fixture không tính là PA model thật. Cập nhật README preset, attach cache và resume.
- [ ] Khi thực thi mới đổi planned → running, chọn account alias theo AGENTS.local.md, điền version/revision/input/code/config hashes. Theo quy trình hiện có chỉ upload private, chưa auto-run; người dùng khởi chạy khi triển khai được yêu cầu. Không dùng username trong file tracked.
- [ ] Khi hết phiên: lưu output version, đo GPU-hours thật, cập nhật card/INDEX cả khi lỗi hoặc pending. Phiên resume tạo card mới và liên kết parent, không đổi cấu hình để cứu điểm giữa chừng.
- [ ] Khi đủ dev: INDEX ghi PA, C/W/A, coverage, n câu/shop và artifact version; thêm insights có nguồn thật. Nếu PA≤0, luôn-abstain là mốc 0 cho so sánh và ưu tiên tìm lỗi trước public, không tuyên bố helper đã tốt hơn.
- [ ] Chọn bước tiếp từ số đo: lỗi loader tập trung → sửa loader; business/time/ranking sai → sửa helper/rule retrieval; code generation sai → raw-agent/model comparison cùng manifest; nhiều wrong có tín hiệu kiểm chứng → thử abstain gate. Mỗi nhánh là run mới một thay đổi; chưa chạy raw/model lớn/verifier/holdout/Public LB trong experiment này.
