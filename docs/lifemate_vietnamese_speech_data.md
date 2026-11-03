# Nguồn dữ liệu âm thanh tiếng Việt đa vùng miền cho LifeMate

Ngày kiểm tra nguồn: **2026-11-03**. Phạm vi: **LifeMate**, bổ sung dữ liệu cho ASR; không áp dụng cho DataMate. Đây là ghi chú nguồn và hướng cân nhắc, chưa phải kết quả thử nghiệm với dữ liệu ngoài. Mọi run và kết quả tiếp tục ghi trong [tracking LifeMate](../.vscode/history_plan/lifemate.md).

## 1. Cần loại dữ liệu nào?

Huấn luyện ASR cần cặp **audio → transcript chính xác**, không cần nhãn tool call. Vì vậy có thể tận dụng các bộ tiếng Việt ngoài cuộc thi để mở rộng người nói, phương ngữ và điều kiện thu âm.

Tuy nhiên, độ đa dạng vùng miền và độ phù hợp nội dung là hai yếu tố riêng:

- Audio thật đa vùng miền giúp bổ sung biến thể phát âm, giọng nói và cách nói.
- Câu lệnh đúng miền LifeMate giúp luyện tên người, địa chỉ, số điện thoại, số tiền, ngày giờ, tên bài hát và yêu cầu nhiều bước.
- Dữ liệu tin tức hoặc podcast có thể bổ sung giọng nhưng không tự thay thế được dữ liệu lệnh trợ lý.

## 2. Những gì dự án đã có

- Kiểm tra local ngày 2026-11-03: train có **1.545 mẫu**, **5.036 file WAV**, tổng khoảng **6,83 giờ**, tính cả các lượt audio lịch sử.
- Run `lm-B-022-oracle-transcript` lần 2: agent trên 417 mẫu dev đạt **36,63 điểm** với ASR Gipformer cũ và **63,13 điểm** khi dùng transcript chuẩn. Đây là chẩn đoán ảnh hưởng của ASR trong cấu hình đó.
- Run `lm-B-033-asr-ft-agent-dev`: cùng checkpoint agent và cấu hình đánh giá, ASR Gipformer FT đạt **44,12 điểm**, Whisper-medium FT đạt **51,19 điểm**. So với mốc ASR cũ là khác run; so Whisper FT với Gipformer FT là trong cùng run.
- Run `lm-A-034-whisper-full-public`: Public LB **48,2 theo user**. **51,19 là điểm dev, không phải điểm public**.

Nguồn kết quả: [tracking LifeMate](../.vscode/history_plan/lifemate.md), các run 022, 033 và 034. Các kết quả này hỗ trợ việc thích nghi ASR với dữ liệu cuộc thi; chưa chứng minh thêm dữ liệu ngoài sẽ tăng điểm.

## 3. Nguồn dữ liệu đã kiểm tra

### ViMD — ưu tiên khảo sát về phương ngữ

- Nguồn: [nguyendv02/ViMD_Dataset](https://huggingface.co/datasets/nguyendv02/ViMD_Dataset).
- Theo dataset card: **102,56 giờ**, **18.949 đoạn**, bao phủ **63 phương ngữ tỉnh** theo thiết kế bộ dữ liệu.
- Có transcript và metadata vùng `North` / `Central` / `South`, tỉnh, người nói, giới tính.
- Nguồn thu thập: chương trình của các đài phát thanh/truyền hình địa phương; không phải bộ chuyên về câu lệnh trợ lý.
- Metadata license: **CC BY-NC-ND 4.0**; tác giả mô tả phục vụ nghiên cứu. Cần làm rõ quyền sử dụng cho bài thi và các nghĩa vụ liên quan trước khi đưa vào pipeline hoặc phân phối dữ liệu/checkpoint; không coi việc tải công khai là xác nhận đã đủ quyền.
- Bài giới thiệu gốc: [Multi-Dialect Vietnamese, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.426/).

### Bud500 — nguồn lớn có giọng ba miền

- Nguồn: [linhtran92/viet_bud500](https://huggingface.co/datasets/linhtran92/viet_bud500).
- Dataset card mô tả khoảng **500 giờ** audio tiếng Việt với transcript; có giọng Bắc, Trung, Nam và nhiều chủ đề như podcast, du lịch, sách, ẩm thực.
- Có điều kiện truy cập (gated): cần đăng nhập và chấp nhận điều kiện của chủ dữ liệu.
- **License đang không nhất quán trên trang**: metadata ghi `cc-by-nc-sa-4.0`, mục License trong nội dung card lại ghi Apache 2.0; card cũng nói dữ liệu dành cho nghiên cứu. Cần xác nhận điều khoản thực sự áp dụng trước khi dùng; không chọn cách diễn giải có lợi hơn khi chưa rõ.
- Chưa tải và chưa đánh giá tác động lên LifeMate. Không cần tải toàn bộ để bắt đầu khảo sát một phần train.

### FLEURS tiếng Việt — nguồn có license rõ hơn

- Nguồn: [google/fleurs](https://huggingface.co/datasets/google/fleurs), cấu hình tiếng Việt **`vi_vn`**.
- Có audio và transcript; license metadata **CC BY 4.0**.
- Card mô tả các tập train theo ngôn ngữ có khoảng 10 giờ giám sát; đây là mô tả chung, chưa đo riêng thời lượng tiếng Việt trong repo này.
- Chủ yếu là đọc câu. Không coi đây là bộ chuyên về phương ngữ hoặc hội thoại tự nhiên.
- Có thể dùng split train để khảo sát bổ sung; giữ nguyên các split dùng làm benchmark nếu muốn báo cáo kết quả trên benchmark đó.

### VIVOS — nguồn bổ sung nhỏ

- Nguồn: [AILAB-VNUHCM/vivos](https://huggingface.co/datasets/AILAB-VNUHCM/vivos).
- Dataset card mô tả **15 giờ** tiếng Việt; có `speaker_id`, audio 16 kHz và transcript `sentence`.
- License metadata: **CC BY-NC-SA 4.0**; cần kiểm điều kiện phù hợp với mục đích bài thi và nghĩa vụ khi phân phối.
- Không có căn cứ từ card đã kiểm để khẳng định bộ này đại diện đầy đủ các vùng miền.

Các thông tin trên lấy từ repository/card của chủ dữ liệu, kiểm tra ngày 2026-11-03. Chưa xác nhận các nguồn này bổ sung bao nhiêu dữ liệu mới so với dữ liệu pretraining của ASR hiện tại.

## 4. Các cách bổ sung sát LifeMate

### Audio người thật

Một quy mô thu thử được đề xuất là **15 người × 100 câu mới = 1.500 audio**, với người nói từ nhiều vùng. Đây là ví dụ khởi đầu, không phải mức tối ưu đã kiểm chứng.

- Viết câu lệnh mới từ phạm vi tác vụ và train được phép dùng; ưu tiên nhóm lỗi đã quan sát trên train/dev.
- Nội dung nên có số tiền, giờ/ngày, số điện thoại, tên riêng, địa danh, tên tiếng Anh và yêu cầu nhiều bước.
- Có cả đọc câu và diễn đạt tự nhiên; transcript phải khớp lời thực sự nói, kể cả khi khác kịch bản.
- Thu metadata người nói/vùng giọng để kiểm tra độ phủ; có sự đồng ý phù hợp của người thu âm.
- Chia train/validation theo người nói khi có thể để tránh đo khả năng nhớ giọng thay vì tổng quát hóa.

### TTS cho nội dung mới

- Sinh câu lệnh mới và transcript, dùng nhiều giọng TTS để tạo audio bổ sung trong giai đoạn phát triển.
- Giọng TTS là nguồn bổ sung; cần đánh giá trên audio người thật để biết có cải thiện nhận dạng phương ngữ hay không.
- Đổi cao độ hoặc tốc độ của cùng một giọng **không tạo ra phương ngữ mới**.

### Tăng cường audio sẵn có

Có thể cân nhắc thêm nhiễu, vang, thay đổi âm lượng hoặc tốc độ ở mức giữ được nội dung. Đây là hướng thử, chưa được đo trong ghi chú này. Tránh biến đổi mạnh làm mất phụ âm/dấu thanh nhưng vẫn giữ transcript cũ như thể nhãn còn chính xác.

## 5. Cách quyết định nguồn nào đáng dùng

- Giữ một baseline và tập đánh giá ổn định từ train cuộc thi; không dùng checkpoint đã train cả dev để báo cáo điểm dev.
- Khảo sát một phần dữ liệu ngoài đã rõ quyền sử dụng trước; chưa ấn định tỷ lệ trộn hoặc khẳng định cần toàn bộ corpus.
- Theo dõi WER cùng lỗi tên riêng, số tiền, ngày giờ và **điểm hành động của agent**. Run 033 cho thấy ASR có WER thấp hơn chưa chắc tạo điểm hành động cao hơn.
- Nếu có metadata vùng/người nói, tách kết quả theo nhóm để tránh mức trung bình che giấu nhóm còn yếu.
- Chỉ mở rộng dữ liệu sau khi có bằng chứng trên đánh giá hợp lệ; không suy ra hiệu quả chỉ từ số giờ audio.

## 6. Ranh giới public/private test

Theo [LifeMate → Rule → Technical requirements → Allowed data and resources](https://challenge.zalo.ai/portal/lifemate):

- Được sinh dữ liệu tổng hợp bằng dịch vụ/model khác trong development theo điều kiện của track.
- Không được gán nhãn `public_test` / `private_test`, bằng tay hoặc model/dịch vụ bên ngoài, để huấn luyện hay tinh chỉnh giải pháp.
- Vì vậy, không lấy audio public/private test để chép transcript rồi đưa vào dữ liệu train ASR, kể cả gọi đó là pseudo-label hoặc synthetic.
- Hệ thống dự thi phải suy luận offline và tuân thủ giới hạn model; quyền dùng dịch vụ để sinh dữ liệu mới không đồng nghĩa với quyền gọi dịch vụ ngoài khi inference.

Thể lệ có thể được BTC cập nhật; trước khi sử dụng nguồn mới, đối chiếu [thể lệ chung](https://dl-challenge.zalo.ai/zalo-ai-challenge-regulations-vi.pdf) và tab Rule chính thức của LifeMate.

## 7. Model có sẵn để clone giọng từ audio mẫu

Bổ sung ngày **2026-11-03**. Bài toán cần tìm là **zero-shot voice-cloning TTS**: audio mẫu của người nói + văn bản mới → audio mới có giọng tương tự. Không cần huấn luyện lại một model riêng cho từng người để thử khả năng này.

Đã kiểm trực tiếp model card và Hugging Face API. Chưa chạy weights hoặc đo chất lượng clone/phương ngữ trong dự án. Chưa có benchmark chung về clone giọng tiếng Việt được xác minh để xếp hạng các ứng viên; thứ tự thử dưới đây là đề xuất theo tính năng và độ thuận tiện.

### Các ứng viên chính

| Model | Audio mẫu / transcript | Điểm phù hợp | License và giới hạn đã kiểm |
|---|---|---|---|
| [pnnbao-ump/VieNeu-TTS-v3-Turbo](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo) | Theo tài liệu: **3–8 giây**, **không cần transcript mẫu** ở v3 | Tiếng Việt/Anh, có đường CPU ONNX và GPU; nên thử trước vì không cần chép lời audio reference | Card ghi **Apache 2.0**; chưa xác nhận chất lượng clone thực tế trên từng phương ngữ |
| [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) | Tài liệu khuyến nghị **3–10 giây**; có thể truyền `ref_text`, hoặc để SDK dùng Whisper tự chép lời | Có tiếng Việt, zero-shot clone, lưu/reuse voice prompt; phù hợp làm đối chứng với VieNeu | **Mã nguồn Apache 2.0, weights CC-BY-NC** theo mục License của model card; không gọi chung cả hai là Apache |
| [hynt/F5-TTS-Vietnamese-ViVoice](https://huggingface.co/hynt/F5-TTS-Vietnamese-ViVoice) | Ví dụ chính thức dùng `ref_audio` + `ref_text` + `gen_text` | Bản F5-TTS fine-tune trên khoảng 1.000 giờ tiếng Việt theo tác giả; có checkpoint tải sẵn | **CC BY-NC-SA 4.0**, card giới hạn nghiên cứu phi thương mại; chưa xác nhận số tham số từ checkpoint `.pt` |
| [capleaf/viXTTS](https://huggingface.co/capleaf/viXTTS) | Theo card: clip khoảng **6 giây** | XTTS-v2 đã mở rộng tiếng Việt, có demo và notebook | **Coqui Public Model License (CPML)**; tác giả ghi chất lượng kém với câu tiếng Việt **dưới 10 từ**, là hạn chế đáng chú ý cho câu lệnh ngắn |

Hướng dẫn gốc: [VieNeu voice cloning](https://docs.vieneu.io/docs/sdk/voice-cloning/) và [OmniVoice voice cloning](https://github.com/k2-fsa/OmniVoice#voice-cloning).

### Metadata và revision đã kiểm

Các số tham số dưới đây là số phần tử tensor từ metadata `safetensors` của HF, không phải phép đếm toàn bộ pipeline bao gồm codec, speaker encoder hoặc ASR tự động. Ngày tạo repo không tự chứng minh thời điểm công khai đầu tiên; các repo chính đều có cập nhật cuối trước mốc 26/10/2026.

| Model | Repo tạo ngày | Cập nhật cuối | Số tham số HF | Revision | File weights thấy trong metadata |
|---|---|---|---|---|---|
| VieNeu v3 Turbo | 2026-06-05 | 2026-09-23 | 130.907.520 | `61b85e3d937fbbacb387714180e8182823512523` | `model.safetensors`, `speaker_encoder.onnx`, các graph ONNX |
| OmniVoice | 2026-03-30 | 2026-07-03 | 612.577.288, gồm 8 phần tử I64 | `c5fdb5ccb189668d56333f77ba2629f4cd7535f4` | `model.safetensors`, `audio_tokenizer/model.safetensors` |
| F5-TTS-Vietnamese-ViVoice | 2025-07-10 | 2026-09-27 | Chưa xác nhận; không suy từ tên kiến trúc | `5c14cdd4e3f62b69e75140558bea2b2e293ee8f1` | `model_last.pt` |
| viXTTS | 2024-04-04 | 2024-04-07 | Chưa xác nhận | `c06f4378883110615941aab481532a9802440b05` | `model.pth` |

Nguồn metadata: API `https://huggingface.co/api/models/<model-id>` tương ứng các link model phía trên. Không tải checkpoint `.pt`/`.pth` để đếm tham số trong lượt tra cứu này.

### Hai biến thể khác đã tìm thấy

- [splendor1811/omnivoice-vietnamese](https://huggingface.co/splendor1811/omnivoice-vietnamese): card mô tả fine-tune OmniVoice trên Dolly-Audio, có API clone từ reference. Repo tạo 2026-04-05, cập nhật 2026-04-09; revision `f70093f90d78a5b790ff0f066b3629ebcc33cb5d`, HF safetensors 612.577.288 phần tử. **Card ghi Apache 2.0 nhưng upstream OmniVoice ghi weights CC-BY-NC**, dataset cũng phi thương mại; cần làm rõ quyền kế thừa, không mặc định bản fine-tune đã bỏ hạn chế upstream.
- [thivux/XTTS-v2-vietnamse](https://huggingface.co/thivux/XTTS-v2-vietnamse) (slug chính xác có chữ `vietnamse`): checkpoint XTTS-v2 fine-tune bằng PhoAudiobook, ví dụ chạy nhận `speaker_audio`. Repo tạo 2025-12-30, cập nhật 2026-01-03; revision `c39fa77ae8ea8513f1ab0ac35b7c92d2e8096513`, file `best_model.pth`; số tham số chưa xác nhận. Metadata ghi BSD-3-Clause, nhưng cần đối chiếu điều kiện của model gốc XTTS-v2 và tài nguyên kèm theo trước khi kết luận quyền sử dụng toàn bộ.

Không chọn model chỉ vì có từ “TTS”: cần có API nhận audio reference thật. Tracking trước đó đã ghi nhánh ZeroTTS khảo sát không công bố voice encoder để clone audio trực tiếp; nhánh V-TTS ONNX đang có cũng không phải nhánh zero-shot.

## 8. Dataset bổ sung cho TTS / nhiều người nói

### viVoice

- [capleaf/viVoice](https://huggingface.co/datasets/capleaf/viVoice): **1.016,97 giờ**, **887.772 mẫu**, 24 kHz theo card; hướng đến tổng hợp giọng nhiều người nói.
- Nguồn từ **186 kênh YouTube**; số kênh **không đồng nghĩa** với số speaker hoặc số phương ngữ đã xác minh.
- License **CC BY-NC-SA 4.0**, chỉ nghiên cứu. Chủ dữ liệu yêu cầu email trường/tổ chức/công việc để xin truy cập, không nhận email phổ thông.
- Tài khoản HF của phiên này **chưa có quyền đọc file dataset**. Chỉ đã xác minh mô tả công khai; chưa tải audio.

### PhoAudiobook

- [thivux/phoaudiobook](https://huggingface.co/datasets/thivux/phoaudiobook): thiết kế cho zero-shot TTS tiếng Việt.
- Theo card, train có **940 giờ audio dài + 554 giờ audio ngắn tăng cường = 1.494 giờ**, **710 speaker**. Không diễn giải 1.494 giờ là toàn bộ audio gốc độc lập; phần tăng cường đã được nêu riêng.
- Gated; điều kiện sử dụng cho nghiên cứu/giáo dục, **không phân phối lại toàn bộ hoặc một phần dataset, cả nguyên bản lẫn đã sửa đổi**. Không tự upload sang Kaggle để vượt điều kiện truy cập.
- Tài khoản HF của phiên này chưa có quyền đọc file dataset. Nguồn mô tả: card công khai; [bài giới thiệu ACL 2025](https://aclanthology.org/2025.acl-short.81/).

### Dolly-Audio

- [dolly-vn/dolly-audio-1000h-vietnamese](https://huggingface.co/datasets/dolly-vn/dolly-audio-1000h-vietnamese): card công bố gần **1.000 giờ**, **664.125 mẫu**, **152 speaker** đa vùng; có `voice_id`, `text`, `audio`.
- Card nêu **CC BY-NC-SA 4.0**, chỉ nghiên cứu phi thương mại và yêu cầu email tổ chức để xét truy cập. Đọc được README không có nghĩa đã được duyệt tải audio.
- Metadata có tag `synthetic`; cần làm rõ nguồn audio thật/synthetic trước khi dùng để đánh giá phương ngữ hoặc coi là tập giọng người thật.
- Tuyên bố “WER gần 0” là của tác giả từ lấy mẫu thủ công, chưa được dự án kiểm chứng. Không dùng tuyên bố đó thay bước kiểm nhãn.

### ViVoice-34 preview

- [anonymous-vivoice34/ViVoice34](https://huggingface.co/datasets/anonymous-vivoice34/ViVoice34): **381 mẫu preview**, metadata có `speaker_id`, `province`, `gender`, `age_group`, transcript; license **CC BY 4.0** theo card.
- Đây là **bản preview**, không khẳng định đã có đủ 34 tỉnh chỉ từ tên dataset. Metadata về tỉnh/người nói vẫn cần kiểm chất lượng.
- Có thể khảo sát trước khi xin các corpus lớn; chưa đánh giá độ sạch hoặc độ đúng nhãn của audio. License dataset không tự xác nhận phạm vi đồng ý clone giọng của từng người nói.

## 9. Cách thử clone phục vụ bổ sung dữ liệu ASR

Đề xuất đầu tiên là **VieNeu v3 Turbo**, với **OmniVoice làm đối chứng**. Lý do: cả hai hỗ trợ reference audio, VieNeu v3 không cần transcript mẫu, và repo đã có notebook khảo sát hai engine. Đây chưa phải kết luận chất lượng.

- Chọn reference **từ train hoặc nguồn có quyền sử dụng**, một người nói, ít nhiễu và không nhạc. Nếu dùng reference từ train, giữ reference và mọi audio sinh từ giọng đó trong cùng phía train/validation để tránh rò người nói.
- Với OmniVoice/F5, dùng transcript reference đã kiểm nếu có. Tự ASR reference là cách tiện lợi nhưng có thể đưa lỗi nhận dạng vào conditioning.
- Sinh **văn bản mới**, tập trung số tiền, giờ, tên riêng và câu lệnh LifeMate; lưu model revision, nguồn reference, văn bản đầu vào và seed để truy vết.
- Kiểm nội dung được nói ra, lỗi lặp/bỏ chữ, độ giống giọng và độ giữ phương ngữ. Không mặc định văn bản đưa vào TTS luôn là transcript đúng của audio đầu ra.
- Khởi đầu bằng một nhóm nhỏ nhiều giọng để nghe so sánh và kiểm ASR; chỉ sinh quy mô lớn sau khi có đánh giá. Audio clone giống âm sắc không tự chứng minh nó bổ sung đúng phát âm vùng miền hoặc tăng điểm agent.
- Nếu muốn dùng **public/private test làm audio reference để sinh data train**, cần BTC xác nhận riêng; điều khoản đã đọc cấm gán nhãn test để train/tune, chưa phải một sự cho phép rõ ràng cho cách chuyển giọng từ test. Ghi chú này không đề xuất dùng test làm reference.

Hiện chưa tải corpus/weights, chưa cài thêm thư viện, chưa gửi audio lên demo ngoài và chưa sinh bản clone. Chưa lựa chọn reference cụ thể.
