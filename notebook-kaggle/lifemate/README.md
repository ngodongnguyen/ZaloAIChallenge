# LifeMate — notebook baseline

Tracking run và kết quả: [`.vscode/history_plan/lifemate.md`](../../.vscode/history_plan/lifemate.md).

`agent/lifemate_agent_public.ipynb`: ASR (PhoWhisper-medium) → agent Qwen3-4B-Instruct-2507 gọi tool qua `zaic_tools.ToolEnv` (mỗi mẫu một env, validate trước khi call) → `submission.jsonl`.

## Chạy để có file nộp Public Test

1. Kaggle → New Notebook → File → Import Notebook → `agent/lifemate_agent_public.ipynb`.
2. Add Input: dataset `thanhkhil/zaloai-lifemate-dataset`. Settings: **GPU T4 x2**, Internet **On**.
3. **Save Version → Save & Run All**. Xong tải `/kaggle/working/submission.jsonl` ở tab Output → nộp Public Leaderboard.

`CONFIG["mode"]`: `public` (mặc định) · `diagnostic` (train, transcript sạch; đặt `split="train"`, `subset="dev_seen"`) · `e2e` (train, transcript ASR) · `sft` (LoRA trên `train_fit` rồi eval dev).

Mỗi experiment một thư mục, push được ngay (`kaggle kernels push -p <thư mục>`; `kernel-metadata.json` chỉ có ở local, sửa `<owner>` trong `id` thành tài khoản sẽ push). Mỗi thư mục một tên file notebook riêng (`lifemate_agent_<biến thể>.ipynb`) để Import Notebook trên Kaggle không lẫn:

| Thư mục | `CONFIG_PRESET` | Run |
|---|---|---|
| `agent/` (`lifemate_agent_public.ipynb`) | `public` | lm-B-014-fewshot-public (Add Input `<owner>/lifemate-asr-cache` nếu có) |
| `agent-dev/` (`lifemate_agent_dev.ipynb`) | `dev` | lm-B-010-fewshot-dev (mốc so sánh) |
| `agent-sft/` (`lifemate_agent_sft.ipynb`) | `sft` | lm-sft-013-lora-gold-vs-asr (không few-shot) |
| `agent-qwen35/` (`lifemate_agent_qwen35.ipynb`) | `dev_qwen35` | lm-B-011-fewshot-qwen35-dev |
| `agent-gipformer/` (`lifemate_agent_gipformer.ipynb`) | `dev_gipformer` | lm-A-012-fewshot-gipformer-dev |
| `agent-guard/` (`lifemate_agent_guard.ipynb`) | `dev_guard` | lm-B-015-guards-gipformer-qwen35-dev (bản sao `agent-qwen35/` + guard) |
| `agent-sft-mix/` (`lifemate_agent_sft_mix.ipynb`) | `sft_mix` | lm-sft-016-lora-asrmix-guard (bản sao `agent-sft/` + guard + trộn transcript) |
| `agent-sft-public/` (`lifemate_agent_sft_public.ipynb`) | `public_sft` | lm-B-017-sft-public (adapter `mix_aug` của 016 trên public → mốc LB) |
| `agent-sft-cont/` (`lifemate_agent_sft_cont.ipynb`) | `sft_cont` | lm-sft-018-cont-syn (warm start từ 016, `cont` vs `cont_syn`) |
| `agent-sft-weight/` (`lifemate_agent_sft_weight.ipynb`) | `sft_weight` | lm-sft-019-cont-weight (warm start từ 016, `cont` vs `cont_w` lặp mẫu hỏi lại ×3 / ≥4 call ×2; không cần synthetic) |
| `agent-sft-ground/` (`lifemate_agent_sft_ground.ipynb`) | `dev_ground` | lm-B-020-ground-xref (adapter 016 trên dev: `base` chạy lại eval 016 vs `ground` + guard `ground_xref`) |
| `agent-sft-ep2/` (`lifemate_agent_sft_ep2.ipynb`) | `sft_ep2` | lm-sft-021-ep2-drop (train từ đầu 2 epoch qua 2 phiên, `ep2` vs `ep2_drop`; không cần output 016) |
| `agent-sft-oracle/` (`lifemate_agent_sft_oracle.ipynb`) | `dev_oracle` | lm-B-022-oracle-transcript (checkpoint 021 `sft_ep2/ckpt-000380`, dev 36.47; không train: `gold` transcript chuẩn vs `asr` gipformer; Add Input output phiên hoàn tất 380 bước của `lifemate-agent-sft-ep2`) |
| `agent-sft-full/` (`lifemate_agent_sft_full.ipynb`) | `sft_full` | lm-sft-023-full-ddp (công thức `ep2` của 021 trên **toàn bộ train**, 2 GPU train chung một model, 2 phiên; train xong chạy public_test + `ground_xref` → `submission.jsonl`; không cần input riêng) |
| `asr-ft/` (`lifemate_asr_ft.ipynb`) | — (notebook riêng, không dùng `lmlib`) | lm-A-030-asr-ft-whisper (fine-tune PhoWhisper-small / -medium trên audio `train_fit`, mỗi GPU một biến thể; chọn epoch theo WER dev 417 mẫu, báo WER `final_holdout`, chép transcript dev + holdout + public_test theo định dạng cache `asr_<split>_<variant>.shard0.jsonl`; chỉ cần dataset cuộc thi) |
| `asr-ft-gipformer/` (`lifemate_asr_ft_gipformer.ipynb`) | — (notebook riêng, không dùng `lmlib`) | lm-A-031-asr-ft-gipformer15 (fine-tune gipformer 1.5 bằng icefall + k2, mỗi GPU một learning rate; cùng split / dev / holdout với lm-A-030; đo thêm zero-shot sherpa-onnx gipformer 1.0 và 1.5; cache `asr_<split>_gipft_<variant>.shard0.jsonl`, model `model_best.pt` định dạng icefall; chỉ cần dataset cuộc thi + Internet) |
| `agent-asr-ft-eval/` (`lifemate_agent_asr_ft_eval.ipynb`) | `dev_asr_ft` | lm-B-033-asr-ft-agent-dev (không train/ASR lại: cùng checkpoint 021 và 417 dev, lần lượt dùng cache Gipformer FT `lr15` epoch 4 và Whisper-medium FT epoch 5; hướng dẫn input bên dưới) |
| `asr-full-public/` (`lifemate_whisper_full_public.ipynb`) | — | lm-A-034-whisper-full-public: Whisper-medium 5 epoch trên toàn bộ 1545 train / 5036 audio, DDP 2×T4; ASR public 1933 audio rồi agent full `ckpt-000610` chạy 634 mẫu → `submission.jsonl`; dataset + agent full + wheels đã nhúng vào `.ipynb` |
| `agent-qwen35-smoke/` (`lifemate_qwen35_sft_smoke.ipynb`) | — (`lmlib` + `qwen35smoke`) | lm-sft-038-qwen35-smoke: Qwen3.5-4B có train LoRA công thức 021 trên T4 được không (kernel `fla` vs torch, s/step + VRAM so với Qwen3-4B cùng 192 example), merge vào checkpoint gốc rồi vLLM 0.32 sinh JSON; kết quả `lm_q35smoke/SMOKE.json` (GO / NO-GO); dataset + wheels đã nhúng vào `.ipynb` |
| `agent-sft-ep3/` (`lifemate_agent_sft_ep3.ipynb`) | — (`lmlib` của 033 + `ep3`) | lm-sft-039-ep3-cont: train tiếp 1 epoch từ checkpoint 021 (`ep3` lr 1e-4 / `ep3_lr5e5` lr 5e-5, mỗi GPU một biến thể), chấm dev bằng Whisper-medium FT như 033 và so cặp với 51.19; input là output gốc pin version (sft-ep2, asr-ft, asr-ft-eval, wheels), kiểm sha256 |

### Cải thiện tiếp từ mốc Public LB48.2

Bản CPU đã tạo: [`agent-action-fix/results/public_replay/submission.jsonl`](agent-action-fix/results/public_replay/submission.jsonl) (gitignored). Replay toàn bộ634mẫu qua ToolEnv mới, sửa tham số giờ ở2mẫu, giữ1547call, CLI chính thức0lỗi. File48.2 gốc giữ nguyên. Dev replay033:51.19→53.11,9mẫu tăng/0giảm;974gold train_fit không đổi. Đây là kết quả phát triển quy tắc trên dev, chưa có điểm LB mới, không phải model sinh lại hành động.

Có thể tái tạo bằng Import [`agent-action-fix/lifemate_action_fix_public.ipynb`](agent-action-fix/lifemate_action_fix_public.ipynb) → Save & Run All. Dataset và output `lifemate-whisper-full-public` **v2/kernelVersion364801062** đã gắn sẵn; không cần train/model/ASR/wheels. Dùng account có quyền output private. Kết quả ở `/kaggle/working/submission.jsonl` và `verification.json`. Toàn bộ tác vụ CPU; metadata giữ T4 theo quy ước repo, có thể chọn None để tiết kiệm quota.

Thí nghiệm tiếp: Import [`agent-fewshot-dev/lifemate_agent_fewshot_dev.ipynb`](agent-fewshot-dev/lifemate_agent_fewshot_dev.ipynb) → Save & Run All trên T4x2. Thử1và3ví dụ truy hồi sau SFT, cùngcheckpoint021/ASR/417dev với033, so cachedcontrol k=0. Pool chỉ974train_fit, không chứa dev/holdout; chưa chạy GPU/chưa có điểm mới. Không ghép rule035 để đo riêng ảnh hưởng few-shot. Đã gắn dataset và output `lifemate-asr-ft`v4(364609062), `lifemate-agent-sft-ep2`v4(364021094), `lifemate-agent-asr-ft-eval`v2(364738806), `build-wheels`v4(363490930); preflight kiểm đúng checkpoint/cache, không train hay ASR lại. Xem `lm_fewshot_dev/experiment_summary.json`; file dev không nộp public.

Thí nghiệm độc lập với036: [`agent-dual-asr-dev/lifemate_agent_dual_asr_dev.ipynb`](agent-dual-asr-dev/lifemate_agent_dual_asr_dev.ipynb), run037. Import → T4x2/InternetOn → Save & Run All. Đã nhúng **6input**: LifeMate v1, `lifemate-asr-ft`v4(364609062), `lifemate-asr-ft-gipformer`v3(364653019), `lifemate-agent-sft-ep2`v4(364021094), `lifemate-agent-asr-ft-eval`v2(364738806), `build-wheels`v4(363490930). Dùng account có quyền output private. Không train/ASR lại; agent k0 thấy hai transcript của cùngaudio, ở cả câu cuối và history, ưu tiên Whisper khi chưa có căn cứ sửa. Không dùng nhãn để chọn transcript, giữ nguyên18module033. Preflight kiểm checkpoint/cacheSHA và đủ417dev/1627audio; một model/worker/GPU. So cachedcontrols Whisper51.19 và Gipformer44.12 bằng điểm từngID/CI95. Chưa chạyGPU/chưa có điểm mới. Xem `lm_dual_asr_dev/experiment_summary.json`; filedev không nộp public. CPUtest: `python3 -m unittest discover -s notebook-kaggle/lifemate -p 'test_dual_asr_dev.py' -v`.

Public k=1 (run 040): Import [`agent-fewshot-public/lifemate_agent_fewshot_public.ipynb`](agent-fewshot-public/lifemate_agent_fewshot_public.ipynb) → T4x2/Internet On → Save & Run All. Đã nhúng 4 input: LifeMate v1, `lifemate-agent-sft-full` v9 (364434494), `lifemate-whisper-full-public` **v2** (364801062, không dùng v3 đã hủy), `build-wheels` v4 (363490930). Giống bản LB 48.39 (034 + 035), chỉ thêm 1 ví dụ truy hồi từ toàn bộ 1545 train; preflight kiểm SHA checkpoint 610 và cache ASR khớp 034. Xuất `/kaggle/working/submission.jsonl` đã sửa giờ 035.

Qwen3.5-4B SFT (run 041): Import [`agent-qwen35-sft/lifemate_qwen35_sft.ipynb`](agent-qwen35-sft/lifemate_qwen35_sft.ipynb) → T4x2/Internet On → Save & Run All. Input gắn sẵn như run 039: LifeMate v1, `lifemate-agent-sft-ep2` v4 (364021094), `lifemate-asr-ft` v4 (364609062), `lifemate-agent-asr-ft-eval` v2 (364738806), `build-wheels` v4 (363490930). Train LoRA công thức 021 trên Qwen3.5-4B, 2 epoch, hai GPU cùng train một model (`lmlib/q35ddp.py`), rồi chấm 417 dev như 033. Thường cần **2 phiên**: phiên 2 Add Input output version phiên 1 của chính notebook → Save & Run All (tự resume).

### Chạy tiếp sau hai notebook ASR fine-tune

Import [`agent-asr-ft-eval/lifemate_agent_asr_ft_eval.ipynb`](agent-asr-ft-eval/lifemate_agent_asr_ft_eval.ipynb) → **Save Version → Save & Run All**. File `.ipynb` đã gắn dataset LifeMate, **T4 x2**, **Internet On** và các Notebook Output sau (pin version đã kiểm tra 2026-11-02):

1. `lifemate-asr-ft-gipformer` version 3 (`kernelVersion` 364653019): chứa `gip_ft/lr15/asr_train_gipft_lr15.shard0.jsonl`.
2. `lifemate-asr-ft` version 4 (`kernelVersion` 364609062): chứa `asr_ft/medium/asr_train_medium.shard0.jsonl`.
3. `lifemate-agent-sft-ep2` version 4 (`kernelVersion` 364021094): chứa `lm_agent/sft_ep2/ckpt-000380/{COMPLETE,adapter_state.pt,adapter_config.json}`. Không dùng checkpoint `sft_full` để đo dev.
4. `build-wheels` version 4 (`kernelVersion` 363490930): cài vLLM nhanh trong venv riêng.

Các nguồn Notebook Output được nhúng bằng `{"sourceType": "kernelVersion", "sourceId": <version_id>}` trong `metadata.kaggle.dataSources`, không chỉ trong metadata CLI. Dùng tài khoản có quyền đọc các output private; chỉ Add Input thủ công nếu Kaggle báo thiếu quyền hoặc bỏ nguồn khi import. Chưa kiểm thao tác Import trên UI đăng nhập. Notebook kiểm đúng checkpoint, đúng epoch cache và đủ 1627 transcript (417 mẫu, gồm history) trước khi cài/nạp model. Mỗi lượt eval chạy một model/worker/GPU, giữ nguyên prompt/guard của run 022. Không cần sửa config hoặc chạy lại hai notebook ASR.

Chạy xong tải `lm_asr_ft_eval/`: xem `eval_variants_summary.json` (điểm, hash, nhóm lỗi) và `paired_report.json` (Gipformer FT trừ Whisper FT, CI95); các thư mục `eval_*/` chứa điểm từng ID, trace, log và timing/VRAM. File submission trong thư mục eval là **dev**, không dùng nộp public. Kiểm CPU: `python3 -m unittest discover -s notebook-kaggle/lifemate -p 'test_*.py' -v`.

### Train Whisper full và xuất bài nộp public

Import [`asr-full-public/lifemate_whisper_full_public.ipynb`](asr-full-public/lifemate_whisper_full_public.ipynb) → **Save Version → Save & Run All**. `.ipynb` đã gắn dataset LifeMate, T4 x2, Internet On và hai output private đã kiểm tra file qua API:

- `lifemate-agent-sft-full` version 9 (`kernelVersion` 364434494), có checkpoint `sft_full/ckpt-000610`.
- `build-wheels` version 4 (`kernelVersion` 363490930), smoke trước đó toàn OK, có bitsandbytes 0.50.2 và vLLM 0.32.0.

Notebook kiểm đủ 1545 train / 5036 audio (cả history và phần dev/holdout cũ), 634 public / 1933 audio, rồi smoke Whisper DDP + FP16 + AdamW8bit trước lượt train dài. Train từ PhoWhisper-medium gốc 5 epoch, lr 6.25e-6, seed 2026, global batch 16 (mỗi GPU batch 4 × accumulation 2); không chọn model theo public, không dùng lại ASR checkpoint split để train thêm 5 epoch. Không drop audio ở cuối epoch; sampler kiểm đủ tập audio mỗi epoch. Sau đó mỗi GPU một worker ASR public, rồi mỗi GPU một worker agent full (k=0, guard như public 023, có `ground_xref`, không thêm router).

Chỉ xuất `/kaggle/working/submission.jsonl` khi đủ 5 epoch, đủ cache public, đủ 634 kết quả và CLI chính thức báo `634 lines checked, 0 schema errors` (CLI có thể trả exit 0 ngay cả khi có lỗi nên notebook kiểm cả báo cáo). Giữ thêm `lm_whisper_full/asr/model/`, `training_summary.json`, `COMPLETE`, `verification.json`, các cache/trace/log và summary timing/VRAM. Version2 đã chạy thành công, đủ5epoch,634public, Public LB48.2 theo user; output cần pin364801062 vì latestv3 đã hủy. Không cần train lại agent hoặc build lại wheels.

`router.py`: Bộ định tuyến BM25 thử nghiệm, chưa bật trong pipeline. Đo lại bằng974train_fit và ASRdev thật: Top4 chỉ84.65% recall đầy đủ app cần dùng, chưa đủ để cắt cứng tool. Con số98.91% trước đây không phải phép đo này; chưa có bằng chứng router tăng điểm agent.

`agent-sft-full/` thêm vào `sft.py` chế độ data parallel (`SFT["ddp"]`: một tiến trình / GPU, mỗi bước chia `idx[rank::world]`, cộng gradient trước unscale/clip → cập nhật như 1 GPU; rank 0 quyết định checkpoint/dừng và ghi checkpoint, rank khác thêm `rng_rank<r>.pt`) và cell kiểm `ddp_selftest_launch` trước ASR; kiểm CPU: `test_sft_full_notebook.py` (gloo, cần torch + transformers + peft). Chưa đồng bộ sang notebook khác.

`agent-sft-public/`, `agent-sft-cont/`, `agent-sft-weight/`, `agent-sft-ground/`, `agent-sft-ep2/`, `agent-sft-oracle/` là bản sao `agent-sft-mix/` (chỉ khác `CONFIG_PRESET`), thêm: `init_adapter` trong `sft.py` (warm start: nạp trọng số LoRA của run khác, optimizer/schedule mới, không vào config hash), dữ liệu synthetic cho biến thể có `"syn": True` (`SFT["syn_glob"]`, catalog train ∪ synthetic, bỏ mẫu dùng app dev_unseen), resume theo đúng thư mục `sft_<biến thể>`, `oversample(rows, ask=, calls4=)` (biến thể có `"oversample"`), guard `AgentConfig.ground_xref` (chặn call có ID không xuất hiện trong kết quả tool nào), `CONFIG["eval_variants"]` (một model, nhiều cấu hình agent chấm lần lượt; khoá `"config"` trong một biến thể ghi đè cấu hình cấp trên, vd `mode`), `SFT["eval_when_finished"]` (phiên chưa train xong thì bỏ merge/eval, phiên sau gắn output làm input để resume). Bản đang lưu trên Kaggle của `sft-public`/`sft-weight`/`sft-cont` là bản trước các thay đổi này; khác biệt không ảnh hưởng preset của chúng. Cả hai cần Add Input output `lifemate-agent-sft-mix` (adapter + transcript gipformer train, khỏi chạy ASR train). Kiểm CPU: `test_sft_cont_notebooks.py` (phần warm start cần torch + transformers + peft).

`agent-guard/` và `agent-sft-mix/` có thêm guard trong `agent.py` (`AgentConfig.block_xref_ask`, `read_repeat_feedback`, `nudge_done_after_read`, `max_guard`; mặc định tắt, preset bật qua `GUARDS`) và trường `guards` trong traces. `agent-sft-mix/` còn có `build_examples(asr_prob=, aug=)` trong `sft.py`. Hai thay đổi này chưa đồng bộ sang các notebook khác. Kiểm CPU: `test_guard_notebooks.py`.

**Few-shot** (`CONFIG["fewshot"]`, mặc định k=3): mỗi mẫu được kèm 3 mẫu train giống nhất (transcript gold + tool call gold). Trên split train chỉ lấy từ `train_fit`, nên dev không thấy nhãn dev; trên public lấy cả train (thư mục `train/` cạnh `public_test/`). Các baseline adapter SFT dùng `fewshot.k=0`; riêng036 thửk1/k3 trên dev trước khi quyết định dùng public. Docker private test phải đóng gói kèm `train.jsonl`.

Các notebook trong `agent-*/` là **bản sao của `agent/lifemate_agent_public.ipynb`, chỉ khác dòng `CONFIG_PRESET`**. Sửa code thì sửa `agent/` rồi chép lại sang các thư mục kia (giữ preset), để không lệch code giữa các experiment.

Ngoại lệ theo yêu cầu người dùng: bản sửa tương thích PEFT/torchao và kiểm lỗi worker SFT hiện chỉ áp dụng trong `agent-sft/lifemate_agent_sft.ipynb`; bản sửa bộ nhớ Qwen3.5 và accelerate ở worker venv chỉ áp dụng trong `agent-qwen35/lifemate_agent_qwen35.ipynb`. Không tự đồng bộ các bản sửa này sang notebook khác; giữ phần riêng này khi đồng bộ code chung.

Riêng Qwen3.5 trên T4: dùng `enforce_eager=True`, `language_model_only=True`, tối đa 4 sequence và 1024 token prefill/bước (chunked prefill, context vẫn 16K). `wrapt`/`accelerate` cài trực tiếp vào worker venv; không cần rebuild wheels. Worker lỗi làm notebook dừng trước khi chấm, tránh hiểu `0/417 completed` là điểm chất lượng model.

`CONFIG_PRESET` (cuối cell config) đặt sẵn bộ config, mỗi version Kaggle chạy một preset:
- `public`: toàn bộ public_test → `submission.jsonl`. Add Input dataset private `<owner>/lifemate-asr-cache` (cache PhoWhisper public) thì bỏ qua ASR (~30 phút).
- `dev`: e2e trên train `dev_seen+dev_unseen` (417 mẫu), điểm local ở `lm_agent/merge_report.json`.
- `sft`: ASR toàn bộ train → 2 biến thể LoRA (transcript gold / ASR) song song mỗi GPU → eval dev từng biến thể (`lm_agent/sft_eval_summary.json`). Dùng adapter cho public: Add Input output version `sft`, đặt `CONFIG["llm"]["adapter"]` = thư mục `.../lm_agent/sft_<biến thể>`.

Riêng notebook `agent-sft/lifemate_agent_sft.ipynb`: setup tự gỡ `torchao <0.16.0` không dùng (xung đột với PEFT), rồi kiểm gắn LoRA + forward/backward/merge trên model nhỏ ngẫu nhiên CPU trước ASR; log phải có `PEFT_SMOKE_OK`. Không cần build lại wheels cho sửa lỗi này. Worker train lỗi hoặc thiếu checkpoint sẽ dừng notebook, không xuất báo cáo eval rỗng. Add Input output run SFT cũ có các file `asr_train_<backend>.shard*.jsonl` để tái sử dụng transcript đã chạy.

Kiểm xử lý kết quả SFT bằng CPU, không tải model: `python -m unittest discover -s notebook-kaggle/lifemate -p 'test_*.py' -v`.

Hết giờ giữa chừng: chạy version mới, attach output cũ làm input, điền đường dẫn vào `resume_from_dirs` (cùng config). Mẫu chưa xong không bao giờ bị điền hash giả.

Code nằm trong các cell `%%writefile` (thư viện `lmlib` ghi vào `/kaggle/working/lib`); sửa logic thì sửa cell đó.

## Preset dev cho plan 001/002

Giữ một notebook nguồn. Cell CONFIG có `CONFIG_PRESET`:

| Preset | Phép đo | Output |
|---|---|---|
| `paired_dev` | Qwen3-4B clean/ASR trên đúng `train/dev_seen`, seed 2026 | `/kaggle/working/lm_baseline/{asr,diagnostic,e2e}` |
| `asr_backend_dev` | gipformer CPU trên cùng audio; so với cache PhoWhisper đã lưu | `/kaggle/working/lm_asr_compare/gipformer` |

Các preset `public`, `dev`, `sft`, `dev_qwen35`, `dev_gipformer` vẫn có cấu hình riêng. `paired_dev` giữ budget wall-time 30s theo wave admission để so clean/e2e cùng điều kiện; các preset budgetfix dùng proxy amortized hiện có. Không dùng số đo T4 này xác nhận SLA A30.

Chọn preset ở CONFIG, rồi đặt đường dẫn qua `DEV_OPTIONS` trong cùng cell (các key chỉ có tác dụng với preset dev mới):

```python
DEV_OPTIONS = {
    # "execution_id": "<run-id-mới-cho-phiên-resume>",
    # "parent_execution_id": "<run-id-phiên-trước>",
    # "resume_from_dirs": ["/kaggle/input/<output-trước>/lm_baseline"],
    # "preflight_dirs": ["/kaggle/input/<preflight-đã-lưu>/lm_baseline"],
}
```

`paired_dev` là writer PhoWhisper mặc định. Ở `asr_backend_dev`, đặt `control_dir` tới **đúng thư mục `lm_baseline/asr`** của output version B đã lưu. Khi chưa có control, A vẫn chạy candidate và xuất `comparison_complete=false`; ghép comparison offline khi control tới, không phải nạp lại LLM. Nếu chủ động chạy control tại A, đặt `run_control_here=True` và ghi quyết định vào dòng run trong `.vscode/history_plan/lifemate.md` để B reuse cache A.

Cache mới có `audio_manifest.json`, `cache_manifest.json` và `transcripts.shard*.jsonl`; cache cũ chỉ có tên backend/header legacy không đủ provenance cho preset dev. Resume phải attach nguyên output root, giữ semantic config và số shard. Đổi execution/deadline/mount path không đổi checkpoint identity; đổi input/model/code/budget làm checkpoint bị từ chối. Thiếu history transcript hoặc cache không khớp thì dừng, không lấy transcript sạch để bù.

Add Input dataset và output `build-wheels`; chọn **T4 x2** cho paired-dev, Internet On khi cần tải weights. `asr_backend_dev` mặc định chỉ nạp gipformer CPU; không cài/nạp LLM. Preflight ASR ghi smoke batch1/batch16; preflight LLM kiểm đúng revision 4B và prompt gần 16K trên cả hai GPU, giới hạn calibration chung 1200s. Backend/serving config được đóng băng trước dev. Toàn phiên dev có budget 14400s kể từ setup và reserve export 900s.

Báo cáo chính: `paired_report.json`, `paired_per_id.jsonl`, `asr_comparison.json`, `asr_per_audio.jsonl`, `merge_report.json`, `summary.shard*.json`. Pending vẫn là pending; không xuất hash giả. Analysis có thể gọi lại từ `lmlib.compare.compare_paired(...)` hoặc `lmlib.asr_compare.compare_asr(..., gold=...)` sau khi cả output đã lưu, không cần model.

Kiểm CPU local (Python có pandas/pyarrow/openpyxl/python-calamine/soundfile):

```bash
python notebook-kaggle/verify_notebooks.py lifemate
```

Notebook cũng chạy cùng regression trước GPU. Kiểm CPU dùng model giả; điểm fixture không là điểm model/cuộc thi. Chưa chạy Kaggle thì smoke GPU, tốc độ và metric thực vẫn chưa được xác nhận. Import/upload private không đồng nghĩa nộp Public LB; `kaggle kernels push` tự chạy nên chỉ dùng khi đã được yêu cầu khởi chạy.
