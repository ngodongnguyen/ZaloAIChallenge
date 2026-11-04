# Results

Số liệu tổng hợp. Chi tiết cấu hình và lý do nằm ở `experiment.md` (cột "Exp" trỏ tới mục `E<n>`).
Cách tính offline: `src/eval.py`, tính điểm 0 nếu `state_hash` khác gold; "tool-call" là điểm khi bỏ qua hash.

## Điểm nộp trên web (public_test)
| Ngày | Exp | ASR | LLM | Điểm | Ghi chú |
|---|---|---|---|---|---|
| - | E1 | PhoWhisper-large | Qwen2.5-3B zero-shot | 7 | |
| - | E2 | PhoWhisper-large | Qwen2.5-1.5B SFT | 17.85 | Kaggle T4 |
| 2026-10-28 18:55 | E4 | PhoWhisper-large | Qwen3-4B SFT | 26.61 | "4b test" |
| 2026-10-28 22:48 | E7 | PhoWhisper-large + LoRA (`asr_ft`) | Qwen3-4B SFT | **46.69** | |
| 2026-10-29 | E9 | `asr_final` (LoRA, toàn bộ train) | Qwen3-4B SFT toàn bộ train, prompt v1 | **50.21** | +3.52 so với E7; bản tốt nhất hiện tại |

## Holdout nội bộ
| Exp | ASR | LLM | Tập | n | mean_score | tool-call | hash khớp | OOS đúng |
|---|---|---|---|---|---|---|---|---|
| E2 | transcript gold | 1.5B | seen | 60 | 0.5504 | - | - | - |
| E2 | transcript gold | 1.5B | unseen | 60 | 0.3335 | - | - | - |
| E4 | PhoWhisper gốc | 4B | seen | 100 | 0.4057 | 0.5503 | 54/100 | 7/10 |
| E7 | `asr_ft` | 4B | seen | 100 | **0.6715** | 0.7474 | 79/100 | 7/10 |
| E12 | transcript gold | 4B | seen | 100 | 0.7654 | 0.8257 | 83/100 | 8/10 |
| E11 | `asr_ft` | 4B + prompt v2 | seen | 100 | 0.6308 | 0.7163 | 74/100 | 7/10 |
| E8 | `asr_ft` | 4B | unseen | 187 | **0.5002** | 0.6455 | 112/187 | - (không có gold rỗng) |
| E11 | `asr_ft` | 4B + prompt v2 | unseen | 187 | 0.5027 | 0.6474 | 111/187 | - |
| E10 | `asr_ft` | 4B + đổi tên app/tool | unseen | 187 | 0.4757 | 0.6340 | 105/187 | - |
| E9 | `asr_final` | 4B final | seen (đã nằm trong train, chỉ kiểm tra không hỏng) | 100 | 0.8966 | 0.9256 | 92/100 | 8/10 |

Phân loại lỗi (`src/analyze.py`):
| Tập | ok | wrong_tool | arg_mismatch | extra_calls | empty_pred | spurious |
|---|---|---|---|---|---|---|
| seen (100) | 58% | 12% | 22% | 3% | 2% | 3% |
| unseen (187) | 35% | 29% | 22% | 10% | 4% | - |

Lưu ý: hàng E2 dùng cách tính cũ (chưa kiểm tra `state_hash`), nên không so trực tiếp với E4/E7.

## ASR
| Exp | Model | Tập đo | n | WER | CER |
|---|---|---|---|---|---|
| E5 | PhoWhisper-large | 150 clip đầu train | 150 | 0.275 | 0.205 |
| E5 | Whisper-large-v3-turbo | 150 clip đầu train | 150 | 0.395 | 0.315 |
| E6 | PhoWhisper-large (gốc) | 150 clip holdout | 150 | 0.235 | - |
| E6 | PhoWhisper-large + LoRA | 150 clip holdout | 150 | **0.098** | - |

## Huấn luyện
| Exp | Model | Dữ liệu | Epoch | Thời gian | Loss đầu -> cuối |
|---|---|---|---|---|---|
| E2 | Qwen2.5-1.5B LoRA fp16 | 1258 mẫu | 2 | (Kaggle T4) | - |
| E4 | Qwen3-4B LoRA bf16 | 1241 mẫu | 2 (156 bước) | 3h46 | 0.45 -> 0.02 |
| E6 | PhoWhisper-large LoRA bf16 | 4134 clip | 3 (774 bước) | ~2h (ước tính) | 1.82 -> 0.04 |
| E11 | Qwen3-4B LoRA bf16, prompt v2 | 1235 mẫu | 2 (156 bước) | 3h51 | 0.39 -> 0.03 |
| E10 | Qwen3-4B LoRA bf16, đổi tên app/tool 50% | 1239 mẫu | 2 (156 bước) | 3h48 | 0.40 -> 0.03 |
| E9 | Qwen3-4B LoRA bf16, toàn bộ train | 1525 mẫu | 2 (192 bước) | 4h37 | 0.47 -> 0.03 |
| E9 | PhoWhisper-large LoRA bf16, toàn bộ train | 5036 clip | 3 (943 bước) | 2h31 (9.7 s/bước) | 1.92 -> 0.04 |

## Tốc độ suy luận (RTX 4060 Ti)
- 1-14 s/mẫu trên holdout seen (giới hạn cuộc thi 30 s trên A30).


### E13 web result (2026-10-30)
- Thought model (teacher Qwen3-14B, holdout-excluded training, asr_final, TIME_BUDGET_S=26 on 4060 Ti): web 39.07, below v1 46.69 and E9 50.21. Thought is not adopted; keep E9 (submission_final.jsonl) as the best.
- Suspects: samples cut by the 26 s budget on the slow GPU, student copying wrong/rationalized thoughts, longer outputs. Holdout numbers still to be recorded.

## E19: Qwen3.5-4B, 3 epoch (holdout model, asr_final, public_test)
- Web: **58.41** (02/11/2026 10:28), so với 56.32 của 1 epoch (+2.09, nhiễu ≈ ±1.8).
- Loss cuối 0.00035; train_loss 0.0215; 531 steps. Merge phải chạy riêng (`train/merge_adapter.py`) vì đĩa đầy.
- Chưa có số holdout unseen của bản 3 epoch trong docs (cần đối chiếu với 0.5970 của 1 epoch).

## E20: Qwen3.5-4B, 1 epoch, E17 (base-ASR transcripts) + E16 rewrites + E18 hint (-q35nc)
- Holdout: unseen 0.5967 (1 epoch cũ 0.5970, 3 epoch 0.6054); seen 0.7890 (0.7035 / 0.7688); gold_price 0.69 (3 epoch 0.6586).
- Web: **57.0** (1 epoch cũ 56.32, 3 epoch 58.41). Mức tăng seen trên holdout không chuyển sang web; chênh lệch nằm trong nhiễu (±1.8).
- Hint giá vàng gần như không giúp: train_0197/0224 vẫn sai (0224 do ASR nghe 131 thành 151).
- ASR gold vs thật (3 epoch): unseen 0.7385 vs 0.6054, seen 0.8667 vs 0.7688. Lỗi ASR thật nằm ở từ miền và tên riêng; prompt Whisper (asr_prompt.txt) không giúp (WER 0.105 vs 0.101).

## E22 (dữ liệu tổng hợp cho 11 app lạ): bước tạo dữ liệu
- utter: 435 câu (Qwen3-14B 4bit, 15 phút); lọc tên người dùng còn 376.
- roll (T=1.0, 3 lần chạy, 90 phút): giữ 231 (61%), không đồng thuận 93, hỏi lại 48, lỗi 4. Với T=0.7 thử 10 câu giữ 8/10, nên nhiệt độ cao mới phân biệt được.
- Đọc tay 8 quỹ đạo (T=0.7): đúng tool, ID lấy từ kết quả đọc, không bịa; toàn việc đơn giản 1-3 lệnh.
- Train: 1 epoch Qwen3.5, dữ liệu E17+E18 (sft_asr-para-cond-q35nc) + 2 x 231 dữ liệu tổng hợp (SUFFIX=-synth). So với web 57.0 (E20).

## E22 kết quả: 1 epoch + 2 x 231 quỹ đạo tổng hợp cho 11 app lạ (-synth)
- Holdout unseen 0.6100 (E20 0.5967; +0.013, trong nhiễu). Web **56.31** (E20 57.0, 3 epoch 58.41).
- Kết luận: dữ liệu tự gán nhãn (student tự đồng thuận) không giúp; student chỉ lặp lại điều nó đã biết. Không tiếp tục hướng này.
- Tổng hợp các thí nghiệm 1 epoch trên Qwen3.5: 56.32 (thường), 57.0 (+E17+E18), 56.31 (+E22). Tất cả trong nhiễu; số epoch (3 epoch: 58.41) là thay đổi duy nhất có tác dụng.

## E23: LLM nền khác, probe zero-shot (60 mẫu unseen, transcript gold)
| Model | điểm | hash đúng |
|---|---|---|
| Qwen3.5-4B (mốc) | 0.3528 | 34/60 |
| Qwen3-4B-2507 | 0.2471 | 25/60 |
| Hunyuan-4B-Instruct (LLM_PREFILL='{') | 0.0722 | 7/60 |
| Gemma-SEA-LION-v4-4B-VL (LLM_PREFILL='{') | 0.0167 | 2/60 |
- Không phải lỗi pipeline: model nạp đúng, JSON hợp lệ; lỗi là chọn sai app/tool. gemma-3-4b-it (bị khóa) và Ministral-3-3B (lỗi transformers) chưa thử. Giữ Qwen3.5-4B.

## E24: n-best ASR voting (K=3), holdout_all 287
- Lần đầu 0.6644 không có giá trị (ba bản nghe giống hệt do lỗi cài đặt, đã sửa).
- Sau sửa: 0.6714 (hash 218/287) so với 0.6623 (214/287). Sai->đúng 7, đúng->sai 3 (p ~0.34, không đủ bằng chứng). Chậm gần gấp đôi (18-20 s/mẫu). Giữ như tùy chọn `--nbest 3`, chưa nộp web.
