---
run_id: lm-B-001-baseline-paired-dev
track: lifemate
lane: B
status: planned
parent_run: lm-B-000-local-dryrun
change_vs_parent: thay mock bằng model thật; trong run paired chỉ đổi nguồn transcript clean so với ASR
hypothesis: đo được năng lực planner và độ nhạy với lỗi ASR trên cùng tập dev_seen
account: null
agent: codex
started: null
finished: null
notebook: notebook-kaggle/lifemate/agent/lifemate_agent.ipynb
kaggle_version: null
output_link: null
config_hash: null
dataset_version: resolve_at_preflight
model: Qwen/Qwen3-4B-Instruct-2507
model_revision: resolve_at_preflight
asr_model: vinai/PhoWhisper-medium
asr_revision: resolve_at_preflight
precision: fp16
backend: resolve_at_preflight_vllm_or_hf
hardware: planned_Kaggle_2x_T4_16GB
gpu_hours: null
split: train
subset: dev_seen_all_manifest_seed_2026
public_score: null
tags: [baseline, planned, paired-dev, diagnostic, e2e, asr]
---

# lm-B-001-baseline-paired-dev

## Mục tiêu

Baseline đầu tiên bằng model thật. Hai nhánh dùng cùng ID/config planner, chỉ khác transcript: diagnostic sạch và e2e PhoWhisper cho cả lịch sử lẫn yêu cầu cuối. Không so “tăng điểm” với oracle/mock của parent; parent chỉ là lineage kỹ thuật.

Thiết kế được người dùng duyệt ngày 2026-10-29. Checklist implement: [`planning.md`, mục 9–10](../planning.md#10-lifemate-paired-dev--implementation-plan). Card này mới đăng ký; chưa chạy Kaggle, chưa có điểm.

## Cấu hình dự kiến

| Thuộc tính | Giá trị đóng băng trước run |
|---|---|
| Preset / experiment | `paired_dev` / `lm-B-001-baseline-paired-dev` |
| Split / subset | `train` / toàn bộ `dev_seen`, seed manifest 2026, không limit |
| Nhánh | `diagnostic`, `e2e`; output/config/hash riêng |
| LLM | Qwen3-4B-Instruct-2507, revision commit cụ thể từ preflight |
| ASR | PhoWhisper-medium, revision cụ thể; greedy, toàn bộ user audio |
| Backend / precision | một backend thực đã preflight cho cả hai nhánh; FP16, không quantization |
| Seed / context / thinking | 0 / 16384 token / off |
| Agent | 8 steps, 256 output token/step, max_invalid 2, budget agent 30s, result_chars 1500 |
| Batch / CPU | agent 32, HF internal 8, ASR 16; 1 CPU thread/worker |
| Hardware | 2 T4, 2 shard ID ổn định, chạy ASR/nhánh tuần tự |
| Session | 14400s tổng, reserve export ≥900s, checkpoint/resume; không 4h mỗi pha |

Manifest ID thực, input/code/model/cache hashes và config hai nhánh được lưu ở output. Không dùng final_holdout/dev_unseen/Public Test; không sửa prompt/ASR sau khi xem điểm trong run này.

## Kiểm tra bắt buộc trước chạy lớn

- [ ] Regression fixtures và smoke model thật qua; backend không mock, revisions đã resolve.
- [ ] Dev ID cố định, disjoint final/train_fit; đủ audio history/final, cache fingerprints khớp.
- [ ] Nhánh e2e không được thấy `audio_text`/label; thiếu một transcript là dừng, không fallback gold.
- [ ] Test score pending dùng đủ N, test hash gate/empty/read thừa, test resume và deadline toàn notebook.

## Kết quả

**Chưa có số đo.** Bảng chỉ điền từ artifact sau chạy; `—` không có nghĩa 0 điểm.

| Metric | Diagnostic | E2E | E2E − diagnostic |
|---|---|---|---|
| Expected / completed / pending ID | — | — | — |
| Score hash-gated /100 trên đủ N | — | — | — |
| Hash OK rate trên đủ N | — | — | — |
| Tool F1 trước hash gate | — | — | — |
| Schema errors / invalid proposals / termination | — | — | — |
| Agent latency / peak VRAM mỗi GPU | — | — | — |

ASR: expected/completed/missing audio, WER/CER, recall heuristic name/number/time, timing và VRAM: **chưa đo**. Paired delta và CI bootstrap 2000 lần seed 2026: **chưa đo**. Score theo empty/ask/write/read-only và history/no-history: **chưa đo**.

## Artifact cần lấy sau chạy

`manifest.json`, `preflight.json`, config hai nhánh, `audio_manifest.json`, ASR cache/metrics, predictions/traces/summary/merge_report/per_id_scores của từng nhánh, `paired_report.json`, `paired_per_id.jsonl`, output version và GPU-hours thực. Chỉ ghi link/checksum vào repo; không commit dữ liệu/cache/submission.

## Phân tích lỗi

Chưa có. Sau chạy đếm theo ASR thực thể, write/hash, read thừa/thiếu, empty/ask, lịch sử, prompt/context và timeout; mỗi nhóm có số lượng và 2–3 ID. Không khẳng định unseen-app generalization từ dev_seen.

## Sự cố vận hành

Chưa chạy. Nếu phiên thiếu budget: giữ pending, lưu output version; execution resume có card mới theo NNN kế tiếp, parent là phiên đang tiếp tục và giữ cùng semantic config. Không đổi run_id/card của phiên đã xảy ra.

## Kết luận và bước tiếp

- Chưa chốt baseline cho đến khi cả hai nhánh đủ cùng N, audio coverage 100%, schema_errors=0 và không lẫn nguồn/config.
- Điểm thấp vẫn được ghi nhận; phân tích paired score để chọn sửa planner, ASR hoặc serving ở run sau.
- Chạy trên 2 T4 với ASR cache chưa xác nhận SLA 30 giây/A30. Public là run/card mới, chưa nộp LB trong experiment này.
- Đã thêm kết luận vào `insights.md`: chưa; hiện chưa có bằng chứng thực nghiệm mới.
