---
run_id: dm-D-001-baseline-dev
track: datamate
lane: D
status: planned
parent_run: dm-D-000-local-dryrun
change_vs_parent: thay MockLLM bằng Qwen3-4B thật trên dev_tune cố định, giữ kiến trúc helper
hypothesis: đo PA và phân bố lỗi thực để quyết định giữ helper hay mở đối chứng kiến trúc ở vòng sau
account: null
agent: codex
started: null
finished: null
notebook: notebook-kaggle/datamate/helper-agent/datamate_helper_agent.ipynb
kaggle_version: null
output_link: null
config_hash: null
dataset_version: resolve_at_preflight
model: Qwen/Qwen3-4B-Instruct-2507
model_revision: resolve_at_preflight
precision: fp16
backend: resolve_at_preflight_vllm_or_hf
hardware: planned_Kaggle_2x_T4_16GB
gpu_hours: null
split: train
subset: dev_tune_all_6_shops_manifest_seed_2026
public_score: null
tags: [baseline, planned, helper-agent, dev-tune, pa]
---

# dm-D-001-baseline-dev

## Mục tiêu

Đo baseline model thật của helper-agent: PA có dương so với always-abstain (=0) không, bao nhiêu điểm mất do loader/code/nghiệp vụ/format? `dm-D-000` chỉ kiểm formatter/profiler, không phải baseline PA để so cải thiện.

Thiết kế được người dùng duyệt ngày 2026-10-29. Checklist implement: [`planning.md`, mục 9–10](../planning.md#10-datamate-helper-baseline--implementation-plan). Card này mới đăng ký; chưa chạy Kaggle, chưa có điểm.

## Cấu hình dự kiến

| Thuộc tính | Giá trị đóng băng trước run |
|---|---|
| Preset / experiment / mode | `baseline_dev` / `dm-D-001-baseline-dev` / `helper` |
| Split / subset | `train` / `dev_tune`, seed 2026, tất cả câu của đúng 6 shop từ manifest |
| LLM | Qwen3-4B-Instruct-2507, revision commit cụ thể từ preflight |
| Backend / precision | một backend thực đã preflight, FP16, không quantization |
| Seed / context / thinking | 0 / 16384 token / off |
| Agent | output 1536 token, tối đa 1 repair, rules_k 8, context_chars 9000 |
| Executor | 60s, 6144 MB, 1 executor/GPU worker; tối đa 2 đồng thời toàn session |
| Batch / CPU | batch 16, HF internal 4; 1 CPU thread/worker |
| Cache | input/rulebook/loader/code fingerprints khớp; không chỉ COMPLETE |
| Session / finalize | 14400s tổng, reserve export ≥900s, dev `finalize=False` |

Danh sách shop/ID thật, dữ liệu/source hashes và độ phủ format/ngành điền từ manifest preflight. Không đoán 6 shop cụ thể từ seed khi chưa chạy code trên dataset version thật. Không dùng train_fit/final_holdout để tune trong run này; smoke kỹ thuật riêng không chấm để chọn cấu hình.

## Kiểm tra bắt buộc trước chạy lớn

- [ ] Manifest disjoint, seed/ID/source cố định; backend không mock, revision đã resolve.
- [ ] Label không vào prompt/cache/executor; fixture OS file isolation chặn path ghép chuỗi/symlink, vẫn đọc được shop/calendar/cache cần thiết. Landlock/kernel chưa được kiểm trên Kaggle; unsupported phải báo preflight failed trước load model.
- [ ] Cache attach được kiểm fingerprint và nguồn lỗi parser; không suy unanswerable từ lỗi đọc dữ liệu.
- [ ] Scorer dùng canonical record sau validate như file xuất; test PA/C/W/A, pending, đủ 9 kiểu và resume.

## Kết quả

**Chưa có số đo.** `—` chưa có nghĩa điểm bằng 0.

| Metric | Giá trị | Mốc so |
|---|---|---|
| N expected / completed / pending | — | đủ manifest |
| Correct / wrong / abstain | — | C+W+A=N khi hoàn tất |
| PA=(C−W)/N | — | always-abstain: 0 |
| Coverage=(C+W)/N | — | không dùng riêng để chọn baseline |
| Accuracy answered=C/(C+W) | — | báo NA nếu chưa trả lời câu nào |
| Invalid replaced / runtime error / repair / timeout | — | phân biệt abstain chủ động và lỗi |
| Setup/profile/load/infer/export; GPU-hours | — | số đo toàn notebook |
| Peak VRAM/GPU, RAM/CPU | — | số đo thực, không snapshot cuối |

Theo shop/gold type và nhóm lỗi: **chưa đo**. Không có group hiếm trong dev thì ghi n=0/NA, không mở holdout để bù. PA partial dùng đủ N, pending được báo riêng, chưa gọi baseline hoàn tất.

## Artifact cần lấy sau chạy

`manifest.json`, `preflight.json`, `config.json`, `cache_manifest.json`, profile/lineage/checks, predictions/traces/summary/merge_report/per_id_scores, `baseline_report.json`, `error_groups.jsonl`, output version, config/input/code/model hashes và GPU-hours. Repo chỉ ghi link/checksum, không commit dữ liệu/weights/submission.

## Phân tích lỗi

Chưa có. Sau inference mới dùng gold phân tích: loader/join, lọc đơn, doanh thu, kỳ thời gian, xếp hạng/đơn vị, nợ, kho/giá vốn, hiểu câu hỏi, code/format/unanswerable. Gán một nhóm chính/câu, tag phụ tùy chọn; lỗi loader đếm cả shop và số câu bị ảnh hưởng.

## Sự cố vận hành

Chưa chạy. Nếu timeout phiên, giữ pending và resume cùng config ở execution/card mới theo NNN kế tiếp. Thay model/backend/prompt/loader/rule policy sau khi đã xem dev làm đổi semantics → run mới, không ghi đè baseline.

## Kết luận và bước tiếp

- Chưa chốt baseline cho đến khi đủ mọi ID/pending=0, file-score khớp và validity checks qua; PA thấp hoặc âm vẫn phải báo.
- PA≤0: helper chưa chứng minh tốt hơn always-abstain trên dev này. Chọn run tiếp dựa nhóm lỗi; chưa tự thêm raw-agent/model lớn/verifier.
- 2 T4 chưa xác nhận ngân sách 6h/2 RTX 5090. Public/holdout là execution với card/config riêng sau đó, chưa nộp LB trong experiment này.
- Đã thêm kết luận vào `insights.md`: chưa; hiện chưa có bằng chứng thực nghiệm mới.
