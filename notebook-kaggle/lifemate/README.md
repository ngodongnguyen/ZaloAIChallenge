# LifeMate — notebook baseline

Tracking run và kết quả: [`.vscode/history_plan/lifemate.md`](../../.vscode/history_plan/lifemate.md).

`agent/lifemate_agent.ipynb`: ASR (PhoWhisper-medium) → agent Qwen3-4B-Instruct-2507 gọi tool qua `zaic_tools.ToolEnv` (mỗi mẫu một env, validate trước khi call) → `submission.jsonl`.

## Chạy để có file nộp Public Test

1. Kaggle → New Notebook → File → Import Notebook → `agent/lifemate_agent.ipynb`.
2. Add Input: dataset `thanhkhil/zaloai-lifemate-dataset`. Settings: **GPU T4 x2**, Internet **On**.
3. **Save Version → Save & Run All**. Xong tải `/kaggle/working/submission.jsonl` ở tab Output → nộp Public Leaderboard.

`CONFIG["mode"]`: `public` (mặc định) · `diagnostic` (train, transcript sạch; đặt `split="train"`, `subset="dev_seen"`) · `e2e` (train, transcript ASR) · `sft` (LoRA trên `train_fit` rồi eval dev).

Mỗi experiment một thư mục, push được ngay (`kaggle kernels push -p <thư mục>`; `kernel-metadata.json` chỉ có ở local, sửa `<owner>` trong `id` thành tài khoản sẽ push):

| Thư mục | `CONFIG_PRESET` | Run |
|---|---|---|
| `agent/` | `public` | lm-B-005-budgetfix-public (Add Input `<owner>/lifemate-asr-cache` nếu có) |
| `agent-dev/` | `dev` | lm-B-006-budgetfix-dev |
| `agent-sft/` | `sft` | lm-sft-007-lora-gold-vs-asr |
| `agent-qwen35/` | `dev_qwen35` | lm-B-008-qwen35-4b-dev |
| `agent-gipformer/` | `dev_gipformer` | lm-A-009-gipformer-dev |

Các notebook trong `agent-*/` là **bản sao của `agent/lifemate_agent.ipynb`, chỉ khác dòng `CONFIG_PRESET`**. Sửa code thì sửa `agent/` rồi chép lại sang các thư mục kia (giữ preset), để không lệch code giữa các experiment.

`CONFIG_PRESET` (cuối cell config) đặt sẵn bộ config, mỗi version Kaggle chạy một preset:
- `public`: toàn bộ public_test → `submission.jsonl`. Add Input dataset private `<owner>/lifemate-asr-cache` (cache PhoWhisper public) thì bỏ qua ASR (~30 phút).
- `dev`: e2e trên train `dev_seen+dev_unseen` (417 mẫu), điểm local ở `lm_agent/merge_report.json`.
- `sft`: ASR toàn bộ train → 2 biến thể LoRA (transcript gold / ASR) song song mỗi GPU → eval dev từng biến thể (`lm_agent/sft_eval_summary.json`). Dùng adapter cho public: Add Input output version `sft`, đặt `CONFIG["llm"]["adapter"]` = thư mục `.../lm_agent/sft_<biến thể>`.
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
