# DataMate — notebook baseline

Tracking run và kết quả: [`.vscode/history_plan/datamate.md`](../../.vscode/history_plan/datamate.md).

`helper-agent/datamate_helper_agent.ipynb`: làm sạch dữ liệu mọi shop (CPU) → Qwen3-4B-Instruct-2507 viết code pandas trên bảng sạch + helper → chạy code trong process riêng → format 9 kiểu đáp án → `submission.jsonl`.

## Chạy để có file nộp Public Test

1. Kaggle → New Notebook → File → Import Notebook → `helper-agent/datamate_helper_agent.ipynb`.
2. Add Input: dataset `thanhkhil/zaloai-datamate-dataset`. Settings: **GPU T4 x2**, Internet **On**.
3. **Save Version → Save & Run All**. Xong tải `/kaggle/working/submission.jsonl` ở tab Output → nộp Public Leaderboard.

Mặc định `split="public_test"`, `finalize=True` (câu chưa kịp xử lý ghi `abstain`, 0 điểm). Đo trên train: `split="train"`, `subset="dev_tune"` → `merge_report.json` có PA theo shop / kiểu đáp án.
Hết giờ giữa chừng: chạy version mới, attach output cũ làm input, điền đường dẫn vào `resume_from_dirs` (cùng config).

Code nằm trong các cell `%%writefile` (thư viện `dmlib` ghi vào `/kaggle/working/lib`); sửa logic thì sửa cell đó.

## Preset dev cho helper/raw

Chọn `CONFIG_PRESET` trong cell CONFIG:

| Preset | Phép đo | Output |
|---|---|---|
| `baseline_dev` | Helper Qwen3-4B trên đủ câu của 6 shop dev_tune | `/kaggle/working/dm_baseline` |
| `raw_dev` | Cùng manifest/model/budget, tự đọc file thô | `/kaggle/working/dm_raw_baseline` |

Cả hai giữ `finalize=False`, seed 2026, FP16, context16384/output1536, một repair và một executor mỗi GPU. Public vẫn là preset riêng. Dùng `DEV_OPTIONS` trong cell CONFIG để đặt `execution_id`, `parent_execution_id`, `resume_from_dirs`, `preflight_dirs`, `control_dir`, `profiler_cache_dirs`.

Chuẩn bị preflight dùng chung bằng `baseline_dev`. Có thể đặt `DEV_OPTIONS={"preflight_only": True}` để xuất preflight mà chưa chạy đủ helper dev. Preflight kiểm **cả raw và helper**, model thật trên hai T4 và executor CSV/Excel/SQLite/parquet. `raw_dev` attach thư mục preflight đã lưu qua `preflight_dirs`; chỉ smoke ngắn raw và không build/attach cache helper. Raw không phải chờ điểm helper. Khi thiếu/mismatch preflight, raw dừng trước model để tránh tạo phép so sánh khác cấu hình.

Ví dụ chạy raw sau khi đã có output control:

```python
DEV_OPTIONS = {
    "preflight_dirs": ["/kaggle/input/<control-output>/dm_baseline"],
    "control_dir": "/kaggle/input/<control-output>/dm_baseline",
    # "resume_from_dirs": ["/kaggle/input/<raw-output-trước>/dm_raw_baseline"],
    # "execution_id": "<run-id-mới>",
    # "parent_execution_id": "<run-id-trước>",
}
```

`control_dir` dùng comparison offline. So sánh chỉ complete khi hai bên đủ cùng ID, common contract khớp và canonical predictions nguyên vẹn. Helper partial không cản raw inference; comparison đợi control đủ. Không ghép output từ code/model/budget khác nhau.

Profiler cache kiểm source/rulebook/loader code và checksum artifact, không chỉ marker `COMPLETE`. Raw không đọc bảng chuẩn/helper modules. Executor cài Landlock trong process riêng **trước preload/code**; chỉ đọc đúng shop/calendar/runtime và cache nếu helper, chỉ ghi scratch. Probe không enforce được thì dừng trước model; không hạ xuống regex. CPU local hỗ trợ Landlock không chứng minh kernel Kaggle cũng hỗ trợ.

`states.shard*.jsonl` lưu `first_generated`, `repair_pending`, `completed`; mỗi câu được flush ngay. Backend lỗi lúc repair giữ pending và first reply; resume chỉ làm repair còn thiếu. `canonical_predictions.jsonl` là nguồn chung của JSONL xuất và scorer: record invalid thay bằng abstain cũng được chấm chính abstain đó. Report tách pending, abstain chủ động, abstain sau lỗi, C/W/A/coverage/PA và nhóm lỗi.

Add Input dataset + `build-wheels`, T4 x2. Calibration tối đa1200s, phiên14400s tính cả setup, reserve export900s. Giữ original Kaggle environment; vLLM ở venv riêng, executor dùng môi trường gốc. Không có dependency production mới ngoài danh sách wheels hiện tại.

Kiểm CPU local và cùng regression nhúng trong notebook:

```bash
python notebook-kaggle/verify_notebooks.py datamate
```

Regression dùng model giả và dữ liệu fixture; PA của fixture không phải PA model thật. GPU smoke, đầy đủ dev và hiệu lực Landlock Kaggle vẫn cần kiểm khi thực thi. Giữ dòng run `planned` trong `.vscode/history_plan/datamate.md` tới lúc bắt đầu Kaggle; không tự nộp leaderboard.

## Theo dõi khi console đứng sau profiler

Notebook hiện stream `worker*.log` ra console trong lúc worker còn chạy, kèm heartbeat mặc định mỗi 15 giây. `status.shard*.json` cho biết stage, số câu hoàn tất và thời điểm cập nhật gần nhất. Có log riêng lúc nạp model, backend thực tế sau khi nạp, bắt đầu/kết thúc generation và từng câu đã ghi checkpoint.

Ví dụ: `stage=load_model done=0/500 last_update=60s ago` cho thấy worker vẫn ở bước nạp model; `stage=generation` là đang sinh code; `done` tăng sau khi prediction đã được flush. Heartbeat xác nhận process còn sống; tiến độ thực phải xem `done`, stage và tuổi của lần cập nhật. Log đầy đủ vẫn giữ trong `worker*.log`. Preflight và profiler cache cũng có thông báo tiến độ.

Cấu hình inference hiện giữ nguyên: public chạy toàn bộ input, batch16/GPU, executor1/GPU, timeout60s/attempt và tối đa một repair. Nếu cả16 câu đều timeout ở cả hai attempt, riêng executor có thể tốn32 phút cho một batch/GPU (chưa tính generation); đây là giới hạn suy từ config, không phải tốc độ đã đo. Log mới giúp phân biệt nạp model, generation và chờ executor trước khi tối ưu cấu hình.

Thay đổi quan sát này dùng thư viện chuẩn, không cần bổ sung wheels. Cần version notebook mới để có log mới; session Kaggle đã khởi chạy vẫn dùng source của version đó.
