# Planning — Zalo AI Challenge 2026

```
planning-research/
├── _RUN_TEMPLATE.md            mẫu run card (dùng chung 2 track)
├── lifemate/ | datamate/
│   ├── planning.md             chiến lược, thí nghiệm dự kiến, vận hành Kaggle, mốc chốt
│   ├── insights.md             kiến thức đúc kết: đã xác nhận / ngõ cụt / nhóm lỗi / giả định chưa kiểm / ý tưởng
│   └── experiments/
│       ├── INDEX.md            bảng mọi run + tốt nhất hiện tại + log nộp Public LB + GPU-hour theo account
│       └── YYYY-MM-DD_<run_id>.md   run card của từng lần chạy
```

Đọc theo thứ tự trước khi làm gì mới: `planning.md` → `insights.md` → `experiments/INDEX.md`. Đặc tả cuộc thi ở `.vscode/docs/`, notebook ở `notebook-kaggle/<track>/`.

## Lịch sử phiên bản plan

`planning.md` là bản đang hiệu lực của mỗi track. **Trước mỗi đợt sửa hoặc thay thế**, lưu nguyên văn bản hiện tại vào [`.vscode/history_plan/`](../.vscode/history_plan/README.md), kiểm SHA-256, rồi ghi [sổ phiên bản](../.vscode/history_plan/INDEX.md) gồm thời điểm, nguồn, lý do và bản kế tiếp. Chỉ sau khi lưu thành công mới edit; không ghi đè hoặc xóa snapshot cũ.

Snapshot dùng tên `<track>/YYYY-MM-DD_HHMMSS+0700__<tên-file>__<lý-do>.md`; plan mới liên kết ngược về lịch sử. Lưu cả thay đổi chưa commit trên ổ đĩa, không thay bằng HEAD/staging. Đây là lịch sử **kế hoạch**, tách với run card thực nghiệm ở `experiments/`; Codex, Claude Code và agy cùng tuân thủ mục tương ứng trong [AGENTS.md](../AGENTS.md).

## Quy ước ghi thực nghiệm

1. **Bắt đầu run**: tạo `experiments/YYYY-MM-DD_<run_id>.md` từ `_RUN_TEMPLATE.md` (status `running`) và thêm một dòng vào `INDEX.md`. Có thể đăng ký trước với status `planned`; khi đó thời điểm chạy, account, version, hash và kết quả chưa có để `null`, chỉ đổi sang `running` lúc thực thi thật.
2. **`run_id`**: `<lm|dm>-<lane>-<NNN>-<mô-tả-ngắn>`, NNN tăng dần theo track (vd `lm-B-001-baseline-public`, `dm-D-002-qwen3-14b-4bit`). Không đổi run_id sau khi tạo.
3. **Một thay đổi chính mỗi run**, ghi `parent_run` + `change_vs_parent` để so được và dựng được cây lineage.
4. **Account**: chỉ ghi bí danh `acc1`…`acc9` (số thứ tự trong pool). Username thật chỉ nằm trong `AGENTS.local.md`.
5. **Kết thúc run**: điền kết quả từ output notebook (`merge_report.json`, `summary.shard*.json`, `config_hash`), link output version, GPU-hour; cập nhật dòng INDEX và bảng "Tốt nhất hiện tại".
6. **Nộp Public LB**: thêm một dòng vào bảng nộp trong INDEX (ngày, lần thứ mấy trong ngày, run_id, điểm). Tối đa 5 lần/ngày/track.
7. **Rút kinh nghiệm**: điều đã kiểm chứng, ngõ cụt, nhóm lỗi lặp lại → một dòng trong `insights.md` có ngày + run_id.
8. Run hỏng / bỏ dở vẫn giữ card (status `failed` / `abandoned` + lý do) để agent sau không chạy lại cùng cấu hình.
9. Chỉ Markdown ở đây; trace, checkpoint, submission nằm trong output Kaggle, không commit.

## Hai experiment tiếp theo — thiết kế đã duyệt 2026-10-29

Ưu tiên hiện tại là đo baseline bằng model thật trên notebook có sẵn. Hai dry-run cũ dùng mock/oracle, chưa là điểm baseline. Checklist triển khai nằm trong mục 9–10 của từng file planning; chưa sửa code/chạy GPU trong đợt lập kế hoạch này.

| Track | Experiment planned | Câu hỏi cần đo | Plan / run card |
|---|---|---|---|
| LifeMate | `lm-B-001-baseline-paired-dev` | Cùng dev_seen: planner với transcript sạch so với PhoWhisper cả history/final | [plan](lifemate/planning.md#10-lifemate-paired-dev--implementation-plan) · [card](lifemate/experiments/2026-10-29_lm-B-001-baseline-paired-dev.md) |
| DataMate | `dm-D-001-baseline-dev` | Helper-agent Qwen3-4B trên 6 shop dev_tune: PA, C/W/A, coverage và lỗi | [plan](datamate/planning.md#10-datamate-helper-baseline--implementation-plan) · [card](datamate/experiments/2026-10-29_dm-D-001-baseline-dev.md) |

Mỗi execution Kaggle/version đều có card riêng, kể cả resume. `experiment` định danh phép đo; `run_id` định danh một execution và tăng NNN theo track. Không ghép output khác semantic config. Public/holdout là các lần chạy riêng về sau, chưa tự nộp LB.
