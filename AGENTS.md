# AGENTS.md — Zalo AI Challenge 2026

Rules chung cho mọi agent làm việc trong repo này (Codex, Claude Code, Antigravity `agy`, DeepSeek Harness `dsh`).
Cuộc thi có 2 track: **LifeMate** (track 1) và **DataMate** (track 2).

## Ý nghĩa từng folder

### `.vscode/docs/` — Thông tin cuộc thi (CHỈ ĐỌC)
- Chứa quy chế, thể lệ, đặc tả kỹ thuật 2 track, công thức chấm điểm, prompt mẫu.
- Đây là nguồn sự thật. Khi cần biết luật, format nộp bài, giới hạn model, metric → đọc ở đây, không đoán.
- Không sửa, không thêm file vào đây trừ khi user yêu cầu.

| File | Nội dung |
| :--- | :--- |
| `COMPETITION_OVERVIEW_2026.md` | Tổng quan, timeline, quy chế nộp bài & Docker |
| `AGENT_CHEATSHEET.md` | Tra cứu nhanh + các cạm bẫy |
| `TRACK1_LIFEMATE_SPECIFICATION.md` | Đặc tả track LifeMate |
| `TRACK2_DATAMATE_SPECIFICATION.md` | Đặc tả track DataMate |
| `OFFICIAL_METRICS_AND_MATCHING_RULES.md` | Công thức chấm điểm, quy tắc so khớp |
| `PROMPT_TEMPLATES_AND_EXAMPLES.md` | Prompt & few-shot mẫu |
| `zalo-ai-challenge-regulations-vi.pdf` | Thể lệ gốc — ưu tiên cao nhất khi có mâu thuẫn |

### `.vscode/extracted_images/` — Ảnh công thức trích từ đề (CHỈ ĐỌC)

### `planning-research/` — Nơi lên kế hoạch
- Mỗi track một thư mục con: `planning-research/lifemate/`, `planning-research/datamate/`.
- Dùng cho: chiến lược, kế hoạch thử nghiệm, nghiên cứu, ghi kết quả và phân tích lỗi.
- **Chỉ chứa tài liệu (Markdown)**, không chứa code chạy được.
- Cách tổ chức file bên trong: xem `planning-research/README.md`.

### `notebook-kaggle/` — Nơi implement & chạy thử nghiệm
- Mỗi track một thư mục con: `notebook-kaggle/lifemate/`, `notebook-kaggle/datamate/`.
- Chứa code thử nghiệm + `kernel-metadata.json` để push lên Kaggle.
- Code ở đây **chạy trên Kaggle**, không phải môi trường local. Đường dẫn dữ liệu dùng `/kaggle/input/...`.
- Mỗi thư mục track có `README.md` mô tả kernel của track đó.

### `Link_data.txt` — Link dataset Kaggle của 2 track

## Luồng làm việc

1. **Plan** trong `planning-research/<track>/` — đọc docs liên quan trong `.vscode/docs/` trước.
2. **Implement** theo plan vào `notebook-kaggle/<track>/`.
3. **Chạy trên Kaggle** (push kernel, lấy output).
4. **Ghi kết quả** quay lại `planning-research/<track>/`.

## Kaggle CLI & pool 9 tài khoản

Máy đã cài sẵn `kaggle` CLI và công cụ quản lý nhiều tài khoản `kaggle-acc` (alias `kaggle-switch`).

**Danh sách tài khoản**: nằm trong file local `AGENTS.local.md` ở root repo (gitignored, chỉ có trên máy chủ repo). Nếu file tồn tại, **đọc nó trước khi dùng Kaggle**. Nếu không có, chạy `kaggle-acc list` để xem các tài khoản trên máy.

**Không bao giờ ghi username Kaggle vào file được git theo dõi** (AGENTS.md, README, docs, plan, commit message). Trong tài liệu dùng placeholder `<user>` / `<owner>`.

**Lệnh quản lý tài khoản**

```bash
kaggle-acc list                          # xem 9 tài khoản + tài khoản đang active
kaggle-acc current                       # tài khoản đang active
kaggle-acc run <user> kaggle <args...>   # chạy 1 lệnh dưới tài khoản <user> (KHÔNG đổi tài khoản global)
kaggle-switch <user_or_number>           # đổi tài khoản global (ghi đè ~/.kaggle/kaggle.json)
```

**Lệnh chạy thử nghiệm** (`<dir>` là thư mục chứa `kernel-metadata.json`, ví dụ `notebook-kaggle/lifemate`)

```bash
kaggle kernels push -p <dir>                       # đẩy code lên và chạy kernel
kaggle kernels status <owner>/<kernel-slug>        # xem trạng thái lần chạy mới nhất
kaggle kernels output <owner>/<kernel-slug> -p <dir>/output   # tải output (submission.jsonl, log) về
kaggle datasets download -d thanhkhil/zaloai-lifemate-dataset # tải dataset (tương tự cho datamate)
```

**Rules khi dùng Kaggle**

- Ưu tiên `kaggle-acc run <user> ...` thay cho `kaggle-switch`. `kaggle-switch` đổi tài khoản cho cả máy, làm hỏng việc của agent khác đang chạy song song.
- `kernel-metadata.json` **bị gitignore** vì chứa username. Git chỉ lưu `kernel-metadata.example.json` (có placeholder `<owner>`). Nếu chưa có `kernel-metadata.json`, copy từ file `.example.json` rồi thay `<owner>` bằng username sẽ push. Thêm field mới vào metadata thì cập nhật cả file `.example.json` (giữ `<owner>`).
- `id` trong metadata có dạng `<owner>/<slug>`. Muốn push bằng tài khoản khác thì phải đổi `<owner>` thành đúng username đó, nếu không push sẽ lỗi.
- Hết quota GPU / giới hạn rate → chuyển sang tài khoản khác trong pool rồi push lại. Ghi lại đã chạy bằng tài khoản nào khi log kết quả vào `planning-research/<track>/`.
- Không in, không commit API key Kaggle; không sửa `~/.kaggle/kaggle.json` bằng tay.

## Rules chung

- Luôn xác định đang làm track nào; không trộn code/plan của LifeMate và DataMate.
- Code thuộc về `notebook-kaggle/`, kế hoạch/ghi chép thuộc về `planning-research/`. Không đặt sai chỗ.
- Không commit dữ liệu, model weights, file audio, `submission.jsonl`.

## Git: commit & push

- Chỉ commit / push khi user yêu cầu.
- **Commit chỉ đứng tên user.** Tuyệt đối **không** thêm dòng `Co-Authored-By:` (hay bất kỳ trailer nào) ghi tên agent/AI vào commit message.
- **Không** ghi attribution của agent trong commit message hoặc mô tả PR (ví dụ "Generated with ...", "🤖 ...", tên model/agent).
- Không sửa `git config user.name` / `user.email`. Không force-push trừ khi user yêu cầu rõ ràng.
- Commit message theo kiểu `feat:`, `fix:`, `docs:`, `chore:` như lịch sử repo.
