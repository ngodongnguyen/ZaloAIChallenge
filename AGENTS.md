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

### `.vscode/history_plan/` — Lịch sử phiên bản kế hoạch
- Lưu nguyên văn plan trước mỗi đợt chỉnh sửa/thay thế, tách theo track.
- `INDEX.md` ghi nguồn, thời điểm lưu, lý do thay đổi, SHA-256 và liên kết tới bản kế tiếp.
- Snapshot cũ là bất biến; đây là lịch sử để tracing/monitor, không thay plan đang hiệu lực trong `planning-research/`.

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

## Thư viện cài thêm: wheels dùng chung

- Mọi thư viện ngoài image Kaggle được tải sẵn thành wheel bởi `notebook-kaggle/wheels/build_wheels.ipynb` (Kaggle: `<owner>/zalo2026-wheels`). Hai danh sách duy nhất trong notebook đó: `requirements.txt` (cài vào môi trường gốc, không được kéo torch/CUDA mới) và `requirements-vllm.txt` (cài vào venv riêng `/tmp/venv_vllm`).
- **Không bao giờ `pip install vllm` vào môi trường gốc Kaggle**: nó thay torch (2.13+cu130 trên image 2026-10) và làm hỏng torchaudio → ASR transformers chết. Worker LLM chạy bằng python của venv (`LLM_PYTHON`), ASR và fallback transformers dùng môi trường gốc.
- Notebook thí nghiệm Add Input output `zalo2026-wheels` (metadata: `kernel_sources`) và cài bằng hàm `pip_install(...)` ở cell cài đặt: ưu tiên `--no-index --find-links /kaggle/input/**/wheels`, lỗi hoặc không có thì cài từ mạng. Không viết `pip install` trần trong notebook mới.
- **Thêm thư viện mới**: thêm dòng vào đúng danh sách → chạy lại `zalo2026-wheels` (version mới) → xem `wheels/SMOKE.json` phải OK → notebook dùng nó lấy version output mới. Khi đã biết bản chạy được thì ghim phiên bản (`vllm==x.y.z`).
- Image Kaggle đổi (Python / CUDA / torch khác) thì build lại wheels; notebook nên đặt *Environment: Pin to original environment*.

## Ghi lịch sử thực nghiệm (bắt buộc với mọi agent)

- Trước khi đề xuất / chạy thí nghiệm: đọc `planning-research/<track>/insights.md` và `experiments/INDEX.md` để không lặp lại run cũ hay ngõ cụt.
- Mỗi lần chạy trên Kaggle (kể cả hỏng) có một run card `planning-research/<track>/experiments/YYYY-MM-DD_<run_id>.md` theo `planning-research/_RUN_TEMPLATE.md` và một dòng trong `INDEX.md`. Quy ước đặt `run_id`, account bí danh `acc<N>`, log nộp Public LB: xem `planning-research/README.md`.
- Kết luận đã kiểm chứng → một dòng trong `insights.md` (ngày + run_id + nguồn).

## Lưu lịch sử plan trước khi sửa (bắt buộc với mọi agent)

Áp dụng cho **Codex, Claude Code, Antigravity (`agy`), `dsh` và mọi subagent** khi sửa hoặc thay thế `planning-research/<track>/planning.md` hay một tài liệu plan đang hiệu lực khác. **Không sửa trực tiếp plan cũ khi chưa lưu bản trước đó.** “Mỗi track một planning.md” nghĩa là một bản đang hiệu lực; các phiên bản trước phải được giữ lại.

1. **Trước khi edit**, đọc bản hiện tại trên ổ đĩa và lưu nguyên văn vào `.vscode/history_plan/<track>/YYYY-MM-DD_HHMMSS+0700__<tên-file>__<lý-do>.md`. Thay toàn bộ plan thì chuyển bản cũ vào lịch sử trước khi tạo bản mới; chỉnh một phần thì snapshot đầy đủ trước khi sửa. Một snapshot/file cho mỗi đợt chỉnh sửa liên tục; nếu có thay đổi từ agent khác, đọc lại và lưu preimage mới.
2. Tạo file archive mới, **không ghi đè** khi trùng tên; thêm hậu tố duy nhất. Kiểm bytes/SHA-256 của archive khớp bản trước-edit rồi mới sửa. Lưu thất bại → giữ nguyên plan và báo lỗi, không tiếp tục ghi đè.
3. Thêm dòng `.vscode/history_plan/INDEX.md`: thời điểm thực có múi giờ, track, đường dẫn nguồn, snapshot, SHA-256, lý do và bản kế tiếp. Nếu khôi phục từ Git thì ghi rõ nguồn/revision và thời điểm khôi phục; không giả mốc thời gian gốc.
4. Plan đang hiệu lực phải liên kết ngược về archive/INDEX. Không chỉnh hoặc xóa snapshot cũ, kể cả sửa link; đính chính metadata bằng dòng mới trong INDEX.
5. **Không lấy HEAD/staging thay cho bản trên ổ đĩa trong quy trình bình thường**: phải giữ cả thay đổi chưa commit của người dùng/agent khác. Git chỉ là nguồn khôi phục khi cần và phải ghi provenance. Không thay đổi Git index hoặc tự commit/push để lưu lịch sử.
6. Quy tắc này áp dụng cả việc bổ sung nội dung vào plan; không yêu cầu snapshot đệ quy của chính archive INDEX/README hoặc mỗi cập nhật vận hành run card/insights. Run card vẫn có lịch sử riêng theo phần trên.

Quy trình đầy đủ: [`.vscode/history_plan/README.md`](.vscode/history_plan/README.md). Claude Code đọc quy tắc này qua `CLAUDE.md`; agy có entry point `GEMINI.md`; Codex đọc `AGENTS.md`.

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
