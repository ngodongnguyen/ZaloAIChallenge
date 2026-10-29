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

### `.vscode/history_plan/` — Tracking (mỗi track một file)
- `lifemate.md`, `datamate.md`: bảng mọi run Kaggle (cách làm → `run_id` → kết quả → kết luận), log nộp Public LB, và kiến thức đã kiểm chứng / ngõ cụt / nhóm lỗi / ý tưởng.
- Không tạo thêm file: không run card riêng, không plan riêng, không snapshot. Quy ước: `.vscode/history_plan/README.md`.

### `notebook-kaggle/` — Nơi implement & chạy thử nghiệm
- Mỗi track một thư mục con: `notebook-kaggle/lifemate/`, `notebook-kaggle/datamate/`.
- Chứa code thử nghiệm + `kernel-metadata.json` để push lên Kaggle.
- Code ở đây **chạy trên Kaggle**, không phải môi trường local. Đường dẫn dữ liệu dùng `/kaggle/input/...`.
- Mỗi thư mục track có `README.md` mô tả kernel của track đó.

### `Link_data.txt` — Link dataset Kaggle của 2 track

## Luồng làm việc

1. **Plan** — đọc docs liên quan trong `.vscode/docs/` và file tracking `.vscode/history_plan/<track>.md` của track. Kế hoạch trao đổi trực tiếp với user, không tạo file plan.
2. **Implement** theo plan vào `notebook-kaggle/<track>/`.
3. **Chạy trên Kaggle** (push kernel, lấy output).
4. **Ghi kết quả** vào `.vscode/history_plan/<track>.md`.

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
- `kernel-metadata.json` **bị gitignore** vì chứa username, chỉ có trên máy local; không tạo file `.example.json`. Thư mục nào chưa có thì viết mới (copy từ thư mục notebook khác rồi sửa `id`, `title`, `code_file`).
- `id` trong metadata có dạng `<owner>/<slug>`. Muốn push bằng tài khoản khác thì phải đổi `<owner>` thành đúng username đó, nếu không push sẽ lỗi.
- Hết quota GPU / giới hạn rate → chuyển sang tài khoản khác trong pool rồi push lại. Ghi lại đã chạy bằng tài khoản nào khi log kết quả vào `.vscode/history_plan/<track>.md`.
- Không in, không commit API key Kaggle; không sửa `~/.kaggle/kaggle.json` bằng tay.

## Tận dụng Kaggle 2× T4 (bắt buộc khi viết notebook GPU)

- Model vừa một T4 (16 GB, FP16, không BF16) → **một worker / GPU**, mỗi worker một bản model, shard dữ liệu theo hash ID ổn định (`CUDA_VISIBLE_DEVICES=i`). Chỉ chia model qua 2 card khi model không vừa một card (2 T4 không có NVLink, tensor-parallel chậm).
- Gom batch để vLLM sinh nhiều câu một lượt; **không để GPU chờ CPU**: việc CPU (chạy code, đọc file, tool call) phải chạy song song với lượt sinh kế tiếp hoặc đủ nhỏ.
- Mỗi run ghi `s/mẫu`, peak VRAM, thời gian load model vào `summary.shard*.json` → dòng run trong `.vscode/history_plan/<track>.md`; chỗ GPU còn rảnh đã biết: mục "Chưa kiểm / ý tưởng chờ thử" trong cùng file.

## Thư viện cài thêm: wheels dùng chung (bắt buộc với mọi agent)

Mục tiêu: notebook thí nghiệm **không tải lại thư viện từ mạng mỗi lần chạy**. Mọi thư viện ngoài image Kaggle nằm sẵn trong bộ wheel của notebook `notebook-kaggle/wheels/build_wheels.ipynb` (trên Kaggle: `<owner>/build-wheels`).

- **Hai danh sách duy nhất**, ở 2 cell đầu của `build_wheels.ipynb` (mỗi dòng ghi notebook nào dùng):
  - `requirements.txt` → cài vào **môi trường gốc** Kaggle. Không được thêm gói kéo theo torch/CUDA mới.
  - `requirements-vllm.txt` → cài vào **venv riêng** `/tmp/venv_vllm` (vLLM và mọi gói worker LLM cần). Ghim phiên bản đã chạy được (`vllm==0.32.0`).
- **Không bao giờ `pip install vllm` vào môi trường gốc**: nó thay torch (2.13+cu130 trên image 2026-10) → torchaudio lệch → ASR / transformers chết (run lm-B-002, dm-D-002).
- Notebook thí nghiệm: Add Input output `build-wheels` (metadata `kernel_sources: ["<owner>/build-wheels"]`) và chỉ cài bằng hàm `pip_install(...)` có sẵn ở cell cài đặt (ưu tiên `--no-index --find-links /kaggle/input/**/wheels`, lỗi thì cài từ mạng). **Không viết `pip install` trần.**

**Checklist khi experiment mới cần thư viện mới** (làm trong cùng thay đổi, không để lần sau):
1. Thêm dòng vào đúng danh sách trong `build_wheels.ipynb` (môi trường gốc hay venv vLLM), kèm tên notebook dùng nó.
2. Thêm một dòng kiểm tra import / chạy thử vào dict `checks` của cell smoke test.
3. Trong notebook thí nghiệm: gọi `pip_install("<gói>")`, không cài trần.
4. Báo người dùng chạy lại `build-wheels` (version mới) và xem `wheels/SMOKE.json` phải toàn OK; ghi phiên bản đã chạy được vào mục "Đã xác nhận" của `.vscode/history_plan/<track>.md` và ghim trong danh sách.
5. Image Kaggle đổi (Python / CUDA / torch khác) → build lại wheels. Notebook nên đặt *Environment: Pin to original environment*.

## Tracking thực nghiệm (bắt buộc với mọi agent)

Mỗi track **một file** `.vscode/history_plan/<track>.md`. Áp dụng cho Codex, Claude Code, `agy`, `dsh` và mọi subagent.

- Trước khi đề xuất / chạy thí nghiệm: đọc file tracking của track để không lặp lại run cũ hay ngõ cụt.
- Mỗi lần chạy trên Kaggle (kể cả hỏng) = **một dòng** trong bảng Runs; nộp Public LB = một dòng trong bảng nộp. Quy ước `run_id` và account bí danh `acc<N>`: xem `.vscode/history_plan/README.md`.
- Kết luận đã kiểm chứng → một dòng trong mục "Đã xác nhận" (ngày + run_id/nguồn).
- **Không tạo file mới** cho plan, run card, snapshot; không tạo lại thư mục `planning-research/`. Bản cũ xem bằng `git log -p`.

## Tra cứu model và paper (MCP có sẵn ở cả 4 agent)

- **Hugging Face MCP** (`huggingface`): kiểm model trước khi dùng — số tham số thật, ngày phát hành (LifeMate: trước 26/10/2026; DataMate: trước 01/10/2026), license, revision, file weights. Ghi kết quả vào mục "Đã xác nhận" của `.vscode/history_plan/<track>.md`.
- **arXiv MCP** (`arxiv`, gói `@cyanheads/arxiv-mcp-server`): tìm và đọc paper (`arxiv_search_papers`, `arxiv_read_paper`). Dẫn link arXiv trong `.vscode/history_plan/<track>.md` khi dùng ý tưởng từ paper.
- Không có MCP trong phiên hiện tại thì dùng `fetch` tới `huggingface.co` / `arxiv.org`; không đoán số tham số hay ngày phát hành từ trí nhớ.

## Rules chung

- Luôn xác định đang làm track nào; không trộn code/plan của LifeMate và DataMate.
- Code thuộc về `notebook-kaggle/`, ghi chép kết quả thuộc về `.vscode/history_plan/<track>.md`. Không đặt sai chỗ.
- Không commit dữ liệu, model weights, file audio, `submission.jsonl`.

## Git: commit & push

- Chỉ commit / push khi user yêu cầu.
- **Commit chỉ đứng tên user.** Tuyệt đối **không** thêm dòng `Co-Authored-By:` (hay bất kỳ trailer nào) ghi tên agent/AI vào commit message.
- **Không** ghi attribution của agent trong commit message hoặc mô tả PR (ví dụ "Generated with ...", "🤖 ...", tên model/agent).
- Không sửa `git config user.name` / `user.email`. Không force-push trừ khi user yêu cầu rõ ràng.
- Commit message theo kiểu `feat:`, `fix:`, `docs:`, `chore:` như lịch sử repo.
