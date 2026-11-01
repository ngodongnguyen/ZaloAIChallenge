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
- **Tên file notebook không được trùng** trong toàn bộ `notebook-kaggle/`, kể cả khi ở khác thư mục (Kaggle Import Notebook và tab Output chỉ hiện tên file). Đặt `<track>_<cách làm>_<biến thể>.ipynb`, ví dụ `lifemate_agent_dev.ipynb`, `lifemate_agent_sft.ipynb`. Tạo bản sao cho experiment mới thì đổi tên file luôn, và sửa `code_file` trong `kernel-metadata.json` cho khớp (sai tên thì `kaggle kernels push` lỗi).
- Mọi notebook của track phải có sẵn input dataset cuộc thi: xem mục "Input dataset trong notebook" bên dưới.

### `Link_data.txt` — Link dataset Kaggle của 2 track

### `dataset/` — Bản local dataset cuộc thi của 2 track (CHỈ ĐỌC, gitignored)
- Đã tải sẵn trên máy, cùng nội dung với 2 dataset Kaggle `thanhkhil/zaloai-<track>-dataset`. Đường dẫn tính từ root repo:

| Track | Thư mục local | Nội dung |
| :--- | :--- | :--- |
| LifeMate | `dataset/lifemate/train+public_test/` | `train/`, `public_test/`, `README.md`, `VERSION` |
| DataMate | `dataset/datamate/` | `train/` (`questions.jsonl`, `answers.jsonl`, `calendar.csv`, `rulebook.md`, `data/`), `public_test/` (như train, không có `answers.jsonl`) |

- Muốn xem format dữ liệu, làm EDA, đếm mẫu, đọc câu hỏi / đáp án mẫu, hay thử nhanh logic không cần GPU → đọc trực tiếp ở đây, **không tải lại** bằng `kaggle datasets download`.
- Chỉ dùng thư mục của track đang làm; không trộn dữ liệu 2 track.
- Không sửa, không xóa, không thêm file vào `dataset/`; output, cache, file tạm để ở scratchpad hoặc thư mục ngoài repo.
- Không commit (`dataset/` đã nằm trong `.gitignore`).
- Notebook trong `notebook-kaggle/` chạy trên Kaggle nên vẫn đọc `/kaggle/input/...`; không hard-code đường dẫn `dataset/...` vào notebook. Script chạy local thì nhận đường dẫn qua tham số / biến môi trường, mặc định trỏ vào `dataset/<track>/`.

## Luồng làm việc

1. **Plan** — đọc docs liên quan trong `.vscode/docs/` và file tracking `.vscode/history_plan/<track>.md` của track. Kế hoạch trao đổi trực tiếp với user, không tạo file plan.
2. **Implement** theo plan vào `notebook-kaggle/<track>/`.
3. **Chạy trên Kaggle** (push kernel, lấy output).
4. **Ghi kết quả** vào `.vscode/history_plan/<track>.md`.

## Sinh dữ liệu synthetic LifeMate (khi user bảo "sinh dữ liệu" / "sinh du lieu" / "sinh data" / "gen data")

User chỉ cần nói một câu ngắn, có dấu hoặc không dấu, ví dụ "sinh dữ liệu", "sinh du lieu", "sinh thêm data lifemate", "gen data app tv, music 20 mau". Khi đó:

1. Mở `synthetic-data-llm-generate/prompt_generate.md`, coi toàn bộ nội dung là yêu cầu của user và làm theo từ đầu đến cuối, không hỏi lại.
2. User có nói app / trọng tâm / số mẫu thì truyền vào `claim_group.py` (`--apps`, `--focus`, `--n`) đúng như mục 1 của prompt; không nói thì để script tự chọn.
3. Làm theo ba vai trò author → solver ở phiên mới → reviewer ở phiên mới tại mục 6 của prompt. Chỉ release mẫu đã duyệt; không tự nhận vai trò khác để duyệt mẫu mình viết. Phần TTS là pipeline riêng.

Chi tiết công cụ: `synthetic-data-llm-generate/README.md`.

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

## Input dataset trong notebook (bắt buộc với mọi agent)

User **Import Notebook thủ công** trên Kaggle; Import chỉ đọc file `.ipynb`, không đọc `kernel-metadata.json`. Vì vậy mỗi notebook `notebook-kaggle/<track>/**/*.ipynb` (trừ `wheels/build_wheels.ipynb`) phải có khối `metadata.kaggle` dưới đây trong metadata của notebook (cùng cấp `kernelspec`), để Kaggle tự gắn dataset, GPU T4 x2 và Internet On. Áp dụng khi tạo notebook mới, sao chép notebook, hay ghi lại notebook bằng script (đừng để `nbformat` / `json.dump` làm mất khối này).

LifeMate (`thanhkhil/zaloai-lifemate-dataset`):

```json
"kaggle": {
 "accelerator": "nvidiaTeslaT4",
 "dataSources": [
  {"databundleVersionId": 22447570, "datasetId": 12872794, "sourceId": 21271194, "sourceType": "datasetVersion"}
 ],
 "isGpuEnabled": true,
 "isInternetEnabled": true,
 "language": "python",
 "sourceType": "notebook"
}
```

DataMate (`thanhkhil/zaloai-datamate-dataset`): như trên, thay phần tử `dataSources` bằng

```json
{"databundleVersionId": 22447572, "datasetId": 12872796, "sourceId": 21271196, "sourceType": "datasetVersion"}
```

- Mỗi notebook chỉ gắn dataset của track mình, không trộn 2 track.
- Đồng thời giữ `"dataset_sources": ["thanhkhil/zaloai-<track>-dataset"]` trong `kernel-metadata.json` (cho `kaggle kernels push`).
- Id ở trên là version 1 của 2 dataset (kiểm 2026-10-29). BTC ra version mới thì lấy lại id (trang dataset → request `GetDatasetBasics`: `datasetVersionId` = `sourceId`, `versionId` = `databundleVersionId`) rồi sửa ở đây và mọi notebook.
- Không thêm input private (`build-wheels`, `lifemate-asr-cache`) vào khối này: account khác không gắn được.
- Kiểm tra trước khi báo xong:

```bash
python3 -c "import json,glob;[print(p, [d['datasetId'] for d in json.load(open(p))['metadata'].get('kaggle',{}).get('dataSources',[])] or 'THIẾU') for p in sorted(glob.glob('notebook-kaggle/*/*/*.ipynb'))]"
```

## Tận dụng Kaggle 2× T4 (bắt buộc khi viết notebook GPU)

- Model vừa một T4 (16 GB, FP16, không BF16) → **một worker / GPU**, mỗi worker một bản model, shard dữ liệu theo hash ID ổn định (`CUDA_VISIBLE_DEVICES=i`). Chỉ chia model qua 2 card khi model không vừa một card (2 T4 không có NVLink, tensor-parallel chậm).
- Gom batch để vLLM sinh nhiều câu một lượt; **không để GPU chờ CPU**: việc CPU (chạy code, đọc file, tool call) phải chạy song song với lượt sinh kế tiếp hoặc đủ nhỏ.
- Mỗi run ghi `s/mẫu`, peak VRAM, thời gian load model vào `summary.shard*.json` → dòng run trong `.vscode/history_plan/<track>.md`; chỗ GPU còn rảnh đã biết: mục "Chưa kiểm / ý tưởng chờ thử" trong cùng file.

## Thư viện cài thêm: chỉ dùng wheels cho gói nặng (bắt buộc với mọi agent)

**Quy ước người dùng, 2026-10-29 — áp dụng cho Codex, Claude Code, DeepSeek Harness `dsh` và Antigravity `agy`, cả LifeMate và DataMate:** chỉ đưa thư viện nặng, tải/cài lâu hoặc cần biên dịch phức tạp vào bộ wheels dùng chung. Thư viện nhỏ, nhẹ, cài nhanh thì cài trực tiếp trong notebook; **không yêu cầu build lại wheels chỉ vì thêm một gói nhỏ**.

- Xét cả dependency kéo theo, không chỉ kích thước gói chính. Gói nhỏ nhưng kéo torch/CUDA hay nhiều dependency nặng phải xử lý như gói nặng.
- **Gói nhẹ** (ví dụ `wrapt`): cài từ mạng ngay trong cell setup của notebook, vào đúng môi trường sử dụng. Có thể dùng `pip_install(...)` sẵn có hoặc `%pip install ...` / `python -m pip install ...` với đúng Python. Không bắt buộc thêm vào `build_wheels.ipynb` hay smoke test của bộ wheels. Nếu đã có wheel thì vẫn có thể tận dụng, không cần xóa khỏi bộ cũ.
- **Gói nặng** (ví dụ vLLM và stack torch/CUDA đi kèm): dùng output `notebook-kaggle/wheels/build_wheels.ipynb` (trên Kaggle: `<owner>/build-wheels`). Notebook Add Input output này (`kernel_sources: ["<owner>/build-wheels"]`); ưu tiên cài offline bằng `--no-index --find-links /kaggle/input/**/wheels`, lỗi thì có thể fallback mạng.
- **Hai danh sách wheels**, ở 2 cell đầu của `build_wheels.ipynb` (mỗi dòng ghi notebook nào dùng):
  - `requirements.txt` → cài vào **môi trường gốc** Kaggle. Không thêm gói kéo theo torch/CUDA mới.
  - `requirements-vllm.txt` → cài vào **venv riêng** `/tmp/venv_vllm`. Ghim phiên bản nặng đã chạy được (`vllm==0.32.0`); gói nhẹ worker cần có thể cài thêm trực tiếp vào venv.
- **Không bao giờ `pip install vllm` vào môi trường gốc**: nó thay torch → torchaudio lệch → ASR / transformers chết (run lm-B-002, dm-D-002). Gói dùng bởi worker vLLM phải cài bằng Python của `/tmp/venv_vllm`, không chỉ cài vào môi trường gốc.

**Khi experiment cần thư viện mới:**
1. Kiểm tra image Kaggle đã có chưa, rồi xét độ nặng và dependency. Gói nhẹ → thêm lệnh cài trực tiếp trong notebook, không bắt người dùng chờ build wheels.
2. Chỉ với gói nặng mới thêm vào đúng danh sách trong `build_wheels.ipynb`, kèm tên notebook dùng nó, và thêm import / chạy thử vào dict `checks` của smoke test trong cùng thay đổi.
3. Khi bộ wheels nặng cần thay đổi, báo người dùng chạy lại `build-wheels` một lần; kiểm `wheels/SMOKE.json` phải toàn OK, ghi phiên bản đã xác nhận vào `.vscode/history_plan/<track>.md` và ghim phiên bản tương ứng.
4. Image Kaggle đổi (Python / CUDA / torch khác) → kiểm tra tương thích, build lại các wheels bị ảnh hưởng khi cần. Notebook nên đặt *Environment: Pin to original environment*.

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
