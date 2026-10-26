# ZALO AI CHALLENGE 2026 — MULTI-AGENT DEVELOPMENT REPOSITORY
## TRACK 1: LIFEMATE & TRACK 2: DATAMATE

Repository chuẩn hóa cho đội tuyển tham gia cuộc thi **Zalo AI Challenge 2026**, được thiết kế để vận hành đồng bộ và tối ưu hóa năng suất trên cả **4 nền tảng AI Agent**:
- **OpenAI Codex** (CLI / VS Code / Copilot Agent)
- **Claude Code** (Anthropic CLI)
- **Antigravity CLI (`agy`)**
- **DeepSeek Harness (DSH)** (Agent Teams, Skills & Subagents)

---

## 1. CẤU TRÚC REPOSITORY (REPOSITORY ANATOMY)

```text
├── AGENTS.md                       # Single Source of Truth cho Codex, Agy, DSH, Claude Code
├── CLAUDE.md                       # Direct entrypoint & CLI reference cho Claude Code
├── README.md                       # Hướng dẫn tổng quan repository
├── .gitignore                      # Fine-grained rules: cho phép code/docs, chặn wav/data
│
├── .vscode/
│   ├── docs/                       # Toàn bộ quy chế, đặc tả kỹ thuật, 12 cạm bẫy, prompt mẫu
│   │   ├── COMPETITION_OVERVIEW_2026.md
│   │   ├── AGENT_CHEATSHEET.md
│   │   ├── TRACK1_LIFEMATE_SPECIFICATION.md
│   │   ├── TRACK2_DATAMATE_SPECIFICATION.md
│   │   ├── OFFICIAL_METRICS_AND_MATCHING_RULES.md
│   │   └── PROMPT_TEMPLATES_AND_EXAMPLES.md
│   │
│   └── planning-research/          # Trung tâm kế hoạch, chiến lược & nhật ký thử nghiệm
│       ├── lifemate/               # strategy.md, experiment_log.md, error_analysis.md, checklist
│       └── datamate/               # strategy.md, experiment_log.md, error_analysis.md, checklist
│
├── notebook-kaggle/                # Môi trường phát triển & thực nghiệm trên Kaggle
│   ├── scripts/
│   │   ├── kaggle_sync.py          # Tool đồng bộ 9 tài khoản Kaggle, push/pull kernel, download data
│   │   └── requirements.txt        # Thư viện Python phụ thuộc
│   ├── lifemate/                   # Kernel Kaggle cho LifeMate (ASR + LLM + ToolEnv)
│   └── datamate/                   # Kernel Kaggle cho DataMate (Code Gen Sandbox + Formatter)
│
└── tests/                          # Bộ kiểm thử tự động (Unit Tests & Integration Tests)
```

---

## 2. BẢNG THAM CHIẾU NHANH HAI BẢNG ĐẤU

| Tiêu chí | Track 1: LifeMate | Track 2: DataMate |
| :--- | :--- | :--- |
| **Bản chất** | Voice-to-Action Agent (Audio $\rightarrow$ Tool Calls) | Tabular Data Agent (Messy Excel/CSV $\rightarrow$ Answer) |
| **File nộp bài** | `submission.jsonl` | `submission.jsonl` |
| **Cấu trúc dòng** | `{"id": "...", "tool_calls": [...], "state_hash": "..."}` | `{"id": "...", "type": "...", "answer": ...}` |
| **Giới hạn tham số** | **Toàn hệ thống < 6B, riêng LLM < 5B parameters** | **Toàn hệ thống $\le$ 15B parameters** |
| **Mô hình đề xuất** | `whisper-large-v3-turbo` (809M) + `Qwen2.5-3B-Instruct` (3.09B) | `Qwen2.5-Coder-14B` (hoặc `7B`) + Python Code Sandbox |
| **Metric đánh giá** | $F_1$ với **Cổng chặn State Hash** | **Penalized Accuracy (PA)**: Đúng **+1**, Abstain **0**, Sai **-1** |
| **Điều kiện 0 điểm** | $\text{hash}_{\text{sys}} \neq \text{hash}_{\text{gold}}$ hoặc $M = 0$ | Bỏ trống hoặc nộp `{"type": "abstain"}` |
| **Tính tất định** | Bắt buộc chạy lại khớp kết quả | $|\Delta \text{PA}| \le 0.03$, bắt buộc **Greedy Decoding (temp=0)** |

---

## 3. CHECKLIST 12 CẠM BẪY CHẾT NGƯỜI (TÓM TẮT)

- **Track 1 LifeMate**:
  1. Cấm model $\ge$ 5B (Chỉ dùng `Qwen2.5-3B-Instruct`).
  2. Cấm đọc trực tiếp `state.json` (Luôn đi qua `ToolEnv`).
  3. Luôn `env.validate()` trước khi `env.call()`.
  4. Tuân thủ `x-ref`: tra cứu ID trước, cấm bịa ID.
  5. Thiếu thông tin $\rightarrow$ gọi `sys.ask_missing_info`.
  6. Ngoài phạm vi $\rightarrow$ trả về `tool_calls: []` kèm hash ban đầu.
- **Track 2 DataMate**:
  7. CẤM FIFO giá vốn ($Cost = quantity \times unit\_cost$ trên dòng bán).
  8. Phân biệt `unanswerable` (+1 nếu đúng) và `abstain` (0 điểm).
  9. Đơn có trả hàng (kể cả 1 phần) $\rightarrow$ LOẠI CẢ ĐƠN khỏi doanh thu.
  10. Đơn `đang giao` $\rightarrow$ VẪN TÍNH vào doanh thu.
  11. Bán chạy nhất $\rightarrow$ Xếp theo `sum(quantity_base)`.
  12. Format cực chuẩn 9 kiểu dữ liệu (Percent là `12.4`, cấm `0.124`, cấm `%`).

---

## 4. QUẢN LÝ POOL 9 TÀI KHOẢN KAGGLE

Hệ thống tích hợp sẵn 9 tài khoản Kaggle toàn cầu để tối ưu hóa quota GPU và lượt nộp:
```bash
# Xem danh sách & trạng thái 9 tài khoản
kaggle-acc list

# Xem tài khoản đang active
kaggle-acc current

# Chuyển đổi tài khoản đang active
kaggle-switch <user>
kaggle-switch 4

# Sử dụng công cụ đồng bộ tự động
python notebook-kaggle/scripts/kaggle_sync.py list-accounts
python notebook-kaggle/scripts/kaggle_sync.py switch-account <user>
python notebook-kaggle/scripts/kaggle_sync.py push-kernel --track lifemate
python notebook-kaggle/scripts/kaggle_sync.py push-kernel --track datamate
```

---

## 5. CHẠY KIỂM THỬ (TESTING)

```bash
# Chạy toàn bộ test suite
python3 -m unittest discover tests/ -v
```

---

## 6. HƯỚNG DẪN DÀNH CHO CÁC AGENT

- **OpenAI Codex**: Đọc [`AGENTS.md`](<AGENTS.md>). Kích hoạt `multi_agent = true` trong `~/.codex/config.toml`.
- **Claude Code**: Bắt đầu từ [`CLAUDE.md`](<CLAUDE.md>).
- **Antigravity CLI (`agy`)**: Đọc [`AGENTS.md`](<AGENTS.md>). Sử dụng **Task Artifacts** để theo dõi kế hoạch (`write_to_file` với `IsArtifact: true`, `ArtifactType: "task"`).
- **DeepSeek Harness**: Sử dụng Agent Teams (tối đa 8 teammates song song), Subagents và các Skills: `kaggle`, `brainstorming`, `writing-plans`, `subagent-driven-development`.
