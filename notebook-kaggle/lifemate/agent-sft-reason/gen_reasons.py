"""lm-sft-042 data: one short Vietnamese reasoning line per SFT target step of the LifeMate train set.

For every train sample the SFT targets are: call 0..n-1 of label.tool_calls, then {"done": true} unless the last call is
sys.ask_missing_info (same as lmlib.sft.build_examples). A teacher LLM (headless `claude -p`, default Haiku 5.5) writes
the reason the assistant would think BEFORE each target, from the request, context, history and the results of earlier
steps only. Train set only (never public/private test). Runs locally; output feeds the Kaggle notebook.

    python3 gen_reasons.py --train dataset/lifemate/train+public_test/train/train.jsonl \
        --out notebook-kaggle/lifemate/agent-sft-reason/results/reasons.jsonl --workers 6

Resumable: ids already in --out are skipped. Every reason is checked (check_reasons) before it is written: right count,
length, no braces (the agent parser takes the first '{'), and no ID / value that only appears in a later tool result.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

MAX_CHARS = 200
ID_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")  # bill_04, con_12, gro_11, trk_003 ...

PROMPT = """Bạn viết dữ liệu huấn luyện cho một trợ lý gọi tool trên điện thoại (tiếng Việt).
Mỗi MẪU dưới đây là một yêu cầu đã được giải ĐÚNG: danh sách BƯỚC là chuỗi lệnh đúng theo thứ tự, kèm kết quả trả về.
Với MỖI BƯỚC, viết đúng một dòng "suy nghĩ" mà trợ lý nghĩ NGAY TRƯỚC KHI làm bước đó.

Quy tắc cho mỗi dòng suy nghĩ:
- Tiếng Việt, 1–2 câu ngắn, tối đa 160 ký tự (DONE tối đa 90 ký tự). Không xuống dòng. Không dùng dấu ngoặc nhọn, không viết JSON, không chép lại lệnh.
- CHỈ dùng thông tin có TRƯỚC bước đó: yêu cầu cuối, ngữ cảnh, lịch sử hội thoại, kết quả của các bước TRƯỚC.
  TUYỆT ĐỐI không nhắc tới ID, tên, số liệu chỉ xuất hiện trong kết quả của chính bước đó hoặc các bước sau.
- Nói rõ LÝ DO quyết định, nhất là chỗ dễ sai:
  * tool đọc: cần tra thông tin gì và vì sao (vd "Cần ID của Lan để gửi tin → tìm trong danh bạ").
  * chọn đối tượng từ kết quả trước: so sánh các ứng viên theo điều kiện người dùng nói, nêu ID được chọn và vì sao loại các mục khác.
  * điều kiện (nếu giá/thời tiết/số dư ...): so sánh con số cụ thể rồi kết luận làm hay không.
  * thời gian: quy đổi cụm từ người dùng nói theo current_time (vd "9 giờ sáng mai = 2026-11-24 09:00").
  * hỏi lại (sys.ask_missing_info): nói rõ thiếu thông tin gì hoặc mơ hồ ở đâu.
  * DONE: vì sao dừng (đã làm đủ các việc được yêu cầu / điều kiện không thỏa nên không làm / ngoài khả năng app đã cài / tán gẫu).
- Văn phong ngắn gọn kiểu ghi chú, có thể dùng "→".

Trả về ĐÚNG một dòng JSON cho mỗi mẫu, theo thứ tự, không thêm chữ nào khác:
{{"id": "<id mẫu>", "reasons": ["<suy nghĩ bước 1>", "<suy nghĩ bước 2>", ...]}}
Số phần tử của reasons phải bằng đúng số BƯỚC của mẫu đó.

{samples}
"""


def _j(v, n: int) -> str:
    s = json.dumps(v, ensure_ascii=False, separators=(",", ":"))
    return s if len(s) <= n else s[: n - 1] + "…"


def targets(row: dict) -> list[dict | None]:
    """The SFT targets of a sample in order: label calls, then None for {"done": true} unless the last call asks."""
    calls = row["label"]["tool_calls"]
    return list(calls) + ([] if calls and calls[-1]["name"] == "sys.ask_missing_info" else [None])


def render_sample(row: dict, result_chars: int = 900) -> str:
    ctx = row["context"]
    prof = ctx.get("user_profile") or {}
    lines = [f"### MẪU {row['id']}",
             f"current_time: {ctx.get('current_time')} | người dùng: {prof.get('name')} | nhà: {prof.get('home_address')} | công ty: {prof.get('work_address')}",
             "app đã cài: " + ", ".join(ctx.get("installed_apps", {}))]
    hist = []
    for t in row.get("history", []):
        if t.get("role") == "user":
            hist.append(f"  Người dùng: {t.get('audio_text', '')}")
        else:
            hist.append(f"  Trợ lý: {t.get('text', '')}")
            for c, r in zip(t.get("tool_calls") or [], (t.get("tool_results") or []) + [None] * 10):
                hist.append(f"    đã gọi {_j(c, 300)}" + (f" → {_j(r, result_chars)}" if r is not None else ""))
    lines.append("lịch sử:\n" + ("\n".join(hist) if hist else "  (không có)"))
    lines.append(f"YÊU CẦU CUỐI: {row['audio_text']}")
    results = row["label"].get("tool_results") or []
    for k, c in enumerate(targets(row)):
        if c is None:
            lines.append(f"BƯỚC {k + 1}: DONE (kết thúc)")
        else:
            res = results[k] if k < len(results) else None
            lines.append(f"BƯỚC {k + 1}: {_j(c, 400)}" + (f"\n  kết quả: {_j(res, result_chars)}" if res is not None else ""))
    return "\n".join(lines)


def _ids(x) -> set[str]:
    return set(ID_RE.findall(json.dumps(x, ensure_ascii=False))) if x is not None else set()


def check_reasons(row: dict, reasons, catalog_names: set[str]) -> list[str]:
    """Problems with the reasons of one sample; [] = usable."""
    tg = targets(row)
    if not isinstance(reasons, list) or len(reasons) != len(tg):
        return [f"count {len(reasons) if isinstance(reasons, list) else type(reasons).__name__} != {len(tg)}"]
    errs = []
    results = row["label"].get("tool_results") or []
    known = _ids(row["audio_text"]) | _ids(row.get("history")) | _ids(row["context"])
    for k, (c, r) in enumerate(zip(tg, reasons)):
        if not isinstance(r, str) or not r.strip():
            errs.append(f"step {k}: empty")
            continue
        if len(r) > MAX_CHARS or "\n" in r:
            errs.append(f"step {k}: {len(r)} chars / newline")
        if "{" in r or "}" in r:
            errs.append(f"step {k}: brace")
        allowed = known | (_ids(c.get("arguments")) if c else set())
        later = set().union(*(_ids(x) for x in results[k:])) if k < len(results) else set()
        leak = {i for i in _ids(r) if i in later and i not in allowed and i not in catalog_names}
        if leak:
            errs.append(f"step {k}: hindsight {sorted(leak)}")
        if c is not None and k < len(results):
            known |= _ids(results[k])  # result of step k is visible from step k+1 on
    return errs


def _catalog_names(catalog_path: Path) -> set[str]:
    cat = json.loads(catalog_path.read_text(encoding="utf-8"))
    names = set()
    for a in cat["apps"]:
        names.add(a["id"])
        for t in a["tools"]:
            names.update(t["name"].split("."))
            names.update((t.get("parameters") or {}).get("properties", {}))
    return names


def ask_teacher(prompt: str, model: str, timeout: int) -> str:
    with tempfile.TemporaryDirectory() as cwd:  # empty cwd: no project CLAUDE.md in the teacher's context
        p = subprocess.run(["claude", "-p", "--model", model, "--no-session-persistence", "--tools", "",
                            "--system-prompt", "Bạn là người viết dữ liệu huấn luyện cẩn thận. Chỉ trả về các dòng JSON được yêu cầu."],
                           input=prompt, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    if p.returncode != 0:
        raise RuntimeError(f"claude exit {p.returncode}: {p.stderr[-500:]}")
    return p.stdout


def parse_lines(text: str) -> dict[str, list]:
    out = {}
    for line in text.splitlines():
        line = line.strip().strip("`")
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("id"), str):
            out[obj["id"]] = obj.get("reasons")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="dataset/lifemate/train+public_test/train/train.jsonl")
    ap.add_argument("--catalog", default="dataset/lifemate/train+public_test/train/train_tools_catalog.json")
    ap.add_argument("--out", default="notebook-kaggle/lifemate/agent-sft-reason/results/reasons.jsonl")
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=3, help="passes over the samples still missing")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args(argv)

    rows = [json.loads(l) for l in open(a.train, encoding="utf-8") if l.strip()][: a.limit]
    names = _catalog_names(Path(a.catalog))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in open(out, encoding="utf-8")} if out.exists() else set()
    lock = threading.Lock()
    rejects = out.with_name(out.stem + ".rejects.jsonl")

    def run_batch(batch: list[dict]) -> tuple[int, int]:
        prompt = PROMPT.format(samples="\n\n".join(render_sample(r) for r in batch))
        try:
            got = parse_lines(ask_teacher(prompt, a.model, a.timeout))
        except Exception as e:  # noqa: BLE001 - one failed batch is retried next round
            print(f"batch {batch[0]['id']}: {e}", file=sys.stderr, flush=True)
            return 0, len(batch)
        ok = bad = 0
        with lock:
            with open(out, "a", encoding="utf-8") as f, open(rejects, "a", encoding="utf-8") as g:
                for r in batch:
                    reasons = got.get(r["id"])
                    errs = check_reasons(r, reasons, names)
                    if errs:
                        bad += 1
                        g.write(json.dumps({"id": r["id"], "errors": errs, "reasons": reasons}, ensure_ascii=False) + "\n")
                    else:
                        ok += 1
                        f.write(json.dumps({"id": r["id"], "reasons": [x.strip() for x in reasons], "teacher": a.model},
                                           ensure_ascii=False) + "\n")
        return ok, bad

    for rnd in range(a.rounds):
        todo = [r for r in rows if r["id"] not in done]
        if not todo:
            break
        batches = [todo[i: i + a.batch] for i in range(0, len(todo), a.batch)]
        print(f"round {rnd + 1}: {len(todo)} samples, {len(batches)} batches", flush=True)
        tot_ok = tot_bad = 0
        with cf.ThreadPoolExecutor(a.workers) as ex:
            for i, (ok, bad) in enumerate(ex.map(run_batch, batches), 1):
                tot_ok += ok; tot_bad += bad
                if i % 10 == 0 or i == len(batches):
                    print(f"  {i}/{len(batches)} batches: ok {tot_ok}, rejected {tot_bad}", flush=True)
        done = {json.loads(l)["id"] for l in open(out, encoding="utf-8")} if out.exists() else set()
    missing = [r["id"] for r in rows if r["id"] not in done]
    print(json.dumps({"samples": len(rows), "with_reasons": len(rows) - len(missing), "missing": len(missing),
                      "missing_ids": missing[:20]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
