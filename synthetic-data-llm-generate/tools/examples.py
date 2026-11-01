"""In ví dụ nhãn train gốc có dùng một tool / app để agent sinh dữ liệu học đúng quy ước.

Chỉ lấy mẫu train_fit (tools/split_ids.json): không cho agent xem dev_seen / dev_unseen / final_holdout,
nếu không dữ liệu synthetic sẽ chép lại mẫu dev và điểm dev bị thổi phồng.

    python3 tools/examples.py banking.transfer        # 5 mẫu có gọi banking.transfer
    python3 tools/examples.py --app bill_payment -k 8
    python3 tools/examples.py --kind ask              # mẫu kết thúc bằng sys.ask_missing_info
    python3 tools/examples.py --kind empty            # mẫu nhãn rỗng
    python3 tools/examples.py --kind history -k 3     # mẫu có history dài
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TRAIN = Path(os.environ.get("LM_TRAIN_DIR", REPO / "dataset/lifemate/train+public_test/train"))


def short(v, n=160):
    s = json.dumps(v, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + "…"


def show(r: dict, full_results: bool) -> None:
    print(f"=== {r['id']}  time={r['context']['current_time']}  city={r['context']['user_profile'].get('city')}  apps={len(r['context']['installed_apps'])}")
    for t in r.get("history") or []:
        if t["role"] == "user":
            print(f"  [u] {t['audio_text']}")
        else:
            calls = " ; ".join(f"{c['name']} {short(c['arguments'], 120)}" for c in t.get("tool_calls") or [])
            print(f"  [a] {t.get('text', '')}" + (f"   <calls: {calls}>" if calls else ""))
    print(f"  USER CUỐI: {r['audio_text']}")
    lab = r["label"]
    if not lab["tool_calls"]:
        print("  NHÃN: []")
    for c, res in zip(lab["tool_calls"], lab["tool_results"]):
        print(f"  -> {c['name']} {json.dumps(c['arguments'], ensure_ascii=False)}")
        print(f"       = {json.dumps(res, ensure_ascii=False) if full_results else short(res)}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tool", nargs="?", help="tên tool, ví dụ banking.transfer")
    ap.add_argument("--app")
    ap.add_argument("--kind", choices=["ask", "empty", "history", "multi"])
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full", action="store_true", help="in đầy đủ tool_results")
    a = ap.parse_args()

    split = json.load(open(HERE / "split_ids.json", encoding="utf-8"))
    allowed = set(split["ids"]["train_fit"])
    rows = [json.loads(l) for l in open(TRAIN / "train.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["id"] in allowed]

    def names(r):
        return [c["name"] for c in r["label"]["tool_calls"]]

    if a.tool:
        rows = [r for r in rows if a.tool in names(r) or any(a.tool == c["name"] for t in r.get("history") or [] for c in t.get("tool_calls") or [])]
    if a.app:
        rows = [r for r in rows if any(n.split(".")[0] == a.app for n in names(r))]
    if a.kind == "ask":
        rows = [r for r in rows if "sys.ask_missing_info" in names(r)]
    elif a.kind == "empty":
        rows = [r for r in rows if not names(r)]
    elif a.kind == "history":
        rows = sorted((r for r in rows if r.get("history")), key=lambda r: -len(r["history"]))[: max(a.k * 4, 20)]
    elif a.kind == "multi":
        rows = [r for r in rows if len(names(r)) >= 4]

    if not rows:
        held = split["holdout_apps"]
        print(f"Không có ví dụ train_fit. App giấu để đo khả năng tổng quát {held} cố ý không có ví dụ: "
              "đọc schema trong catalog + code zaic_tools/apps/<app>.py và suy quy ước từ các app tương tự.")
        return
    random.Random(a.seed).shuffle(rows)
    print(f"# {len(rows)} mẫu train_fit khớp, in {min(a.k, len(rows))}\n")
    for r in rows[: a.k]:
        show(r, a.full)


if __name__ == "__main__":
    main()
