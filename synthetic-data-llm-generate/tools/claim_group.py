"""Claim a text/state authoring group atomically; --finish seals complete drafts.

    python3 tools/claim_group.py                                    # nhóm kế tiếp trong PLAN
    python3 tools/claim_group.py --apps "tv, music" --focus "..."   # người dùng chỉ định app
    python3 tools/claim_group.py --n 20

Hết PLAN thì tự chọn 3 app có ít call nhất trong dữ liệu synthetic hiện có (trừ app giấu trong dev_unseen).
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path
from pipeline import Pipeline, PipelineError, atomic_json, FAMILIES, HOLDOUT

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
import os
OUT = Path(os.environ.get("SYN_OUT", ROOT / "train"))
PARTS = OUT / "parts"
UNSEEN = HOLDOUT  # chỉ sinh khi user chỉ định experiment riêng

# Ưu tiên theo lỗi đo ở lm-sft-016 (xem README.md)
PLAN = [
    ("g7", "banking, bill_payment, ewallet",
     "lọc hoá đơn theo hạn + số tiền; chuyển tiền khi có 2 tài khoản; điều kiện theo số dư"),
    ("g8", "calendar, reminders, alarm, tasks",
     "giờ tiếng Việt (kém, rưỡi, tối mai, thứ bảy tuần sau, trước giờ hẹn nửa tiếng); dời/huỷ nhiều sự kiện"),
    ("g9", "contacts, sms, phone, chat_ott, email",
     "tên gần giống → search rồi hỏi contact_id; nhắn nhiều người; đổi ý giữa chừng"),
    ("g10", "smart_home, weather, notes",
     "nhiều thiết bị cùng loại khác phòng; nếu trời mưa/nóng thì..."),
    ("g11", "food_delivery, grocery, shopping_list, pantry, recipes",
     "chuỗi 4–6 call: tra công thức → kiểm tủ → mua phần thiếu"),
    ("g12", "ride_hailing, maps, public_transit, parking",
     "địa chỉ nhà/công ty lấy từ user_profile; giờ đón tương đối"),
    ("g13", "ecommerce, stocks, gold_price, fuel_price, translator",
     "thiếu số lượng/mã → hỏi; huỷ/trả đơn theo điều kiện"),
    ("g14", "habit_tracker, event_tickets, train_bus_tickets",
     "history 4–12 cặp lượt, lượt cuối dùng ID từ kết quả đọc cũ; hỏi → thôi khỏi; được kết hợp app bất kỳ"),
]


def used_ids() -> set:
    ids = {p.name for p in (OUT / "train_environment").glob("syn_*")}
    for f in glob.glob(str(PARTS / "g*.jsonl")):
        ids |= {json.loads(l)["id"] for l in open(f, encoding="utf-8") if l.strip()}
    return ids


def least_covered(known: set) -> str:
    cnt = Counter({a: 0 for a in known - UNSEEN - {"sys"}})
    for f in glob.glob(str(PARTS / "g*.jsonl")):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                for c in json.loads(l)["label"]["tool_calls"]:
                    app = c["name"].split(".")[0]
                    if app in cnt:
                        cnt[app] += 1
    return ", ".join(a for a, _ in sorted(cnt.items(), key=lambda x: (x[1], x[0]))[:3])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apps")
    ap.add_argument("--focus")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--finish", help="seal một group sau khi hoàn tất ghi mẫu")
    a = ap.parse_args()
    if a.finish:
        print(json.dumps(Pipeline(OUT).seal(a.finish), ensure_ascii=False))
        return
    if not 1 <= a.n <= 999:
        sys.exit("--n phải trong 1..999 để các dải ID không chồng nhau")

    known = {x["id"] for x in json.load(open(OUT / "train_tools_catalog.json", encoding="utf-8"))["apps"]}
    if a.apps:
        bad = [x.strip() for x in a.apps.split(",") if x.strip() not in known]
        if bad:
            sys.exit(f"app không có trong catalog: {bad}")

    PARTS.mkdir(parents=True, exist_ok=True)
    taken = {int(m.group(1)) for p in PARTS.glob("g*.jsonl") if (m := re.fullmatch(r"g(\d+)\.jsonl", p.name))}
    if a.apps:
        cands = [(f"g{k}", a.apps, a.focus) for k in range(max(taken | {14}) + 1, 10_000)]
    else:
        cands = [x for x in PLAN if int(x[0][1:]) not in taken]
        cands += [(f"g{k}", None, None) for k in range(max(taken | {14}) + 1, 10_000)]

    # Only complete, atomic part files are read; the claim itself reserves its full ID range.
    used = used_ids()
    for group, apps, focus in cands:
        k = int(group[1:])
        ids = [f"syn_{k * 1000 + 1 + i:04d}" for i in range(a.n)]
        if set(ids) & used:
            continue
        try:  # mở "x": nguyên tử, agent khác đã tạo thì bỏ qua
            open(PARTS / f"{group}.jsonl", "x").close()
        except FileExistsError:
            continue
        apps = apps or least_covered(known)
        focus = a.focus or focus or "theo các nhóm lỗi và phân bố trong prompt_generate.md"
        claim = {"REPO": str(HERE.parents[1]), "GROUP": group, "APPS": apps, "FOCUS": focus,
                 "ID_FIRST": ids[0], "ID_LAST": ids[-1], "N": a.n,
                 "FAMILIES": sorted(FAMILIES), "COUNTERFACTUAL_PAIRS": max(1, a.n // 6) if a.n >= 2 else 0,
                 "GUIDANCE": "ưu tiên lỗi lm-sft-016; goals độc lập; 6–14 app; history nhiều lượt; không sinh TTS"}
        atomic_json(PARTS / f"{group}.claim.json", claim)
        print(json.dumps(claim, ensure_ascii=False, indent=1))
        return
    sys.exit("không tìm được nhóm trống")


if __name__ == "__main__":
    try:
        # Serialize claim and least-covered counting with writers/builders.
        if any(arg == "--finish" or arg.startswith("--finish=") for arg in sys.argv):
            main()
        else:
            with Pipeline(OUT).lock():
                main()
    except (PipelineError, OSError, ValueError, KeyError) as e:
        sys.exit(f"ERR: {e}")
