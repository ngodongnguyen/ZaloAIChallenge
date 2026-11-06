"""Đóng gói release synthetic đã duyệt (check_sample.py --verified-path) thành dataset Kaggle cho notebook `agent-sft-cont` (preset `sft_cont`).

    python3 tools/pack_kaggle.py --user <user>        # ghi train/kaggle_upload/ (gitignored)
    kaggle-acc run <user> kaggle datasets create -p train/kaggle_upload        # lần đầu (private)
    kaggle-acc run <user> kaggle datasets version -p train/kaggle_upload -m "g7-g9"   # các lần sau

Chỉ gồm text (notebook train bằng audio_text, chưa cần audio TTS): syn_train.jsonl + syn_tools_catalog.json.
Tên file khác train.jsonl để notebook không nhầm với split train của dataset cuộc thi.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "train"
DST = OUT / "kaggle_upload"
SLUG = "lifemate-syn"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="tài khoản Kaggle sở hữu dataset (cùng tài khoản chạy notebook)")
    a = ap.parse_args()

    # chỉ release đã duyệt (author → solver → reviewer), checksum do check_sample.py --verified-path kiểm
    r = subprocess.run([sys.executable, str(HERE / "check_sample.py"), "--verified-path"], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("chưa có release đã duyệt: chạy `check_sample.py --all --build` sau khi review\n" + (r.stdout + r.stderr)[-1500:])
    DST.mkdir(exist_ok=True)
    shutil.copy(r.stdout.strip().splitlines()[-1], DST / "syn_train.jsonl")
    shutil.copy(OUT / "train_tools_catalog.json", DST / "syn_tools_catalog.json")
    meta = {"title": SLUG, "id": f"{a.user}/{SLUG}", "licenses": [{"name": "CC0-1.0"}]}
    (DST / "dataset-metadata.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    n = sum(1 for l in open(DST / "syn_train.jsonl", encoding="utf-8") if l.strip())
    print(f"{n} mẫu -> {DST}")


if __name__ == "__main__":
    main()
