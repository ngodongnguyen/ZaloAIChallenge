"""Turn validated scenarios into train-like items plus a merged catalog, for build_sft.py --synth-dir.

Each item installs the scenario's apps and real distractor apps, so its prompt looks like a real one. Real apps from
Nguyen's holdout are never installed: their catalog text must stay out of runs measured on splits/unseen_apps.json.

  python synth/to_items.py --apps synth/apps.json --scen synth/ok.jsonl --out synth/out
"""
import argparse
import json
import random
from pathlib import Path

DATA = Path("C:/Project/zalo_ai/data/LifeMate/train+public_test/train")
HELD = {"appliances", "door_lock", "gold_price", "home_automation", "package_tracking", "robot_vacuum"}
CITIES = {"Hà Nội": (21.0278, 105.8342), "Đà Nẵng": (16.0544, 108.2022), "Cần Thơ": (10.0452, 105.7469), "Huế": (16.4637, 107.5909)}
DEFAULT_LOC = (10.7769, 106.7009)  # Ho Chi Minh City


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apps", required=True)
    ap.add_argument("--scen", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    real = json.load(open(DATA / "train_tools_catalog.json", encoding="utf-8-sig"))
    synth = json.loads(Path(args.apps).read_text(encoding="utf-8"))["apps"]
    tools_of = {a["id"]: [t["name"] for t in a["tools"]] for a in real["apps"] + synth}
    pool = sorted(a["id"] for a in real["apps"] if a["id"] != "sys" and a["id"] not in HELD)
    train = [json.loads(line) for line in open(DATA / "train.jsonl", encoding="utf-8-sig") if line.strip()]
    sizes = [len(r["context"]["installed_apps"]) for r in train]

    rng = random.Random(0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in open(args.scen, encoding="utf-8")]
    with open(out / "items.jsonl", "w", encoding="utf-8") as f:
        for n, s in enumerate(rows):
            used = {c["name"].split(".")[0] for c in s["tool_calls"] if c["name"] != "sys.ask_missing_info"}
            for h in s.get("history") or []:
                used |= {c["name"].split(".")[0] for c in h.get("tool_calls") or [] if c["name"] != "sys.ask_missing_info"}
            apps = sorted(used | set(s["apps"]))
            extra = max(0, rng.choice(sizes) - len(apps))
            apps += rng.sample(pool, min(extra, len(pool)))
            rng.shuffle(apps)
            city = s["user_profile"].get("city", "")
            lat, lng = next((v for k, v in CITIES.items() if k in city), DEFAULT_LOC)
            hist = [{"role": "user", "audio_text": h["text"]} if h["role"] == "user" else h for h in s.get("history") or []]
            item = {"id": f"synth_{n:04d}", "audio_text": s["utterance"], "history": hist,
                    "context": {"current_time": s["current_time"], "timezone": "Asia/Ho_Chi_Minh",
                                "location": {"lat": lat, "lng": lng}, "user_profile": s["user_profile"],
                                "installed_apps": {a: {"tools": tools_of[a]} for a in apps}},
                    "label": {"tool_calls": s["tool_calls"], "tool_results": s["tool_results"]}}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    catalog = dict(real, apps=real["apps"] + [dict(a, tools=[dict(t, app=a["id"]) for t in a["tools"]]) for a in synth])
    (out / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
    print("items", len(rows), "apps in catalog", len(catalog["apps"]))


if __name__ == "__main__":
    main()
