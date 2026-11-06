"""Check synthetic apps and scenarios locally. A scenario is kept only if every rule passes.

  python synth/validate.py --apps synth/apps_pilot.json --scen synth/scen_pilot.jsonl --out synth/ok_pilot.jsonl
"""
import argparse
import collections
import json
import re
from pathlib import Path

DATA = Path("C:/Project/zalo_ai/data/LifeMate/train+public_test")
ISO = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d$")
TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}
VI = re.compile(r"[àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]", re.I)


def real_app_ids():
    ids = set()
    for p in ("train/train_tools_catalog.json", "public_test/public_test_tools_catalog.json"):
        ids |= {a["id"] for a in json.load(open(DATA / p, encoding="utf-8-sig"))["apps"]}
    return ids


def check_app(app, real):
    bad = []
    if app["id"] in real or any(r != "sys" and (r in app["id"] or app["id"] in r) for r in real):
        bad.append("overlaps a real app id")
    kinds = {t.get("kind") for t in app["tools"]}
    if not {"read", "write"} <= kinds:
        bad.append("needs read and write tools")
    for t in app["tools"]:
        if not t["name"].startswith(app["id"] + "."):
            bad.append(f"bad tool name {t['name']}")
        props = t.get("parameters", {}).get("properties", {})
        for r in t.get("parameters", {}).get("required", []):
            if r not in props:
                bad.append(f"{t['name']}: required {r} is not a property")
    return bad


def check_calls(calls, results, tools, seen, where):
    """`seen` is every string already visible to the assistant; each call may use the IDs in it."""
    errs = []
    for i, c in enumerate(calls):
        name, args = c.get("name"), c.get("arguments") or {}
        if name == "sys.ask_missing_info":
            if i != len(calls) - 1:
                errs.append(f"{where}: ask_missing_info is not last")
            elif args.get("tool_name") not in tools or not args.get("params"):
                errs.append(f"{where}: bad ask_missing_info")
            else:
                known = tools[args["tool_name"]].get("parameters", {}).get("properties", {})
                errs += [f"{where}: ask for unknown param {p}" for p in args["params"] if p not in known]
            continue
        if name not in tools:
            errs.append(f"{where}: unknown tool {name}")
            continue
        spec = tools[name].get("parameters", {})
        props = spec.get("properties", {})
        errs += [f"{where}: {name} missing {r}" for r in spec.get("required", []) if r not in args]
        for k, v in args.items():
            if k not in props:
                errs.append(f"{where}: {name} unknown param {k}")
                continue
            s = props[k]
            if v in ("", None, [], {}):
                errs.append(f"{where}: {name}.{k} is empty")
            want = TYPES.get(s.get("type"))
            if want and (not isinstance(v, want) or (s.get("type") in ("integer", "number") and isinstance(v, bool))):
                errs.append(f"{where}: {name}.{k} has the wrong type")
            if "enum" in s and v not in s["enum"]:
                errs.append(f"{where}: {name}.{k}={v} is not in the enum")
            if s.get("format") == "date-time" and not (isinstance(v, str) and ISO.match(v)):
                errs.append(f"{where}: {name}.{k}={v} is not full ISO 8601")
            if s.get("x-ref"):
                for x in (v if isinstance(v, list) else [v]):
                    if str(x) not in seen:
                        errs.append(f"{where}: {name}.{k}={x} was not read before use")
        if i < len(results):
            seen.add_result(results[i])
    return errs


class Seen(set):
    def add_result(self, obj):
        if isinstance(obj, dict):
            for v in obj.values():
                self.add_result(v)
        elif isinstance(obj, list):
            for v in obj:
                self.add_result(v)
        else:
            self.add(str(obj))


def check_scenario(s, tools):
    errs = []
    u = s.get("utterance", "")
    if len(u) < 4 or not VI.search(u):
        errs.append("utterance is not Vietnamese")
    if re.search(r"\b[a-z]+_\d+\b|\b\w+\.\w+_\w+\b", u):
        errs.append("utterance reads an ID or tool name")
    if not ISO.match(s.get("current_time", "")):
        errs.append("bad current_time")
    if not all(s.get("user_profile", {}).get(k) for k in ("name", "home_address", "city")):
        errs.append("incomplete user_profile")
    seen = Seen()
    for n, h in enumerate(s.get("history") or []):
        if h.get("role") == "user":
            if not h.get("text"):
                errs.append(f"history {n}: empty user turn")
        elif h.get("role") == "assistant":
            calls, res = h.get("tool_calls") or [], h.get("tool_results") or []
            if len(res) != len(calls):
                errs.append(f"history {n}: results do not match calls")
            errs += check_calls(calls, res, tools, seen, f"history {n}")
        else:
            errs.append(f"history {n}: bad role")
    calls, res = s.get("tool_calls") or [], s.get("tool_results") or []
    ask = bool(calls) and calls[-1].get("name") == "sys.ask_missing_info"
    if len(res) != len(calls) - ask:
        errs.append("results do not match calls")
    if any(not (isinstance(r, dict) and "ok" in r) for r in res):
        errs.append("result without ok")
    return errs + check_calls(calls, res, tools, seen, "turn")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apps", required=True)
    ap.add_argument("--scen", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--apps-out", default=None, help="write the kept apps here")
    args = ap.parse_args()
    real = real_app_ids()
    apps = json.loads(Path(args.apps).read_text(encoding="utf-8"))["apps"]
    good = {}
    for a in apps:
        bad = check_app(a, real)
        if bad:
            print("DROP app", a["id"], bad)
        else:
            good[a["id"]] = a
    print("apps kept", len(good), "of", len(apps))
    if args.apps_out:
        Path(args.apps_out).write_text(json.dumps({"apps": list(good.values())}, ensure_ascii=False, indent=1), encoding="utf-8")
    if not args.scen:
        return
    tools = {t["name"]: t for a in good.values() for t in a["tools"]}
    rows = [json.loads(line) for line in open(args.scen, encoding="utf-8")]
    reasons, kept = collections.Counter(), []
    for r in rows:
        errs = check_scenario(r, tools)
        if errs:
            reasons.update(re.sub(r"^(history \d+|turn): ", "", e).split(" ")[-3] + " " + e.split(" ")[-1] for e in errs[:1])
            print("REJECT:", r["utterance"][:60], "|", errs[:2])
        else:
            kept.append(r)
    print("scenarios kept", len(kept), "of", len(rows))
    n = max(1, len(kept))
    print("calls", sorted(collections.Counter(len(r["tool_calls"]) for r in kept).items()), "ask", round(sum(any(c["name"] == "sys.ask_missing_info" for c in r["tool_calls"]) for r in kept) / n, 2),
          "history", round(sum(bool(r.get("history")) for r in kept) / n, 2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in kept)


if __name__ == "__main__":
    main()
