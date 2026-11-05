"""Prompt construction shared by SFT data building and inference (must stay identical)."""
import json

from condition import condition_hint
from config import COND_HINT, HISTORY_RESULT_CHARS, MONEY_HINT, RESULT_DIGEST, TOOL_RESULT_CHARS

SYSTEM = """Bạn là trợ lý giọng nói tiếng Việt thực hiện hành động qua các công cụ (tool) của ứng dụng đã cài trên thiết bị.
Mỗi lượt chỉ xuất MỘT đối tượng JSON, không giải thích:
- Gọi tool: {"name": "<app.tool>", "arguments": {...}}
- Kết thúc: {"done": true}
Quy tắc:
1. Chỉ dùng tool trong danh sách. Tham số có x-ref phải là ID lấy từ kết quả tool đọc (read), không được bịa.
2. Cần ID/thông tin mà chưa có: gọi tool read trước rồi mới gọi tool write.
3. Chỉ hỏi lại khi THỰC SỰ thiếu thông tin người dùng chưa nói và không tra được: trước hết dùng tool read để tìm (danh bạ, thiết bị, lịch, danh sách...), dùng ngữ cảnh/hồ sơ (nhà, nơi làm việc, giờ hiện tại). Chỉ khi vẫn không xác định được một tham số bắt buộc mới gọi {"name": "sys.ask_missing_info", "arguments": {"tool_name": "<tool định gọi>", "params": ["<tham số thiếu>"]}} rồi kết thúc.
4. Không có ứng dụng nào xử lý được yêu cầu: kết thúc ngay {"done": true}, không gọi tool nào.
5. Không gọi tool thừa. Không truyền tham số tùy chọn mà người dùng không nói đến; không dùng giá trị rỗng ("", null). Thời gian dùng định dạng ISO 8601 theo múi giờ của người dùng."""


def _param_line(name, spec, required):
    t = spec.get("type", "any")
    if "enum" in spec:
        t = "|".join(map(str, spec["enum"]))
    flags = []
    if name in required:
        flags.append("bắt buộc")
    if spec.get("x-ref"):
        flags.append(f"x-ref={spec['x-ref']}")
    if spec.get("x-match") == "fuzzy":
        flags.append("văn bản tự do")
    desc = spec.get("description", "")
    extra = f" [{', '.join(flags)}]" if flags else ""
    return f"    - {name}: {t}{extra} {desc}".rstrip()


def render_tool(tool):
    params = tool.get("parameters", {}) or {}
    props = params.get("properties", {}) or {}
    required = set(params.get("required", []) or [])
    lines = [f"* {tool['name']} ({tool.get('kind', '')}): {tool.get('description', '')}"]
    for n, s in props.items():
        lines.append(_param_line(n, s, required))
    if params.get("x-one-of"):
        lines.append(f"    (cần ít nhất một nhóm: {json.dumps(params['x-one-of'], ensure_ascii=False)})")
    if tool.get("returns"):
        lines.append(f"    trả về: {tool['returns']}")
    return "\n".join(lines)


def installed_tool_names(item):
    """Tool names allowed for the sample; None means 'all tools of the installed apps'."""
    apps = item["context"].get("installed_apps", {}) or {}
    names, app_ids = set(), set()
    if isinstance(apps, dict):
        for app_id, v in apps.items():
            app_ids.add(app_id)
            for t in (v or {}).get("tools", []) or []:
                names.add(t if isinstance(t, str) else t.get("name"))
    else:  # list of app ids
        app_ids = set(apps)
    return app_ids, {n for n in names if n}


def tools_for_item(item, catalog):
    app_ids, names = installed_tool_names(item)
    out = []
    for app in catalog["apps"]:
        if app["id"] not in app_ids:
            continue
        for t in app["tools"]:
            if not names or t["name"] in names:
                out.append(t)
    return out


def allowed_names(item, catalog):
    return {t["name"] for t in tools_for_item(item, catalog)} | {"sys.ask_missing_info"}


def truncate(obj, limit=TOOL_RESULT_CHARS):
    s = json.dumps(obj, ensure_ascii=False)
    return s if len(s) <= limit else s[:limit] + "…"


def render_history(history):
    """History turns: user -> text (audio_text), assistant -> text + executed calls (results truncated)."""
    out = []
    for h in history or []:
        if h.get("role") == "user":
            out.append({"role": "user", "text": h.get("audio_text", "")})
        else:
            turn = {"role": "assistant", "text": h.get("text", "")}
            if h.get("tool_calls"):
                turn["tool_calls"] = h["tool_calls"]
                turn["tool_results"] = [truncate(r, HISTORY_RESULT_CHARS) for r in h.get("tool_results", [])]
            out.append(turn)
    return out


_FEWSHOT = None


def fewshot_block(item, transcript):
    """FEWSHOT=k (teacher/probe only, not used in SFT): the k train requests with the most word overlap, with their gold
    calls, so a model that never saw this data learns its conventions (look up IDs instead of asking, no optional extras).
    FEWSHOT_EXCLUDE: file of ids never shown (e.g. cache/holdout_all.txt, so held-out apps stay unseen)."""
    import os
    import re
    global _FEWSHOT
    k = int(os.environ.get("FEWSHOT", 0))
    if not k:
        return ""
    if _FEWSHOT is None:
        excl = set(open(os.environ["FEWSHOT_EXCLUDE"]).read().split()) if os.environ.get("FEWSHOT_EXCLUDE") else set()
        rows = [json.loads(l) for l in open(os.environ.get("FEWSHOT_DATA", "data/train/train.jsonl"), encoding="utf-8")]
        _FEWSHOT = [(r["id"], r["audio_text"], r["label"]["tool_calls"]) for r in rows if r["id"] not in excl]
    words = lambda s: set(re.findall(r"\w+", str(s).lower()))
    q = words(transcript)
    scored = sorted(_FEWSHOT, key=lambda e: -len(q & words(e[1])) / (len(q | words(e[1])) or 1))
    picked = [e for e in scored if e[0] != item.get("id")][:k]
    lines = [f"- Yêu cầu: {t}\n  Các lệnh đúng: {json.dumps(c, ensure_ascii=False)}" for _, t, c in picked]
    return ("## Ví dụ cách làm đúng (ứng dụng khác, chỉ để học quy ước: tra ID bằng tool read thay vì hỏi lại, "
            "không thêm tham số tùy chọn người dùng không nói)\n" + "\n".join(lines) + "\n\n")


def initial_messages(item, transcript, catalog):
    ctx = item["context"]
    tools = tools_for_item(item, catalog)
    tool_txt = "\n".join(render_tool(t) for t in tools)
    meta = {
        "current_time": ctx.get("current_time"),
        "timezone": ctx.get("timezone"),
        "location": ctx.get("current_location") or ctx.get("location"),
        "user_profile": ctx.get("user_profile"),
    }
    user = (
        f"## Công cụ khả dụng\n{tool_txt}\n"
        f"* sys.ask_missing_info (meta): hỏi lại người dùng. Tham số: tool_name: string, params: array<string>\n\n"
        f"## Ngữ cảnh\n{json.dumps(meta, ensure_ascii=False)}\n\n"
        f"## Lịch sử hội thoại\n{json.dumps(render_history(item.get('history')), ensure_ascii=False)}\n\n"
        f"{fewshot_block(item, transcript)}"
        f"## Yêu cầu của người dùng (từ giọng nói)\n{transcript}"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def call_msg(name, arguments, thought=None):
    obj = {"thought": thought} if thought else {}
    obj.update({"name": name, "arguments": arguments})
    return {"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)}


def done_msg(thought=None):
    obj = {"thought": thought} if thought else {}
    obj["done"] = True
    return {"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)}


def money_hint(obj, limit=6):
    """Large amounts restated in millions ("sell=126.6 triệu"), so a threshold said as "127 triệu" is easy to compare."""
    found = []

    def walk(o, key=""):
        if len(found) >= limit:
            return
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, k)
        elif isinstance(o, list):
            for v in o[:6]:
                walk(v, key)
        elif isinstance(o, (int, float)) and not isinstance(o, bool) and abs(o) >= 1_000_000:
            found.append(f"{key}={o / 1_000_000:.2f}".rstrip("0").rstrip(".") + " triệu")

    walk(obj)
    return "\n(quy đổi: " + ", ".join(found) + ")" if found else ""


def device_digest(obj):
    """Devices/locks regrouped by floor in one line each, so "all lights on floor 2 except rooms with the AC on" is a lookup."""
    if not isinstance(obj, dict):
        return ""
    lines = {}
    for d in obj.get("devices") or []:
        if isinstance(d, dict) and "floor" in d and isinstance(d.get("state"), dict):
            st = ", ".join(f"{k}={v}" for k, v in d["state"].items())
            lines.setdefault(f"tầng {d['floor']}", []).append(f"{d.get('id')} ({d.get('name')}; {st})")
    for d in obj.get("locks") or []:
        if isinstance(d, dict) and "location" in d:
            lines.setdefault(str(d["location"]), []).append(f"{d.get('id')} ({d.get('name')}; locked={d.get('locked')})")
    if not lines:
        return ""
    return "\n(tóm tắt: " + " | ".join(f"{k}: " + "; ".join(v) for k, v in sorted(lines.items())) + ")"


def result_msg(result, transcript=None):
    digest = device_digest(result) if RESULT_DIGEST else ""
    cond = condition_hint(transcript, result) if COND_HINT and transcript else ""
    return {"role": "user", "content": "KẾT QUẢ: " + truncate(result) + (money_hint(result) if MONEY_HINT else "") + digest + cond}
