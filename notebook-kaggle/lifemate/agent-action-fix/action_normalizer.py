"""Conservative action-time arithmetic using only user input and observed tool results.

Does not read labels or simulator state. Never adds/removes a tool call or chooses a new ID.
Ambiguous clocks/events are left to the original agent.
"""
from __future__ import annotations

import copy
import re
import unicodedata
from datetime import datetime, timedelta

WORDS = {"không": 0, "một": 1, "mốt": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4,
         "năm": 5, "lăm": 5, "sáu": 6, "bảy": 7, "tám": 8, "chín": 9}
NUMBER = r"(?:\d{1,3}|(?:(?:không|một|mốt|hai|ba|bốn|tư|năm|lăm|sáu|bảy|tám|chín|mười|mươi|trăm|linh|lẻ)\s*){1,7})"


def text(s):
    return " ".join(unicodedata.normalize("NFC", s or "").lower().split())


def number(s):
    tokens = text(s).split()
    if not tokens:
        return None
    if len(tokens) == 1 and tokens[0].isdigit():
        return int(tokens[0])
    tokens = [t for t in tokens if t not in {"linh", "lẻ"}]
    if "trăm" in tokens:
        i = tokens.index("trăm")
        if i != 1 or tokens[0] not in WORDS:
            return None
        tail = number(" ".join(tokens[2:])) if tokens[2:] else 0
        return WORDS[tokens[0]] * 100 + tail if tail is not None else None
    if tokens[0] == "mười":
        return 10 + WORDS[tokens[1]] if len(tokens) == 2 and tokens[1] in WORDS else 10 if len(tokens) == 1 else None
    if "mươi" in tokens:
        if len(tokens) not in (2, 3) or tokens[1] != "mươi" or tokens[0] not in WORDS:
            return None
        return WORDS[tokens[0]] * 10 + (WORDS.get(tokens[2], 0) if len(tokens) == 3 else 0)
    if len(tokens) == 1:
        return WORDS.get(tokens[0])
    if len(tokens) == 2 and tokens[0] in WORDS and tokens[1] == "lăm":
        return WORDS[tokens[0]] * 10 + 5
    return None


def _datetime(value):
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return result if result.tzinfo else None
    except (ValueError, AttributeError, TypeError):
        return None


def _date(request, now):
    # ASR lowercases names too: "cô Mai" is a person, not a date word.
    request = re.sub(r"\b(?:cô(?: giáo)?|chị|anh|bạn|chú|bác|dì|ông|bà|gọi)\s+mai\b", "người được nhắc", request)
    match = re.search(rf"ngày\s+({NUMBER})\s+tháng\s+({NUMBER})", request)
    if match:
        day, month = number(match[1]), number(match[2])
        if day is not None and month is not None:
            try:
                return now.replace(month=month, day=day).date()
            except ValueError:
                return None
    if re.search(r"\b(?:ngày|hôm|sáng|chiều|trưa|tối|đêm)\s+mai\b", request):
        return (now + timedelta(days=1)).date()
    if re.search(rf"(?:^|[,.;]\s*)mai\s+(?:lúc\s+)?{NUMBER}\s*giờ", request):
        return (now + timedelta(days=1)).date()
    if re.search(r"(?:hôm|sáng|chiều|trưa|tối|đêm) nay", request):
        return now.date()
    weekdays = {"hai": 0, "2": 0, "ba": 1, "3": 1, "tư": 2, "4": 2,
                "năm": 3, "5": 3, "sáu": 4, "6": 4, "bảy": 5, "7": 5}
    match = re.search(r"thứ\s+(hai|ba|tư|năm|sáu|bảy|[2-7])|chủ nhật", request)
    if match:
        weekday = weekdays[match[1]] if match[1] else 6
        if "tuần sau" in request:
            return (now + timedelta(days=7 - now.weekday() + weekday)).date()
        if "tuần này" in request:
            return (now + timedelta(days=weekday - now.weekday())).date()
        return (now + timedelta(days=(weekday - now.weekday()) % 7)).date()
    return None


def _clock(request, now, previous, arrival=False):
    # Arrival deadlines, self-corrections and dates relative to another day need
    # additional reasoning; a bare wall clock must not replace those calculations.
    if re.search(r"à không|ý là|không phải", request):
        return None
    if not arrival and re.search(r"sao cho|có mặt|kịp|tới nơi|đến nơi|rời|ra khỏi|khởi hành|xuất phát", request):
        return None
    if not arrival and re.search(r"\b(?:trước|sau|sớm|muộn)\b", request):
        return None
    if re.search(r"(?:hôm|ngày|sáng|chiều|tối)\s+(?:trước|sau)\b", request):
        return None
    pattern = re.compile(rf"(?P<h>{NUMBER})\s*giờ\s*(?:(?:kém\s*(?P<minus>{NUMBER}))|(?P<half>rưỡi)|(?P<plus>{NUMBER})(?:\s*phút)?)?\s*(?P<period>sáng|chiều|tối|trưa|đêm)?")
    found = []
    for match in pattern.finditer(request):
        if re.search(r"(?:trước|sau|muộn|sớm)\s*$", request[max(0, match.start() - 12):match.start()]):
            continue
        if re.search(r"từ\s*$", request[max(0, match.start() - 6):match.start()]):
            continue
        hour = number(match["h"])
        minute = number(match["minus"] or match["plus"]) if match["minus"] or match["plus"] else 30 if match["half"] else 0
        if hour is None or minute is None or not 0 <= hour <= 23 or not 0 <= minute < 60:
            continue
        period = match["period"]
        if period == "đêm":
            continue  # midnight/day-boundary semantics are deliberately left unchanged
        old = _datetime(previous)
        segment = re.split(r"[,.;]", request[match.start():match.end() + 25], maxsplit=1)[0]
        if len(set(re.findall(r"\b(?:sáng|chiều|tối|trưa|đêm)\b", segment))) > 1:
            continue
        if period is None:
            global_periods = set(re.findall(r"\b(sáng|chiều|tối|trưa|đêm)\s+nay\b", request))
            if len(global_periods) == 1:
                period = next(iter(global_periods))
            elif hour < 12 and old is not None:
                hour = hour % 12 + (12 if old.hour >= 12 else 0)
            elif hour < 12:
                continue
        if period in {"chiều", "tối"} and hour < 12:
            hour += 12
        if period == "trưa" and hour < 6:
            hour += 12
        elif period == "trưa" and hour < 11:
            continue
        if period == "sáng" and hour == 12:
            hour = 0
        prefix = re.split(r"[,.;]", request[max(0, match.start() - 12):match.start()])[-1]
        suffix = re.split(r"[,.;]", request[match.end():match.end() + 50], maxsplit=1)[0]
        local = prefix + match.group(0) + suffix
        day = _date(local, now)
        if day is None:
            if old is None:
                continue
            day = old.date()
        value = datetime.combine(day, datetime.min.time(), tzinfo=now.tzinfo).replace(hour=hour)
        value += timedelta(minutes=-minute if match["minus"] else minute)
        found.append(value)
    return found[0] if len(found) == 1 else None


def _observed_events(row, steps):
    events = {}
    observations = []
    for turn in row.get("history", []):
        if turn.get("role") == "assistant":
            observations.extend((call, result, True) for call, result in zip(turn.get("tool_calls") or [], turn.get("tool_results") or []))
    observations.extend(({"name": step["name"], "arguments": step.get("arguments", {})}, step["result"], False) for step in steps
                        if step.get("kind") == "call" and "result" in step)
    current_initial = set()
    for call, result, historical in observations:
        if not isinstance(result, dict):
            continue
        name = call.get("name")
        if name == "calendar.list_events":
            for event in result.get("events") or []:
                if isinstance(event, dict) and event.get("id") and _datetime(event.get("start")):
                    if historical or event["id"] not in current_initial:
                        events[event["id"]] = event
                    if not historical:
                        current_initial.add(event["id"])
        elif historical and result.get("ok") is True:
            args = call.get("arguments") or {}
            if name == "calendar.cancel_event":
                events.pop(args.get("event_id"), None)
            elif name in {"calendar.create_event", "calendar.update_event"}:
                event = result.get("item") or result.get("event")
                if isinstance(event, dict) and event.get("id") and _datetime(event.get("start")):
                    events[event["id"]] = event
                elif args.get("event_id") in events:
                    events[args["event_id"]] = dict(events[args["event_id"]], **{k: v for k, v in args.items() if k != "event_id" and v is not None})
    return list(events.values())


def _event(row, request, steps, args, now):
    events = _observed_events(row, steps)
    if args.get("event_id"):
        return next((e for e in events if e["id"] == args["event_id"]), None)
    day = _date(request, now)
    if day is None and _datetime(args.get("schedule_time")):
        day = _datetime(args["schedule_time"]).date()
    if day:
        events = [e for e in events if _datetime(e["start"]).date() == day]
    if len(events) == 1 and re.search(r"\b(?:hẹn|lịch)\b", request):
        return events[0]
    tokens = set(re.findall(r"\w+", request))
    ranked = sorted(((len(tokens & set(re.findall(r"\w+", text(e.get("title"))))), i, e)
                     for i, e in enumerate(events)), reverse=True, key=lambda item: item[0])
    if ranked and ranked[0][0] >= 2 and (len(ranked) == 1 or ranked[0][0] > ranked[1][0]):
        return ranked[0][2]
    return None


def _duration(request, patterns):
    values = []
    for pattern in patterns:
        for match in re.finditer(pattern, request):
            value = number(match["n"])
            if value is not None:
                values.append(value * (60 if match["unit"] in {"tiếng", "giờ"} else 1))
    unique = set(values)
    return next(iter(unique)) if len(unique) == 1 else None


MEASURE = rf"(?P<n>{NUMBER})\s*(?P<unit>phút|tiếng|giờ)"
RIDE_LEAD = [rf"trước\s+(?:giờ\s+)?hẹn\s+{MEASURE}", rf"đón\s+sớm\s+{MEASURE}",
             rf"sớm\s+{MEASURE}\s+so với giờ hẹn"]
PICKUP_LEAD = [rf"trước\s+(?:lúc|giờ|khi)\s+(?:xe\s+(?:đón|tới)|đón\s+xe)\s+{MEASURE}",
               rf"{MEASURE}\s+trước\s+(?:(?:lúc|giờ|khi)\s+)?(?:ra\s+)?xe\s+(?:đón|tới)"]
EVENT_LEAD = [rf"báo thức.{{0,40}}?trước\s+(?:đó|giờ hẹn)\s+{MEASURE}",
              rf"báo thức\s+{MEASURE}\s+trước\s+đó"]


def _departure(row, request, steps, now, previous):
    if not re.search(r"có mặt|kịp|sao cho|tới tiệm|tới nơi|đến nơi", request):
        return None
    if not re.search(r"nhắc.{0,70}(?:rời|đi|ra khỏi|khởi hành|xuất phát|chạy)", request):
        return None
    routes = []
    for turn in row.get("history", []):
        for call, result in zip(turn.get("tool_calls") or [], turn.get("tool_results") or []):
            if call.get("name") == "maps.get_route" and isinstance(result, dict):
                routes.append(result.get("route") or {})
    for step in steps:
        if step.get("kind") == "call" and step.get("name") == "maps.get_route" and isinstance(step.get("result"), dict):
            routes.append(step["result"].get("route") or {})
    if len(routes) != 1:
        return None
    duration = routes[0].get("duration_min")
    deadline = _clock(request, now, previous, arrival=True)
    if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration < 0 or deadline is None:
        return None
    early = _duration(request, [rf"(?:tới|đến)\s+sớm\s+{MEASURE}"]) or 0
    return deadline - timedelta(minutes=duration + early)


def normalize_action(row, request, steps, name, arguments):
    args = copy.deepcopy(arguments)
    audit = []
    if name not in {"calendar.update_event", "ride_hailing.book", "reminders.create", "alarm.create"}:
        return args, audit
    request = text(request)
    now = _datetime(row.get("context", {}).get("current_time"))
    if now is None:
        return args, audit

    def update(key, value, reason):
        if value is None:
            return
        value = value.isoformat(timespec="seconds")
        if args.get(key) != value:
            audit.append({"parameter": key, "before": args.get(key), "after": value, "reason": reason})
            args[key] = value

    event_args = args
    if name == "alarm.create":
        updates = [step["arguments"] for step in steps if step.get("kind") == "call"
                   and step.get("name") == "calendar.update_event"]
        if len(updates) == 1:
            event_args = {"event_id": updates[0].get("event_id")}
    event = _event(row, request, steps, event_args, now)
    anchor = _datetime(event["start"]) if event else None
    if event:
        updates = [step for step in steps if step.get("kind") == "call" and step.get("name") == "calendar.update_event"
                   and step.get("arguments", {}).get("event_id") == event["id"]]
        if updates:
            anchor = _datetime(updates[-1]["arguments"].get("start")) or anchor
    if name == "calendar.update_event" and event:
        match = re.search(rf"(?<!đón )(?P<direction>muộn|trễ|sớm)\s+(?P<n>{NUMBER})\s*(?P<unit>phút|tiếng|giờ)", request)
        if match and not re.search(r"đón\s*$", request[max(0, match.start() - 8):match.start()]):
            amount = number(match["n"])
            if amount is not None:
                delta = timedelta(minutes=amount * (60 if match["unit"] in {"tiếng", "giờ"} else 1)
                                  * (-1 if match["direction"] == "sớm" else 1))
                update("start", _datetime(event["start"]) + delta, "explicit event shift")
                end = _datetime(event.get("end"))
                if end and not re.search(r"giữ.{0,12}(?:kết thúc|giờ cuối)", request):
                    update("end", end + delta, "preserve event duration")
    elif name == "ride_hailing.book":
        lead = _duration(request, RIDE_LEAD)
        if anchor is not None and lead is not None:
            update("schedule_time", anchor - timedelta(minutes=lead), "explicit pickup before observed event")
    else:
        lead = _duration(request, PICKUP_LEAD)
        bookings = [step["arguments"] for step in steps if step.get("kind") == "call"
                    and step.get("name") == "ride_hailing.book" and _datetime(step.get("arguments", {}).get("schedule_time"))]
        target = None
        if name == "reminders.create" and lead is not None and len(bookings) == 1:
            target = _datetime(bookings[0]["schedule_time"]) - timedelta(minutes=lead)
        if name == "alarm.create" and anchor is not None:
            lead = _duration(request, EVENT_LEAD)
            if lead is not None:
                target = anchor - timedelta(minutes=lead)
        if name == "reminders.create" and target is None:
            target = _departure(row, request, steps, now, args.get("time"))
        if target is None and ((name == "reminders.create" and "nhắc" in request)
                               or (name == "alarm.create" and "báo thức" in request)):
            target = _clock(request, now, args.get("time"))
        update("time", target, "explicit clock or observed temporal anchor")
    if name in {"alarm.create", "reminders.create"} and args.get("repeat") == "once":
        args.pop("repeat")
        audit.append({"parameter": "repeat", "before": "once", "after": None, "reason": "omit simulator default"})
    return args, audit
