"""TIME_HINT: weekday/date table and minutes-until table computed from context.current_time.

The 3-epoch model loses points on pure date/time arithmetic (weekday -> date, "3 giờ chiều nay" -> delay_minutes,
"trước 15 phút"). Both tables turn that arithmetic into a lookup. Must be identical in SFT data building and inference."""
from datetime import datetime, timedelta

WEEKDAYS = ["thứ hai", "thứ ba", "thứ tư", "thứ năm", "thứ sáu", "thứ bảy", "chủ nhật"]
DAYS_AHEAD = 14
GRID_MIN = 30
HOURS_AHEAD = 24


def time_hint(current_time):
    try:
        now = datetime.fromisoformat(current_time)
    except (TypeError, ValueError):
        return ""
    days = []
    for i in range(DAYS_AHEAD):
        d = now + timedelta(days=i)
        label = {0: "hôm nay", 1: "mai", 2: "ngày kia"}.get(i, "")
        days.append(f"{WEEKDAYS[d.weekday()]} {d:%Y-%m-%d}" + (f" ({label})" if label else ""))
    start = now.replace(second=0, microsecond=0)
    first = start + timedelta(minutes=(-(start.minute + start.hour * 60)) % GRID_MIN or GRID_MIN)
    steps = []
    t = first
    while t <= start + timedelta(hours=HOURS_AHEAD):
        steps.append(f"{t:%H:%M}{'' if t.date() == now.date() else '+1' if (t.date() - now.date()).days == 1 else '+2'}={int((t - start).total_seconds() // 60)}")
        t += timedelta(minutes=GRID_MIN)
    return ("## Lịch và thời gian (tra cứu, không tự tính)\n"
            f"Ngày: {'; '.join(days)}.\n"
            f"Số phút từ bây giờ ({now:%H:%M}) đến mốc giờ (+1 = hôm sau): {', '.join(steps)}.\n\n")
