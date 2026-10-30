"""Evaluate a price threshold said by the user against the price a tool returned ("bán ra trên 127 triệu" vs sell=126.6M).

The 4B model compares such near-equal figures unreliably, so code does the comparison and states the outcome in the tool
result. Deterministic and generic: any "<comparator> <number> triệu" in the transcript against buy/sell prices.
"""
import re

UNITS = {"không": 0, "một": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4, "năm": 5, "lăm": 5, "sáu": 6, "bảy": 7, "tám": 8, "chín": 9}
ABOVE = r"(?:vượt\s+quá|vượt|cao\s+hơn|lớn\s+hơn|trên|hơn|quá)"
BELOW = r"(?:thấp\s+hơn|nhỏ\s+hơn|ít\s+hơn|dưới)"
NUM_WORD = r"(?:" + "|".join(UNITS) + r"|mười|mươi|mốt|trăm|linh|lẻ)"
NUMBER = rf"(\d+(?:[.,]\d+)?|{NUM_WORD}(?:\s+{NUM_WORD})*)"
PATTERN = re.compile(rf"\b({ABOVE}|{BELOW})\s+(?:mức\s+)?{NUMBER}\s*(?:triệu|tr)\b", re.I)


def words_to_int(text):
    """'một trăm hai mươi' -> 120, 'một trăm linh năm' -> 105 (hundreds and tens only; enough for prices in millions)."""
    total, cur, tens = 0, 0, False
    for w in text.lower().split():
        if w in UNITS:
            cur = UNITS[w]
        elif w == "mười":
            cur, tens = 10, True
        elif w == "mươi":
            cur, tens = cur * 10 if cur else 10, True
        elif w == "mốt":
            cur += 1
        elif w == "trăm":
            total, cur = total + cur * 100, 0
        elif w in ("linh", "lẻ"):
            continue
    return total + cur


def parse_threshold(transcript):
    m = PATTERN.search(transcript)
    if not m:
        return None
    cmp_word, num = m.group(1).lower(), m.group(2)
    if re.fullmatch(r"\d+(?:[.,]\d+)?", num):
        value = float(num.replace(",", "."))
    else:
        value = float(words_to_int(num))
    return ("above" if re.fullmatch(ABOVE, cmp_word) else "below"), value, m.group(0)


def condition_hint(transcript, result):
    """'\n(so sánh: giá bán ra 126.6 triệu không trên 127 triệu -> điều kiện SAI)' or '' when nothing can be decided."""
    if not isinstance(result, dict) or not isinstance(result.get("prices"), list) or len(result["prices"]) != 1:
        return ""
    price = result["prices"][0]
    t = transcript.lower()
    field = "sell" if "bán ra" in t else "buy" if ("mua vào" in t or "mua vô" in t) else None
    parsed = parse_threshold(transcript)
    if not field or not parsed or not isinstance(price.get(field), (int, float)):
        return ""
    kind, thr, phrase = parsed
    val = price[field] / 1_000_000
    ok = val > thr if kind == "above" else val < thr
    label = "bán ra" if field == "sell" else "mua vào"
    return f"\n(so sánh: giá {label} {val:g} triệu so với '{phrase}' -> điều kiện {'ĐÚNG' if ok else 'SAI'})"
