"""Spoken-form Vietnamese text for ASR targets: digits become words, punctuation goes, case is up to the caller.

Gipformer is trained on lower-case text without punctuation and with numbers read out ("một trăm hai mươi bảy triệu"),
while the competition transcripts write "127 triệu", "9 giờ", "09889990002", "45%". Used to turn the gold text into
targets for train/gipformer_ft.py and to score its output."""
import re
import unicodedata

DIGITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
UNITS = ["", "nghìn", "triệu", "tỷ"]


def _group(n, full):
    """Words for 0 < n < 1000; full=True when a higher group came first, so 5 is "không trăm linh năm"."""
    h, r = divmod(n, 100)
    t, u = divmod(r, 10)
    w = []
    if h or full:
        w += [DIGITS[h], "trăm"]
    if t == 0:
        if u:
            if h or full:
                w.append("linh")
            w.append(DIGITS[u])
    elif t == 1:
        w.append("mười")
        if u == 5:
            w.append("lăm")
        elif u:
            w.append(DIGITS[u])
    else:
        w += [DIGITS[t], "mươi"]
        if u == 1:
            w.append("mốt")
        elif u == 5:
            w.append("lăm")
        elif u:
            w.append(DIGITS[u])
    return w


def number_words(n):
    if n == 0:
        return ["không"]
    groups = []
    while n and len(groups) < len(UNITS):
        groups.append(n % 1000)
        n //= 1000
    out = []
    for i in range(len(groups) - 1, -1, -1):
        g = groups[i]
        if g == 0:
            continue
        out += _group(g, full=bool(out) and g < 100)
        if UNITS[i]:
            out.append(UNITS[i])
    return out


def _digits(s):
    return [DIGITS[int(c)] for c in s]


def _sub_number(m):
    s = m.group(0)
    if re.fullmatch(r"0\d{8,11}", s):  # phone number: digit by digit
        return " " + " ".join(_digits(s)) + " "
    if re.fullmatch(r"\d{1,2}:\d{2}", s):  # 8:30 -> tám giờ ba mươi
        hh, mm = s.split(":")
        w = number_words(int(hh)) + ["giờ"]
        if int(mm):
            w += number_words(int(mm))
        return " " + " ".join(w) + " "
    if re.fullmatch(r"\d{1,3}(\.\d{3})+", s):  # 1.290.000
        return " " + " ".join(number_words(int(s.replace(".", "")))) + " "
    if re.fullmatch(r"\d+[,.]\d+", s):  # 12,8 -> mười hai phẩy tám
        a, b = re.split(r"[,.]", s)
        return " " + " ".join(number_words(int(a)) + ["phẩy"] + _digits(b)) + " "
    return " " + " ".join(number_words(int(s))) + " "


NUMBER = re.compile(r"0\d{8,11}|\d{1,2}:\d{2}|\d{1,3}(?:\.\d{3})+|\d+[,.]\d+|\d+")


def spoken(text):
    """Lower-case spoken form without punctuation (what the model should output)."""
    s = unicodedata.normalize("NFC", text).lower()
    s = s.replace("%", " phần trăm ")
    s = NUMBER.sub(_sub_number, s)
    s = re.sub(r"[^\w\s]|_", " ", s)
    return " ".join(s.split())
