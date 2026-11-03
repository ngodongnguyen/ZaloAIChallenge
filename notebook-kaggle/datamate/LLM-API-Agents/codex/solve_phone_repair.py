"""Direct, data-only calculations for DataMate dev_011/dev_012/dev_014.

No local LLM/API calls. Source spreadsheets are read-only. Results are written
beside this file as an id -> submission-object mapping plus compact evidence.
"""
from __future__ import annotations

import calendar
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from common import DATA, OUT, book, dt, norm, num, rnd, tab, text

SHOPS = {"dev_011", "dev_012", "dev_014"}
QUESTIONS = [q for q in __import__("common").QUESTIONS if q["business_id"] in SHOPS]
EVIDENCE: dict[str, dict] = {}


def end_month(y: int, m: int) -> date:
    return date(y, m, calendar.monthrange(y, m)[1])


def month_range(y: int, m: int) -> tuple[date, date]:
    return date(y, m, 1), end_month(y, m)


def val_date(v, fallback=None):
    x = dt(v)
    if pd.isna(x):
        return fallback
    return x.date()


def first_value(values, default=None):
    for v in values:
        if text(v):
            return v
    return default


def safe_int(v, default=0):
    try:
        return int(num(v))
    except Exception:
        return default


def sheet_period(name: str) -> tuple[int, int] | None:
    m = re.search(r"T(\d{1,2})-(\d{4})", name, re.I)
    return (int(m.group(2)), int(m.group(1))) if m else None


def model_key(value):
    raw=text(value).lower()
    n=norm(value)
    if "alpha" in raw or "α" in raw or n in ("wave", "alpha"):
        return "wave alpha"
    aliases={"ab":"air blade", "airblade":"air blade", "shmode":"sh mode",
             "ex":"exciter", "rsx":"wave rsx", "vision 110":"vision",
             "future 125":"future", "vario 125":"vario", "sirius fi":"sirius"}
    return aliases.get(n,n)


def get_sales(shop: str):
    path = DATA / "data" / shop / "SoBanHang.xlsx"
    wb = book(path)
    aliases = tab(path, "Ghi tắt")
    alias_name = {norm(r["Ghi tắt"]): text(r["Tên hàng"]) for _, r in aliases.iterrows()}
    alias_type = {norm(r["Ghi tắt"]): norm(r["Loại"]) for _, r in aliases.iterrows()}
    for _, r in aliases.iterrows():
        alias_name[norm(r["Tên hàng"])] = text(r["Tên hàng"])
        alias_type[norm(r["Tên hàng"])] = norm(r["Loại"])
    all_lines = []
    for sheet in wb:
        if norm(sheet) == norm("Ghi tắt"):
            continue
        p = sheet_period(sheet)
        df = tab(path, sheet)
        if "Ngày" not in df or "STT" not in df:
            continue
        dates = df["Ngày"].map(lambda v: val_date(v, date(*p, 1) if p else None)).ffill()
        stts = df["STT"].map(lambda x: safe_int(x, None) if text(x) else None).ffill()
        for i, row in df.iterrows():
            stt = stts.iloc[i]
            if stt is None or pd.isna(stt):
                continue
            d = dates.iloc[i]
            if d is None or pd.isna(d):
                continue
            sh_key = sheet
            item_short = text(row.get("Hàng"))
            if not item_short:
                continue
            nitem = norm(item_short)
            item_full = alias_name.get(nitem, item_short)
            typ = alias_type.get(nitem, "")
            imei = text(row.get("IMEI"))
            if not typ:
                if imei:
                    # Some books use the full product name instead of a
                    # `Ghi tắt` code. Those names mark used phones with `cũ`;
                    # the remaining IMEI-bearing rows are new phones.
                    typ = norm("máy cũ") if re.search(r"\bcu\b", norm(item_full)) else norm("máy mới")
                else:
                    # The shop READMEs state that phone sales have an IMEI;
                    # accessory lines do not.
                    typ = norm("phụ kiện")
            all_lines.append({
                "shop": shop, "sheet": sh_key, "date": d,
                "stt": int(stt), "key": f"{sh_key}:{int(stt)}",
                "customer": text(row.get("Khách")),
                "short": item_short, "item": item_full, "item_norm": norm(item_full),
                "kind": typ, "imei": imei,
                "qty": num(row.get("SL")), "unit_price": num(row.get("Giá")),
                "amount": num(row.get("Thành tiền")),
                "discount_raw": row.get("Bớt"), "trade_raw": row.get("Thu máy cũ"),
                "paid_raw": row.get("Khách trả"), "note": text(row.get("Ghi chú")),
                "raw_sig": tuple(text(row.get(c)) for c in ("Ngày","STT","Khách","Hàng","IMEI","SL","Giá","Thành tiền","Bớt","Thu máy cũ","Khách trả","Ghi chú")),
            })
    # Exact duplicate detail rows inside the same order are export artifacts.
    seen = set(); lines = []
    for r in all_lines:
        sig = (r["key"], r["raw_sig"])
        if sig in seen:
            continue
        seen.add(sig); lines.append(r)
    grouped = defaultdict(list)
    for r in lines:
        grouped[r["key"]].append(r)
    orders = []
    for key, rows in grouped.items():
        first = rows[0]
        note = first_value((r["note"] for r in rows), "")
        ret = any(any(k in norm(r["note"]) for k in ("tra lai", "hoan tra", "doi tra", "mang tra", "khach tra", "tra hang")) for r in rows)
        discounts = [num(r["discount_raw"]) for r in rows if text(r["discount_raw"])]
        tradevalues = [num(r["trade_raw"]) for r in rows if text(r["trade_raw"])]
        paidvalues = [num(r["paid_raw"]) for r in rows if text(r["paid_raw"])]
        revenue = sum(r["amount"] for r in rows)
        discount = discounts[0] if discounts else 0
        tradein = tradevalues[0] if tradevalues else 0
        orders.append({
            "key": key, "sheet": first["sheet"], "date": first["date"],
            "stt": first["stt"], "customer": first_value((r["customer"] for r in rows), ""),
            "note": note, "return": ret, "revenue": revenue,
            "discount": discount, "tradein": tradein,
            "payable": revenue - discount - tradein,
            "paid": paidvalues[0] if paidvalues else 0,
            "rows": rows,
        })
    return lines, orders, {"source": str(path), "sheets": list(wb), "line_rows": len(lines), "order_count": len(orders)}


def load_repairs():
    shop = "dev_012"; path = DATA / "data" / shop / "SoSuaXe.xlsx"; wb = book(path)
    reps = []
    for sheet in wb:
        p = sheet_period(sheet)
        df = tab(path, sheet)
        for _, row in df.iterrows():
            xe = text(row.get("Xe"))
            if not xe:
                continue
            year, month = p or (2025, 1)
            d = val_date(row.get("Ngày"), date(year, month, 1))
            if d is None: continue
            work = text(row.get("Làm gì")); note = text(row.get("Ghi chú"))
            plate = re.search(r"\b\d{2}[A-Z0-9]{1,2}\d?[-.]\d{3}[.]\d{2}\b", xe, re.I)
            model = xe[:plate.start()].strip() if plate else re.sub(r"\s+\d.*$", "", xe).strip()
            reps.append({"date": d, "sheet": sheet, "vehicle": xe, "plate": plate.group(0).upper() if plate else norm(xe),
                         "model": model or xe, "model_norm": model_key(model or xe), "work": work, "work_norm": norm(work),
                         "tech_alias": text(row.get("Thợ")), "money": num(row.get("Tiền")), "note": note,
                         "warranty": "bao hanh" in norm(note) and num(row.get("Tiền")) == 0})
    staffdf = tab(DATA / "data" / shop / "DanhSachTho.xlsx")
    staff = {norm(r["Họ tên"]): {"name": text(r["Họ tên"]), "rate": num(r["Hoa hồng (%)"])} for _, r in staffdf.iterrows()}
    aliases = {"an": norm("Đàm Lan An"), "phuc": norm("Đoàn Bảo Phúc"), "dat": norm("Hà Loan Đạt"), "duc": norm("Lương Chi Đức")}
    for r in reps:
        a = norm(r["tech_alias"])
        if a.startswith("a "): a = a[2:].strip()
        r["tech"] = aliases.get(a, "")
        r["tech_name"] = staff.get(r["tech"], {}).get("name", r["tech_alias"])
        r["commission"] = r["money"] * staff.get(r["tech"], {}).get("rate", 0) / 100
    return reps, staff, {"source": str(path), "sheet_rows": len(reps), "sheets": list(wb)}


def load_retail_extra():
    shop = "dev_014"; base = DATA / "data" / shop
    wb = book(base / "SoNo.xlsx")
    def flatten(sheet):
        return tab(base / "SoNo.xlsx", sheet)
    # Warranty rows
    warranty = []
    for _, r in flatten("Bảo hành").iterrows():
        warranty.append({"date": val_date(r.get("Ngày nhận")), "imei": re.sub(r"\D", "", text(r.get("IMEI"))),
                         "customer": text(r.get("Khách")), "phone": re.sub(r"\D", "", text(r.get("SĐT"))),
                         "model": text(r.get("Máy")), "model_norm": norm(r.get("Máy"))})
    # Trade-ins
    ti = []
    for _, r in tab(base / "ThuMayCu.xlsx").iterrows():
        d = val_date(r.get("Ngày"))
        ti.append({"date": d, "stt": safe_int(r.get("Trừ vào (STT sổ bán)")), "amount": num(r.get("Giá thu")),
                   "imei": re.sub(r"\D", "", text(r.get("IMEI"))), "customer": text(r.get("Khách"))})
    # Installments: one record per sale contract, payments are its nonblank dated rows.
    inst_df = flatten("Trả góp")
    inst_df["contract_date"] = inst_df["Ngày"].map(val_date).ffill()
    inst_df["stt_fill"] = inst_df["STT"].map(lambda x: safe_int(x, None) if text(x) else None).ffill()
    inst_df["customer_fill"] = inst_df["Khách"].ffill()
    inst_df["phone_fill"] = inst_df["SĐT"].ffill()
    groups = defaultdict(list)
    for _, r in inst_df.iterrows():
        d = r["contract_date"]
        s = r["stt_fill"]
        if d is None or pd.isna(d) or s is None or pd.isna(s): continue
        groups[(d, int(s))].append(r)
    contracts = []
    for (buydate, stt), rows in groups.items():
        first = next((r for r in rows if text(r.get("Ngày"))), rows[0])
        price = num(first.get("Giá")); down = num(first.get("Trả trước")); n = safe_int(first.get("Số kỳ"))
        per = num(first.get("Góp/tháng")); due_day = safe_int(first.get("Hạn góp (ngày)"), 1)
        payments = []
        for r in rows:
            pdte = val_date(r.get("Ngày góp"))
            amount = num(r.get("Đã góp"))
            if pdte and amount:
                payments.append((pdte, amount))
        contracts.append({"date": buydate, "stt": stt, "customer": text(first.get("Khách")) or text(first.get("customer_fill")),
                          "phone": re.sub(r"\D", "", text(first.get("SĐT")) or text(first.get("phone_fill"))),
                          "model": text(first.get("Máy")), "price": price, "down": down, "n": n,
                          "per": per, "due_day": due_day, "payments": payments})
    # General credit ledger: each order begins when date is populated; following rows are payments.
    debt_df = flatten("Sổ nợ")
    debt_df["buy_date"] = debt_df["Ngày"].map(val_date).ffill()
    debt_df["customer_fill"] = debt_df["Khách"].ffill(); debt_df["phone_fill"] = debt_df["SĐT"].ffill(); debt_df["debt_fill"] = debt_df["Nợ"].ffill()
    debts = []
    groups = defaultdict(list)
    for _, r in debt_df.iterrows():
        d = r["buy_date"]
        if d is None or pd.isna(d): continue
        sttmatch = re.search(r"STT\s*(\d+)", text(r.get("Ghi chú")), re.I)
        groups[(d, sttmatch.group(1) if sttmatch else text(r.get("customer_fill")) + text(r.get("phone_fill")))].append(r)
    for (buydate, ident), rows in groups.items():
        first = next((r for r in rows if text(r.get("Nợ"))), rows[0])
        payments = []
        for r in rows:
            pdte = val_date(r.get("Ngày trả")); amount = num(r.get("Đã trả"))
            if pdte and amount: payments.append((pdte, amount))
        debts.append({"date": buydate, "customer": text(first.get("Khách")) or text(first.get("customer_fill")),
                      "phone": re.sub(r"\D", "", text(first.get("SĐT")) or text(first.get("phone_fill"))),
                      "initial": num(first.get("Nợ")), "payments": payments})
    return warranty, ti, contracts, debts, {"source": str(base), "warranty_rows": len(warranty), "tradein_rows": len(ti), "installment_contracts": len(contracts), "credit_orders": len(debts)}


SALES = {s: get_sales(s) for s in ("dev_011", "dev_014")}
REPAIRS, STAFF, REPAIR_META = load_repairs()
WARRANTY, TRADEINS, INSTALLMENTS, DEBTS, EXTRA_META = load_retail_extra()
_tradein_by_order = defaultdict(float)
for _ti in TRADEINS:
    if _ti["date"]:
        _tradein_by_order[(_ti["date"].year, _ti["date"].month, _ti["stt"]) ] += _ti["amount"]
for _o in SALES["dev_014"][1]:
    _o["tradein"] = _tradein_by_order.get((_o["date"].year, _o["date"].month, _o["stt"]), 0)
    _o["payable"] = _o["revenue"] - _o["discount"] - _o["tradein"]


def as_date(q):
    return date.fromisoformat(q["as_of_date"])


def parse_window(q):
    raw = q["question"]
    s = norm(raw); today = as_date(q)
    # Keep slash and hyphen separators while removing Vietnamese diacritics.
    folded = unicodedata.normalize("NFD", raw.lower()).replace("đ", "d")
    folded = "".join(c for c in folded if unicodedata.category(c) != "Mn")
    folded = re.sub(r"\s+", " ", folded)
    # Handle an explicit month-to-month interval before the single-month
    # matcher below, which would otherwise stop at the first month.
    month_span = re.search(
        r"(?:thang|th)?\s*(\d{1,2})\s*(?:[/\-]\s*|nam\s*)(20\d{2})\s*"
        r"(?:den|toi)\s*(?:het\s*)?(?:thang|th)?\s*"
        r"(\d{1,2})\s*(?:[/\-]\s*|nam\s*)(20\d{2})",
        folded,
    )
    if month_span:
        start_month, start_year, end_month_number, end_year = map(int, month_span.groups())
        if 1 <= start_month <= 12 and 1 <= end_month_number <= 12:
            return date(start_year, start_month, 1), end_month(end_year, end_month_number)
    if "tet" in s or "tet nguyen dan" in s:
        m = re.search(r"tet(?: nguyen dan| am lich)?\s*(20\d{2})", s)
        y = int(m.group(1)) if m else today.year
        c = pd.read_csv(DATA / "calendar.csv")
        c["date"] = pd.to_datetime(c["date"]).dt.date
        start = c.loc[(c.lunar_year == y - 1) & (c.lunar_month == 12) & (c.lunar_day == 23), "date"].iloc[0]
        end = c.loc[(c.lunar_year == y) & (c.lunar_month == 1) & (c.lunar_day == 10), "date"].iloc[0]
        return start, end
    if "tu dau nam" in s or "nam nay tinh toi" in s or re.search(r"tu\s+(?:ngay\s+)?1[/\-]1\s+den", folded):
        return date(today.year, 1, 1), today
    qm = re.search(r"(?:quy|q)\s*([1-4])\s*(?:nam\s*)?[/\- ]\s*(20\d{2})", folded)
    if qm:
        quarter = int(qm.group(1)); year = int(qm.group(2)); month = (quarter - 1) * 3 + 1
        return date(year, month, 1), end_month(year, month + 2)
    # Explicit month, with or without a year.
    mm = re.search(
        r"(?:thang|th)\s*(\d{1,2})\s*(?:(?:[/\-]\s*|nam\s*)(\d{2,4}))?",
        folded,
    )
    if not mm:
        # Standalone M/YY and M/YYYY forms (for example `8/25`). The
        # lookarounds keep us from mistaking the month/year suffix of a full
        # D/M/YY date for a month period.
        mm = re.search(
            r"(?<![\d/])(\d{1,2})\s*[/\-]\s*(20\d{2}|\d{2})(?![\d/])",
            folded,
        )
    if mm:
        month = int(mm.group(1)); ys = mm.group(2)
        if 1 <= month <= 12:
            if ys:
                year = int(ys); year = year + 2000 if year < 100 else year
            else:
                year = today.year if month <= today.month else today.year - 1
            return month_range(year, month)
    if any(k in s for k in ("thang nay", "thang ni", "th nay", "th ni", "thang nay")):
        return date(today.year, today.month, 1), today
    if "tuan truoc" in s or "tuan roi" in s:
        monday = today - timedelta(days=today.weekday())
        return monday - timedelta(days=7), monday - timedelta(days=1)
    # Explicit calendar day or relative today/yesterday.
    if any(k in s for k in ("hom qua", "hqua", "bua qua")):
        d = today - timedelta(days=1); return d, d
    if any(k in s for k in ("hom nay", "bua ni", "ngay nay")):
        return today, today
    dm = re.search(r"(?:ngay\s*)?(\d{1,2})/(\d{1,2})/(20\d{2})", folded)
    if dm:
        d = date(int(dm.group(3)), int(dm.group(2)), int(dm.group(1))); return d, d
    # Default covers a bare "this month" variant and otherwise month through as-of date.
    return date(today.year, today.month, 1), today


def in_window(d, w):
    return d is not None and w[0] <= d <= w[1]


def sale_data(shop):
    lines, orders, meta = SALES[shop]
    return lines, orders


def valid_orders(shop, w):
    return [o for o in sale_data(shop)[1] if in_window(o["date"], w) and not o["return"]]


def valid_lines(shop, w):
    lines, orders = sale_data(shop)
    valid = {o["key"] for o in orders if in_window(o["date"], w) and not o["return"]}
    return [r for r in lines if r["key"] in valid]


def revenue(shop, w):
    return sum(o["revenue"] for o in valid_orders(shop, w))


def pct(cur, base):
    return rnd((cur - base) / base * 100, 2) if base else 0


def phone_lines(shop, w, kind=None):
    rows = valid_lines(shop, w)
    rows = [r for r in rows if r["kind"] in ("may moi", "may cu", "may") or r["imei"]]
    if kind:
        rows = [r for r in rows if r["kind"] == norm(kind)]
    return rows


def category_revenue(shop, w, cat):
    return sum(r["amount"] for r in valid_lines(shop, w) if r["kind"] == norm(cat))


def day_revenue(shop, w):
    by = defaultdict(float)
    for o in valid_orders(shop, w): by[o["date"]] += o["revenue"]
    if not by: return None
    return sorted(by.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def tradein_window(w):
    return [r for r in TRADEINS if in_window(r["date"], w)]


def month_overlap_window(today, previous=False, mtd=False):
    if previous:
        y, m = today.year, today.month - 1
        if m == 0: y, m = y - 1, 12
        end = end_month(y, m) if not mtd else date(y, m, min(today.day, calendar.monthrange(y,m)[1]))
    else:
        y, m = today.year, today.month
        end = today if mtd else end_month(y,m)
    return date(y,m,1), end


def month_indices(text_q, today):
    w = parse_window({"question": text_q, "as_of_date": today.isoformat()})
    return w


def money_answer(v): return {"type": "number", "answer": rnd(v)}
def count_answer(v): return {"type": "number", "answer": int(v)}
def percent_answer(v): return {"type": "percent", "answer": rnd(v, 2)}
def list_answer(v): return {"type": "list", "answer": v}
def string_answer(v): return {"type": "string", "answer": v}
def date_answer(v): return {"type": "date", "answer": v.isoformat() if isinstance(v, date) else v}
def unanswerable(reason): return {"type": "unanswerable", "reason": reason}


def total_payable(orders): return sum(o["payable"] for o in orders)


def group_lines_by_item(shop, w, keep=None):
    c = defaultdict(float)
    for r in valid_lines(shop, w):
        if keep and not keep(r): continue
        c[r["item"]] += r["qty"]
    return c


def phone_model_line(name):
    """The sales-sheet item prefix before storage capacity defines a phone line."""
    item=text(name)
    m=re.search(r"\b\d+\s*GB\b",item,re.I)
    return item[:m.start()].strip() if m else item


def phone_revenue(shop, w, kind=None):
    return sum(r["amount"] for r in phone_lines(shop, w, kind))


def phone_qty(shop, w, kind=None):
    return sum(r["qty"] for r in phone_lines(shop, w, kind))


def installment_calc(s, today):
    def due_dates(c):
        dates=[]; y,m=c["date"].year,c["date"].month
        for i in range(1,c["n"]+1):
            idx=y*12+(m-1)+i; yy=idx//12; mm=idx%12+1
            dates.append(date(yy,mm,min(c["due_day"],calendar.monthrange(yy,mm)[1])))
        return dates
    def paid_until(c, cutoff): return sum(v for d,v in c["payments"] if d<=cutoff)
    def unpaid_total(c, cutoff): return max(0,c["price"]-c["down"]-paid_until(c,cutoff))
    def overdue_amount(c, cutoff):
        required=min(c["price"]-c["down"], sum(c["per"] for d in due_dates(c) if d<cutoff))
        return max(0,required-paid_until(c,cutoff))
    contracts=[c for c in INSTALLMENTS if c["date"]<=today]
    if any(k in s for k in ("bao nhieu khach", "bnhiu khach", "may khach", "bao nhieu nguoi", "bnhiu nguoi", "may nguoi", "co may nguoi", "mi nguoi", "mi ngui")) and "qua han" in s:
        people={c["phone"] or norm(c["customer"]) for c in contracts if overdue_amount(c,today)>0}
        return count_answer(len(people)), f"distinct installment customers with scheduled unpaid amount past due as of {today}: {len(people)}"
    # Count scheduled installments that have reached their due date for a named customer.
    if "may ky" in s and any(k in s for k in ("toi han", "ky han", "gop toi han")):
        c=next((c for c in contracts if norm(c["customer"]) in s),None)
        n=sum(d<=today for d in due_dates(c)) if c else 0
        return count_answer(n), f"scheduled due installments for {c['customer'] if c else 'no matched customer'} by {today}"
    if "qua han" in s or "cham gop" in s or "da qua han" in s:
        v=sum(overdue_amount(c,today) for c in contracts)
        return money_answer(v), f"scheduled installment amounts due by {today} less contributions received by {today}"
    matched=[c for c in contracts if norm(c["customer"]) in s]
    chosen=matched if matched else contracts
    v=sum(unpaid_total(c,today) for c in chosen)
    return money_answer(v), f"contract prices less down payments and contributions through {today}; {len(chosen)} contract(s)"


def credit_calc(s, today):
    if "tuoi no" in s or re.search(r"0\s*7", s):
        buckets=[0,0,0]
        for d in DEBTS:
            if d["date"]>today: continue
            age=(today-d["date"]).days
            remain=max(0,d["initial"]-sum(v for pdte,v in d["payments"] if pdte<=today))
            if remain>0:
                idx=0 if age<=7 else 1 if age<=15 else 2; buckets[idx]+=remain
        for c in INSTALLMENTS:
            if c["date"]>today: continue
            age=(today-c["date"]).days
            remain=max(0,c["price"]-c["down"]-sum(v for pdte,v in c["payments"] if pdte<=today))
            if remain>0:
                idx=0 if age<=7 else 1 if age<=15 else 2; buckets[idx]+=remain
        return list_answer([rnd(x) for x in buckets]), f"general-credit and installment balances grouped by age at {today}: 0-7, 8-15, >15 days"
    if "bao nhieu khach" in s or "may khach" in s or "bnhiu khach" in s:
        people={d["phone"] or norm(d["customer"]) for d in DEBTS if d["date"]<=today and max(0,d["initial"]-sum(v for pdte,v in d["payments"] if pdte<=today))>0}
        people|={c["phone"] or norm(c["customer"]) for c in INSTALLMENTS if c["date"]<=today and max(0,c["price"]-c["down"]-sum(v for pdte,v in c["payments"] if pdte<=today))>0}
        return count_answer(len(people)), f"distinct customers with positive unpaid general-credit or installment balances as of {today}"
    v=sum(max(0,d["initial"]-sum(a for pdte,a in d["payments"] if pdte<=today)) for d in DEBTS if d["date"]<=today)
    v+=sum(max(0,c["price"]-c["down"]-sum(a for pdte,a in c["payments"] if pdte<=today)) for c in INSTALLMENTS if c["date"]<=today)
    return money_answer(v), f"general-credit Sổ nợ plus installment balances remaining after payments through {today}"


def warranty_calc(q, s, shop, w, today):
    war=[r for r in WARRANTY if r["date"] and in_window(r["date"],w) and r["date"]<=today]
    if "chiem" in s or "phan tram" in s:
        qty=phone_qty(shop,w)
        return percent_answer(rnd(len(war)/qty*100,2) if qty else 0), f"warranty visits {len(war)}/phone units sold {qty}"
    if any(k in s for k in ("dòng máy nào", "dong may nao", "nhieu luot nhat")):
        counts=Counter(); labels=defaultdict(Counter)
        for r in war:
            label=phone_model_line(r["model"]); key=norm(label)
            if key: counts[key]+=1; labels[key][label]+=1
        key=sorted(counts,key=lambda n:(-counts[n],n))[0] if counts else ""
        name=sorted(labels[key],key=lambda n:(-labels[key][n],norm(n),n))[0] if key else ""
        return string_answer(name), f"warranty records by normalized model line; winner {counts.get(key,0)}"
    if "may ban" in s or "may da duoc mang" in s or "ban thang" in s:
        sold=[]
        for r in valid_lines(shop,w): sold += re.findall(r"\d{15}",r["imei"])
        warr={r["imei"] for r in WARRANTY if r["imei"] and r["date"] and r["date"]<=today}
        return count_answer(len(set(sold)&warr)), f"sold unique IMEIs {len(set(sold))}, matched warranty IMEIs {len(set(sold)&warr)} as of {today}"
    return count_answer(len(war)), f"warranty visit rows in period: {len(war)}"


def score_question(q):
    qid, shop, raw = q["id"], q["business_id"], q["question"]
    s = norm(raw); today = as_date(q); w = parse_window(q)
    method = ""
    ans = None

    if shop == "dev_012":
        rows = [r for r in REPAIRS if in_window(r["date"], w)]
        is_com = any(k in s for k in ("hoa hong", "huê hong", "huê hồng", "commission", "lanh bao nhieu tien hue hong"))
        if "tang tien" in s and "hoa hong" in s: is_com = True
        by_tech_money = defaultdict(float); by_tech_tickets = Counter()
        for r in rows:
            by_tech_money[r["tech_name"]] += r["money"]
            by_tech_tickets[r["tech_name"]] += 1
        named_staff = next((STAFF[n]["name"] for n in STAFF if n in s), None)
        if ("trung binh" in s or "binh quan" in s) and "phieu" in s:
            vals=[r["money"] for r in rows if r["money"]>0 and not r["warranty"]]
            ans=money_answer(sum(vals)/len(vals) if vals else 0)
            method=f"average paid, non-warranty repair ticket; {len(vals)} positive tickets"
        elif "trung binh moi thu tu" in s or "moi thu tu" in s:
            wed = [d for d in pd.date_range(w[0], w[1]) if d.weekday() == 2]
            value = sum(1 for r in rows if r["date"].weekday() == 2) / max(1, len(wed))
            v = rnd(value, 2)
            ans = {"type": "number", "answer": int(v) if v == int(v) else v}; method = f"Wednesday repair rows {int(value*len(wed))}/{len(wed)} Wednesdays"
        elif "moi thu ba" in s or "thu 3 thang nay" in s:
            tue = [d for d in pd.date_range(w[0], w[1]) if d.weekday() == 1]
            value = sum(1 for r in rows if r["date"].weekday() == 1) / max(1, len(tue))
            v = rnd(value, 2)
            ans = {"type": "number", "answer": int(v) if v == int(v) else v}; method = f"Tuesday repair rows / {len(tue)} Tuesdays"
        elif "dong xe" in s and ("nhieu tien" in s or "nhieu nhat" in s):
            by=defaultdict(float); labels=defaultdict(Counter)
            for r in rows:
                by[r["model_norm"]]+=r["money"]
                labels[r["model_norm"]][r["model"]]+=1
            key=sorted(by,key=lambda n:(-by[n],n))[0] if by else ""
            display={"wave alpha":"Wave Alpha","air blade":"Air Blade","sh mode":"SH Mode","wave rsx":"Wave RSX","lead":"Lead","vision":"Vision","sirius":"Sirius","exciter":"Exciter","future":"Future","vario":"Vario","janus":"Janus","grande":"Grande","winner":"Winner"}.get(key)
            name=display or (sorted(labels[key],key=lambda n:(-labels[key][n],norm(n),n))[0] if key else "")
            ans=string_answer(name); method=f"max repair revenue by normalized model {rnd(by.get(key,0))}; aliases merged under {key}"
        elif is_com and ("bang" in s or "xep" in s):
            names = [x["name"] for x in STAFF.values()]
            data = [(n, rnd(by_tech_money[n] * STAFF[norm(n)]["rate"] / 100)) for n in names]
            data.sort(key=lambda kv: (-kv[1], kv[0]))
            ans = {"type": "table", "answer": {"columns": ["thợ", "tiền hoa hồng"], "rows": [[n, v] for n,v in data]}}
            method = f"commission = technician monthly ticket total × listed rate; {len(rows)} repairs"
        elif is_com:
            names = [x["name"] for x in STAFF.values()]
            comm = {n: by_tech_money[n] * STAFF[norm(n)]["rate"] / 100 for n in names}
            name = sorted(names, key=lambda n: (-comm[n], n))[0]
            # Named employee asks take amount; otherwise asks highest mechanic.
            named = next((n for n in names if norm(n) in s), None)
            if named:
                ans = money_answer(comm[named]); method = f"commission {named}: eligible repair revenue {rnd(by_tech_money[named])} × {STAFF[norm(named)]['rate']}%"
            elif any(k in s for k in ("bao nhieu tien", "tong bao nhieu tien")) and any(n in s for n in ["dam lan an", "ha loan dat", "doan bao phuc", "luong chi duc"]):
                named = next(n for n in names if norm(n) in s); ans = money_answer(comm[named]); method = "named technician commission"
            else:
                ans = string_answer(name); method = f"highest commission {rnd(comm[name])}; {len(rows)} repairs"
        elif named_staff and any(k in s for k in ("phieu", "luot xe", "luot", "sua duoc", "lam duoc")) and not any(k in s for k in ("bao nhieu tien", "bnhiu tien", "tien hang", "dthu", "doanh thu", "hoa hong")):
            ans=count_answer(by_tech_tickets[named_staff]); method=f"repair visits assigned to {named_staff}; {len(rows)} monthly rows"
        elif "tho" in s and not named_staff and any(k in s for k in ("nhieu tien", "dthu cao nhat", "doanh thu cao nhat", "lam ra nhieu tien")):
            name=sorted(by_tech_money,key=lambda n:(-by_tech_money[n],n))[0] if by_tech_money else ""
            ans=string_answer(name); method=f"highest monthly repair-ticket revenue: {rnd(by_tech_money.get(name,0))}"
        elif "nhiet tinh" in s and any(k in s for k in ("nhat", "nhut", "ai la")):
            ans=unanswerable("Dữ liệu chỉ ghi phiếu sửa, doanh thu và thợ thực hiện; không ghi nhận tiêu chí đánh giá chủ quan thợ nhiệt tình.")
            method="employee enthusiasm is subjective and absent from the released records"
        elif "cham chi" in s and any(k in s for k in ("nhat", "nhut", "ai la")):
            workdays=defaultdict(set)
            for r in rows: workdays[r["tech_name"]].add(r["date"])
            counts={name:len(days) for name,days in workdays.items()}
            name=sorted(counts,key=lambda n:(-counts[n],n))[0] if counts else ""
            ans=string_answer(name); method=f"most technician workdays: {counts.get(name,0)}"
        elif named_staff and any(k in s for k in ("bao nhieu luot", "luot xe", "sua duoc", "nhan nhieu phieu")):
            ans=count_answer(by_tech_tickets[named_staff]); method=f"repair visits assigned to {named_staff}; {len(rows)} monthly rows"
        elif ("thang" in s or "thien tinh" in s) and "nhat" in s and any(k in s for k in ("ai", "tho")):
            workdays=defaultdict(set)
            for r in rows: workdays[r["tech_name"]].add(r["date"])
            counts={name:len(days) for name,days in workdays.items()}
            name=sorted(counts,key=lambda n:(-counts[n],n))[0] if counts else ""
            ans=string_answer(name); method=f"most technician workdays: {counts.get(name,0)}"
        elif any(k in s for k in ("cao hay thap", "tang hay giam", "tang giam", "so voi thang", "tang hay giam bao nhieu")) or ("tang" in s and ("bao nhieu" in s or "bnhiu" in s) and "thang" in s):
            pcur=w; d1=pcur[0]; prev_end=(pd.Timestamp(d1)-pd.Timedelta(days=1)).date()
            if len(pd.date_range(pcur[0],pcur[1])) < 27:
                prev_start=date(prev_end.year,prev_end.month,1)
                prev_end=date(prev_end.year,prev_end.month,min(pcur[1].day,calendar.monthrange(prev_end.year,prev_end.month)[1]))
            else: prev_start,prev_end=month_range(prev_end.year,prev_end.month)
            base=sum(r["money"] for r in REPAIRS if prev_start<=r["date"]<=prev_end)
            cur=sum(r["money"] for r in rows)
            ans=percent_answer(pct(cur,base)); method=f"repair revenue change {rnd(cur)} vs {rnd(base)}"
        elif "doanh thu" in s or "dthu" in s or "tien sua" in s or "tien hang" in s or "lam ra" in s or "ban tong cong" in s or "bao nhieu tien" in s or "tong tien" in s:
            # Named mechanic amount = summed repair ticket money, not share.
            named = next((n for n in STAFF if n in s), None)
            if named:
                v = by_tech_money[STAFF[named]["name"]]; ans = money_answer(v); method = f"sum ticket amounts assigned to {STAFF[named]['name']} ({len(rows)} monthly rows)"
            elif any(k in s for k in ("tang hay giam", "tang giam", "so voi", "%")):
                today = as_date(q)
                # Current period / prior comparable period.
                pcur = w
                d1 = pcur[0]; prev_end = (pd.Timestamp(d1) - pd.Timedelta(days=1)).date()
                if len(pd.date_range(pcur[0],pcur[1])) < 27:
                    prev_start = date(prev_end.year, prev_end.month, 1)
                    prev_end = date(prev_end.year, prev_end.month, min(pcur[1].day, calendar.monthrange(prev_end.year,prev_end.month)[1]))
                else: prev_start, prev_end = month_range(prev_end.year, prev_end.month)
                base = sum(r["money"] for r in REPAIRS if prev_start <= r["date"] <= prev_end)
                cur = sum(r["money"] for r in rows)
                ans = percent_answer(pct(cur, base)); method = f"repair revenue change {rnd(cur)} vs {rnd(base)}"
            else:
                ans = money_answer(sum(r["money"] for r in rows)); method = f"sum repair-ticket amounts; {len(rows)} source rows"
        elif "ngay nao" in s and ("nhieu tien" in s or "doanh thu" in s or "ban ra" in s):
            by = defaultdict(float)
            for r in rows: by[r["date"]] += r["money"]
            if by:
                d, v = sorted(by.items(), key=lambda kv:(-kv[1],kv[0]))[0]
                ans = date_answer(d); method = f"max daily repair-ticket amount {rnd(v)}"
            else: ans = count_answer(0); method = "no repair rows in window"
        elif "luot xe" in s or "xe vao" in s or "xe vo" in s or "vao sua" in s or "nhieu luot" in s or "nhieu phieu" in s or "phieu sua" in s or "bao hanh" in s or "phieu thu tien" in s or "phieu co thu tien" in s or "sh mode" in s or "air blade" in s or "wave alpha" in s:
            if "bao hanh" in s and ("chiem" in s or "phan tram" in s):
                warranty = sum(r["warranty"] for r in rows); paid = sum(r["money"] > 0 for r in rows)
                ans = percent_answer(rnd(warranty/paid*100,2) if paid else 0); method = f"warranty visits {warranty}/paid tickets {paid}"
            elif "bao hanh" in s:
                v = sum(r["warranty"] for r in rows); ans = count_answer(v); method = f"zero-charge warranty return rows; source rows {len(rows)}"
            elif "phieu thu tien" in s:
                vals=[r["money"] for r in rows if r["money"]>0]
                ans=money_answer(sum(vals)/len(vals) if vals else 0); method=f"average among {len(vals)} positive repair receipts"
            elif "trung binh" in s:
                vals=[r["money"] for r in rows if r["money"]>0]
                ans=money_answer(sum(vals)/len(vals) if vals else 0); method=f"average paid repair ticket; {len(vals)} paid rows"
            elif "sh mode" in s or "air blade" in s or "wave alpha" in s:
                target = "sh mode" if "sh mode" in s else "air blade" if "air blade" in s else "wave alpha"
                matched=[r for r in rows if target in r["model_norm"]]
                if "khac nhau" in s or "moi xe tinh mot lan" in s or "moi xe" in s or "khac" in s or "bao nhieu chiec" in s and "khac" in s:
                    ans=count_answer(len({r["plate"] for r in matched})); method=f"distinct license plates matching {target}; {len(matched)} visits"
                else: ans=count_answer(len(matched)); method=f"repair visits for {target}"
            elif "xe khach quen" in s or "khach quen" in s:
                paid_rows=[r for r in rows if r["money"]>0]
                by=Counter(r["plate"] for r in paid_rows); ans=count_answer(sum(v>=2 for v in by.values())); method=f"vehicles with >=2 paid repair visits; {len(by)} distinct plates"
            elif "tu 2 lan" in s:
                paid_rows=[r for r in rows if r["money"]>0]
                by=Counter(r["plate"] for r in paid_rows); ans=count_answer(sum(v>=2 for v in by.values())); method=f"vehicles with >=2 paid repair visits; {len(by)} distinct plates"
            elif "tho" in s and any(k in s for k in ("nhieu", "nhieu nhat", "nhieu don")):
                counts=Counter(r["tech_name"] for r in rows); name=sorted(counts,key=lambda n:(-counts[n],n))[0]
                ans=string_answer(name); method=f"highest repair visit count {counts[name]}"
            elif "tho" in s and ("bao nhieu luot xe" in s or "sua duoc" in s):
                named=next((n for n in STAFF if n in s),None)
                if named: v=by_tech_tickets[STAFF[named]["name"]]; ans=count_answer(v); method=f"repair visits for {STAFF[named]['name']}"
                else: ans=count_answer(len(rows)); method="repair ticket count"
            elif "thay dau lap" in s or "thay nhot lap" in s or "dau lap" in s or "nhot lap" in s:
                v=sum((any(k in r["work_norm"] for k in ("nhot lap", "dau lap", "nhot hop so", "dau hop so")) or "lap" in r["work_norm"].split()) and not r["warranty"] for r in rows); ans=count_answer(v); method=f"paid repair rows mentioning gear oil: {v}"
            elif "va ruot" in s:
                v=sum((any(k in r["work_norm"] for k in ("va ruot", "va xe", "va")) or "va" in r["work_norm"].split()) and not r["warranty"] for r in rows); ans=count_answer(v); method=f"paid repair rows mentioning inner tube patch: {v}"
            elif "dien" in s:
                v=sum(("dien" in r["work_norm"] or "sua dien" in r["work_norm"]) and not r["warranty"] for r in rows); ans=count_answer(v); method=f"paid repair rows mentioning electrical work: {v}"
            else:
                ans=count_answer(len(rows)); method=f"repair ticket rows in window: {len(rows)}"
        else:
            ans={"type":"abstain"}; method="classifier unresolved; abstain"

    elif shop in ("dev_011", "dev_014"):
        lines, orders = sale_data(shop)
        valid = valid_orders(shop, w)
        valid_detail = valid_lines(shop, w)
        if shop == "dev_014" and any(k in s for k in ("tra gop", "tien gop", "ky gop", "ky qua han")):
            ans, method = installment_calc(s,today)
        elif shop == "dev_014" and "bao hanh" in s:
            ans, method = warranty_calc(q,s,shop,w,today)
        elif shop == "dev_014" and any(k in s for k in ("con no", "cong no", "so no", "tien no", "tuoi no", "no shop")):
            # The non-installment credit sheet is the only general debt ledger.
            ans, method = credit_calc(s,today)
        elif shop == "dev_014" and ("loi nhieu nhat" in s or "loi nhat" in s):
            ans=unanswerable("Sổ bán hàng không có giá vốn trên dòng bán hoặc bảng giá vốn nên không thể tính lãi gộp.")
            method="product cost is absent from the released sales and lookup sheets"
        elif shop == "dev_014" and "duong nguyen van linh" in s:
            ans=unanswerable("Sổ phát hành chỉ có tên và số điện thoại khách; không có địa chỉ khách hàng.")
            method="customer address is absent from the released sheets"
        elif shop == "dev_014" and ("thu may cu" in s or "may cu cua khach" in s) and ("chiem" in s or "phan tram" in s or "%" in raw):
            orders_with_trade=sum(any(r["date"].year==o["date"].year and r["date"].month==o["date"].month and r["stt"]==o["stt"] for r in tradein_window(w)) for o in valid)
            ans=percent_answer(rnd(orders_with_trade/len(valid)*100,2) if valid else 0)
            method=f"valid orders linked to a trade-in {orders_with_trade}/{len(valid)}"
        elif ("tien thu may cu" in s or "thu mua may cu" in s or "may cu tru vao don" in s) and not any(k in s for k in ("khach phai tra", "khach can tra", "chiem", "phan tram")):
            if shop=="dev_014":
                tis=tradein_window(w); v=sum(r["amount"] for r in tis); method=f"sum trade-in ledger amounts for {len(tis)} dated receipts"
            else:
                v=sum(o["tradein"] for o in valid); method=f"sum legacy handset trade-in entries on {len(valid)} valid orders"
            ans=money_answer(v)
        elif shop == "dev_014" and ("thu may cu" in s or "may cu cua khach" in s or "may cu tru vao don" in s) and not any(k in s for k in ("khach phai tra", "khach can tra")):
            tis=tradein_window(w)
            ans=money_answer(sum(r["amount"] for r in tis)); method=f"sum trade-in ledger amounts for {len(tis)} dated receipts"
        elif "tang gia" in s or "thay doi bao nhieu %" in s and "so luong" in s:
            # Product's price-change question: compare N-day after window incl change date to N prior dates.
            qdate=date(2025,12,1); target=norm("Cáp Type-C sang Lightning 1m")
            n=14; after=(qdate,qdate+timedelta(days=n-1)); before=(qdate-timedelta(days=n),qdate-timedelta(days=1))
            def qty_range(rng): return sum(r["qty"] for r in valid_lines(shop,rng) if r["item_norm"]==target)
            a,b=qty_range(after),qty_range(before); ans=percent_answer(pct(a,b)); method=f"14-day quantity after {a} vs preceding 14 days {b}"
        elif ("trung binh" in s and re.search(r"\bdon\b",s)) or "binh quan mot don" in s:
            if "doanh thu" in s:
                v=sum(o["revenue"] for o in valid)/len(valid) if valid else 0
                ans=money_answer(v); method=f"valid sales revenue {rnd(sum(o['revenue'] for o in valid))}/{len(valid)} orders"
            else:
                v=total_payable(valid)/len(valid) if valid else 0
                ans=money_answer(v); method=f"payable after discount and trade-in {rnd(total_payable(valid))}/{len(valid)} orders"
        elif "khach phai tra" in s or "khach tra tong cong" in s or "khach phai tra" in s or "khach can tra" in s:
            ans=money_answer(total_payable(valid)); method=f"valid orders; line totals less one order discount and trade-in; {len(valid)} orders"
        elif "top 3" in s or "3 dong may nao" in s:
            items=defaultdict(float); variants=defaultdict(Counter)
            for r in valid_lines(shop,w):
                if r["kind"] in ("may moi","may cu","may") or r["imei"]:
                    label=phone_model_line(r["item"]); key=norm(label)
                    items[key]+=r["qty"]; variants[key][label]+=r["qty"]
            top=sorted(items,key=lambda key:(-items[key],key))[:3]
            labels=[sorted(variants[key],key=lambda label:(-variants[key][label],norm(label),label))[0] for key in top]
            ans=list_answer(labels)
            method=f"phone line quantities aggregated before storage capacity; {len(valid_detail)} sale lines"
        elif "ban chay nhat" in s or "ban duoc nhieu may nhat" in s or "mat hang nao" in s and "ban chay" in s:
            items=group_lines_by_item(shop,w)
            if "nhom may moi" in s: items=group_lines_by_item(shop,w,lambda r:r["kind"]=="may moi")
            name=sorted(items,key=lambda n:(-items[n],norm(n)))[0] if items else ""
            ans=string_answer(name); method=f"largest item quantity {items.get(name,0)}"
        elif "loi nhieu nhat" in s or "loi nhat" in s:
            ans=unanswerable("Giá vốn của hàng bán không được ghi trong sổ của shop nên không thể xếp hạng lãi gộp.")
            method="no unit cost data is released for the retailer"
        elif any(k in s for k in ("bao nhieu tien", "thu ve bao nhieu tien", "ban duoc bao nhieu tien", "ban ra bao nhieu tien", "bnhiu tien")) and not ("trung binh" in s and "may" in s) and not any(k in s for k in ("nhom may moi", "nhom may cu", "nhom phu kien")):
            candidates=[]
            for r in valid_detail:
                itemn=norm(r["item"])
                if itemn and itemn in s: candidates.append(r["item"])
            if not candidates:
                for r in lines:
                    itemn=norm(r["item"])
                    if itemn and itemn in s: candidates.append(r["item"])
            if not candidates:
                # Match a distinctive contiguous phrase while excluding generic words.
                qtokens=[t for t in s.split() if t not in {"thang","nam","nay","nhi","nhe","rua","duoc","ban","bao","nhieu","tien","thu","ve","hang","may","cua","shop","tiem","hom","qua","doanh","tong","cong"}]
                scores=Counter()
                for r in lines:
                    words=set(norm(r["item"]).split()); score=sum(t in words for t in qtokens)
                    if score>=2: scores[r["item"]]=score
                candidates=[n for n,sc in scores.items() if sc==max(scores.values(),default=0)]
            if candidates:
                names=set(candidates)
                v=sum(r["amount"] for r in valid_detail if r["item"] in names)
                ans=money_answer(v); method=f"line revenue for requested item(s) {sorted(names)}; {len(valid_detail)} valid rows"
            elif "nhom may cu" in s:
                v=category_revenue(shop,w,"máy cũ"); ans=money_answer(v); method="used phone line revenue"
            elif any(k in s for k in ("tong cong", "doanh thu", "dthu", "bua ni", "hom qua")):
                v=revenue(shop,w); ans=money_answer(v); method=f"general valid-order revenue; {len(valid)} orders"
            else:
                ans={"type":"abstain"}; method="unmatched item wording; abstain"
        elif any(k in s for k in ("ngay nao", "bua nao")) and any(k in s for k in ("nhieu tien", "nhieu nhat", "doanh thu")):
            pair=day_revenue(shop,w)
            if pair: ans=date_answer(pair[0]); method=f"max daily revenue {rnd(pair[1])}"
            else: ans={"type":"abstain"}; method="no sale dates in requested window"
        elif "nhom phu kien" in s and ("tien" in s or "dthu" in s):
            ans=money_answer(category_revenue(shop,w,"phụ kiện")); method="accessory line revenue"
        elif "nhom may cu" in s and ("tien" in s or "doanh thu" in s):
            ans=money_answer(category_revenue(shop,w,"máy cũ")); method="used phone line revenue"
        elif shop=="dev_014" and "phu kien" in s and "bo" in s and "may moi" in s:
            a=category_revenue(shop,w,"máy mới"); b=phone_revenue(shop,w)
            ans=percent_answer(rnd(a/b*100,2) if b else 0); method=f"new phone revenue {rnd(a)}/all phone revenue {rnd(b)}"
        elif "may moi" in s and "chiem" in s and ("phan tram" in s or "tien ban may" in s or "%" in raw):
            a=category_revenue(shop,w,"máy mới"); b=phone_revenue(shop,w)
            ans=percent_answer(rnd(a/b*100,2) if b else 0); method=f"new phone revenue {rnd(a)}/all phone revenue {rnd(b)}"
        elif shop=="dev_014" and "samsung" in s and "chiem" in s:
            allq=phone_qty(shop,w); samsung=sum(r["qty"] for r in phone_lines(shop,w) if "samsung" in r["item_norm"])
            ans=percent_answer(rnd(samsung/allq*100,2) if allq else 0); method=f"Samsung phone units {samsung}/all phone units {allq}"
        elif shop=="dev_014" and "don co thu lai may cu" in s and "chiem" in s:
            tradeorders=sum(any(r["date"].year==o["date"].year and r["date"].month==o["date"].month and r["stt"]==o["stt"] for r in tradein_window(w)) for o in valid)
            ans=percent_answer(rnd(tradeorders/len(valid)*100,2) if valid else 0); method=f"valid trade-in orders {tradeorders}/{len(valid)}"
        elif any(k in s for k in ("bao nhieu chiec", "bao nhieu may", "bnhiu chiec", "bnhiu may", "ban dc may", "ban duoc may", "may chiec", "may cai", "bnhiu cai", "bao nhieu cai")) and any(k in s for k in ("iphone", "samsung", "xiaomi", "redmi", "realme", "oppo", "the nho", "op lung")):
            # Match specific model/product names, allowing storage/color suffixes.
            exact={r["item"] for r in lines if r["item_norm"] and r["item_norm"] in s}
            candidates=list(exact)
            if not exact and any(brand in s for brand in ("iphone","samsung","xiaomi","redmi","realme","oppo","the nho")):
                qfrag=next((frag for frag in re.findall(r"(?:iphone|samsung galaxy|xiaomi redmi note|xiaomi redmi|xiaomi|realme|oppo|redmi)\s+[a-z0-9 ]+",s) if len(frag.split())>=2),"")
                qfrag=re.sub(r"\b(thang|th|nay|ni|ban|duoc|bao|nhieu|chiec|may|bnhiu|dc|roi|the|da|den|5g)\b.*$","",qfrag).strip()
                for r in lines:
                    ni=r["item_norm"]
                    if qfrag and qfrag in ni:
                        suffix=ni[len(qfrag):].strip()
                        if not any(x in suffix.split() for x in ("pro","max","plus","mini","ultra")) or any(x in qfrag.split() for x in ("pro","max","plus","mini","ultra")):
                            candidates.append(r["item"])
            if not exact and "op lung" in s:
                candidates += [r["item"] for r in lines if "op lung silicon iphone" in r["item_norm"] and "op lung silicon iphone" in s]
            if candidates:
                names=set(candidates); v=sum(r["qty"] for r in valid_detail if r["item"] in names)
                ans=count_answer(v); method=f"requested model units in {sorted(names)}; {len(valid_detail)} valid rows"
            else:
                ans={"type":"abstain"}; method="unmatched item wording; abstain"
        elif any(k in s for k in ("bao nhieu chiec", "bao nhieu dien thoai", "bao nhieu may", "bnhiu chiec", "bnhiu may", "may chiec", "may may", "bao nhieu may ban", "ban dc may", "ban duot may", "khong ke phu kien")) and any(k in s for k in ("dien thoai", "dieu thoai", "may moi", "may cu", "phu kien")):
            if "khong ke phu kien" in s or "khong tinh phu kien" in s or "ca may moi" in s or "tat ca" in s:
                v=phone_qty(shop,w); method=f"new+used phone units excluding accessories: {v}"
            elif "may cu" in s or "dien thoai cu" in s:
                v=phone_qty(shop,w,"máy cũ"); method=f"used phone units: {v}"
            elif "may moi" in s:
                v=phone_qty(shop,w,"máy mới"); method=f"new phone units: {v}"
            else:
                v=phone_qty(shop,w); method=f"phone units excluding accessories: {v}"
            ans=count_answer(v)
        elif "nhom may moi" in s and ("tien" in s or "doanh thu" in s):
            ans=money_answer(category_revenue(shop,w,"máy mới")); method=f"new phone line revenue; {len(valid_detail)} valid detail lines"
        elif "nhom phu kien" in s and ("tien" in s or "dthu" in s):
            ans=money_answer(category_revenue(shop,w,"phụ kiện")); method="accessory line revenue"
        elif "may moi" in s and "chiem" in s and "%" in s or "may moi" in s and "phan tram" in s:
            a=category_revenue(shop,w,"máy mới"); b=phone_revenue(shop,w)
            ans=percent_answer(pct(a,b) if False else (rnd(a/b*100,2) if b else 0)); method=f"new phone revenue {rnd(a)}/all phone revenue {rnd(b)}"
        elif "phu kien" in s and "bo" in s and "may moi" in s:
            a=category_revenue(shop,w,"máy mới"); b=phone_revenue(shop,w)
            ans=percent_answer(rnd(a/b*100,2) if b else 0); method=f"new phone revenue {rnd(a)}/non-accessory phone revenue {rnd(b)}"
        elif "samsung" in s and "chiem" in s and "%" in s:
            allq=phone_qty(shop,w); samsung=sum(r["qty"] for r in phone_lines(shop,w) if "samsung" in r["item_norm"])
            ans=percent_answer(rnd(samsung/allq*100,2) if allq else 0); method=f"Samsung phone units {samsung}/all phone units {allq}"
        elif "trung binh" in s and "may dt" in s or "trung binh" in s and "may" in s and "gia bao nhieu" in s:
            qty=phone_qty(shop,w); amt=phone_revenue(shop,w)
            ans=money_answer(amt/qty if qty else 0); method=f"phone revenue {rnd(amt)}/phone units {qty}"
        elif "trung binh" in s and "moi may" in s:
            qty=phone_qty(shop,w); amt=phone_revenue(shop,w)
            ans=money_answer(amt/qty if qty else 0); method=f"phone revenue {rnd(amt)}/phone units {qty}"
        elif "trung binh" in s and any(k in s for k in ("may dien thoai", "may dt", "moi may")):
            qty=phone_qty(shop,w); amt=phone_revenue(shop,w)
            ans=money_answer(amt/qty if qty else 0); method=f"phone revenue {rnd(amt)}/phone units {qty}"
        elif "doanh thu" in s or "dthu" in s or "tien ban" in s or "ban duoc tong cong bao nhieu tien" in s or "ban tong cong" in s or "ban duoc bao nhieu tien" in s or "ban duoc tong cong" in s:
            if any(k in s for k in ("tang hay giam", "cao hay thap", "tang hay giam bao nhieu", "tang giam", "thay doi", "so voi thang", "so voi cung ky")):
                # Explicit previous full month or matched month-to-date.
                if "tet" in s and "17 ngay" in s:
                    cur=w; prev=(w[0]-timedelta(days=17),w[0]-timedelta(days=1))
                elif "tu dau thang" in s or "cung ky thang truoc" in s:
                    cur=w; prev=month_overlap_window(today,previous=True,mtd=True)
                else:
                    cur=w; prevend=(pd.Timestamp(w[0])-pd.Timedelta(days=1)).date(); prev=month_range(prevend.year,prevend.month)
                a=revenue(shop,cur); b=revenue(shop,prev); ans=percent_answer(pct(a,b)); method=f"revenue growth {rnd(a)} vs prior comparable period {rnd(b)}"
            elif "thu tu" in s or "lan luot" in s and "th7" in s and "th8" in s and "th9" in s:
                values=[]
                for m in (7,8,9): values.append(rnd(revenue(shop,month_range(2026,m))))
                ans=list_answer(values); method="July, August, and September 2026 revenues in order"
            else:
                v=revenue(shop,w); ans=money_answer(v); method=f"valid-order line revenue; {len(valid)} orders, {len(valid_detail)} detail rows"
        elif (re.search(r"\bdon\b",s) or "dh" in s) and (any(k in s for k in ("bao nhieu", "bnhiu", "may don", "so dh", "don hang")) or "so dh" in s):
            # Order count, or count above specified goods threshold.
            rawf=unicodedata.normalize("NFD",raw.lower()).replace("đ","d")
            rawf="".join(c for c in rawf if unicodedata.category(c)!="Mn")
            m=re.search(r"(?:dat|tu)\s*(\d[\d.,]*)\s*(?:dong|d)",rawf)
            if m:
                limit=num(m.group(1)); v=sum(o["revenue"]>=limit for o in valid)
                ans=count_answer(v); method=f"valid orders with line total >= {rnd(limit)}"
            elif shop=="dev_014" and "may cu" in s and ("phan tram" in s or "chiem" in s or "%" in raw):
                tradeorders=sum(o["key"].split(":")[-1] and any(r["date"].year==o["date"].year and r["date"].month==o["date"].month and r["stt"]==o["stt"] for r in tradein_window(w)) for o in valid)
                ans=percent_answer(rnd(tradeorders/len(valid)*100,2) if valid else 0); method=f"valid trade-in orders {tradeorders}/{len(valid)}"
            else:
                ans=count_answer(len(valid)); method=f"distinct valid orders; {len(valid)} orders"
        elif shop=="dev_014" and "bao hanh" in s:
            # Warranty visit count, device distinct count, or model rank / ratio.
            war=[r for r in WARRANTY if in_window(r["date"],w) and (not r["date"] or r["date"]<=today)]
            if "chiem" in s and "%" in s:
                q=phone_qty(shop,w)
                ans=percent_answer(rnd(len(war)/q*100,2) if q else 0); method=f"warranty visits {len(war)}/phone units sold {q}"
            elif "dòng máy nào" in raw.lower() or "dong may nao" in s or "nhieu luot nhat" in s:
                counts=Counter(r["model"] for r in war); name=sorted(counts,key=lambda n:(-counts[n],norm(n)))[0] if counts else ""
                ans=string_answer(name); method=f"warranty records by model; winner {counts.get(name,0)}"
            elif "may ban thang" in s or "may ban ra thang" in s or "bao nhieu may da duoc mang" in s:
                # Link each IMEI sold in the stated period to warranty entries by the as-of date.
                sold=[]
                for r in valid_lines(shop,w):
                    sold += re.findall(r"\d{15}",r["imei"])
                warr={r["imei"] for r in WARRANTY if r["imei"] and r["date"] and r["date"]<=today}
                ans=count_answer(len(set(sold)&warr)); method=f"sold unique IMEIs {len(set(sold))}, matched warranty IMEIs {len(set(sold)&warr)} as of {today}"
            else:
                ans=count_answer(len(war)); method=f"warranty visit rows in period: {len(war)}"
        elif shop=="dev_014" and ("gop" in s or "ky han" in s or "ky qua han" in s or "tra gop" in s):
            asof=today
            def due_dates(c):
                dates=[]; y,m=c["date"].year,c["date"].month
                for i in range(1,c["n"]+1):
                    idx=y*12+(m-1)+i; yy=idx//12; mm=idx%12+1
                    dates.append(date(yy,mm,min(c["due_day"],calendar.monthrange(yy,mm)[1])))
                return dates
            def paid_until(c, cutoff): return sum(v for d,v in c["payments"] if d<=cutoff)
            def unpaid_total(c, cutoff): return max(0,c["price"]-c["down"]-paid_until(c,cutoff))
            def overdue_amount(c, cutoff):
                ds=due_dates(c); required=min(c["price"]-c["down"], sum(c["per"] for d in ds if d<cutoff))
                return max(0,required-paid_until(c,cutoff))
            asks_overdue_people = any(k in s for k in ("bao nhieu khach", "bnhiu khach", "may khach", "bao nhieu nguoi", "bnhiu nguoi", "may nguoi", "mi nguoi", "mi ngui"))
            if asks_overdue_people and ("qua han" in s or "con ky qua han" in s):
                people={c["phone"] or norm(c["customer"]) for c in INSTALLMENTS if overdue_amount(c,asof)>0}
                ans=count_answer(len(people)); method=f"distinct installment customers with scheduled unpaid amount past due as of {asof}: {len(people)}"
            elif "con chua dong bao nhieu" in s or "tien cham gop" in s or "qua han khach chua tra" in s or "da qua han" in s or "chua dong bao nhieu tien" in s:
                v=sum(overdue_amount(c,asof) for c in INSTALLMENTS)
                ans=money_answer(v); method=f"sum scheduled installment amounts due by {asof} less contributions received by {asof}"
            elif "may ky" in s and ("tien toi han" in s or "toi han may ky" in s or "toi han" in s):
                named=next((c for c in INSTALLMENTS if norm(c["customer"]) in s),None)
                ds=due_dates(named) if named else []
                ans=count_answer(sum(d<=asof for d in ds)); method=f"scheduled due installments for {named['customer'] if named else 'no matched customer'} by {asof}"
            elif "con lai bao nhieu" in s or "chua gop" in s or "con no" in s or "chua tra" in s or "con thieu" in s or "chua dong" in s or "con lai" in s:
                named=[c for c in INSTALLMENTS if norm(c["customer"]) in s] if any(norm(c["customer"]) in s for c in INSTALLMENTS) else None
                chosen=named if named is not None else INSTALLMENTS
                v=sum(unpaid_total(c,asof) for c in chosen)
                ans=money_answer(v); method=f"contract price less down payments and contributions through {asof}; {len(chosen)} contract(s)"
            else:
                # Any not otherwise classified installment question is total outstanding balance.
                v=sum(unpaid_total(c,asof) for c in INSTALLMENTS)
                ans=money_answer(v); method=f"total installment balance through {asof}; {len(INSTALLMENTS)} contracts"
        elif shop=="dev_014" and ("no" in s or "con no" in s):
            if "tuoi no" in s or "0 7" in s:
                buckets=[0,0,0]
                for d in DEBTS:
                    age=(today-d["date"]).days
                    remain=max(0,d["initial"]-sum(v for pdte,v in d["payments"] if pdte<=today))
                    if remain>0:
                        idx=0 if age<=7 else 1 if age<=15 else 2; buckets[idx]+=remain
                ans=list_answer([rnd(x) for x in buckets]); method=f"general-credit balances grouped by age at {today}: 0-7, 8-15, >15 days"
            elif "bao nhieu khach" in s:
                people={d["phone"] or norm(d["customer"]) for d in DEBTS if max(0,d["initial"]-sum(v for pdte,v in d["payments"] if pdte<=today))>0}
                ans=count_answer(len(people)); method=f"distinct general-credit customers with a positive unpaid balance as of {today}"
            else:
                v=sum(max(0,d["initial"]-sum(a for pdte,a in d["payments"] if pdte<=today)) for d in DEBTS)
                ans=money_answer(v); method=f"general-credit balances from Sổ nợ remaining after payments through {today}"
        elif "may" in s and ("ban duoc bao nhieu" in s or "ban ra may" in s or "ban duoc may" in s or "ban dc may" in s):
            # Item/unit quantity, with accessory and new/used filters.
            exact = None
            for r in valid_detail:
                n=norm(r["item"])
                if n and n in s: exact=r["item"]; break
            if exact:
                v=sum(r["qty"] for r in valid_detail if r["item"]==exact); ans=count_answer(v); method=f"sold units of {exact}; valid sales rows"
            elif "may cu" in s and "khong tinh phu kien" not in s:
                v=phone_qty(shop,w,"máy cũ"); ans=count_answer(v); method=f"used phone units from sale rows; valid rows {len(valid_detail)}"
            elif "khong tinh phu kien" in s:
                v=phone_qty(shop,w); ans=count_answer(v); method=f"phone units excluding accessories: {v}"
            elif "dieu thoai" in s or "dien thoai" in s or "chiec may" in s:
                v=phone_qty(shop,w); ans=count_answer(v); method=f"phone unit count excluding accessories: {v}"
            else:
                v=phone_qty(shop,w); ans=count_answer(v); method=f"phone unit count: {v}"
        elif "address" in s or "duong nguyen van linh" in s or "loi nhat" in s:
            ans=unanswerable("Không có địa chỉ khách hàng hoặc giá vốn trong dữ liệu phát hành của cửa hàng.")
            method="required customer address or product cost is not present in released sheets"
        else:
            ans={"type":"abstain"}; method="classifier unresolved; abstain"

    if ans is None:
        ans = {"type":"abstain"}
        method = method or "no classifier matched; abstain"
    meta = {"dev_011": SALES.get("dev_011", (None,None,{}))[2],
            "dev_014": {**SALES.get("dev_014", (None,None,{}))[2], **EXTRA_META},
            "dev_012": REPAIR_META}.get(shop,{})
    EVIDENCE[qid] = {"business_id": shop, "question": raw, "source_counts": meta,
                     "calculation": method, "result": ans}
    return ans


def qa_summary(answers):
    sales_lines_014, sales_orders_014, _ = SALES["dev_014"]
    sale_path=DATA/"data/dev_014/SoBanHang.xlsx"
    footer_checks=[]
    for sheet in book(sale_path):
        if norm(sheet)==norm("Ghi tắt"): continue
        df=tab(sale_path,sheet)
        footer=df[df["Ngày"].astype(str).str.contains("Tổng cộng",case=False,na=False)]
        if footer.empty: continue
        expected=num(footer.iloc[0].get("Thành tiền"))
        observed=sum(r["amount"] for r in sales_lines_014 if r["sheet"]==sheet)
        footer_checks.append({"sheet":sheet,"detail_line_sum":rnd(observed),"footer_total":rnd(expected),"difference":rnd(observed-expected)})
    sale_keys={(o["date"].year,o["date"].month,o["stt"]) for o in sales_orders_014}
    trade_keys={(r["date"].year,r["date"].month,r["stt"]) for r in TRADEINS if r["date"]}
    trade_joined=[o for o in sales_orders_014 if o["tradein"]>0]
    debt_initial=sum(d["initial"] for d in DEBTS)
    debt_paid=sum(a for d in DEBTS for _,a in d["payments"])
    debt_remaining_raw=sum(d["initial"]-sum(a for _,a in d["payments"]) for d in DEBTS)
    installment_financed=sum(c["price"]-c["down"] for c in INSTALLMENTS)
    installment_scheduled=sum(c["n"]*c["per"] for c in INSTALLMENTS)
    installment_paid=sum(a for c in INSTALLMENTS for _,a in c["payments"])
    installment_remaining_raw=sum(c["price"]-c["down"]-sum(a for _,a in c["payments"]) for c in INSTALLMENTS)
    raw_repair_rows=0; kept_repair_rows=0; blank_vehicle_rows=0
    repair_path=DATA/"data/dev_012/SoSuaXe.xlsx"
    for sheet in book(repair_path):
        df=tab(repair_path,sheet); raw_repair_rows+=len(df)
        blank_vehicle_rows+=sum(not text(x) for x in df["Xe"])
        kept_repair_rows+=sum(bool(text(x)) for x in df["Xe"])
    return {
        "answer_coverage":{"assigned_questions":len(QUESTIONS),"answer_ids":len(answers),"all_ids_present":set(answers)=={q["id"] for q in QUESTIONS},
                           "types":dict(Counter(v["type"] for v in answers.values()))},
        "dev_014_sales_footer_reconciliation":{"checks":len(footer_checks),"all_match":all(x["difference"]==0 for x in footer_checks),"months":footer_checks,
                                             "line_rows_after_exact_duplicate_removal":len(sales_lines_014),"orders":len(sales_orders_014)},
        "dev_014_tradein_join":{"ledger_rows":len(TRADEINS),"ledger_amount":rnd(sum(r["amount"] for r in TRADEINS),0),
                               "distinct_link_keys":len(trade_keys),"linked_orders":len(trade_joined),"linked_amount":rnd(sum(o["tradein"] for o in trade_joined)),
                               "unmatched_ledger_rows":sum(1 for r in TRADEINS if r["date"] and (r["date"].year,r["date"].month,r["stt"]) not in sale_keys)},
        "dev_014_credit_reconciliation":{"credit_orders":len(DEBTS),"initial_debt":rnd(debt_initial),"recorded_payments":rnd(debt_paid),
                                         "remaining_before_clamp":rnd(debt_remaining_raw),"negative_orders":sum(d["initial"]-sum(a for _,a in d["payments"])<0 for d in DEBTS)},
        "dev_014_installment_reconciliation":{"contracts":len(INSTALLMENTS),"financed_amount":rnd(installment_financed),"scheduled_amount":rnd(installment_scheduled),
                                              "max_schedule_difference":rnd(max((abs((c["price"]-c["down"])-c["n"]*c["per"]) for c in INSTALLMENTS),default=0)),
                                              "recorded_contributions":rnd(installment_paid),"remaining_before_clamp":rnd(installment_remaining_raw),
                                              "negative_contracts":sum(c["price"]-c["down"]-sum(a for _,a in c["payments"])<0 for c in INSTALLMENTS)},
        "dev_012_repair_rows":{"source_rows_after_header":raw_repair_rows,"rows_with_vehicle":kept_repair_rows,"blank_vehicle_footer_rows_skipped":blank_vehicle_rows,
                               "loaded_rows":len(REPAIRS)},
    }


def main():
    answers = {q["id"]: score_question(q) for q in QUESTIONS}
    EVIDENCE["__qa__"] = qa_summary(answers)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "answers_phone_repair.json"
    ev = OUT / "evidence_phone_repair.json"
    out.write_text(json.dumps(answers, ensure_ascii=False, indent=2) + "\n")
    ev.write_text(json.dumps(EVIDENCE, ensure_ascii=False, indent=2) + "\n")
    counts = Counter(v["type"] for v in answers.values())
    print(json.dumps({"questions": len(answers), "types": counts, "answers": str(out), "evidence": str(ev)}, ensure_ascii=False))
    abstained = [(q["id"], q["question"]) for q in QUESTIONS if answers[q["id"]]["type"] == "abstain"]
    for item in abstained: print("ABSTAIN", json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
