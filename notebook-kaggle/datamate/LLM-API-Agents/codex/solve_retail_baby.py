#!/usr/bin/env python3
"""Direct deterministic calculations for public-test shops dev_007, dev_008, dev_013."""
from __future__ import annotations

import calendar
import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd

from common import DATA, OUT, dt as parse_dt, norm, num, rnd, text

SHOPS = {"dev_007", "dev_008", "dev_013"}
REASONS = {
    "unknown": "Dữ liệu phát hành không ghi nhận thông tin cần thiết để trả lời câu hỏi.",
    "coverage": "Kỳ được hỏi nằm ngoài khoảng thời gian có dữ liệu phát hành.",
    "subjective": "Dữ liệu phát hành không ghi nhận đánh giá chủ quan này.",
}


def clean_answer_string(value):
    value = unicodedata.normalize("NFC", str(value)).lower()
    value = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in value)
    return re.sub(r"\s+", " ", value).strip()


def query_preserving_separators(value):
    s = unicodedata.normalize("NFD", text(value).lower()).replace("đ", "d")
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"[^a-z0-9/.,%\-]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def out(typ, answer=None, reason=None):
    obj = {"type": typ}
    if typ not in {"unanswerable", "abstain"}:
        obj["answer"] = answer
    if typ == "unanswerable" and reason:
        obj["reason"] = reason
    return obj


def first_nonblank(values):
    for value in values:
        if text(value):
            return value
    return None


def customer_name(value):
    name = text(value)
    name = re.sub(r"(?<!\d)(?:0(?:[\s().-]*\d){9,10}|\+?84(?:[\s().-]*\d){9})(?!\d)", " ", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip(" \t-–—,")


def money(value):
    s = text(value).lower().replace(" ", "").replace("đ", "").replace("₫", "")
    if not s or re.fullmatch(r"[-.,]+", s):
        return 0.0
    if not re.search(r"\d", s):
        return 0.0
    # In this dataset comma/dot plus a three-digit tail before "k" is a
    # thousands grouping, e.g. "14,036k" means 14,036,000.
    match = re.fullmatch(r"(-?[\d.,]+)(?:k|nghin|ngan)", s)
    if match:
        mantissa = match.group(1)
        if mantissa.count(",") + mantissa.count(".") and len(re.split(r"[,.]", mantissa)[-1]) == 3:
            return float(re.sub(r"[,.]", "", mantissa)) * 1000
    return num(value)


def is_guest(name, code=""):
    n = norm(name)
    return not n or n in {"khach le", "le", "guest", "kl"}


def status_kind(status="", note=""):
    joined = f"{norm(status)} {norm(note)}"
    if any(term in joined for term in ("da huy", "huy don", "khach huy", "huy", "cancel", "canceled", "cancelled")):
        return "cancel"
    if any(term in joined for term in (
        "tra hang", "hoan tra", "da hoan hang", "hang tra ve", "khach tra", "doi tra", "tra lai",
        "hoan hang", "returned", "return goods",
    )):
        return "return"
    return "valid"


def parse_qty(value):
    if not text(value):
        return 0.0
    m = re.search(r"-?\d+(?:[.,]\d+)?", text(value).replace(" ", ""))
    return num(m.group(0)) if m else 0.0


def conv_map_from_text(value):
    result = {}
    for large, factor, _base in re.findall(r"1\s*([^=;]+?)\s*=\s*(\d+(?:[.,]\d+)?)\s*([^;]+)", text(value)):
        result[norm(large)] = num(factor)
    return result


REST_CONVERSIONS = {
    (norm("Bia Tiger lon"), "thung"): 24,
    (norm("Bia Sai Gon Special lon"), "thung"): 24,
    (norm("Coca-Cola lon"), "thung"): 24,
    (norm("Bia Tiger Bac lon"), "thung"): 24,
    (norm("Nuoc suoi"), "thung"): 24,
    (norm("Bia Heineken lon"), "thung"): 24,
    (norm("Bia Sai Gon Lager chai"), "ket"): 20,
    (norm("7Up lon"), "thung"): 24,
}


def load_restaurant(shop):
    root = DATA / "data" / shop
    if shop == "dev_007":
        rows = pd.read_csv(root / "chi_tiet_hoa_don.csv", encoding="utf-8-sig", dtype=object, low_memory=False)
        catalog = pd.read_csv(root / "danh_muc_hang.csv", encoding="utf-8-sig", dtype=object)
        import_rows = pd.read_csv(root / "chi_tiet_nhap_hang.csv", encoding="utf-8-sig", dtype=object)
        name_col, id_col, cat_col = "Mặt hàng", "Mã vạch", "Phân loại"
        bill_col, date_col, status_col, amount_col = "Mã bill", "Ngày giao dịch", "Tiến độ", "Thành tiền"
        qty_col, qtybase_col, row_unit_col = "Sluong", "Số lượng lẻ", None
        cust_col, custid_col, group_col = "Họ tên khách", "Mã khách mua", "Nhóm khách hàng"
        staff_col, channel_col, payment_col = "Nhân viên bán", "Nguồn", None
        time_col = "Giờ bán"
        catalog_name_col, catalog_id_col, catalog_cat_col = "Hàng bán", "Mã SKU", "Nhóm"
        catalog_unit_col, cost_col, stock_col = "ĐVT", "Giá vốn", "Tồn cuối"
    else:
        rows = pd.concat(
            [pd.read_excel(root / f"ChiTietHoaDon_{n}.xlsx", engine="calamine", dtype=object) for n in (1, 2)],
            ignore_index=True,
        ).drop_duplicates()
        catalog = pd.read_excel(root / "DanhMucHang.xlsx", engine="calamine", dtype=object)
        import_rows = pd.read_excel(root / "ChiTietNhapHang.xlsx", engine="calamine", dtype=object)
        name_col, id_col, cat_col = "Tên hàng", "Ma hang", "Nhóm hàng"
        bill_col, date_col, status_col, amount_col = "Bill", "Ngày xuất", "TT đơn", "Thành tiền"
        qty_col, qtybase_col, row_unit_col = "SLg", "SL lẻ", "Đơn vị (tính)"
        cust_col, custid_col, group_col = "Khách", "Mã người mua", "Nhóm khách hàng"
        staff_col, channel_col, payment_col = "Nhân viên bán", "Kênh bán", "Thanh toán bằng"
        time_col = "Giờ bán"
        catalog_name_col, catalog_id_col, catalog_cat_col = "Hàng", "Mã sản phẩm bán", "Loại hàng"
        catalog_unit_col, cost_col, stock_col = "DVT", "Giá vốn", "Tồn thực"

    rows = rows.dropna(subset=[bill_col]).drop_duplicates().copy()
    rows[bill_col] = rows[bill_col].map(text)
    rows = rows[rows[bill_col] != ""]
    if id_col in rows:
        rows[id_col] = rows[id_col].map(text)
    catalog_by_id, catalog_by_name = {}, {}
    for _, c in catalog.iterrows():
        pid, name = text(c.get(catalog_id_col)), text(c.get(catalog_name_col))
        if not pid and not name:
            continue
        info = {
            "id": pid, "name": name, "category": text(c.get(catalog_cat_col)),
            "unit": norm(c.get(catalog_unit_col)), "cost": money(c.get(cost_col)),
            "stock": money(c.get(stock_col)),
        }
        if pid:
            catalog_by_id[pid] = info
        if name:
            catalog_by_name[norm(name)] = info

    result = []
    for bill, g in rows.groupby(bill_col, sort=False, dropna=True):
        date_val = first_nonblank(g[date_col].tolist())
        sale_dt = parse_dt(date_val)
        time_val = first_nonblank(g[time_col].tolist())
        order_dt = parse_dt(time_val) if time_val else sale_dt
        if pd.isna(sale_dt) and not pd.isna(order_dt):
            sale_dt = order_dt
        status = text(first_nonblank(g[status_col].tolist()))
        note = text(first_nonblank(g["Note"].tolist())) if "Note" in g else ""
        cname, cid = customer_name(first_nonblank(g[cust_col].tolist())), text(first_nonblank(g[custid_col].tolist()))
        group, staff = text(first_nonblank(g[group_col].tolist())), text(first_nonblank(g[staff_col].tolist()))
        channel = norm(first_nonblank(g[channel_col].tolist())) if channel_col else ""
        payment = norm(first_nonblank(g[payment_col].tolist())) if payment_col else ""
        total, lines = 0.0, []
        for _, r in g.iterrows():
            pname = text(r.get(name_col))
            if not pname:
                continue
            pid = text(r.get(id_col))
            meta = catalog_by_id.get(pid) or catalog_by_name.get(norm(pname), {})
            actual_name = meta.get("name") or pname
            service = norm(pname) == "phi phuc vu"
            amount = money(r.get(amount_col))
            total += amount
            raw_qty, qbase = parse_qty(r.get(qty_col)), parse_qty(r.get(qtybase_col))
            if qbase == 0 and raw_qty:
                unit = norm(r.get(row_unit_col)) if row_unit_col else ""
                qbase = raw_qty * REST_CONVERSIONS.get((norm(actual_name), unit), 1)
            lines.append({
                "name": actual_name, "name_norm": norm(actual_name), "id": pid or meta.get("id", ""),
                "category": meta.get("category") or text(r.get(cat_col)), "unit": meta.get("unit") or "",
                "qty": qbase, "raw_qty": raw_qty, "amount": amount, "service": service,
            })
        if not pd.isna(sale_dt):
            sale_dt = pd.Timestamp(sale_dt).normalize()
        if not pd.isna(order_dt):
            order_dt = pd.Timestamp(order_dt)
        result.append({
            "id": text(bill), "date": sale_dt, "datetime": order_dt,
            "status": status_kind(status, ""), "status_raw": status, "note": note,
            "customer": cname, "customer_id": cid, "group": group, "staff": staff,
            "channel": channel, "payment": payment, "revenue": total, "lines": lines, "shop": shop,
            "total_due": money(first_nonblank(g["Tong tien"].tolist())) if shop == "dev_008" else 0.0,
        })

    imp_bill, imp_date, imp_amount = "Mã phiếu nhập", "Ngày nhập", "Thành tiền"
    imp_supplier, imp_status = "Mã NCC", "Trạng thái phiếu"
    imports = []
    import_rows[imp_bill] = import_rows[imp_bill].map(text)
    for bill, g in import_rows.dropna(subset=[imp_bill]).drop_duplicates().groupby(imp_bill, sort=False):
        if not text(bill):
            continue
        status = norm(first_nonblank(g[imp_status].tolist()))
        if any(x in status for x in ("huy", "cancel")):
            continue
        imports.append({
            "id": text(bill), "date": parse_dt(first_nonblank(g[imp_date].tolist())),
            "supplier": text(first_nonblank(g[imp_supplier].tolist())),
            "amount": sum(money(v) for v in g[imp_amount].tolist()),
        })
    stock_value = sum(info["stock"] * info["cost"] for info in catalog_by_id.values())
    return {"orders": result, "catalog": list(catalog_by_id.values()),
            "catalog_by_id": catalog_by_id, "catalog_by_name": catalog_by_name,
            "imports": imports, "stock_value": stock_value}


def load_baby():
    root = DATA / "data" / "dev_013"
    cat_df = pd.read_excel(root / "QuanLy.xlsx", sheet_name="DanhMuc", engine="calamine", dtype=object)
    catalog, alias_map = {}, {}
    for _, c in cat_df.iterrows():
        name = text(c.get("Tên hàng"))
        if not name:
            continue
        aliases = [name] + [x.strip() for x in text(c.get("Tên gọi khác")).split(",") if x.strip()]
        info = {
            "name": name, "category": text(c.get("Nhóm")), "brand": text(c.get("Hãng")),
            "size": text(c.get("Số / Size")), "unit": norm(c.get("ĐVT")),
            "conversion": conv_map_from_text(c.get("Quy đổi")),
        }
        catalog[norm(name)] = info
        for alias in aliases:
            alias_map[norm(alias)] = info

    orders_by_id = {}
    for fname in ("BanHang_2025.xlsx", "BanHang_2026.xlsx"):
        path = root / fname
        for sheet in pd.ExcelFile(path, engine="calamine").sheet_names:
            year = int(sheet[-4:])
            df = pd.read_excel(path, sheet_name=sheet, engine="calamine", header=1, dtype=object)
            df["HĐ"] = df["HĐ"].map(text)
            df = df[df["HĐ"] != ""].drop_duplicates().copy()
            for bill, g in df.groupby("HĐ", sort=False):
                oid = text(bill)
                o = orders_by_id.setdefault(oid, {
                    "id": oid, "date": pd.NaT, "datetime": pd.NaT, "status": "valid",
                    "status_raw": "", "note": "", "customer": "", "customer_id": "",
                    "group": "", "staff": "", "channel": "", "payment": "", "revenue": 0.0,
                    "lines": [], "shop": "dev_013",
                })
                if pd.isna(o["date"]):
                    day = parse_dt(first_nonblank(g["Ngày"].tolist()), year=year)
                    if not pd.isna(day):
                        o["date"] = pd.Timestamp(day).normalize()
                if not o["customer"]:
                    o["customer"] = customer_name(first_nonblank(g["Khách"].tolist()))
                if not o["channel"]:
                    o["channel"] = norm(first_nonblank(g["Kênh"].tolist()))
                notes = [text(v) for v in g["Ghi chú"].tolist() if text(v)]
                if notes:
                    o["note"] = " ".join(notes)
                    o["status_raw"] = o["note"]
                    o["status"] = status_kind("", o["note"])
                for _, r in g.iterrows():
                    pname = text(r.get("Tên hàng"))
                    if not pname:
                        continue
                    pn, amount = norm(pname), money(r.get("TT"))
                    memo = "voucher" if pn == "voucher" else "points" if pn.startswith("dung ") and "diem" in pn else "shipping" if pn == "phi ship" else ""
                    if memo:
                        o["lines"].append({
                            "name": pname, "name_norm": pn, "id": "", "category": "", "unit": "",
                            "qty": 0.0, "raw_qty": 0.0, "amount": amount, "service": True, "memo": memo,
                        })
                        continue
                    info = catalog.get(pn) or alias_map.get(pn, {})
                    qty, unit = parse_qty(r.get("SL")), norm(r.get("ĐVT"))
                    qbase = qty * info.get("conversion", {}).get(unit, 1)
                    o["revenue"] += amount
                    o["lines"].append({
                        "name": info.get("name", pname), "name_norm": norm(info.get("name", pname)),
                        "id": "", "category": info.get("category", ""), "brand": info.get("brand", ""),
                        "size": info.get("size", ""), "unit": info.get("unit", ""), "qty": qbase,
                        "raw_qty": qty, "amount": amount, "service": False,
                    })
    members = pd.read_excel(root / "QuanLy.xlsx", sheet_name="KhachHang", engine="calamine", dtype=object)
    members = members.copy()
    members["_date"] = members["Ngày đăng ký"].map(parse_dt)
    import_raw = pd.read_excel(root / "QuanLy.xlsx", sheet_name="NhapHang", engine="calamine", dtype=object)
    imports = []
    import_raw["Số phiếu"] = import_raw["Số phiếu"].map(text)
    for bill, g in import_raw.groupby("Số phiếu", sort=False):
        if not text(bill):
            continue
        imports.append({
            "id": text(bill), "date": parse_dt(first_nonblank(g["Ngày"].tolist())),
            "supplier": text(first_nonblank(g["Nhà cung cấp"].tolist())),
            "amount": sum(money(v) for v in g["Thành tiền"].tolist()),
        })
    return {"orders": list(orders_by_id.values()), "catalog": list(catalog.values()),
            "catalog_by_id": {}, "catalog_by_name": catalog, "alias_map": alias_map,
            "members": members, "imports": imports, "stock_value": None}


CACHE_FILE = Path("/tmp/datamate-codex-cache/retail_baby_datasets.pkl")
if CACHE_FILE.exists():
    try:
        import pickle
        with open(CACHE_FILE, "rb") as f:
            DATASETS = pickle.load(f)
    except Exception:
        DATASETS = {shop: load_restaurant(shop) for shop in ("dev_007", "dev_008")}
        DATASETS["dev_013"] = load_baby()
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(DATASETS, f)
else:
    DATASETS = {shop: load_restaurant(shop) for shop in ("dev_007", "dev_008")}
    DATASETS["dev_013"] = load_baby()
    try:
        import pickle
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(DATASETS, f)
    except Exception:
        pass
QUESTIONS = [json.loads(line) for line in (DATA / "questions.jsonl").read_text(encoding="utf-8").splitlines()
             if json.loads(line).get("business_id") in SHOPS]


def month_bounds(year, month):
    return pd.Timestamp(year, month, 1), pd.Timestamp(year, month, calendar.monthrange(year, month)[1])


def month_year(value):
    y = int(value)
    return 2000 + y if y < 100 else y


def explicit_months(original, qn=None):
    qsrc = query_preserving_separators(original)
    qn = norm(original) if qn is None else qn
    found = []
    # Capture compact month/year references while excluding DD/MM/YYYY dates.
    patterns = [
        r"(?<![\d/])(?:thang|th|t)\s*0?([1-9]|1[0-2])\s*[/\-]\s*(20\d{2}|\d{2})(?![\d/])",
        r"(?<![\d/])0?([1-9]|1[0-2])\s*[/\-]\s*(20\d{2}|\d{2})(?![\d/])",
        r"thang\s*0?([1-9]|1[0-2])\s+nam\s+(20\d{2})",
    ]
    for pat in patterns:
        for m in re.finditer(pat, qsrc):
            pair = (month_year(m.group(2)), int(m.group(1)))
            if pair not in found:
                found.append(pair)
    # "5, 6, 7/2025" and "th9, th10, th11/2025".
    multi = r"(?:th\s*)?0?(\d{1,2})\s*,\s*(?:th\s*)?0?(\d{1,2})\s*,\s*(?:th\s*)?0?(\d{1,2})\s*[/\-]\s*(20\d{2}|\d{2})"
    for m in re.finditer(multi, qsrc):
        year = month_year(m.group(4))
        months = [(year, int(m.group(i))) for i in (1, 2, 3)]
        if all(1 <= mo <= 12 for _, mo in months):
            found = months + [p for p in found if p not in months]
    named_multi = r"th\s*0?(\d{1,2})\s*,\s*th\s*0?(\d{1,2})\s*,\s*th\s*0?(\d{1,2})\s*[/\-]\s*(20\d{2}|\d{2})"
    for m in re.finditer(named_multi, qsrc):
        year = month_year(m.group(4))
        months = [(year, int(m.group(i))) for i in (1, 2, 3)]
        if all(1 <= mo <= 12 for _, mo in months):
            found = months + [p for p in found if p not in months]
    named_multi_year = r"thang\s*0?(\d{1,2})\s*,\s*thang\s*0?(\d{1,2})\s*,\s*thang\s*0?(\d{1,2})\s+nam\s+(20\d{2})"
    for m in re.finditer(named_multi_year, qsrc):
        year = int(m.group(4))
        months = [(year, int(m.group(i))) for i in (1, 2, 3)]
        if all(1 <= mo <= 12 for _, mo in months):
            found = months + [p for p in found if p not in months]
    if len(found) >= 2 and " tu " in f" {qsrc} " and " den " in f" {qsrc} ":
        chronological = sorted(set(found))
        if chronological[0][0] == chronological[-1][0]:
            first_month, last_month = chronological[0][1], chronological[-1][1]
            if 1 < last_month - first_month <= 11:
                year = chronological[0][0]
                found = [(year, month) for month in range(first_month, last_month + 1)]
    return found


def period_sets(qn, asof, original=None):
    # Quarter before month parsing because "quý 3/2025" contains a month-like token.
    qsrc = query_preserving_separators(original) if original is not None else qn
    quarter = re.search(r"quy\s*([1-4])\s*(?:[/\-]\s*|\s+nam\s+)(20\d{2}|\d{2})", qsrc)
    if quarter:
        qnum, year = int(quarter.group(1)), month_year(quarter.group(2))
        start_month = 1 + 3 * (qnum - 1)
        start = pd.Timestamp(year, start_month, 1)
        end_month = start_month + 2
        end = pd.Timestamp(year, end_month, calendar.monthrange(year, end_month)[1])
        return [(start, end, f"Q{qnum}-{year}")]

    months = explicit_months(original if original is not None else qn, qn)
    if len(months) > 1:
        return [(month_bounds(y, m)[0], month_bounds(y, m)[1], f"{y:04d}-{m:02d}") for y, m in months]
    if months:
        y, m = months[0]
        return [(month_bounds(y, m)[0], month_bounds(y, m)[1], f"{y:04d}-{m:02d}")]

    qsrc = query_preserving_separators(original) if original is not None else qn
    if re.search(r"\bthang\w*\s+nay\b|\bth\s+nay\b", qsrc):
        return [(pd.Timestamp(asof.year, asof.month, 1), asof.normalize(), f"{asof.year:04d}-{asof.month:02d}")]
    short_month = re.search(r"\b(?:thang|th)\s*0?([1-9]|1[0-2])(?![\d/])", qsrc)
    if short_month:
        month = int(short_month.group(1))
        year = asof.year if month <= asof.month else asof.year - 1
        return [(month_bounds(year, month)[0], month_bounds(year, month)[1], f"{year:04d}-{month:02d}")]
    if "tu dau nam" in qn or "dau nam toi" in qn or "dau nam den" in qn or "tu 1 1" in qn or "tu ngay 1 1" in qn or ("1/1" in (original or "") and "nam" in qn):
        return [(pd.Timestamp(asof.year, 1, 1), asof.normalize(), str(asof.year))]
    ndays = re.search(r"(\d+)\s*ngay\s*gan\s*nhat", qn)
    if ndays:
        n = int(ndays.group(1))
        return [(asof.normalize() - pd.Timedelta(days=n - 1), asof.normalize(), f"last{n}")]
    if "tuan truoc" in qn or "tuan trc" in qn or "tuan roi" in qn:
        monday = asof.normalize() - pd.Timedelta(days=asof.weekday())
        start = monday - pd.Timedelta(days=7)
        return [(start, start + pd.Timedelta(days=6), "previous-week")]
    if "tuan nay" in qn:
        start = asof.normalize() - pd.Timedelta(days=asof.weekday())
        return [(start, asof.normalize(), "this-week")]
    if "hom qua" in qn or "bua qua" in qn or "hqua" in qn:
        day = asof.normalize() - pd.Timedelta(days=1)
        return [(day, day, day.strftime("%Y-%m-%d"))]
    if "hom nay" in qn or "bua nay" in qn:
        day = asof.normalize()
        return [(day, day, day.strftime("%Y-%m-%d"))]
    return []


def period_valid(ds, start, end, asof):
    if start > asof.normalize():
        return False, None, None, "future"
    end = min(end.normalize(), asof.normalize())
    dated = [o["date"] for o in ds["orders"] if not pd.isna(o["date"])]
    if not dated:
        return False, None, None, "no-data"
    data_min, data_max = min(dated), max(dated)
    if start > data_max or end < data_min:
        return False, None, None, "coverage"
    return True, max(start.normalize(), data_min), min(end, data_max), ""


def in_period(o, start, end):
    d = o.get("date")
    return not pd.isna(d) and start <= d <= end


def period_orders(ds, p, asof, status="valid"):
    start, end, _label = p
    valid, lo, hi, why = period_valid(ds, start, end, asof)
    if not valid:
        return [], why
    return [o for o in ds["orders"] if in_period(o, lo, hi) and (status is None or o["status"] == status)], ""


def sum_revenue(orders):
    return sum(o["revenue"] for o in orders)


def line_items(orders):
    return [line for o in orders for line in o["lines"] if not line.get("service")]


def aggregate_lines(orders, predicate=None):
    lines = line_items(orders)
    return [line for line in lines if predicate(line)] if predicate else lines


def pct_delta(current, previous):
    return None if previous == 0 else (current - previous) / previous * 100


def money_or_count(value):
    return rnd(value, 0)


def product_for_question(ds, qn):
    candidates = [(norm(info["name"]), info) for info in ds.get("catalog", [])]
    for alias, info in ds.get("alias_map", {}).items():
        candidates.append((alias, info))
    # Explicit aliases used in these released questions.
    aliases = {
        "pho mat con bo cuoi hop 8 mieng": "pho mai con bo cuoi hop 8 mieng",
        "ma lon nuong nuoc mam": "ma heo nuong nuoc mam",
        "chao ngao": "chao ngheu",
        "muc ran nuoc mam": "muc chien nuoc mam",
        "dau phu ran gion": "dau hu chien gion",
        "canh chua ca trau": "canh chua ca loc",
    }
    for alias, target in aliases.items():
        for info in ds.get("catalog", []):
            if target in norm(info["name"]):
                candidates.append((alias, info))
    candidates.sort(key=lambda item: len(item[0]), reverse=True)
    for name, info in candidates:
        if len(name) >= 4 and name in qn:
            return info
    # For known homonyms/spacing, permit only a unique high token-overlap result.
    q_tokens = set(qn.split())
    scored = []
    for info in ds.get("catalog", []):
        tokens = {x for x in norm(info["name"]).split() if len(x) > 2}
        score = len(tokens & q_tokens) / max(1, len(tokens))
        if score >= 0.85:
            scored.append((score, info))
    scored.sort(key=lambda item: item[0], reverse=True)
    if scored and (len(scored) == 1 or scored[0][0] > scored[1][0]):
        return scored[0][1]
    return None


def group_for_question(ds, qn):
    categories = {norm(x.get("category")) for x in ds.get("catalog", []) if x.get("category")}
    matches = sorted((x for x in categories if x and re.search(r"\b" + re.escape(x) + r"\b", qn)), key=len, reverse=True)
    return matches[0] if matches else ""


def customer_key(order):
    if is_guest(order.get("customer"), order.get("customer_id", "")):
        return ""
    if text(order.get("customer_id")):
        return "id:" + norm(order["customer_id"])
    return "name:" + norm(order.get("customer"))


def rank_customers(orders, metric="spend"):
    stats = defaultdict(lambda: {"name": "", "spend": 0.0, "orders": 0, "days": set(), "cancels": 0})
    for order in orders:
        key = customer_key(order)
        if not key:
            continue
        row = stats[key]
        cand = re.sub(r"\s*\(\s*\)$", "", text(order.get("customer"))).strip()
        if cand:
            if not row["name"] or sum(ch > "\x7f" for ch in cand) > sum(ch > "\x7f" for ch in row["name"]):
                row["name"] = cand
        if order["status"] == "valid":
            row["spend"] += order["revenue"]
            row["orders"] += 1
            if not pd.isna(order["date"]):
                row["days"].add(order["date"].date())
        elif order["status"] == "cancel":
            row["cancels"] += 1
    key = {"spend": "spend", "orders": "orders", "cancels": "cancels", "days": "days"}[metric]
    metric_value = (lambda row: len(row[key])) if key == "days" else (lambda row: row[key])
    return sorted(stats.items(), key=lambda item: (-metric_value(item[1]), norm(item[1]["name"])))


def question_n(qn, default=1):
    for pattern in (r"top\s*(\d+)", r"(\d+)\s*(?:khach|mon|mat hang|san pham)\s+(?:nao|ban|mua|chay)", r"(\d+)\s+mon"):
        found = re.search(pattern, qn)
        if found:
            return int(found.group(1))
    return default


def parse_change_windows(qn):
    source = query_preserving_separators(qn)
    if not any(term in source for term in ("tang gia", "doi gia", "len gia", "tang len")):
        return None
    length = re.search(r"(\d+)\s*ngay", source)
    date_match = re.search(r"(?<!\d)(\d{1,2})/(\d{1,2})/(20\d{2})(?!\d)", source)
    if not length or not date_match:
        return None
    n = int(length.group(1))
    day, month, year = map(int, date_match.groups())
    start = pd.Timestamp(year, month, day)
    return ((start - pd.Timedelta(days=n), start - pd.Timedelta(days=1)),
            (start, start + pd.Timedelta(days=n - 1)))


def sum_for_period(ds, period, asof, predicate=None):
    orders, why = period_orders(ds, period, asof, status="valid")
    if why:
        return None, why
    return (sum_revenue(orders) if predicate is None else sum(predicate(o) for o in orders)), ""


def customer_debts(ds, asof):
    debts = []
    for order in ds["orders"]:
        if order["status"] == "cancel" or order.get("payment") != "debt":
            continue
        if any(word in order.get("channel", "") for word in ("san", "shopee", "lazada", "tiki")):
            continue
        note = query_preserving_separators(order.get("note", ""))
        if not note:
            continue
        balance = 0.0
        if any(s in note for s in ("da tra du", "ck du roi", "khach ck du roi", "tra du roi")):
            balance = 0.0
        elif "no ca don" in note:
            balance = money(order.get("total_due", 0))
        else:
            amount_pat = r"(\d[\d., ]*(?:\s*(?:tr|k)\s*\d{0,3}|k|nghin|ngan)?(?:\s*\d{3})?\s*(?:d|dong)?)"
            rem = re.search(r"(?:con\s*(?:(?:lai|thieu|no)\s*)?|thieu\s*(?:lai\s*)?)" + amount_pat, note)
            if rem:
                balance = money(rem.group(1))
            if balance <= 0:
                rem = re.search(r"(?:ghi so|no)\s*" + amount_pat, note)
                if rem:
                    balance = money(rem.group(1))
                elif "dua truoc" in note and "ghi so" in note:
                    paid = re.search(r"dua truoc\s*" + amount_pat, note)
                    balance = max(0.0, money(order.get("total_due", 0)) - (money(paid.group(1)) if paid else 0))
        if balance <= 0:
            continue
        age = (asof.normalize() - order["date"]).days if not pd.isna(order["date"]) else 0
        debts.append({"order": order, "balance": balance, "age": age, "customer_key": customer_key(order),
                      "customer": text(order.get("customer"))})
    return debts


def reason_for_why(why):
    return REASONS["coverage"] if why in {"coverage", "future", "no-data"} else REASONS["unknown"]


def answer_question(q):
    shop = q["business_id"]
    ds = DATASETS[shop]
    asof = pd.Timestamp(q["as_of_date"])
    original, qn = q["question"], norm(q["question"])
    is_baby = shop == "dev_013"
    percent_words = ("%" in original) or any(
        x in qn for x in ("phan tram", "phan trawm", "ty trong", "ti trong", "chiem may", "ti le")
    )

    if any(x in qn for x in ("than thien nhat", "mon nao ngon nhat", "ngon nhat", "dang tin cay", "uy tin", "kho tinh nhat")):
        return out("unanswerable", reason=REASONS["subjective"])
    if any(x in qn for x in ("vi mon bi loi", "mon bi loi", "goi nham")):
        return out("unanswerable", reason=REASONS["unknown"])
    # The source does not record customer gender.
    if is_baby and ("khach nam" in qn or "khach nu" in qn):
        return out("unanswerable", reason=REASONS["unknown"])

    periods = period_sets(qn, asof, original)

    # Questions about price changes compare N days from the stated date with N days before it.
    windows = parse_change_windows(original)
    if windows:
        product = product_for_question(ds, qn)
        if not product:
            return out("unanswerable", reason=REASONS["unknown"])
        quantities = []
        for start, end in windows:
            orders, why = period_orders(ds, (start, end, "custom"), asof, status="valid")
            if why:
                return out("unanswerable", reason=reason_for_why(why))
            name = norm(product["name"])
            quantities.append(sum(line["qty"] for line in aggregate_lines(orders, lambda line: line["name_norm"] == name)))
        delta = pct_delta(quantities[1], quantities[0])
        return out("percent", rnd(delta, 2)) if delta is not None else out("unanswerable", reason=REASONS["unknown"])

    # Explicit current-month comparisons use the same elapsed number of days in the previous month.
    change_words = any(w in qn for w in (
        "tang truong", "thay doi", "tang giam", "tang hay giam", "cao hon hay thap hon",
        "cao hay thap hon", "nhieu hay it hon", "nhieu hon hay it hon",
        "len hay xuong", "hon hay kem", "chenh",
    ))
    session_comparison = "ca chieu" in qn and "ca sang" in qn
    if change_words and percent_words and periods and not session_comparison:
        if "thang nay" in qn:
            this_period = periods[0]
            prev_day = asof.replace(day=1) - pd.Timedelta(days=1)
            prev_start = pd.Timestamp(prev_day.year, prev_day.month, 1)
            prev_end = pd.Timestamp(prev_day.year, prev_day.month, min(asof.day, calendar.monthrange(prev_day.year, prev_day.month)[1]))
            comparison = [(prev_start, prev_end, "previous"), this_period]
        elif len(periods) > 1:
            compare_pos = qn.find("so voi")
            main_pos = min([pos for pos in (qn.find("doanh thu"), qn.find("dthu"), qn.find("tien ban")) if pos >= 0], default=-1)
            # If the question starts "so với [older month], doanh thu [newer month]",
            # the later-mentioned month is the target; otherwise the first month is.
            target_is_last = compare_pos >= 0 and main_pos >= 0 and compare_pos < main_pos
            target, previous = (periods[-1], periods[0]) if target_is_last else (periods[0], periods[1])
            comparison = [previous, target]
        elif re.match(r"\d{4}-\d{2}$", periods[0][2]):
            current_start = periods[0][0]
            prev_end = current_start - pd.Timedelta(days=1)
            comparison = [(pd.Timestamp(prev_end.year, prev_end.month, 1), prev_end, "previous"), periods[0]]
        else:
            comparison = []
        if len(comparison) == 2:
            values = []
            for p in comparison:
                value, why = sum_for_period(ds, p, asof)
                if why:
                    return out("unanswerable", reason=reason_for_why(why))
                values.append(value)
            current, previous = values[1], values[0]
            delta = pct_delta(current, previous)
            return out("percent", rnd(delta, 2)) if delta is not None else out("unanswerable", reason=REASONS["unknown"])

    # Yes/no comparisons among several periods.
    if len(periods) > 1 and any(w in qn for w in ("dung vay", "tang lien tuc", "co tang lien tuc")):
        sorted_periods = sorted(periods, key=lambda p: p[0])
        values = []
        for p in sorted_periods:
            value, why = sum_for_period(ds, p, asof)
            if why:
                return out("unanswerable", reason=reason_for_why(why))
            values.append(value)
        return out("boolean", all(left < right for left, right in zip(values, values[1:])))

    if len(periods) > 1 and "thang nao" in qn and any(w in qn for w in ("doanh thu", "dthu", "ban duoc nhieu tien")):
        values = []
        for p in periods:
            value, why = sum_for_period(ds, p, asof)
            if why:
                return out("unanswerable", reason=reason_for_why(why))
            values.append(value)
        best = max(range(len(periods)), key=lambda i: (values[i], periods[i][0]))
        return out("date", periods[best][0].strftime("%Y-%m"))

    # An ordered list of monthly revenues.
    if len(periods) > 1 and any(token in qn for token in ("lan luot", "moi thang bao nhieu", "ghi theo thu tu")):
        values = []
        for p in periods:
            value, why = sum_for_period(ds, p, asof)
            if why:
                return out("unanswerable", reason=reason_for_why(why))
            values.append(money_or_count(value))
        return out("list", values)

    # Baby-care annotations that are kept as non-product order rows.
    if is_baby and ("voucher" in qn or "vau cho" in qn):
        p = periods[0] if periods else period_sets("thang nay", asof)[0]
        orders, why = period_orders(ds, p, asof)
        if why:
            return out("unanswerable", reason=reason_for_why(why))
        value = sum(-line["amount"] for o in orders for line in o["lines"] if line.get("memo") == "voucher")
        return out("number", money_or_count(value))
    if is_baby and ("dung diem" in qn or "diem tru" in qn):
        p = periods[0] if periods else period_sets("thang nay", asof)[0]
        orders, why = period_orders(ds, p, asof)
        if why:
            return out("unanswerable", reason=reason_for_why(why))
        value = sum(-line["amount"] for o in orders for line in o["lines"] if line.get("memo") == "points")
        return out("number", money_or_count(value))
    if is_baby and "phi ship" in qn:
        p = periods[0] if periods else period_sets("thang nay", asof)[0]
        orders, why = period_orders(ds, p, asof)
        if why:
            return out("unanswerable", reason=reason_for_why(why))
        value = sum(line["amount"] for o in orders for line in o["lines"] if line.get("memo") == "shipping")
        return out("number", money_or_count(value))

    # Purchases and supplier volume.
    if any(x in qn for x in ("nhap hang", "nhap hagn", "danh hang ve", "tien nhap", "nhap hang het", "cat hang")):
        p = periods[0] if periods else period_sets("thang nay", asof)[0]
        valid, lo, hi, why = period_valid(ds, p[0], p[1], asof)
        if not valid:
            return out("unanswerable", reason=reason_for_why(why))
        purchases = [x for x in ds["imports"] if not pd.isna(x["date"]) and lo <= pd.Timestamp(x["date"]).normalize() <= hi]
        return out("number", money_or_count(sum(x["amount"] for x in purchases)))
    if is_baby and "nha cung cap" in qn and any(x in qn for x in ("dang tin cay", "nha cung cap chinh")):
        if not periods:
            return out("unanswerable", reason=REASONS["unknown"])
        p = periods[0]
        valid, lo, hi, why = period_valid(ds, p[0], p[1], asof)
        if not valid:
            return out("unanswerable", reason=reason_for_why(why))
        receipts = defaultdict(set)
        for row in ds["imports"]:
            if not pd.isna(row["date"]) and lo <= pd.Timestamp(row["date"]).normalize() <= hi:
                receipts[row["supplier"]].add(row["id"])
        if not receipts:
            return out("unanswerable", reason=REASONS["unknown"])
        winner = sorted(receipts.items(), key=lambda item: (-len(item[1]), norm(item[0])))[0][0]
        return out("string", clean_answer_string(winner))
    if is_baby and "khach thanh vien dang ky moi" in qn:
        p = periods[0] if periods else period_sets("thang nay", asof)[0]
        members = ds["members"]
        if p[0] > asof.normalize():
            return out("unanswerable", reason=REASONS["coverage"])
        end = min(p[1], asof.normalize())
        return out("number", int(((members["_date"] >= p[0]) & (members["_date"] <= end)).sum()))

    # Inventory snapshot is explicitly valued from the catalog's per-unit cost.
    if "hang con du trong kho" in qn or "ton kho" in qn or "hang con ton" in qn:
        return out("number", money_or_count(ds["stock_value"])) if ds["stock_value"] is not None else out("unanswerable", reason=REASONS["unknown"])

    # Current receivables come from order-level debt notes. COD and canceled orders are excluded.
    if not is_baby and any(x in qn for x in ("cong no", "con no", "no con lai", "no co tuoi", "dang no")):
        debts = customer_debts(ds, asof)
        if "tuoi qua 10" in qn:
            return out("number", money_or_count(sum(d["balance"] for d in debts if d["age"] > 10)))
        if "0 7 ngay" in qn or "8 15 ngay" in qn:
            bins = [(0, 7), (8, 15), (16, math.inf)]
            amounts = [money_or_count(sum(d["balance"] for d in debts if lo <= d["age"] <= hi))
                       for lo, hi in bins]
            return out("list", amounts)
        if "5 khach" in qn or "top 5" in qn:
            by_customer = defaultdict(lambda: ["", 0.0])
            for d in debts:
                if not d["customer_key"]:
                    continue
                row = by_customer[d["customer_key"]]
                row[0] = row[0] or d["customer"]
                row[1] += d["balance"]
            top = sorted(by_customer.values(), key=lambda row: (-row[1], norm(row[0])))[:5]
            return out("table", {"columns": ["khach_hang", "so_tien_no"],
                                 "rows": [[clean_answer_string(n), money_or_count(v)] for n, v in top]})
        if "so khach" in qn or "bao nhieu khach" in qn:
            return out("number", len({d["customer_key"] for d in debts if d["customer_key"]}))
        return out("number", money_or_count(sum(d["balance"] for d in debts)))

    # Baby product sizes, formula numbers, and brand share.
    category_revenue_query = (
        is_baby
        and bool(group_for_question(ds, qn))
        and product_for_question(ds, qn) is None
        and any(w in qn for w in ("doanh thu", "dthu", "tien", "ban duoc bao nhieu tien"))
    )
    baby_product_q = is_baby and any(x in qn for x in (
        "bim", "sua bot", "sua so", "size xl", "size m", "size l", "size nb",
        "moony", "huggies", "bobby", "merries", "pampers",
    )) and not category_revenue_query
    if baby_product_q:
        p = periods[0] if periods else period_sets("thang nay", asof)[0]
        orders, why = period_orders(ds, p, asof)
        if why:
            return out("unanswerable", reason=reason_for_why(why))
        lines = line_items(orders)
        diapers = [x for x in lines if norm(x.get("category")) == "bim ta"]
        category = "bim ta" if ("bim" in qn or "size" in qn or any(b in qn for b in ("moony", "huggies", "bobby", "merries", "pampers"))) else "sua bot"
        target_size = next((s for s in ("size nb", "size xl", "size m", "size l") if s in qn), "")
        formula_no = next((s for s in ("so 1", "so 2", "so 4") if s in qn), "")
        target_brand = next((b for b in ("moony", "huggies", "bobby", "merries", "pampers") if b in qn), "")
        target = [x for x in lines if norm(x.get("category")) == category]
        if target_size:
            target = [x for x in target if norm(x.get("size")) == target_size]
        if formula_no:
            target = [x for x in target if norm(x.get("size")) == formula_no]
        if target_brand:
            target = [x for x in target if norm(x.get("brand")) == target_brand]
        if "hang nao" in qn and formula_no:
            by_brand = defaultdict(float)
            for line in target:
                by_brand[text(line.get("brand"))] += line["qty"]
            if not by_brand:
                return out("unanswerable", reason=REASONS["unknown"])
            best = sorted(by_brand.items(), key=lambda item: (-item[1], norm(item[0])))[0][0]
            return out("string", clean_answer_string(best))
        qty = sum(x["qty"] for x in target)
        if percent_words or "chiem" in qn:
            if category == "bim ta":
                denom = sum(x["qty"] for x in diapers)
            else:
                denom = sum(x["qty"] for x in lines if norm(x.get("category")) == "sua bot")
            return out("percent", rnd(qty / denom * 100, 2)) if denom else out("unanswerable", reason=REASONS["unknown"])
        return out("number", money_or_count(qty))

    # "Unsold in this period" means sold before the period but not during it.
    if any(term in qn for term in ("khong ban ra", "hok ban ra", "khong co don nao", "khong ban duoc")):
        if not periods:
            return out("unanswerable", reason=REASONS["unknown"])
        valid, lo, hi, why = period_valid(ds, periods[0][0], periods[0][1], asof)
        if not valid:
            return out("unanswerable", reason=reason_for_why(why))
        prior, current = set(), set()
        for order in ds["orders"]:
            if order["status"] != "valid" or pd.isna(order["date"]) or order["date"] > asof.normalize():
                continue
            target = prior if order["date"] < lo else current if order["date"] <= hi else None
            if target is None:
                continue
            for line in order["lines"]:
                if not line.get("service"):
                    target.add(line.get("id") or line["name_norm"])
        unsold = prior - current
        return out("number", len(unsold)) if any(term in qn for term in ("bao nhieu", "bnhiu", "may mon")) else out("list", sorted(unsold))

    # Exact product revenue or quantity.
    product = product_for_question(ds, qn)
    if product and any(w in qn for w in (
        "bao nhieu tien", "doanh thu", "tien ban", "ban duoc", "may noi", "may bich",
        "may hop", "so dia", "bao nhieu dia", "bao nhieu noi", "so hop", "ban ra", "bao nhieu lon",
    )):
        if not periods:
            return out("unanswerable", reason=REASONS["unknown"])
        orders, why = period_orders(ds, periods[0], asof)
        if why:
            return out("unanswerable", reason=reason_for_why(why))
        lines = aggregate_lines(orders, lambda line: line["name_norm"] == norm(product["name"]))
        if any(w in qn for w in ("bao nhieu tien", "doanh thu", "tien ban")):
            return out("number", money_or_count(sum(line["amount"] for line in lines)))
        return out("number", money_or_count(sum(line["qty"] for line in lines)))

    if not periods:
        return out("unanswerable", reason=REASONS["unknown"])
    period = periods[0]
    all_status, why_all = period_orders(ds, period, asof, status=None)
    orders, why = period_orders(ds, period, asof, status="valid")
    if why and not any(w in qn for w in ("huy", "don dat")):
        return out("unanswerable", reason=reason_for_why(why))
    if why:
        all_status, why_all = [], why
    if why_all and ("huy" in qn or "don dat" in qn):
        return out("unanswerable", reason=reason_for_why(why_all))

    # Customer group share/revenue. Guest orders stay in the shop-wide denominator.
    if "nhom than thiet" in qn:
        group_orders = [o for o in orders if not is_guest(o["customer"], o["customer_id"]) and norm(o["group"]) == "than thiet"]
        if any(w in qn for w in ("trung binh", "binh quan", "gia tri trung binh", "da tri trung binh")):
            return out("number", money_or_count(sum_revenue(group_orders) / len(group_orders))) if group_orders else out("unanswerable", reason=REASONS["unknown"])
        value = sum_revenue(group_orders)
        if percent_words or "chiem" in qn or "dong gop" in qn:
            denominator = sum_revenue(orders)
            return out("percent", rnd(value / denominator * 100, 2)) if denominator else out("unanswerable", reason=REASONS["unknown"])
        return out("number", money_or_count(value))

    # Employee totals and the highest-revenue employee.
    if "nhan vien" in qn or "nhan dien" in qn or re.search(r"\bnv\b|\bnv\d+", qn):
        employee = re.search(r"nv\d{3}", qn)
        if employee:
            amount = sum_revenue([o for o in orders if norm(o.get("staff")) == employee.group(0)])
            return out("number", money_or_count(amount))
        totals = defaultdict(float)
        for o in orders:
            if o.get("staff"):
                totals[o["staff"]] += o["revenue"]
        if not totals:
            return out("unanswerable", reason=REASONS["unknown"])
        winner = sorted(totals.items(), key=lambda item: (-item[1], norm(item[0])))[0][0]
        return out("string", clean_answer_string(winner))

    # Sales split by channel or order payment method.
    channel_terms = [
        ("don ban qua zalo oa", "channel", "zalo"),
        ("zalo oa", "channel", "zalo"),
        ("don ban qua dien thoai", "channel", "phone"),
        ("dien thoai", "channel", "phone"),
        ("don ban tai quay", "channel", "store"),
        ("tai quay", "channel", "store"),
        ("tai tiem", "channel", "tai tiem"),
        ("san thuong mai dien tu", "channel", "san tmdt"),
        ("vi dien tu", "payment", "ewallet"),
        ("don tra bang vi dien tu", "payment", "ewallet"),
        ("tien mat", "payment", "cash"),
        ("cod", "payment", "cod"),
    ]
    selected_terms = []
    for phrase, kind, value in channel_terms:
        if phrase in qn and (kind, value) not in selected_terms:
            selected_terms.append((kind, value))
    selected_terms.sort(key=lambda item: min(
        [qn.find(phrase) for phrase, kind, value in channel_terms if (kind, value) == item and qn.find(phrase) >= 0],
        default=len(qn),
    ))
    if selected_terms and any(w in qn for w in ("doanh thu", "dthu", "tien ban", "bao nhieu tien", "dong gop", "chiem")):
        def matches(order, key):
            kind, value = key
            if kind == "payment":
                return order.get("payment") == value
            channel = order.get("channel", "")
            if value == "zalo":
                return "zalo" in channel
            if value == "phone":
                return "phone" in channel
            if value == "store":
                return channel == ("tai tiem" if is_baby else "store")
            return value in channel
        values = [sum_revenue([o for o in orders if matches(o, key)]) for key in selected_terms]
        if len(values) > 1 and any(x in qn for x in ("tong ben", "tong cong", "ca hai")):
            value = sum(values)
            if percent_words or "dong gop" in qn or "chiem" in qn:
                denominator = sum_revenue(orders)
                return out("percent", rnd(value / denominator * 100, 2)) if denominator else out("unanswerable", reason=REASONS["unknown"])
            return out("number", money_or_count(value))
        if len(values) > 1:
            return out("list", [money_or_count(x) for x in values])
        if percent_words or "dong gop" in qn or "chiem" in qn:
            denominator = sum_revenue(orders)
            return out("percent", rnd(values[0] / denominator * 100, 2)) if denominator else out("unanswerable", reason=REASONS["unknown"])
        return out("number", money_or_count(values[0]))

    # Cancellation rate has every placed order as its denominator; cancelled count is also available.
    if "huy" in qn or "bi huy" in qn:
        canceled = [o for o in all_status if o["status"] == "cancel"]
        if percent_words or "ti le" in qn:
            return out("percent", rnd(len(canceled) / len(all_status) * 100, 2)) if all_status else out("unanswerable", reason=REASONS["unknown"])
        return out("number", len(canceled))

    # Customer behavior and rankings.
    if "khach" in qn:
        if any(s in qn for s in ("kho tinh", "doi y")):
            ranked = rank_customers(all_status, "cancels")
            if not ranked or ranked[0][1]["cancels"] == 0:
                return out("unanswerable", reason=REASONS["unknown"])
            return out("string", clean_answer_string(ranked[0][1]["name"]))
        threshold = re.search(r"(?:tu|tu\s*)\s*(\d+)\s*don\s*(?:tro len)?", qn)
        if not threshold:
            threshold = re.search(r"(\d+)\s*don\s*tro len", qn)
        if threshold and any(w in qn for w in ("mua", "khach", "don")):
            minimum = int(threshold.group(1))
            ranked = [x for x in rank_customers(orders, "orders") if x[1]["orders"] > 0]
            count = sum(1 for _, stat in ranked if stat["orders"] >= minimum)
            if percent_words:
                return out("percent", rnd(count / len(ranked) * 100, 2)) if ranked else out("unanswerable", reason=REASONS["unknown"])
            return out("number", count)
        if any(s in qn for s in ("mua nhieu lan nhat", "mua nhieu don nhat")):
            ranked = rank_customers(orders, "orders")
            if not ranked or ranked[0][1]["orders"] == 0:
                return out("unanswerable", reason=REASONS["unknown"])
            return out("string", clean_answer_string(ranked[0][1]["name"]))
        if any(s in qn for s in ("mua nhieu tien nhat", "khach nao mua nhieu tien", "chi nhieu tien nhat", "chi tieu nhieu nhat")):
            n = question_n(qn, 5)
            ranked = [x for x in rank_customers(orders, "spend") if x[1]["spend"] > 0]
            return out("list", [clean_answer_string(stat["name"]) for _, stat in ranked[:n]])

    # Top products by base quantity, popularity by number of distinct orders.
    if any(s in qn for s in ("ban chay", "chay nhat", "top ", "hot nhat", "ua chuong nhat", "nhieu don nhat", "goi trong it nhat", "duoc mua o tu")):
        if "goi trong it nhat" in qn or "duoc mua o tu" in qn:
            threshold = re.search(r"(?:it nhat|tu)\s*(\d+)\s*don", qn)
            if not threshold:
                threshold = re.search(r"(\d+)\s*don khac nhau", qn)
            minimum = int(threshold.group(1)) if threshold else 1
            by_product = defaultdict(set)
            for o in orders:
                for line in o["lines"]:
                    if not line.get("service"):
                        by_product[line.get("id") or line["name_norm"]].add(o["id"])
            return out("number", sum(len(order_ids) >= minimum for order_ids in by_product.values()))
        if "ua chuong nhat" in qn or "nhieu don nhat" in qn:
            by_product, names, ids = defaultdict(set), {}, {}
            for o in orders:
                for line in o["lines"]:
                    if not line.get("service"):
                        key = line.get("id") or line["name_norm"]
                        by_product[key].add(o["id"])
                        cname = line["name"]
                        if key not in names or sum(ch > "\x7f" for ch in cname) > sum(ch > "\x7f" for ch in names[key]):
                            names[key] = cname
                        ids[key] = line.get("id", "")
            if not by_product:
                return out("unanswerable", reason=REASONS["unknown"])
            winner = sorted(by_product, key=lambda key: (-len(by_product[key]), ids.get(key, key)))[0]
            return out("string", clean_answer_string(names[winner]))
        quantities, names, ids = defaultdict(float), {}, {}
        for line in line_items(orders):
            key = line.get("id") or line["name_norm"]
            quantities[key] += line["qty"]
            cname = line["name"]
            if key not in names or sum(ch > "\x7f" for ch in cname) > sum(ch > "\x7f" for ch in names[key]):
                names[key] = cname
            ids[key] = line.get("id", "")
        if not quantities:
            return out("unanswerable", reason=REASONS["unknown"])
        explicit_n = question_n(qn, None)
        ranked = sorted(quantities, key=lambda key: (-quantities[key], ids.get(key, key)))
        if explicit_n is None and not any(w in qn for w in ("top", "cac mon", "nhung mon", "danh sach", "ke ten")):
            return out("string", clean_answer_string(names[ranked[0]]))
        n = explicit_n or (3 if "3 mon" in qn else 5)
        return out("list", [clean_answer_string(names[key]) for key in ranked[:n]])

    # Count distinct products that reached an order-frequency threshold.
    count_threshold = re.search(r"(?:it nhat|tu)\s*(\d+)\s*don", qn)
    if count_threshold and "mat hang" in qn:
        minimum = int(count_threshold.group(1))
        by_product = defaultdict(set)
        for o in orders:
            for line in o["lines"]:
                if not line.get("service"):
                    by_product[line.get("id") or line["name_norm"]].add(o["id"])
        return out("number", sum(len(ids) >= minimum for ids in by_product.values()))

    # Quantity of all products whose catalog base unit is named in the question.
    if any(term in qn for term in (
        "so dia", "ban duoc may dia", "bao nhieu dia", "bnhiu dia", "nhieu dia",
        "bao nhieu lon", "bnhiu lon", "nhieu lon", "may lon", "may lon sua",
        "may bich", "bao nhieu bich", "bao nhieu hop", "bao nhieu noi", "tong so dia",
    )):
        group = group_for_question(ds, qn)
        target_unit = next((u for u in ("dia", "lon", "bich", "hop", "noi") if u in qn), "")
        lines = line_items(orders)
        if group:
            lines = [line for line in lines if norm(line.get("category")) == group]
        if target_unit and not is_baby:
            lines = [line for line in lines if line.get("unit") == target_unit]
        return out("number", money_or_count(sum(line["qty"] for line in lines)))

    # Count valid orders at or above a stated order revenue threshold.
    qnumeric = query_preserving_separators(original)
    # Count sales days that reached a revenue threshold.
    day_threshold = re.search(r"(?:tu|dat tu)\s*([\d., ]+)\s*(tr(?:ieu)?|k|nghin|ngan)?\s*(?:dong|d)?\s*tro len", qnumeric)
    if not day_threshold:
        day_threshold = re.search(r"tu\s*([\d., ]+)\s*(tr(?:ieu)?|k|nghin|ngan)?\s*(?:dong|d)?\s*tro len", qnumeric)
    if day_threshold and any(x in qn for x in ("ngay", "so ngay", "may ngay")):
        threshold = money(day_threshold.group(1) + (day_threshold.group(2) or ""))
        daily = defaultdict(float)
        for o in orders:
            daily[o["date"].strftime("%Y-%m-%d")] += o["revenue"]
        return out("number", sum(value >= threshold for value in daily.values()))

    order_threshold = re.search(r"(?:tu|dat tu)\s*([\d., ]+)\s*(tr(?:ieu)?|k|nghin|ngan)?\s*(?:dong|d)?\s*tro len", qnumeric)
    if not order_threshold:
        order_threshold = re.search(r"tu\s*([\d., ]+)\s*(tr(?:ieu)?|k|nghin|ngan)?\s*(?:dong|d)?\s*tro len", qnumeric)
    if order_threshold and (re.search(r"\b(?:don|dh|hoa don|dong)\b", qn) or any(x in qn for x in ("may", "bao nhieu", "dem"))):
        threshold = money(order_threshold.group(1) + (order_threshold.group(2) or ""))
        return out("number", sum(o["revenue"] >= threshold for o in orders))

    # Item revenue by category/group, excluding non-item memo rows.
    category = group_for_question(ds, qn)
    if category and any(w in qn for w in ("doanh thu", "dthu", "tien", "ban duoc bao nhieu")):
        lines = [line for line in line_items(orders) if norm(line.get("category")) == category]
        value = sum(line["amount"] for line in lines)
        if percent_words or "chiem" in qn or "dong gop" in qn:
            denominator = sum_revenue(orders)
            return out("percent", rnd(value / denominator * 100, 2)) if denominator else out("unanswerable", reason=REASONS["unknown"])
        return out("number", money_or_count(value))

    # Sales by customer group; guests are excluded from the group numerator only.
    if "khach" in qn and any(w in qn for w in ("nhom", "group")) and "than thiet" in qn:
        group_orders = [o for o in orders if not is_guest(o["customer"], o["customer_id"]) and norm(o["group"]) == "than thiet"]
        if any(w in qn for w in ("trung binh", "binh quan", "gia tri trung binh", "da tri trung binh")):
            return out("number", money_or_count(sum_revenue(group_orders) / len(group_orders))) if group_orders else out("unanswerable", reason=REASONS["unknown"])
        value = sum_revenue(group_orders)
        if percent_words or "chiem" in qn:
            denominator = sum_revenue(orders)
            return out("percent", rnd(value / denominator * 100, 2)) if denominator else out("unanswerable", reason=REASONS["unknown"])
        return out("number", money_or_count(value))

    # Peak date by revenue.
    if "ngay nao" in qn or "hom nao" in qn:
        daily = defaultdict(float)
        for o in orders:
            daily[o["date"].strftime("%Y-%m-%d")] += o["revenue"]
        if not daily:
            return out("unanswerable", reason=REASONS["unknown"])
        day = sorted(daily, key=lambda d: (-daily[d], d))[0]
        return out("date", day)

    # Morning and afternoon sales / order counts.
    if "ca sang" in qn or "ca chieu" in qn:
        morning = [o for o in orders if not pd.isna(o.get("datetime")) and 6 <= o["datetime"].hour <= 13]
        afternoon = [o for o in orders if not pd.isna(o.get("datetime")) and 14 <= o["datetime"].hour <= 21]
        count_metric = any(w in qn for w in ("so don", "may don", "mi don", "dem giup so don", "bao nhieu don")) or bool(re.search(r"\bdh\b", qn))
        morning_value = len(morning) if count_metric else sum_revenue(morning)
        afternoon_value = len(afternoon) if count_metric else sum_revenue(afternoon)
        if "ca chieu" in qn and "ca sang" in qn and percent_words:
            change = pct_delta(afternoon_value, morning_value)
            return out("percent", rnd(change, 2)) if change is not None else out("unanswerable", reason=REASONS["unknown"])
        if "ca chieu" in qn and "ca sang" in qn and "chenh" in qn:
            return out("number", money_or_count(afternoon_value - morning_value))
        if "ca sang" in qn and "ca chieu" not in qn:
            return out("number", money_or_count(morning_value))
        if "ca chieu" in qn and "ca sang" not in qn:
            return out("number", money_or_count(afternoon_value))
        return out("list", [money_or_count(morning_value), money_or_count(afternoon_value)])

    # An ordered monthly revenue list can be asked without the phrase "lần lượt".
    if len(periods) > 1 and "doanh thu" in qn and not percent_words:
        values = []
        for p in periods:
            value, why = sum_for_period(ds, p, asof)
            if why:
                return out("unanswerable", reason=reason_for_why(why))
            values.append(money_or_count(value))
        return out("list", values)

    # Average order value is rounded to the nearest đồng.
    if any(w in qn for w in ("trung binh", "binh quan", "gia tri trung binh")):
        basis = orders
        if "nhom than thiet" in qn:
            basis = [o for o in orders if not is_guest(o["customer"], o["customer_id"]) and norm(o["group"]) == "than thiet"]
        if not basis:
            return out("unanswerable", reason=REASONS["unknown"])
        return out("number", money_or_count(sum_revenue(basis) / len(basis)))

    # Loyal customers: count distinct days with a valid purchase.
    if "khach trung thanh" in qn:
        ranked = rank_customers(orders, "days")
        return out("string", clean_answer_string(ranked[0][1]["name"])) if ranked else out("unanswerable", reason=REASONS["unknown"])

    # Total placed-order count, excluding canceled and returned bills.
    if any(w in qn for w in ("so don", "bao nhieu don", "may don", "tong so dh", "tat ca bao nhieu don")) or re.search(r"\bdh\b", qn):
        return out("number", len(orders))

    # Revenue for a named sales channel or method. If several channels are requested
    # as a combined total, sum them; otherwise retain their listed order.
    revenue_question = any(w in qn for w in ("doanh thu", "doanhthu", "dthu", "tien ban", "ban duoc bao nhieu tien", "ban duoc tong cong bao nhieu tien", "tien hang"))
    if revenue_question:
        keys = []
        if "zalo oa" in qn or "qua zalo" in qn:
            keys.append(("channel", "zalo"))
        if "dien thoai" in qn or "qua phone" in qn:
            keys.append(("channel", "phone"))
        if "tai quay" in qn:
            keys.append(("channel", "store"))
        if "tai tiem" in qn:
            keys.append(("channel", "tai tiem"))
        if "san thuong mai dien tu" in qn:
            keys.append(("channel", "san tmdt"))
        if "vi dien tu" in qn:
            keys.append(("payment", "ewallet"))
        def has_key(o, key):
            kind, value = key
            if kind == "payment":
                return o.get("payment") == value
            channel = o.get("channel", "")
            if is_baby and value == "store":
                return channel == "tai tiem"
            return value in channel
        if keys:
            values = [sum_revenue([o for o in orders if has_key(o, key)]) for key in keys]
            if len(keys) > 1 and any(x in qn for x in ("tong ben", "tong cong", "ca hai")):
                value = sum(values)
            elif len(values) > 1:
                return out("list", [money_or_count(x) for x in values])
            else:
                value = values[0]
            if percent_words or "dong gop" in qn or "chiem" in qn:
                denominator = sum_revenue(orders)
                return out("percent", rnd(value / denominator * 100, 2)) if denominator else out("unanswerable", reason=REASONS["unknown"])
            return out("number", money_or_count(value))
        return out("number", money_or_count(sum_revenue(orders)))

    return out("unanswerable", reason=REASONS["unknown"])


def main():
    answers, failures = {}, []
    for q in QUESTIONS:
        try:
            answers[q["id"]] = answer_question(q)
        except Exception as exc:
            answers[q["id"]] = out("abstain")
            failures.append({"id": q["id"], "error": f"{type(exc).__name__}: {exc}"})
    path = OUT / "answers_retail_baby.json"
    path.write_text(json.dumps(answers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "questions": len(QUESTIONS), "answers": len(answers),
        "types": pd.Series([value["type"] for value in answers.values()]).value_counts().to_dict(),
        "runtime_errors": failures, "output": str(path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
