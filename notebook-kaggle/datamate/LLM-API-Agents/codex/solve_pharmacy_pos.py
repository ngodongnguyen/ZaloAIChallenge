"""Direct, deterministic solution for DataMate pharmacy shops dev_005/dev_006.

No model or baseline outputs are used. All calculations come from the released
public-test source files and the published rulebook.
"""
from __future__ import annotations

import calendar
import json
import math
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "dataset/datamate").exists())
sys.path.insert(0, str(HERE))
from common import DATA, QUESTIONS, dt, norm, num, rnd, tab, text

P5 = DATA / "data/dev_005"
P6 = DATA / "data/dev_006/pos.sqlite"


def out(kind, answer=None, reason=None):
    r = {"type": kind}
    if kind not in {"unanswerable", "abstain"}:
        r["answer"] = answer
    if reason:
        r["reason"] = reason
    return r


def unanswerable(reason):
    return out("unanswerable", reason=reason)


def money(v):
    return int(rnd(float(v or 0), 0))


def pct(a, b):
    if not b:
        return None
    return float(rnd((float(a) / float(b) - 1.0) * 100.0, 2))


def pct_share(a, b):
    if not b:
        return None
    return float(rnd(float(a) / float(b) * 100.0, 2))


def requests_percent(question, normalized):
    return "%" in question or "phan tram" in normalized


def money_threshold(question):
    """Extract an amount followed by a currency word, retaining grouped digits."""
    pat = r"(?<!\w)(\d[\d.,\s]*?)\s*(đồng|đ|vnd|củ|cu|triệu|trieu|nghìn|nghin|k)(?!\w)"
    for m in re.finditer(pat, question, flags=re.I):
        amount = num(m.group(1))
        unit = norm(m.group(2))
        if unit in {"cu", "trieu"}:
            amount *= 1_000_000
        elif unit in {"nghin", "k"}:
            amount *= 1_000
        return amount
    s = norm(question)
    m = re.search(r"(?<!\d)(\d+(?:\s+\d{3})*)\s*(dong|d|cu|trieu|nghin|k)(?!\w)", s)
    if not m:
        return None
    amount = num(m.group(1))
    unit = m.group(2)
    return amount * (1_000_000 if unit in {"cu", "trieu"} else 1_000 if unit in {"nghin", "k"} else 1)


def parse_qty_unit(value, fallback=""):
    s = text(value)
    m = re.match(r"^\s*(-?\d+(?:[.,]\d+)?)\s*(.*?)\s*$", s)
    if m:
        return num(m.group(1)), (m.group(2).strip() or text(fallback))
    if isinstance(value, (int, float)):
        return float(value), text(fallback)
    return num(value), text(fallback)


def canon_status(value):
    s = norm(value)
    if any(x in s for x in ("huy", "cancel", "canceled", "cancelled")):
        return "cancelled"
    if any(x in s for x in ("tra lai", "tra hang", "hang tra ve", "da hoan hang", "returned", "return")):
        return "returned"
    if s in {"shipping", "dang giao", "dang van chuyen"}:
        return "shipping"
    return "completed"


def is_lump_customer(value):
    return norm(value) in {"", "le", "khach le", "kl", "khach vang lai", "retail"}


def convert_factor_tables(df, product_col, unit_col, ratio_col, base_col, code_col=None):
    """Return (product key -> base unit, (product key, unit) -> units per base)."""
    base = {}
    factors = {}
    links = defaultdict(list)
    if code_col and code_col in df:
        key_col = code_col
    else:
        key_col = product_col
    for _, row in df.iterrows():
        key = text(row.get(key_col, ""))
        unit = text(row.get(unit_col, ""))
        base_unit = text(row.get(base_col, ""))
        ratio_text = text(row.get(ratio_col, ""))
        if not key or not unit:
            continue
        if base_unit:
            base[key] = base_unit
        amount, target = parse_qty_unit(ratio_text)
        if amount == 0:
            continue
        links[key].append((unit, amount, target))

    for key, rows in links.items():
        direct = defaultdict(list)
        for unit, amount, target in rows:
            target_key = norm(target) if target else ""
            direct[norm(unit)].append((float(amount), target_key))
        base_key = norm(base.get(key, ""))
        cache = {base_key: 1.0} if base_key else {}

        def to_base(u, seen=None):
            u = norm(u)
            if not u:
                return None
            if u in cache:
                return cache[u]
            seen = set() if seen is None else set(seen)
            if u in seen:
                return None
            seen.add(u)
            for ratio, target in direct.get(u, []):
                next_unit = target or base_key
                child = to_base(next_unit, seen) if next_unit else None
                if child is not None:
                    cache[u] = ratio * child
                    return cache[u]
            return None

        for unit, _, _ in rows:
            factor = to_base(unit)
            if factor:
                factors[(key, norm(unit))] = float(factor)
    return base, factors


def extract_explicit_months(question, asof):
    """Month/year tokens in the question, excluding day/month/year dates."""
    q = question
    ignored = []
    for m in re.finditer(r"(?<!\d)\d{1,2}[/.\-]\d{1,2}[/.\-]\d{4}(?!\d)", q):
        ignored.append((m.start(), m.end()))
    found = []
    pat = re.compile(r"(?<!\d)(1[0-2]|0?[1-9])\s*[/\-]\s*(20\d{2}|\d{2})(?!\d)")
    for m in pat.finditer(q):
        if any(a <= m.start() < b for a, b in ignored):
            continue
        mo = int(m.group(1)); yy = int(m.group(2))
        year = (2000 + yy if yy < 80 else 1900 + yy) if yy < 100 else yy
        found.append((year, mo, m.start()))
    found.sort(key=lambda z: z[2])
    seq = []
    for y, m, _ in found:
        if (y, m) not in seq:
            seq.append((y, m))
    # Bare month references (e.g. "tháng 4"), interpreted relative to as_of.
    if not seq:
        for m in re.finditer(r"\b(?:thang|th|t)\s*(1[0-2]|0?[1-9])\b", norm(q)):
            mo = int(m.group(1))
            year = asof.year if mo <= asof.month else asof.year - 1
            if (year, mo) not in seq:
                seq.append((year, mo))
    return seq


def month_start(y, m):
    return date(y, m, 1)


def month_end(y, m):
    return date(y, m, calendar.monthrange(y, m)[1])


def shift_month(y, m, delta):
    total = y * 12 + (m - 1) + delta
    return total // 12, total % 12 + 1


def expand_month_sequence(question, asof, allow_relative=True):
    s = norm(question)
    explicit = extract_explicit_months(question, asof)
    if len(explicit) >= 2:
        # When wording gives just start/end months, include every intervening month.
        if len(explicit) == 2 and any(x in s for x in ("tu ", "from ")) and any(x in s for x in ("den ", "to ", "toi ", "toi")):
            a, b = explicit[0][:2], explicit[1][:2]
            ia, ib = a[0] * 12 + a[1] - 1, b[0] * 12 + b[1] - 1
            if 0 <= ib - ia <= 24:
                return [shift_month(a[0], a[1], d) for d in range(ib - ia + 1)]
        return [(y, m) for y, m in explicit]
    if len(explicit) == 1:
        y, m = explicit[0][:2]
        if re.search(r"3\s*thang\s+(?:lien|lien nhau)", s) and any(x in s for x in ("tinh toi", "den ", "to ")):
            return [shift_month(y, m, d) for d in (-2, -1, 0)]
        return [(y, m)]
    if allow_relative:
        if any(x in s for x in ("thang nay", "thang ni", "th nay", "th ni")):
            return [(asof.year, asof.month)]
    return []


def date_tokens(question):
    res = []
    for m in re.finditer(r"(?<!\d)(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})(?!\d)", question):
        try:
            res.append(date(int(m.group(3)), int(m.group(2)), int(m.group(1))))
        except ValueError:
            pass
    return res


def quarter_months(question):
    m = re.search(r"\bquy\s*([1-4])\s*(?:[/\-]\s*|\s+)(20\d{2})", norm(question))
    if not m:
        return []
    q, y = int(m.group(1)), int(m.group(2))
    return [(y, m) for m in range((q - 1) * 3 + 1, (q - 1) * 3 + 4)]


def relative_window(question, asof):
    s = norm(question)
    if "tuan truoc" in s or "tuan roi" in s:
        monday = asof - timedelta(days=asof.weekday() + 7)
        return monday, monday + timedelta(days=6)
    if "tuan nay" in s or "tuan ni" in s:
        monday = asof - timedelta(days=asof.weekday())
        return monday, asof
    n = re.search(r"(\d+)\s*ngay\s*(?:gan nhat|lien tiep)", s)
    if n:
        return asof - timedelta(days=int(n.group(1)) - 1), asof
    if "dau nam" in s or "tu 1/1" in s or "tu 1 1" in s or "tu ngay 1/1" in s or "tu ngay 1 1" in s or ("1/1" in s and "nam" in s):
        return date(asof.year, 1, 1), asof
    if any(x in s for x in ("thang nay", "thang ni", "th nay", "th ni")):
        return date(asof.year, asof.month, 1), asof
    if any(x in s for x in ("hom nay", "hnay", "bua nay", "nay ban", "hien tai")):
        return asof, asof
    if any(x in s for x in ("hom qua", "hqua", "bua qua")):
        d = asof - timedelta(days=1)
        return d, d
    return None


def period_for_question(question, asof, default_full_current=False):
    qms = quarter_months(question)
    if qms:
        return month_start(qms[0][0], qms[0][1]), month_end(qms[-1][0], qms[-1][1])
    rel = relative_window(question, asof)
    if rel:
        return rel
    months = expand_month_sequence(question, asof)
    if months:
        return month_start(*months[0]), month_end(*months[0])
    dates = date_tokens(question)
    if dates:
        d = dates[0]
        return d, d
    if default_full_current:
        return date(asof.year, asof.month, 1), asof
    return asof, asof


def requested_unit(question):
    """Unit following the requested count, ignoring package units in the item name."""
    s=norm(question)
    m=re.search(r"(?:bao nhieu|bnhiu|bn|may|so)\s+(mieng|vien|goi|chai|lo|hop|vi|tuyp)",s)
    return {"mieng":"miếng","vien":"viên","goi":"gói","chai":"chai","lo":"lọ","hop":"hộp","vi":"vỉ","tuyp":"tuýp"}[m.group(1)] if m else None


class Shop005:
    def __init__(self):
        raw_lines = []
        for path in sorted(P5.glob("BanHang_Thang*.xlsx")):
            df = tab(path, "DonHang_ChiTiet")
            cols = ["Mã đơn hàng", "Ngày đặt hàng", "Trạng thái đơn hàng", "Tên khách hàng", "Tên sản phẩm", "Số lượng sản phẩm", "Giá sản phẩm", "Tổng tiền", "Đơn vị tính"]
            df = df.reindex(columns=cols)
            df[cols[:4]] = df[cols[:4]].ffill()
            df = df[df["Mã đơn hàng"].map(text).ne("") & df["Tên sản phẩm"].map(text).ne("")].copy()
            df = df.drop_duplicates()
            for _, r in df.iterrows():
                raw_lines.append({
                    "order_id": text(r["Mã đơn hàng"]),
                    "date": dt(r["Ngày đặt hàng"]),
                    "status": canon_status(r["Trạng thái đơn hàng"]),
                    "customer": text(r["Tên khách hàng"]),
                    "item": text(r["Tên sản phẩm"]),
                    "qty": parse_qty_unit(r["Số lượng sản phẩm"], r["Đơn vị tính"])[0],
                    "sale_unit": parse_qty_unit(r["Số lượng sản phẩm"], r["Đơn vị tính"])[1],
                    "amount": num(r["Tổng tiền"]),
                })
        self.lines = pd.DataFrame(raw_lines)
        self.lines["date"] = pd.to_datetime(self.lines.date).dt.date
        # Product master and medication labels
        prod = tab(P5 / "DanhSachSanPham.xlsx")
        self.prod = prod[["Mã SKU", "Tên sản phẩm", "Đơn vị", "Loại sản phẩm"]].copy()
        self.prod = self.prod.rename(columns={"Mã SKU": "sku", "Tên sản phẩm": "item", "Đơn vị": "base_unit", "Loại sản phẩm": "category"})
        self.prod["sku"] = self.prod.sku.map(text)
        self.prod["item"] = self.prod.item.map(text)
        self.prod["key"] = self.prod.item.map(norm)
        self.prod["base_unit"] = self.prod.base_unit.map(text)
        self.prod["category"] = self.prod.category.map(text)
        self.by_item = {r.key: r for r in self.prod.itertuples()}
        med = tab(P5 / "DanhSachThuoc.xlsx")
        self.med = {text(r["Mã SKU"]): {"ingredient": text(r["Hoạt chất"]), "rx": bool(r["Thuốc kê đơn"])} for _, r in med.iterrows()}
        conv = tab(P5 / "DanhSachQuyDoi.xlsx")
        base_by_name, factor_by_name = convert_factor_tables(conv, "Tên sản phẩm", "Đơn vị quy đổi", "Tỉ lệ quy đổi", "Đơn vị cơ bản")
        # Keep the master unit even for products with no larger unit.
        self.base_unit = {r.key: r.base_unit for r in self.prod.itertuples()}
        self.unit_factor = {}
        for pkey, bunit in self.base_unit.items():
            factor_map = {norm(bunit): 1.0}
            for (name, unit), factor in factor_by_name.items():
                if norm(name) == pkey:
                    factor_map[unit] = factor
            self.unit_factor[pkey] = factor_map
        # Order-level fields from first line, and revenue from eligible item lines.
        self.orders = self.lines.groupby("order_id", sort=False).agg(
            date=("date", "first"), status=("status", "first"), customer=("customer", "first"), revenue=("amount", "sum")
        ).reset_index()
        self.orders["customer_key"] = self.orders.customer.map(lambda x: "" if is_lump_customer(x) else norm(x))
        self.lines = self.lines.merge(self.orders[["order_id", "customer_key"]], on="order_id", how="left")
        self.lines["product_key"] = self.lines.item.map(norm)
        self.lines["base_qty"] = self.lines.apply(self._base_qty, axis=1)
        self.lines["sku"] = self.lines.product_key.map(lambda k: self.by_item[k].sku if k in self.by_item else "")
        self.lines["category"] = self.lines.product_key.map(lambda k: self.by_item[k].category if k in self.by_item else "")
        self.lines["base_unit"] = self.lines.product_key.map(lambda k: self.base_unit.get(k, ""))
        self.lines["ingredient"] = self.lines.sku.map(lambda x: self.med.get(x, {}).get("ingredient", ""))
        self.lines["rx"] = self.lines.sku.map(lambda x: self.med.get(x, {}).get("rx", False))
        self.eligible_ids = set(self.orders.loc[~self.orders.status.isin(["cancelled", "returned"]), "order_id"])
        self.eligible_orders = self.orders[self.orders.order_id.isin(self.eligible_ids)].copy()
        self.sales = self.lines[self.lines.order_id.isin(self.eligible_ids)].copy()
        self.coverage_start = self.orders.date.min()
        self.coverage_end = self.orders.date.max()

    def _base_qty(self, row):
        unit = norm(row.sale_unit)
        base = norm(self.base_unit.get(row.product_key, ""))
        factor = self.unit_factor.get(row.product_key, {}).get(unit, 1.0 if unit == base or not unit else None)
        if factor is None:
            return float(row.qty)
        return float(row.qty) * factor

    def overlap(self, start, end):
        if end < self.coverage_start or start > self.coverage_end:
            return None
        return max(start, self.coverage_start), min(end, self.coverage_end)

    def orders_between(self, start, end, eligible=True):
        actual = self.overlap(start, end)
        if actual is None:
            return None
        df = self.eligible_orders if eligible else self.orders
        return df[(df.date >= actual[0]) & (df.date <= actual[1])].copy()

    def lines_between(self, start, end, extra=None):
        actual = self.overlap(start, end)
        if actual is None:
            return None
        df = self.sales[(self.sales.date >= actual[0]) & (self.sales.date <= actual[1])].copy()
        if extra is not None:
            df = extra(df)
        return df

    def revenue(self, start, end, lines=None):
        df = self.lines_between(start, end) if lines is None else lines
        return None if df is None else float(df.amount.sum())

    def count_orders(self, start, end, eligible=True):
        df = self.orders_between(start, end, eligible)
        return None if df is None else int(len(df))

    def item_row(self, qtext):
        nq = norm(qtext)
        # Prefer exact normalized match, then a full unique substring match.
        if nq in self.by_item:
            return self.by_item[nq]
        matches = [r for r in self.prod.itertuples() if norm(r.item) in nq or nq in norm(r.item)]
        if len(matches) == 1:
            return matches[0]
        # Use distinctive quoted/named pieces that appear in product catalog.
        matches = [r for r in self.prod.itertuples() if len(norm(r.item)) >= 10 and norm(r.item) in nq]
        return max(matches, key=lambda r: len(norm(r.item))) if matches else None

    def customer_aggregate(self, start, end, metric="revenue"):
        od = self.orders_between(start, end, True)
        if od is None:
            return None
        od = od[od.customer_key.ne("")]
        if metric == "orders":
            agg = od.groupby("customer_key").agg(value=("order_id", "nunique"), customer=("customer", "first"))
        else:
            agg = od.groupby("customer_key").agg(value=("revenue", "sum"), customer=("customer", "first"))
        agg["tie"] = agg.index.astype(str)
        return agg


class Shop006:
    def __init__(self):
        con = sqlite3.connect(P6)
        self.tables = {name: pd.read_sql_query(f'SELECT * FROM "{name}"', con) for (name,) in con.execute("select name from sqlite_master where type='table'")}
        con.close()
        for frame in self.tables.values():
            for column in frame.columns:
                if column.startswith("Ma") or column == "IDCuaHang":
                    frame[column] = frame[column].map(text)
        self.inv = self.tables["don_hang"].copy()
        self.inv["time"] = pd.to_datetime(self.inv.ThoiGianLap, errors="coerce")
        self.inv["date"] = self.inv.time.dt.date
        self.inv["status"] = self.inv.TinhTrang.map(canon_status)
        self.inv["total_due"] = self.inv.TongThanhToan.map(num)
        self.inv["customer_id"] = self.inv.MaKhachHang.fillna("").astype(str)
        cust = self.tables["khach_hang"]
        self.customer_name = dict(zip(cust.MaKhachHang.fillna("").astype(str), cust.TenGoi.map(text)))
        self.customer_type = dict(zip(cust.MaKhachHang.fillna("").astype(str), cust.LoaiKhachHang.map(text)))
        self.customer_gender = dict(zip(cust.MaKhachHang.fillna("").astype(str), cust.NamNu.map(text)))
        staff = self.tables["nhan_vien"]
        self.employee_name = dict(zip(staff.MaNhanVien.fillna("").astype(str), staff.TenGoi.map(text)))
        self.inv["customer"] = self.inv.customer_id.map(self.customer_name).fillna("")
        self.inv["cust_type"] = self.inv.customer_id.map(self.customer_type).fillna("")
        self.inv["employee"] = self.inv.MaNhanVien.fillna("").astype(str).map(self.employee_name).fillna("")
        self.inv["channel"] = self.inv.KenhBanHang.fillna("").map(text)
        self.inv["method"] = self.inv.PhuongThucTT.fillna("").map(text)
        self.lines = self.tables["chi_tiet_don"].drop_duplicates().copy()
        self.lines["qty"] = self.lines.SL.map(num)
        self.lines["base_qty"] = self.lines.SLQuyDoi.map(num)
        self.lines["amount"] = self.lines.TriGia.map(num)
        self.lines["cost_unit"] = self.lines.GiaVonLucBan.map(num)
        self.lines["cost"] = self.lines.qty * self.lines.cost_unit
        self.products = self.tables["san_pham"].copy()
        self.products["name"] = self.products.TenGoi.map(text)
        self.products["key"] = self.products.name.map(norm)
        self.products["category"] = self.products.DanhMuc.map(text)
        self.products["base_unit"] = self.products.DonViCoSo.map(text)
        self.products["base_cost"] = self.products.GiaVonHang.map(num)
        self.p_by_code = {text(r.MaHangHoa): r for r in self.products.itertuples()}
        self.p_by_name = {norm(r.name): r for r in self.products.itertuples()}
        med = self.tables["thong_tin_thuoc"]
        self.med = {text(r.MaHangHoa): {"ingredient": text(r.ThanhPhanChinh), "rx": bool(num(r.BanTheoDon))} for r in med.itertuples()}
        self.units = self.tables["don_vi_sp"].copy()
        self.unit_factor = {}
        for r in self.units.itertuples():
            self.unit_factor[(text(r.MaHangHoa), norm(r.DonViTinh))] = float(num(r.HeSoQuyDoi) or 1)
        self.baseunit_by_code = {text(r.MaHangHoa): text(r.DonViCoSo) for r in self.products.itertuples()}
        self.lines = self.lines.merge(self.inv[["MaHoaDon", "date", "time", "status", "customer_id", "customer", "cust_type", "employee", "channel", "method"]], on="MaHoaDon", how="left")
        self.lines = self.lines.merge(self.products[["MaHangHoa", "name", "key", "category", "base_unit"]], on="MaHangHoa", how="left")
        self.lines["ingredient"] = self.lines.MaHangHoa.map(lambda c: self.med.get(text(c), {}).get("ingredient", ""))
        self.lines["rx"] = self.lines.MaHangHoa.map(lambda c: self.med.get(text(c), {}).get("rx", False))
        self.eligible = self.lines[~self.lines.status.isin(["cancelled", "returned"])].copy()
        self.coverage_start = self.inv.date.min()
        self.coverage_end = self.inv.date.max()
        self.payments = self.tables["thanh_toan"].copy()
        self.payments["time"] = pd.to_datetime(self.payments.NgayGioThu, errors="coerce")
        self.payments["date"] = self.payments.time.dt.date
        self.payments["amount"] = self.payments.SoTienThu.map(num)
        self.returns = self.tables["tra_hang"].copy()
        self.returns["time"] = pd.to_datetime(self.returns.NgayGioTra, errors="coerce")
        self.returns["date"] = self.returns.time.dt.date
        self.purchases = self.tables["phieu_nhap"].copy()
        self.purchases["date"] = pd.to_datetime(self.purchases.NgayNhapKho.map(lambda x: dt(x)), errors="coerce").dt.date
        self.purchases["amount"] = self.purchases.TongGiaTriNhap.map(num)
        self.purchases = self.purchases.merge(self.tables["nha_cung_cap"][["MaNhaCungCap", "TenGoi"]].drop_duplicates(), on="MaNhaCungCap", how="left")
        self.purchases["supplier"] = self.purchases.TenGoi.map(text)
        self.stock = self.tables["the_kho"].copy()
        self.stock["time"] = pd.to_datetime(self.stock.NgayGioPhatSinh, errors="coerce")
        self.stock["date"] = self.stock.time.dt.date
        self.stock["balance"] = self.stock.TonCuoi.map(num)
        self.lots = self.build_lots()

    def overlap(self, start, end):
        if end < self.coverage_start or start > self.coverage_end:
            return None
        return max(start, self.coverage_start), min(end, self.coverage_end)

    def lines_between(self, start, end, eligible=True):
        actual = self.overlap(start, end)
        if actual is None:
            return None
        df = self.eligible if eligible else self.lines
        return df[(df.date >= actual[0]) & (df.date <= actual[1])].copy()

    def orders_between(self, start, end, eligible=True):
        actual = self.overlap(start, end)
        if actual is None:
            return None
        df = self.inv[~self.inv.status.isin(["cancelled", "returned"])] if eligible else self.inv
        return df[(df.date >= actual[0]) & (df.date <= actual[1])].copy()

    def build_lots(self):
        lots = self.tables["lo_hang"].copy()
        lots["product"] = lots.MaHangHoa.map(text)
        lots["expiry"] = pd.to_datetime(lots.HanSuDung, errors="coerce").dt.date
        lots["initial"] = lots.SLNhapQuyDoi.map(num)
        lots["unit_price"] = lots.DonGiaNhap.map(num)
        # DonGiaNhap in lo_hang is the purchase-line unit price; convert to basic unit.
        imp = self.tables["chi_tiet_nhap"][["MaLo", "DonViTinh"]].drop_duplicates("MaLo")
        lots = lots.merge(imp, on="MaLo", how="left")
        lots["unit_factor"] = lots.apply(lambda r: self.unit_factor.get((text(r.product), norm(r.DonViTinh)), 1.0), axis=1)
        lots["base_cost"] = lots.apply(lambda r: (r.unit_price / r.unit_factor) if r.unit_factor else r.unit_price, axis=1)
        lo = self.tables["chi_tiet_don_lo"].drop_duplicates().copy()
        lo["qty"] = lo.SLQuyDoi.map(num)
        lo["returned"] = lo.SLTraQuyDoi.map(num)
        lo = lo.merge(self.inv[["MaHoaDon", "status", "date"]], on="MaHoaDon", how="left")
        lo = lo[lo.status.ne("cancelled")]
        sold = lo.groupby("MaLo").agg(sold=("qty", "sum"), returned=("returned", "sum"))
        lots = lots.merge(sold, left_on="MaLo", right_index=True, how="left")
        lots[["sold", "returned"]] = lots[["sold", "returned"]].fillna(0)
        lots["remaining"] = (lots.initial - lots.sold + lots.returned).clip(lower=0)
        lots = lots.merge(self.products[["MaHangHoa", "name"]], left_on="product", right_on="MaHangHoa", how="left")
        return lots

    def stock_asof(self, d):
        f = self.stock[self.stock.date <= d].sort_values(["time", "MaGiaoDichKho"])
        if f.empty:
            return pd.DataFrame(columns=["MaHangHoa", "balance"])
        return f.groupby("MaHangHoa", as_index=False).tail(1)[["MaHangHoa", "balance"]]

    def debt_ledger(self, d):
        inv = self.inv[(self.inv.date <= d) & self.inv.status.ne("cancelled")].copy()
        market = inv.channel.map(lambda s: any(x in norm(s) for x in ("shopee", "lazada", "tiki", "sendo", "marketplace", "san thuong mai")))
        inv = inv[(inv.method.map(norm) != "cod") & ~market]
        paid = self.payments[self.payments.date <= d].groupby("MaHoaDon").amount.sum()
        inv["paid"] = inv.MaHoaDon.map(paid).fillna(0)
        inv["outstanding"] = (inv.total_due - inv.paid).clip(lower=0)
        inv = inv[inv.outstanding > 0].copy()
        inv["age"] = [(d - x).days for x in inv.date]
        return inv

    def item_row(self, qtext):
        nq = norm(qtext)
        if nq in self.p_by_name:
            return self.p_by_name[nq]
        matches = [r for r in self.products.itertuples() if len(norm(r.name)) >= 8 and norm(r.name) in nq]
        return max(matches, key=lambda r: len(norm(r.name))) if matches else None

    def quantity_for_unit(self, lines, itemcode, target_unit):
        base_unit = norm(self.baseunit_by_code.get(itemcode, ""))
        unit = norm(target_unit)
        factor = self.unit_factor.get((itemcode, unit))
        if factor is None:
            aliases={"lo":"chai","chai":"lo"}
            alias=aliases.get(unit)
            if alias:
                factor=self.unit_factor.get((itemcode,alias))
        if factor is None:
            factor = 1.0 if unit == base_unit else None
        if factor is None:
            return None
        base_q = float(lines.loc[lines.MaHangHoa.map(text).eq(itemcode), "base_qty"].sum())
        return base_q / factor


S5 = Shop005()
S6 = Shop006()


def period_series(q, asof):
    months = expand_month_sequence(q, asof)
    if months:
        return [(month_start(y, m), month_end(y, m)) for y, m in months]
    quarter = quarter_months(q)
    if quarter:
        return [(month_start(quarter[0][0], quarter[0][1]), month_end(quarter[-1][0], quarter[-1][1]))]
    w = relative_window(q, asof)
    return [w] if w else []


def month_months_for_change(q, asof):
    months = expand_month_sequence(q, asof)
    return months[0] if months else (asof.year, asof.month)


def get_sales_window(q, asof):
    periods = period_series(q, asof)
    if periods:
        return periods[0]
    return period_for_question(q, asof)


def max_daily_revenue(shop, start, end):
    df = shop.lines_between(start, end)
    if df is None:
        return None, None
    daily = df.groupby("date").amount.sum()
    if daily.empty:
        return None, 0
    maxv = daily.max()
    days = sorted(d for d, v in daily.items() if v == maxv)
    return days[0], float(maxv)


def change_from_price(shop, q, asof):
    nq = norm(q)
    item = shop.item_row(q)
    if not item:
        return None
    if isinstance(shop, Shop005):
        name, code = item.item, item.sku
        # Explicit dates have precedence; a price history isn't present in dev_005.
        explicit = date_tokens(q)
        if not explicit:
            return None
        D = explicit[0]
        n = int(re.search(r"(\d+)\s*ngay", nq).group(1)) if re.search(r"(\d+)\s*ngay", nq) else 30
        targetunit = "vien" if "vien" in nq else ("lo" if "lo" in nq else item.base_unit)
        unit_factor = S5.unit_factor.get(norm(name), {}).get(norm(targetunit), 1.0 if norm(targetunit) == norm(item.base_unit) else None)
        if unit_factor is None:
            return None
        Q = S5.sales[S5.sales.product_key.eq(norm(name))]
        new = Q[(Q.date >= D) & (Q.date < D + timedelta(days=n))].base_qty.sum() / unit_factor
        old = Q[(Q.date >= D - timedelta(days=n)) & (Q.date < D)].base_qty.sum() / unit_factor
    else:
        code, name = text(item.MaHangHoa), item.name
        history = shop.tables["lich_su_gia"].copy()
        history["day"] = pd.to_datetime(history.NgayApDung, errors="coerce").dt.date
        history = history[history.MaHangHoa.map(text).eq(code)].sort_values("day")
        price_target = None
        mprice = re.search(r"(?:tang gia len|len gia|gia len)\s*([\d., ]+?)\s*(?:dong|d|vnd|/)", nq)
        if mprice:
            price_target = num(mprice.group(1))
        if price_target is not None:
            # Select the first day the requested increased price took effect.
            hit = history[history.GiaBanLe.map(num).eq(price_target)]
            if hit.empty:
                return None
            D = hit.iloc[0].day
        else:
            dates = date_tokens(q)
            if not dates:
                return None
            D = dates[0]
        nmatch = re.search(r"(\d+)\s*ngay", nq)
        n = int(nmatch.group(1)) if nmatch else 30
        targetunit = requested_unit(q) or item.base_unit
        unitfactor = shop.unit_factor.get((code, norm(targetunit)), 1.0 if norm(targetunit) == norm(item.base_unit) else None)
        if unitfactor is None:
            return None
        Q = shop.eligible[(shop.eligible.MaHangHoa.map(text) == code)]
        new = Q[(Q.date >= D) & (Q.date < D + timedelta(days=n))].base_qty.sum() / unitfactor
        old = Q[(Q.date >= D - timedelta(days=n)) & (Q.date < D)].base_qty.sum() / unitfactor
    return pct(new, old), (float(new), float(old), D.isoformat(), n)


def answer005(qrow):
    q, asof = qrow["question"], date.fromisoformat(qrow["as_of_date"])
    s = norm(q)

    if "gio" in s and any(x in s for x in ("dong khach", "dong nhat", "cao diem")):
        return unanswerable("Nguồn dev_005 chỉ ghi ngày đặt hàng, không ghi giờ tạo đơn.")
    if any(x in s for x in ("khach nu", "khach hang nu", "phu nu")):
        return unanswerable("Nguồn dev_005 không có mã khách hoặc trường giới tính để xác định khách nữ.")

    if re.search(r"\b(?:tang gia|len gia|doi gia|ngay tang gia)\b", s):
        result = change_from_price(S5, q, asof)
        if result is None:
            return unanswerable("Không đủ dữ liệu giá hoặc đơn vị quy đổi để đối chiếu hai khoảng bán.")
        return out("percent", result[0])

    # Explicit monthly comparisons/trends/lists.
    months = expand_month_sequence(q, asof)
    if ("cung ky" in s or "nam ngoai" in s or "cung thang nam truoc" in s) and "thang truoc" not in s:
        y, m = month_months_for_change(q, asof)
        startcur, endcur = month_start(y, m), month_end(y, m)
        if any(x in s for x in ("thang nay", "thang ni", "th nay", "th ni")):
            endcur = min(asof, endcur)
        prevy = y - 1
        startprev, endprev = month_start(prevy, m), month_end(prevy, m)
        a = S5.revenue(startcur, endcur); b = S5.revenue(startprev, endprev)
        if a is None or b is None:
            return unanswerable("Một trong hai kỳ so sánh nằm hoàn toàn ngoài dữ liệu.")
        val = pct(a, b)
        return unanswerable("Kỳ so sánh có doanh thu bằng 0 nên không xác định được phần trăm.") if val is None else out("percent", val)
    if any(x in s for x in ("so voi thang truoc", "thang lien truoc", "thang lien trc", "thang truoc", "cung so ngay")) and (requests_percent(q,s) or "tang hay giam" in s):
        y, m = month_months_for_change(q, asof)
        curstart = month_start(y, m); curend = min(asof, month_end(y, m)) if any(x in s for x in ("thang nay", "thang ni", "th nay", "th ni")) else month_end(y, m)
        py, pm = shift_month(y, m, -1)
        prevstart = month_start(py, pm)
        prevend = min(month_end(py, pm), prevstart + timedelta(days=(curend - curstart).days))
        a = S5.revenue(curstart, curend); b = S5.revenue(prevstart, prevend)
        if a is None or b is None:
            return unanswerable("Một trong hai kỳ so sánh nằm hoàn toàn ngoài dữ liệu.")
        val = pct(a, b)
        return unanswerable("Kỳ so sánh có doanh thu bằng 0 nên không xác định được phần trăm.") if val is None else out("percent", val)
    if any(x in s for x in ("tang lien tuc", "thang sau deu cao hon", "tang lien tuc khong")):
        if len(months) < 2 and months:
            months = [shift_month(*months[0], d) for d in (-2, -1, 0)]
        if len(months) >= 2:
            vals = []
            for y, m in months:
                v = S5.revenue(month_start(y, m), month_end(y, m))
                if v is None: return unanswerable("Một tháng được hỏi nằm hoàn toàn ngoài dữ liệu.")
                vals.append(v)
            return out("boolean", all(b > a for a, b in zip(vals, vals[1:])))
    if ("thang nao" in s or "thang nao cao" in s) and len(months) >= 2:
        vals = []
        for y, m in months:
            v = S5.revenue(month_start(y, m), month_end(y, m))
            if v is None: return unanswerable("Một tháng được hỏi nằm hoàn toàn ngoài dữ liệu.")
            vals.append((v, y, m))
        best = sorted(vals, key=lambda z: (-z[0], z[1], z[2]))[0]
        return out("date", f"{best[1]:04d}-{best[2]:02d}")
    if ("lan luot" in s or "thu tu" in s or "tung thang" in s or len(months) >= 3) and len(months) >= 2:
        vals=[]
        for y,m in months:
            v=S5.revenue(month_start(y,m),month_end(y,m))
            if v is None:return unanswerable("Một tháng được hỏi nằm hoàn toàn ngoài dữ liệu.")
            vals.append(money(v))
        return out("list", vals)

    # Determine target date range. Month series uses the first period for single metrics.
    start, end = get_sales_window(q, asof)
    whole = S5.overlap(start, end)
    if whole is None:
        return unanswerable("Kỳ được hỏi nằm hoàn toàn ngoài khoảng dữ liệu dev_005 (2025-01-01–2026-05-10).")
    start, end = whole

    # Cancellation rate/count.
    if "huy" in s or "huy" in q.lower():
        all_orders = S5.orders_between(start, end, eligible=False)
        if all_orders is None: return unanswerable("Kỳ được hỏi nằm ngoài dữ liệu.")
        cancelled = all_orders[all_orders.status.eq("cancelled")]
        if "%" in q or "phan tram" in s or "chiam" in s:
            val = 0.0 if len(all_orders) == 0 else float(rnd(len(cancelled) / len(all_orders) * 100, 2))
            return out("percent", val)
        return out("number", int(len(cancelled)))

    # Product ranking / product revenue / item-specific sales.
    item = S5.item_row(q)
    if item and any(x in s for x in ("nhieu tien", "thu ve", "ban duoc", "ban duoc bao nhieu", "tien ban", "ban bao nhieu", "ban duoc may", "bao nhieu tien", "bnhiu", " may ", " bn ")):
        df = S5.lines_between(start, end)
        if df is None: return unanswerable("Kỳ được hỏi nằm ngoài dữ liệu.")
        df = df[df.product_key.eq(norm(item.item))]
        target_requested=requested_unit(q)
        if target_requested or any(x in s for x in ("bao nhieu vien", "bao nhieu lo", "bao nhieu chai", "bao nhieu goi", "bao nhieu hop", "ban ra bnhiu vien", "ban duoc may chai", "ban duoc bnhiu vien")):
            target_unit = target_requested or item.base_unit
            factor = S5.unit_factor.get(norm(item.item), {}).get(norm(target_unit), 1.0 if norm(target_unit) == norm(item.base_unit) else None)
            if factor is None:
                return unanswerable("Không có hệ số đổi sang đơn vị được hỏi trong danh mục quy đổi.")
            return out("number", money(df.base_qty.sum() / factor))
        return out("number", money(df.amount.sum()))

    # Gross revenue categories and prescription/nonprescription/active ingredient.
    df = S5.lines_between(start, end)
    if df is None: return unanswerable("Kỳ được hỏi nằm ngoài dữ liệu.")
    is_non_rx = any(k in s for k in ("khong ke don", "ko ke don", "k ke don", "khong can don", "ko can don", "k can don"))
    if "ke don" in s or is_non_rx:
        rx_target = not is_non_rx
        amount = float(df.loc[df.rx.eq(rx_target), "amount"].sum())
        if ("chiem" in s or "phan tram" in s or "%" in q) and rx_target:
            total = float(df.amount.sum())
            return out("percent", pct_share(amount, total)) if total else out("unanswerable", reason="Kỳ không có doanh thu để làm mẫu số.")
        return out("number", money(amount))
    # Exact active ingredient; combination is treated as one complete label.
    if "hoat chat" in s:
        ing = None
        for candidate in sorted(set(S5.lines.ingredient), key=len, reverse=True):
            if norm(candidate) and norm(candidate) in s:
                ing = norm(candidate); break
        if not ing:
            return unanswerable("Không xác định được hoạt chất trong danh mục thuốc.")
        amount = float(df.loc[df.ingredient.map(norm).eq(ing), "amount"].sum())
        return out("number", money(amount))
    # Group/category money share.
    cat_match = None
    for cat in sorted(set(S5.prod.category), key=len, reverse=True):
        if norm(cat) and norm(cat) in s:
            cat_match = norm(cat); break
    if cat_match:
        if any(x in s for x in ("bao nhieu don", "may don", "bnhiu don", "so don")):
            orders_count = df[df.category.map(norm).eq(cat_match)]["order_id"].nunique()
            return out("number", int(orders_count))
        amount = float(df.loc[df.category.map(norm).eq(cat_match), "amount"].sum())
        if "chiem" in s or "ty trong" in s or "phan tram" in s or "%" in q:
            val = pct_share(amount, float(df.amount.sum()))
            return out("percent", val) if val is not None else unanswerable("Kỳ không có doanh thu để làm mẫu số.")
        # highest-revenue item within the category
        if any(x in s for x in ("mat hang nao", "san pham nao")):
            grouped = df[df.category.map(norm).eq(cat_match)].groupby(["sku", "item"], as_index=False).amount.sum()
            if grouped.empty: return unanswerable("Không có mặt hàng thuộc nhóm này trong kỳ.")
            row = grouped.sort_values(["amount", "sku"], ascending=[False, True]).iloc[0]
            return out("string", row["item"])
        return out("number", money(amount))

    # Quantity by base unit across all products.
    unit_terms = {"goi": "gói", "vien": "viên", "lo": "lọ", "chai": "chai", "mieng": "miếng", "hop": "hộp", "vi": "vỉ"}
    matched_unit = next((v for k,v in unit_terms.items() if re.search(rf"\b{k}\b", s)), None)
    if ("ban duoc tong cong" in s or "bao nhieu " in s) and matched_unit and ("goi" in s or "mieng" in s):
        base_u = norm(matched_unit)
        val = float(df.loc[df.base_unit.map(norm).eq(base_u), "base_qty"].sum())
        return out("number", money(val))

    # Customer frequency and top customers.
    if any(x in s for x in ("khach", "ai mua", "nguoi nao mua", "ai chi")) and any(x in s for x in ("don", "mua nhieu tien", "chi nhieu tien", "tieu nhieu", "nhieu don")):
        if "bao nhieu %" in s or "may %" in s or "phan tram" in s or "%" in q:
            threshold = re.search(r"(?:it nhat|tu)\s*(\d+)\s*(?:don|dh)", s)
            if threshold:
                n = int(threshold.group(1)); agg = S5.customer_aggregate(start, end, "orders")
                if agg is None:return unanswerable("Kỳ được hỏi nằm ngoài dữ liệu.")
                total = len(agg); count = int((agg.value >= n).sum())
                return out("percent", float(rnd(count / total * 100, 2))) if total else out("unanswerable", reason="Kỳ không có khách mua để làm mẫu số.")
        threshold = re.search(r"(?:it nhat|tu)\s*(\d+)\s*(?:don|dh)", s)
        if threshold:
            n=int(threshold.group(1)); agg=S5.customer_aggregate(start,end,"orders")
            if agg is None:return unanswerable("Kỳ được hỏi nằm ngoài dữ liệu.")
            return out("number", int((agg.value>=n).sum()))
        if "top 5" in s or bool(re.search(r"\b5\s*khach\b", s)):
            agg=S5.customer_aggregate(start,end,"revenue")
            if agg is None:return unanswerable("Kỳ được hỏi nằm ngoài dữ liệu.")
            top=agg.sort_values(["value","tie"],ascending=[False,True]).head(5)
            return out("list", list(top.customer))
        agg=S5.customer_aggregate(start,end,"orders" if any(x in s for x in ("nhieu don", "nhiu don", "so don", "may don")) else "revenue")
        if agg is None:return unanswerable("Kỳ được hỏi nằm ngoài dữ liệu.")
        if agg.empty:return unanswerable("Kỳ không có khách đã mua.")
        row=agg.sort_values(["value","tie"],ascending=[False,True]).iloc[0]
        return out("string", row.customer)

    # Top sellers by normalized base-unit quantity.
    if any(x in s for x in ("ban chay", "san pham nao ngon nhat", "ban nhieu nhat")):
        grouped=df.groupby(["sku","item"],as_index=False).base_qty.sum()
        topn=5 if "top 5" in s or bool(re.search(r"\b5\s*(?:mat hang|san pham)\b", s)) else 1
        grouped=grouped.sort_values(["base_qty","sku"],ascending=[False,True]).head(topn)
        if topn==1:
            return out("string", grouped.iloc[0]["item"]) if len(grouped) else unanswerable("Kỳ không có giao dịch mặt hàng.")
        return out("list", list(grouped["item"]))

    # Day questions and order thresholds.
    if any(x in s for x in ("ngay nao", "ngaynao", "hom nao", "homnao")) and any(x in s for x in ("cao nhat", "nhta", "nhieu tien", "dthu", "doanh thu")):
        d,_=max_daily_revenue(S5,start,end)
        return out("date", d.isoformat()) if d else unanswerable("Kỳ không có đơn bán.")
    threshold=money_threshold(q)
    if threshold is not None and ("ngay" in s or "bua" in s):
        lim=threshold; daily=S5.lines_between(start,end)
        if daily is None:return unanswerable("Kỳ nằm ngoài dữ liệu.")
        byday=daily.groupby("date").amount.sum()
        # Only days within requested calendar period are candidates.
        count=sum(v>=lim for v in byday.values)
        return out("number", int(count))
    if threshold is not None and ("don" in s or "dh" in s):
        lim=threshold; od=S5.orders_between(start,end,True)
        if od is None:return unanswerable("Kỳ nằm ngoài dữ liệu.")
        return out("number", int((od.revenue>=lim).sum()))

    # Order count and average order revenue.
    if "don" in s or "dh" in s:
        od=S5.orders_between(start,end,True)
        if od is None:return unanswerable("Kỳ nằm ngoài dữ liệu.")
        if "trung binh" in s or "binh quan" in s or "moi don" in s or "1 don" in s or "gia tri trung binh" in s:
            return out("number", money(od.revenue.mean())) if len(od) else out("number",0)
        return out("number", int(len(od)))

    # Gross revenues default.
    if "doanh thu" in s or "dthu" in s or "tien ban" in s or "tien hang" in s or ("ban duoc" in s and ("bao nhieu" in s or "bnhiu" in s or "ten" in s)) or ("ban" in s and ("tien" in s or "bnhiu" in s or "bao nhieu" in s)):
        v=S5.revenue(start,end)
        return out("number", money(v)) if v is not None else unanswerable("Kỳ nằm ngoài dữ liệu.")

    return out("abstain")


def answer006(qrow):
    q, asof = qrow["question"], date.fromisoformat(qrow["as_of_date"])
    s = norm(q)
    if "ba bau" in s or "mang thai" in s:
        return unanswerable("Dữ liệu thuốc không ghi hướng dẫn sử dụng cho người mang thai.")
    if "khach nu" in s or "khach hang nu" in s:
        return unanswerable("Nguồn dev_006 không ghi giới tính khách hàng (cột NamNu trống).")
    if (re.search(r"\b(?:tang gia|len gia|gia len)\b", s) or ("tu khi" in s and "ban trong" in s)) and any(x in s for x in ("ngay", "ban")):
        result=change_from_price(S6,q,asof)
        if result is None:return unanswerable("Không tìm thấy mốc tăng giá và hệ số đổi đơn vị cần thiết trong dữ liệu phát hành.")
        return out("percent",result[0])

    # Calendar-based Lunar New Year.
    if "tet" in s and "2026" in s:
        cal=pd.read_csv(DATA/"calendar.csv")
        start_row=cal[(cal.lunar_day==23)&(cal.lunar_month==12)&(cal.lunar_year==2025)].iloc[0]
        end_row=cal[(cal.lunar_day==10)&(cal.lunar_month==1)&(cal.lunar_year==2026)].iloc[0]
        ts=date.fromisoformat(start_row.date); te=date.fromisoformat(end_row.date)
        if "ban chay nhat" in s or "sp nao ban chay" in s:
            df=S6.lines_between(ts,te)
            if df is None:return unanswerable("Kỳ Tết nằm ngoài dữ liệu.")
            g=df.groupby(["MaHangHoa","name"],as_index=False).base_qty.sum().sort_values(["base_qty","MaHangHoa"],ascending=[False,True])
            return out("string",g.iloc[0]["name"]) if len(g) else unanswerable("Không có đơn bán trong dịp Tết.")
        # The explicit 17-day pre-Tết comparison refers to the 17 days before the 23rd lunar day.
        pre_start=ts-timedelta(days=17); pre_end=ts-timedelta(days=1)
        a=S6.lines_between(ts,te); b=S6.lines_between(pre_start,pre_end)
        if a is None or b is None:return unanswerable("Một kỳ so sánh Tết nằm hoàn toàn ngoài dữ liệu.")
        val=pct(float(a.amount.sum()),float(b.amount.sum()))
        return out("percent",val) if val is not None else unanswerable("Kỳ trước Tết có doanh thu bằng 0.")

    # Expiration and inventory questions.
    if "lo gan het han" in s or "can date nhut" in s or "gan het han nhat" in s:
        prod=S6.item_row(q)
        if not prod:return unanswerable("Không nhận dạng được thuốc trong danh mục.")
        lots=S6.lots[(S6.lots["product"].eq(text(prod.MaHangHoa))) & (S6.lots.remaining>0) & (S6.lots.expiry>=asof)]
        if lots.empty:return unanswerable("Không có lô còn tồn và chưa quá hạn của thuốc tại ngày hỏi.")
        d=min(lots.expiry)
        return out("date",d.isoformat())
    if "tien von" in s and ("ngay toi" in s or "den het" in s or "han" in s or "dong" in s):
        dates=date_tokens(q)
        if len(dates)>=2:
            start,end=dates[0],dates[1]
        elif len(dates)==1:
            start,end=asof,dates[0]
        else:
            nmatch=re.search(r"(\d+)\s*ngay",s); n=int(nmatch.group(1)) if nmatch else 180
            start,end=asof,asof+timedelta(days=n)
        lots=S6.lots[(S6.lots.remaining>0)&(S6.lots.expiry>=start)&(S6.lots.expiry<=end)]
        value=(lots.remaining*lots.base_cost).sum()
        return out("number",money(value))
    if "lo" in s and ("het han" in s or "han dung" in s) and ("bn" in s or "bao nhieu" in s):
        nmatch=re.search(r"(\d+)\s*ngay",s); n=int(nmatch.group(1)) if nmatch else 90
        end=asof+timedelta(days=n)
        lots=S6.lots[(S6.lots.remaining>0)&(S6.lots.expiry>=asof)&(S6.lots.expiry<=end)]
        return out("number",int(len(lots)))
    if (("sp nao" in s or "nhung sp nao" in s or "cac sp" in s) and any(x in s for x in ("hsd", "han dung", "het han", "lo ton"))):
        dates=date_tokens(q)
        start=dates[0] if len(dates)>=1 else asof
        nmatch=re.search(r"(\d+)\s*ngay",s)
        end=dates[1] if len(dates)>=2 else start+timedelta(days=int(nmatch.group(1)) if nmatch else 90)
        lots=S6.lots[(S6.lots.remaining>0)&(S6.lots.expiry>=start)&(S6.lots.expiry<=end)]
        if lots.empty:return out("list",[])
        order=lots.groupby(["product","name"],as_index=False).expiry.min().sort_values(["expiry","product"])
        return out("list",list(order["name"]))
    if ("ton kho" in s or "ton trong kho" in s) and ("gia von" in s or "tinh gia" in s):
        d=date_tokens(q)[0] if date_tokens(q) else asof
        stock=S6.stock_asof(d).merge(S6.products[["MaHangHoa","base_cost"]],on="MaHangHoa",how="left")
        return out("number",money((stock.balance*stock.base_cost).fillna(0).sum()))
    if any(x in s for x in ("het hang", "chay hang", "dut hang")):
        d=date_tokens(q)[0] if date_tokens(q) else asof
        stock=S6.stock_asof(d)
        # Only products that had stock placed on the ledger by this date can be out of stock.
        prev=S6.stock[S6.stock.date<=d]
        purchased=prev[prev.LoaiGiaoDich.map(norm).str.contains("purchase|nhap",regex=True)].MaHangHoa.map(text).unique() if "LoaiGiaoDich" in prev else []
        had=set(purchased)
        if not had:
            # Use purchase receipts as an equivalent indication of items previously stocked.
            rec=S6.tables["phieu_nhap"]
            rc=S6.tables["chi_tiet_nhap"].merge(rec[["MaPhieuNhap","NgayNhapKho"]],on="MaPhieuNhap",how="left")
            rc["day"]=rc.NgayNhapKho.map(lambda x: dt(x))
            had=set(rc.loc[rc.day<=d,"MaHangHoa"].map(text))
        return out("number",int((stock.MaHangHoa.map(text).isin(had)&stock.balance.eq(0)).sum()))

    # Purchase invoices, suppliers.
    months=expand_month_sequence(q,asof)
    period=period_for_question(q,asof)
    start,end=period
    if re.search(r"\b(?:huy|huy don|don huy|bi huy)\b", s):
        actual=S6.overlap(start,end)
        if actual is None:return unanswerable("Kỳ hủy đơn nằm ngoài dữ liệu.")
        all_orders=S6.inv[(S6.inv.date>=actual[0])&(S6.inv.date<=actual[1])]
        cancelled=all_orders[all_orders.status.eq("cancelled")]
        if "%" in q or "phan tram" in s:
            return out("percent",float(rnd(len(cancelled)/len(all_orders)*100,2))) if len(all_orders) else out("number",0)
        return out("number",int(len(cancelled)))
    if "uy tin" in s and ("nha cung cap" in s or "ncc" in s):
        return unanswerable("Nguồn chỉ ghi phiếu nhập và số tiền, không ghi đánh giá độ uy tín nhà cung cấp.")
    if "nhap" in s or "ncc" in s or "nha cung cap" in s or "cat hang" in s:
        actual=S6.overlap(start,end)
        if actual is None:return unanswerable("Kỳ nhập hàng nằm ngoài dữ liệu.")
        pur=S6.purchases[(S6.purchases.date>=actual[0])&(S6.purchases.date<=actual[1])&~S6.purchases.TinhTrang.map(canon_status).eq("cancelled")]
        if "ncc nao" in s or "nha cung cap nao" in s or "nha cung cap" in s:
            metric="amount" if "nhieu tien" in s else "MaPhieuNhap"
            grp=pur.groupby(["MaNhaCungCap","supplier"],as_index=False)[metric].sum() if metric=="amount" else pur.groupby(["MaNhaCungCap","supplier"],as_index=False).MaPhieuNhap.nunique()
            if grp.empty:return unanswerable("Kỳ không có phiếu nhập.")
            valuecol=metric if metric=="amount" else "MaPhieuNhap"
            best=grp.sort_values([valuecol,"MaNhaCungCap"],ascending=[False,True]).iloc[0]
            return out("string",best.supplier)
        if ("tien hang" in s or "nhap hang" in s or "nhap vo" in s or "ton kho" in s or "cat hang" in s) and not S6.item_row(q):
            return out("number",money(pur.amount.sum()))

    if "nhan vien" in s or "nhan vien" in s:
        staff_name = next((name for _, name in sorted(S6.employee_name.items()) if name and norm(name) in s), None)
        if staff_name:
            employee_lines=S6.lines_between(start,end)
            if employee_lines is None:return unanswerable("Kỳ nằm ngoài dữ liệu.")
            employee_lines=employee_lines[employee_lines.employee.map(norm).eq(norm(staff_name))]
            return out("number",money(employee_lines.amount.sum()))

    # Explicit products and stock movement quantity.
    prod=S6.item_row(q)
    if prod and any(x in s for x in ("bao nhieu", "bnhiu", "bn ", " may ", "chênh", "chenh", "tru so", "tong goi", "nhap ve")):
        code=text(prod.MaHangHoa)
        if "nhap" in s and "ban" in s and ("tru" in s or "chenh" in s):
            actual=S6.overlap(start,end)
            if actual is None:return unanswerable("Kỳ nằm ngoài dữ liệu.")
            recv=S6.tables["lo_hang"].merge(S6.purchases[["MaPhieuNhap","date"]],on="MaPhieuNhap",how="left")
            recv=recv[(recv.MaHangHoa.map(text)==code)&(recv.date>=actual[0])&(recv.date<=actual[1])]
            bought=float(recv.SLNhapQuyDoi.map(num).sum())
            sold=S6.eligible[(S6.eligible.MaHangHoa.map(text)==code)&(S6.eligible.date>=actual[0])&(S6.eligible.date<=actual[1])]
            return out("number",money(bought-float(sold.base_qty.sum())))
        if "nhap" in s:
            actual=S6.overlap(start,end)
            if actual is None:return unanswerable("Kỳ nhập hàng nằm ngoài dữ liệu.")
            recv=S6.tables["lo_hang"].merge(S6.purchases[["MaPhieuNhap","date"]],on="MaPhieuNhap",how="left")
            recv=recv[(recv.MaHangHoa.map(text)==code)&(recv.date>=actual[0])&(recv.date<=actual[1])]
            target_unit=requested_unit(q) or prod.base_unit
            factor=S6.unit_factor.get((code,norm(target_unit)),1.0 if norm(target_unit)==norm(prod.base_unit) else None)
            if factor is None:return unanswerable("Không có hệ số đổi sang đơn vị được hỏi.")
            return out("number",money(recv.SLNhapQuyDoi.map(num).sum()/factor))
        if "ban" in s or "ban ra" in s:
            df=S6.lines_between(start,end)
            if df is None:return unanswerable("Kỳ nằm ngoài dữ liệu.")
            target_unit=requested_unit(q) or prod.base_unit
            val=S6.quantity_for_unit(df,code,target_unit)
            if val is None:return unanswerable("Không có hệ số đổi sang đơn vị được hỏi.")
            return out("number",money(val))

    # Debts.
    if "no" in s or "cong no" in s or "tra no" in s or "no them" in s or "con no" in s or "thieu tien" in s:
        day=date_tokens(q)[0] if date_tokens(q) else asof
        # In-question dates are at the time requested; otherwise use as_of_date.
        debt=S6.debt_ledger(day)
        if "no them" in s or "tra no" in s or "tra bot no" in s or "doi duoc" in s:
            # Only debt-method orders; sales on the purchase date are not repayment.
            inv=S6.inv[(S6.inv.status.ne("cancelled"))&(S6.inv.method.map(norm).eq("debt"))]
            name_match=None
            for cid,name in S6.customer_name.items():
                if name and norm(name) in s:
                    name_match=cid; break
            if name_match:
                inv=inv[inv.customer_id.eq(name_match)]
            if len(months):
                start,end=month_start(*months[0]),month_end(*months[0])
            else:
                start,end=period
            actual=S6.overlap(start,end)
            if actual is None:return unanswerable("Kỳ công nợ nằm ngoài dữ liệu.")
            inv=inv[(inv.date>=actual[0])&(inv.date<=actual[1])]
            pay=S6.payments.merge(S6.inv[["MaHoaDon","date","status","method"]].rename(columns={"date":"order_date"}),on="MaHoaDon",how="left")
            pay=pay[(pay.method.map(norm).eq("debt"))&pay.status.ne("cancelled")]
            due_add=0.0; repayment=0.0
            for _,ir in inv.iterrows():
                p=pay[pay.MaHoaDon.eq(ir.MaHoaDon)]
                same_day=float(p.loc[p.date.eq(ir.date),"amount"].sum())
                due_add += max(0.0,float(ir.total_due)-same_day)
                repayment += float(p.loc[(p.date>ir.date)&(p.date>=actual[0])&(p.date<=actual[1]),"amount"].sum())
            asks_add="no them" in s
            asks_repay=any(x in s for x in ("tra no","tra bot no","doi duoc","thu no"))
            if asks_add and asks_repay:return out("list",[money(due_add),money(repayment)])
            if asks_add:return out("number",money(due_add))
            return out("number",money(repayment))
        if "top 3" in s or "3 khach" in s:
            grp=debt[debt.customer_id.ne("")].groupby(["customer_id","customer"],as_index=False).outstanding.sum()
            grp=grp.sort_values(["outstanding","customer_id"],ascending=[False,True]).head(3)
            return out("table",{"columns":["khach_hang","so_tien_no"],"rows":[[r.customer,money(r.outstanding)] for r in grp.itertuples()]})
        if "tuoi no" in s or "tuoi qua" in s or "qua" in s and "ngay" in s:
            if "0 7 ngay" in s or "0 7" in s or "8 15" in s:
                vals=[debt.loc[debt.age.between(0,7),"outstanding"].sum(),debt.loc[debt.age.between(8,15),"outstanding"].sum(),debt.loc[debt.age>15,"outstanding"].sum()]
                return out("list",[money(x) for x in vals])
            m=re.search(r"(?:qua|hon)\s*(\d+)\s*ngay",s)
            limit=int(m.group(1)) if m else 0
            return out("number",money(debt.loc[debt.age>limit,"outstanding"].sum()))
        if any(x in s for x in ("bao nhieu nguoi", "bao nhieu khach", "may khach", "con may khach", "bnhiu khach", "con bnhiu khach", "may nguoi", "bnhiu nguoi", "so khach")):
            cust=debt[debt.customer_id.ne("")].groupby("customer_id").outstanding.sum()
            return out("number",int((cust>0).sum()))
        if "top" in s or "nhieu nhat" in s:
            grp=debt[debt.customer_id.ne("")].groupby(["customer_id","customer"],as_index=False).outstanding.sum().sort_values(["outstanding","customer_id"],ascending=[False,True])
            return out("table",{"columns":["khach_hang","so_tien_no"],"rows":[[r.customer,money(r.outstanding)] for r in grp.head(3).itertuples()]})
        if "bao nhieu tien" in s or "bnhiu" in s or "tong tien" in s or "con bn" in s or "con lai" in s:
            return out("number",money(debt.outstanding.sum()))

    # Gross profit.
    if "lai gop" in s or "lai nhat" in s or "lai nhieu nhat" in s or "loi nhuan gop" in s:
        df=S6.lines_between(start,end)
        if df is None:return unanswerable("Kỳ nằm ngoài dữ liệu.")
        if "sp nao" in s or "hang mo" in s or "mat hang" in s:
            g=df.assign(profit=df.amount-df.cost).groupby(["MaHangHoa","name"],as_index=False).profit.sum().sort_values(["profit","MaHangHoa"],ascending=[False,True])
            return out("string",g.iloc[0]["name"]) if len(g) else unanswerable("Kỳ không có đơn bán.")
        return out("number",money((df.amount-df.cost).sum()))

    # Expiration inventory value is already handled. Purchase and sales by categories.
    df=S6.lines_between(start,end)
    if df is None:
        return unanswerable("Kỳ được hỏi nằm hoàn toàn ngoài khoảng dữ liệu dev_006 (2025-04-01–2026-08-08).")

    is_non_rx = any(k in s for k in ("khong ke don", "ko ke don", "k ke don", "khong can don", "ko can don", "k can don"))
    if "thuoc ke don" in s or "ke don" in s or is_non_rx:
        rx = not is_non_rx
        amount=float(df.loc[df.rx.eq(rx),"amount"].sum())
        if "phan tram" in s or "%" in q or "may %" in s:
            val=pct_share(amount,float(df.amount.sum()))
            return out("percent",val) if val is not None else unanswerable("Kỳ không có doanh thu để làm mẫu số.")
        return out("number",money(amount))

    # Best-selling products and quantities requested in a base unit.
    if any(x in s for x in ("ban chay", "ban nhieu nhat", "hot nhat")):
        grouped=df.groupby(["MaHangHoa","name"],as_index=False).base_qty.sum().sort_values(["base_qty","MaHangHoa"],ascending=[False,True])
        topn=5 if "top 5" in s or "5 mat hang" in s or "5 san pham" in s else (3 if "top 3" in s or "3 sp" in s else 1)
        if not len(grouped):return unanswerable("Kỳ không có đơn bán.")
        return out("string",grouped.iloc[0]["name"]) if topn==1 else out("list",list(grouped.head(topn)["name"]))
    if any(x in s for x in ("ban duoc tong cong bao nhieu", "ban duoc bao nhieu", "ban ra bnhiu")):
        unit_map={"goi":"gói","vien":"viên","lo":"lọ","chai":"chai","mieng":"miếng","hop":"hộp","vi":"vỉ"}
        requested=next((v for k,v in unit_map.items() if re.search(rf"\b{k}\b",s)),None)
        if requested:
            basekey=norm(requested)
            qty=float(df.loc[df.base_unit.map(norm).eq(basekey),"base_qty"].sum())
            return out("number",money(qty))
    if "hoat chat" in s:
        ing=next((norm(x) for x in sorted(set(df.ingredient),key=len,reverse=True) if norm(x) and norm(x) in s),None)
        if ing is None:return unanswerable("Không tìm thấy hoạt chất trong danh mục thuốc.")
        return out("number",money(df.loc[df.ingredient.map(norm).eq(ing),"amount"].sum()))

    # Date/quarter/monthly revenue comparison/list.
    months=expand_month_sequence(q,asof)
    if ("cung ky" in s or "nam ngoai" in s or "cung thang nam truoc" in s) and "thang truoc" not in s:
        y,m=month_months_for_change(q,asof)
        a=S6.lines_between(month_start(y,m),month_end(y,m)); b=S6.lines_between(month_start(y-1,m),month_end(y-1,m))
        if a is None or b is None:return unanswerable("Một trong hai kỳ so sánh nằm ngoài dữ liệu.")
        val=pct(a.amount.sum(),b.amount.sum())
        return out("percent",val) if val is not None else unanswerable("Kỳ so sánh có doanh thu bằng 0.")
    if len(months)==2 and months[0][1]==months[1][1] and months[0][0]!=months[1][0] and any(x in s for x in ("so voi","tang hay giam","tang giam")):
        a=S6.lines_between(month_start(*months[0]),month_end(*months[0]))
        b=S6.lines_between(month_start(*months[1]),month_end(*months[1]))
        if a is None or b is None:return unanswerable("Một trong hai kỳ so sánh nằm ngoài dữ liệu.")
        val=pct(a.amount.sum(),b.amount.sum())
        return out("percent",val) if val is not None else unanswerable("Kỳ so sánh có doanh thu bằng 0.")
    if ("so voi thang truoc" in s or "thang lien truoc" in s or "thang truoc" in s) and "cung so ngay" not in s:
        y,m=month_months_for_change(q,asof); py,pm=shift_month(y,m,-1)
        a=S6.lines_between(month_start(y,m),month_end(y,m));b=S6.lines_between(month_start(py,pm),month_end(py,pm))
        if a is None or b is None:return unanswerable("Một trong hai kỳ so sánh nằm ngoài dữ liệu.")
        val=pct(a.amount.sum(),b.amount.sum())
        return out("percent",val) if val is not None else unanswerable("Kỳ so sánh có doanh thu bằng 0.")
    if "cung so ngay" in s:
        y,m=asof.year,asof.month; py,pm=shift_month(y,m,-1)
        n=asof.day
        a=S6.lines_between(month_start(y,m),min(asof,month_end(y,m)))
        b=S6.lines_between(month_start(py,pm),min(month_end(py,pm),month_start(py,pm)+timedelta(days=n-1)))
        if a is None or b is None:return unanswerable("Một trong hai kỳ so sánh nằm ngoài dữ liệu.")
        val=pct(a.amount.sum(),b.amount.sum())
        return out("percent",val) if val is not None else unanswerable("Kỳ so sánh có doanh thu bằng 0.")
    if any(x in s for x in ("tang lien tuc", "thang sau deu cao hon", "tang lien tuc khong")):
        if len(months)==1 and "3 thang" in s: months=[shift_month(*months[0],d) for d in (-2,-1,0)]
        vals=[]
        for y,m in months:
            x=S6.lines_between(month_start(y,m),month_end(y,m))
            if x is None:return unanswerable("Một tháng được hỏi nằm ngoài dữ liệu.")
            vals.append(float(x.amount.sum()))
        if len(vals)>1:return out("boolean",all(b>a for a,b in zip(vals,vals[1:])))
    if ("thang nao" in s or "thang nao cao" in s) and len(months)>=2:
        vals=[]
        for y,m in months:
            x=S6.lines_between(month_start(y,m),month_end(y,m))
            if x is None:return unanswerable("Một tháng được hỏi nằm ngoài dữ liệu.")
            vals.append((float(x.amount.sum()),y,m))
        best=sorted(vals,key=lambda z:(-z[0],z[1],z[2]))[0]
        return out("date",f"{best[1]:04d}-{best[2]:02d}")
    if ("lan luot" in s or "thu tu" in s or "tung thang" in s or "tung thang bao nhieu" in s or len(months) >= 3) and len(months)>=2:
        vals=[]
        for y,m in months:
            x=S6.lines_between(month_start(y,m),month_end(y,m))
            if x is None:return unanswerable("Một tháng được hỏi nằm ngoài dữ liệu.")
            vals.append(money(x.amount.sum()))
        return out("list",vals)

    # Day revenue / daily peak / threshold.
    if "ngay nao" in s and ("dthu" in s or "nhieu tien" in s or "cao nhat" in s):
        d,_=max_daily_revenue(S6,start,end)
        return out("date",d.isoformat()) if d else unanswerable("Kỳ không có doanh thu.")
    if "ca chieu" in s and "ca sang" in s:
        all_lines=S6.lines_between(start,end)
        if all_lines is None:return unanswerable("Kỳ nằm ngoài dữ liệu.")
        hours=all_lines.time.dt.hour
        afternoon=float(all_lines.loc[hours.between(14,21),"amount"].sum())
        morning=float(all_lines.loc[hours.between(6,13),"amount"].sum())
        val=pct(afternoon,morning)
        return out("percent",val) if val is not None else unanswerable("Ca sáng không có doanh thu để làm mẫu số.")
    mt=money_threshold(q)
    if mt is not None and "ngay" in s:
        amount=mt; daily=df.groupby("date").amount.sum()
        return out("number",int((daily>=amount).sum()))
    if mt is not None and ("don" in s or "hoa don" in s):
        bills=df.groupby("MaHoaDon").amount.sum()
        return out("number",int((bills>=mt).sum()))
    if "bao nhieu don" in s or "may don" in s or "tong bnhiu dh" in s:
        od=S6.orders_between(start,end,True)
        if od is None:return unanswerable("Kỳ nằm ngoài dữ liệu.")
        return out("number",int(len(od)))

    # Store/customer group/payment-mode shares and totals.
    if "nhom khach" in s or "nhom than thiet" in s or "khach than thiet" in s or "khach si" in s or "nhom si" in s or "khach le" in s or "nhom le" in s:
        group="thân thiết" if "than thiet" in s else ("sỉ" if ("khach si" in s or "nhom si" in s) else "lẻ")
        groupkey=norm(group)
        total=float(df.amount.sum())
        grp=float(df.loc[df.cust_type.map(norm).eq(groupkey),"amount"].sum())
        if any(x in s for x in ("binh quan", "bin quan", "trung binh", "trung bih", "moi don", "bq moi don")):
            gg=df[df.cust_type.map(norm).eq(groupkey)].groupby("MaHoaDon").amount.sum()
            return out("number",money(gg.mean())) if len(gg) else out("number",0)
        if "%" in q or "phan tram" in s or "chiem" in s:
            val=pct_share(grp,total)
            return out("percent",val) if val is not None else unanswerable("Kỳ không có doanh thu để làm mẫu số.")
        return out("number",money(grp))

    # Customer repeat rate, spend/amount averages, ranked customer.
    if "khach" in s and any(x in s for x in ("mua", "chi", "tieu", "sop nhat", "so nhat", "don")):
        od=S6.orders_between(start,end,True)
        if od is None:return unanswerable("Kỳ nằm ngoài dữ liệu.")
        customers=od[od.customer_id.ne("")]
        freq=customers.groupby("customer_id").agg(n=("MaHoaDon","nunique"),name=("customer","first"))
        threshold=re.search(r"(?:it nhat|tu)\s*(\d+)\s*(?:don|dh)",s)
        if threshold:
            n=int(threshold.group(1)); count=int((freq.n>=n).sum())
            if "phan tram" in s or "%" in q or "may %" in s:
                return out("percent",float(rnd(count/len(freq)*100,2))) if len(freq) else unanswerable("Kỳ không có khách mua.")
            return out("number",count)
        if "so nhat" in s or "sop nhat" in s:
            revs=S6.eligible[(S6.eligible.date>=start)&(S6.eligible.date<=end)&S6.eligible.customer_id.ne("")].groupby(["customer_id","customer"],as_index=False).agg(rev=("amount","sum"),orders=("MaHoaDon","nunique"))
            revs=revs[revs.orders>=3]
            if revs.empty:return unanswerable("Kỳ không có khách thoả mãn tiêu chí.")
            revs["avg"]=revs.rev/revs.orders
            row=revs.sort_values(["avg","customer_id"],ascending=[False,True]).iloc[0]
            return out("string",row.customer)
        if "top 3" in s or "3 khach" in s:
            rev=customers.groupby(["customer_id","customer"],as_index=False).MaHoaDon.nunique()
            sums=customers.groupby(["customer_id","customer"],as_index=False).apply(lambda g: float(g.MaHoaDon.map(S6.eligible.groupby("MaHoaDon").amount.sum()).sum()),include_groups=False)
            # Simpler stable revenue rollup from the already filtered line table.
            revs=S6.eligible[(S6.eligible.date>=start)&(S6.eligible.date<=end)&S6.eligible.customer_id.ne("")].groupby(["customer_id","customer"],as_index=False).amount.sum().sort_values(["amount","customer_id"],ascending=[False,True])
            return out("list",list(revs.head(3).customer))
        if "nhieu nhat" in s or re.search(r"(?<!bao |bnhiu )\bnhieu tien\b", s) or "chi nhieu" in s:
            metric="orders" if any(x in s for x in ("nhieu don", "nhiu don", "so don", "may don")) else "revenue"
            if metric=="orders":
                g=customers.groupby(["customer_id","customer"],as_index=False).MaHoaDon.nunique().rename(columns={"MaHoaDon":"value"})
            else:
                g=S6.eligible[(S6.eligible.date>=start)&(S6.eligible.date<=end)&S6.eligible.customer_id.ne("")].groupby(["customer_id","customer"],as_index=False).amount.sum().rename(columns={"amount":"value"})
            if g.empty:return unanswerable("Kỳ không có khách mua.")
            row=g.sort_values(["value","customer_id"],ascending=[False,True]).iloc[0]
            return out("string",row.customer)

    if "tai quay" in s or "kenh ban" in s:
        store=float(df.loc[df.channel.map(norm).eq("store"),"amount"].sum())
        val=pct_share(store,float(df.amount.sum()))
        return out("percent",val) if val is not None else unanswerable("Kỳ không có doanh thu.")
    pay_terms = [("xi o di", "cod"), ("cod", "cod"), ("vi dien tu", "ewallet"), ("chuyen khoan", "transfer"), ("tien mat", "cash"), ("cash", "cash")]
    found_methods = []
    for term, mcode in pay_terms:
        pos = s.find(term)
        if pos != -1 and mcode not in [m for _, m in found_methods]:
            found_methods.append((pos, mcode))
    if found_methods:
        found_methods.sort(key=lambda x: x[0])
        if "lan luot" in s:
            vals = [float(df.loc[df.method.map(norm).eq(mcode), "amount"].sum()) for _, mcode in found_methods]
            return out("list", [money(x) for x in vals])
        target_m = found_methods[0][1]
        return out("number", money(df.loc[df.method.map(norm).eq(target_m), "amount"].sum()))

    # Exact ingredient and category turnover.
    if "hoat chat" in s:
        ing=next((norm(x) for x in sorted(set(df.ingredient),key=len,reverse=True) if norm(x) and norm(x) in s),None)
        if ing is None:return unanswerable("Không tìm thấy hoạt chất trong danh mục thuốc.")
        return out("number",money(df.loc[df.ingredient.map(norm).eq(ing),"amount"].sum()))
    cat=next((norm(x) for x in sorted(set(S6.products.category),key=len,reverse=True) if norm(x) and norm(x) in s),None)
    if cat:
        amount=float(df.loc[df.category.map(norm).eq(cat),"amount"].sum())
        if requests_percent(q,s) or "chiem" in s or "ty trong" in s:
            val=pct_share(amount,float(df.amount.sum()))
            return out("percent",val) if val is not None else unanswerable("Kỳ không có doanh thu để làm mẫu số.")
        return out("number",money(amount))

    # Monthly average order value.
    if "trung binh" in s or "trung bih" in s or "binh quan" in s or "moi don" in s or "bq moi don" in s:
        od=S6.orders_between(start,end,True)
        if od is None:return unanswerable("Kỳ nằm ngoài dữ liệu.")
        amount=float(df.amount.sum()); count=od.MaHoaDon.nunique()
        return out("number",money(amount/count)) if count else out("number",0)

    # Revenue as default.
    if any(x in s for x in ("doanh thu","dthu","tien ban","tien hang","ban duoc","ban dc")) and any(x in s for x in ("tien","bnhiu","bao nhieu","nhieu tien")):
        return out("number",money(df.amount.sum()))
    return out("abstain")


def source_checks():
    """Compact reconciliation evidence for the two loaded pharmacy sources."""
    S=S6
    line_sums=S.lines.groupby("MaHoaDon").amount.sum().rename("line_goods")
    headers=S.inv[["MaHoaDon","TongTienHang"]].copy()
    headers["header_goods"]=headers.TongTienHang.map(num)
    rev=headers.merge(line_sums,left_on="MaHoaDon",right_index=True,how="left").fillna({"line_goods":0})
    rev["difference"]=rev.line_goods-rev.header_goods

    end=S.coverage_end
    inv=S.inv[(S.inv.date<=end)&S.inv.status.ne("cancelled")].copy()
    marketplace=inv.channel.map(lambda x:any(word in norm(x) for word in ("shopee","lazada","tiki","sendo","marketplace","san thuong mai")))
    inv=inv[(inv.method.map(norm)!="cod")&~marketplace]
    paid=S.payments[S.payments.date<=end].groupby("MaHoaDon").amount.sum()
    inv["calculated_debt"]=(inv.total_due-inv.MaHoaDon.map(paid).fillna(0)).clip(lower=0)
    inv["stored_debt"]=inv.ConPhaiThu.map(num)
    inv["debt_difference"]=inv.calculated_debt-inv.stored_debt

    ledger=S.stock_asof(end).groupby("MaHangHoa").balance.sum()
    lots=S.lots.groupby("product").remaining.sum()
    stock=pd.concat([ledger.rename("ledger"),lots.rename("lots")],axis=1).fillna(0)
    stock["difference"]=stock.ledger-stock.lots
    return {
        "dev_005": {
            "orders": int(len(S5.orders)),
            "detail_lines": int(len(S5.lines)),
            "eligible_revenue_vnd": money(S5.sales.amount.sum()),
            "coverage": [str(S5.coverage_start),str(S5.coverage_end)],
        },
        "dev_006": {
            "invoice_line_total_vs_header_vnd": {
                "invoices": int(len(rev)),
                "line_total": money(rev.line_goods.sum()),
                "header_total": money(rev.header_goods.sum()),
                "mismatch_count": int(rev.difference.ne(0).sum()),
                "sum_absolute_difference": money(rev.difference.abs().sum()),
            },
            "final_debt_vs_stored_vnd": {
                "invoices": int(len(inv)),
                "calculated": money(inv.calculated_debt.sum()),
                "stored": money(inv.stored_debt.sum()),
                "mismatch_count": int(inv.debt_difference.abs().gt(0.01).sum()),
            },
            "final_stock_ledger_vs_lots_units": {
                "products": int(len(stock)),
                "ledger_total": money(stock.ledger.sum()),
                "lot_total": money(stock.lots.sum()),
                "mismatch_products": int(stock.difference.ne(0).sum()),
                "total_absolute_difference": money(stock.difference.abs().sum()),
            },
        },
    }


def solve():
    answers = {}
    evidence = {}
    assigned = [q for q in QUESTIONS if q["business_id"] in {"dev_005", "dev_006"}]
    for q in assigned:
        result = answer005(q) if q["business_id"] == "dev_005" else answer006(q)
        answers[q["id"]] = result
        evidence[q["id"]] = {"business_id": q["business_id"], "question": q["question"], "answer": result}
    evidence["__qa__"] = source_checks()
    qa=evidence["__qa__"]["dev_006"]
    assert qa["invoice_line_total_vs_header_vnd"]["mismatch_count"] == 0
    assert qa["final_debt_vs_stored_vnd"]["mismatch_count"] == 0
    assert qa["final_stock_ledger_vs_lots_units"]["mismatch_products"] == 0
    return answers, evidence


if __name__ == "__main__":
    answers, evidence = solve()
    (HERE / "answers_pharmacy_pos.json").write_text(json.dumps(answers, ensure_ascii=False, indent=2) + "\n")
    (HERE / "evidence_pharmacy_pos.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    c = Counter((v["type"] for v in answers.values()))
    print(f"answered={len(answers)} dev_005={sum(q['business_id']=='dev_005' for q in QUESTIONS if q['id'] in answers)} dev_006={sum(q['business_id']=='dev_006' for q in QUESTIONS if q['id'] in answers)}")
    print("types", dict(c))
    print("unresolved", [(qid, v.get("reason")) for qid,v in answers.items() if v["type"] == "unanswerable"])
