"""Validate the export and record every change from the confirmed 0.998 file."""
import hashlib
import json
import os
import re
import sqlite3
from decimal import Decimal, ROUND_HALF_UP
from datetime import date, timedelta
from pathlib import Path

from run import validate_row

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'dataset/datamate').exists())
DATA = Path(os.environ.get('DATAMATE_DATA', ROOT / 'dataset/datamate/public_test'))
BASELINE_SHA = 'ab9c5547e9b5608c95c10d9aeb2af673bc724ccc2786c7926d68b2640e29d11b'

# Values independently calculated from released records, not the solver helpers.
CONFIRMED_PREVIOUS_FIXES = {
    'q_51f4751e4eb3': ('list', [12361000, 30121300],
                     'March 2026 eligible line revenue: COD, then ewallet; rulebook 1.'),
    'q_3b70619cf8cc': ('number', 77101300,
                     'April 2025 NV002: DISTINCT eligible detail sum and 758 invoice headers both give 77101300; rulebooks 1 and 8.'),
    'q_2d5eb3a196b7': ('number', 2294053,
                     'November 2025 receipts after purchase day: 989883 + 1304170 from older orders; rulebook 4.'),
    'q_ca9e9c233325': ('number', 3140750,
                     'September 2025 receipts after purchase day: 1434231 + 1706519 from older orders; rulebook 4.'),
    'q_fbaf2023cfb6': ('number', 1500,
                     'June 2025 received Tiffy lots: 300 + 6*200; explicit question year.'),
    'q_58215441e762': ('percent', 19.18,
                     '(May 2026 revenue 1285826500 / May 2025 revenue 1078858000 - 1) * 100.'),
    'q_82bbd52c42dc': ('number', 43875000,
                     'NV001 August revenue includes 213000 from in-transit orders; rulebook 1.'),
    'q_cda272637d58': ('number', 67001700,
                     'August bank-transfer revenue includes 600900 from in-transit orders; rulebook 1.'),
    'q_f539cf51f360': ('number', 37387100,
                     'NV004 August revenue includes 619800 from in-transit orders; rulebook 1.'),
    'q_fc4f06f3006b': ('percent', -6.43,
                     '(June 2025 revenue 272091800 / full May 2025 revenue 290789000 - 1) * 100; include May 31 revenue 9505800.'),
    'q_4df64734eb4d': ('number', 846,
                     'January 2025: 846 of 1694 valid bills have goods revenue >=410000, including 3 exactly equal; interpret transposed word tro lne as tro len.'),
}

EXPECTED_CHANGES = {
    'q_64e62d4bea7d': ('number', 102978,
                     'Previous Monday–Sunday week, 2026-02-23 through 2026-03-01: 57976800 VND / 563 valid orders, rounded to 102978; recognize tuan trc.'),
}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def check_pos_source():
    """Independent SQL joins; no production solver or money parser reused."""
    def receipt_money(value):
        value = str(value).strip().lower()
        if 'tr' in value:
            head, tail = value.split('tr', 1)
            amount = Decimal(head.replace(',', '.')) * 1000000
            if tail:
                amount += Decimal(tail) * 1000
        else:
            amount = Decimal(re.sub(r'[.đ\s]', '', value))
        assert amount == amount.to_integral_value(), value
        return int(amount)

    with sqlite3.connect((DATA / 'data/dev_006/pos.sqlite').resolve().as_uri() + '?mode=ro', uri=True) as con:
        con.create_function('receipt_vnd', 1, receipt_money)
        eligible = "d.TinhTrang NOT IN ('cancelled','returned')"
        staff = con.execute(f"""
            SELECT SUM(l.TriGia) FROM (SELECT DISTINCT * FROM chi_tiet_don) l
            JOIN don_hang d USING(MaHoaDon)
            WHERE substr(d.ThoiGianLap,1,7)='2025-04'
              AND d.MaNhanVien='NV002' AND {eligible}
        """).fetchone()[0]
        header = con.execute(f"""
            SELECT SUM(TongTienHang), COUNT(*) FROM don_hang d
            WHERE substr(ThoiGianLap,1,7)='2025-04'
              AND MaNhanVien='NV002' AND {eligible}
        """).fetchone()
        assert staff == header[0] == 77101300 and header[1] == 758, (staff, header)
        methods = dict(con.execute(f"""
            SELECT d.PhuongThucTT, SUM(l.TriGia)
            FROM (SELECT DISTINCT * FROM chi_tiet_don) l JOIN don_hang d USING(MaHoaDon)
            WHERE substr(d.ThoiGianLap,1,7)='2026-03'
              AND d.PhuongThucTT IN ('cod','ewallet') AND {eligible}
            GROUP BY d.PhuongThucTT
        """))
        assert methods == {'cod': 12361000, 'ewallet': 30121300}, methods
        repayments = {}
        for month, expected, prior_expected in [('2025-11', 2294053, 1304170),
                                                 ('2025-09', 3140750, 1706519)]:
            total, prior = con.execute("""
                SELECT SUM(receipt_vnd(p.SoTienThu)),
                       SUM(CASE WHEN substr(d.ThoiGianLap,1,7) != ?
                                THEN receipt_vnd(p.SoTienThu) ELSE 0 END)
                FROM thanh_toan p JOIN don_hang d USING(MaHoaDon)
                WHERE d.PhuongThucTT='debt' AND d.TinhTrang!='cancelled'
                  AND substr(p.NgayGioThu,1,7)=?
                  AND date(p.NgayGioThu)>date(d.ThoiGianLap)
            """, (month, month)).fetchone()
            assert (total, prior) == (expected, prior_expected), (month, total, prior)
            repayments[month] = {'total_vnd': total, 'from_older_orders_vnd': prior}
        return {'staff_detail_equals_header_vnd': staff, 'staff_orders': header[1],
                'payment_method_revenue_vnd': methods, 'debt_repayment_vnd': repayments}


def check_full_month_source():
    """Aggregate the two released spreadsheets independently of Shop005."""
    from common import tab, num, norm, dt
    revenues = {}
    last_day_may = 0
    for month in (5, 6):
        rows = tab(DATA / f'data/dev_005/BanHang_Thang{month:02d}_2025.xlsx', 'DonHang_ChiTiet')
        header = ['Mã đơn hàng', 'Ngày đặt hàng', 'Trạng thái đơn hàng', 'Tên khách hàng']
        rows[header] = rows[header].ffill()
        rows = rows[rows['Tên sản phẩm'].notna() & rows['Mã đơn hàng'].notna()].drop_duplicates().copy()
        statuses = rows['Trạng thái đơn hàng'].map(norm)
        rows = rows[~statuses.isin({'huy', 'da huy', 'tra lai', 'tra hang'})].copy()
        rows['value'] = rows['Tổng tiền'].map(num)
        dates = {value: dt(value) for value in rows['Ngày đặt hàng'].unique()}
        rows['date'] = rows['Ngày đặt hàng'].map(dates)
        assert rows.date.dt.year.eq(2025).all() and rows.date.dt.month.eq(month).all()
        revenues[month] = int(rows.value.sum())
        if month == 5:
            last_day_may = int(rows.loc[rows.date.dt.day.eq(31), 'value'].sum())
    assert revenues == {5: 290789000, 6: 272091800}, revenues
    assert last_day_may == 9505800, last_day_may
    growth = ((Decimal(revenues[6]) / Decimal(revenues[5]) - 1) * 100).quantize(
        Decimal('0.01'), rounding=ROUND_HALF_UP)
    assert growth == Decimal('-6.43'), growth
    return {'may_revenue_vnd': revenues[5], 'june_revenue_vnd': revenues[6],
            'may_31_revenue_vnd': last_day_may, 'growth_percent': float(growth)}


def check_previous_week_source():
    """Reconcile raw invoice totals with distinct line totals for the full week."""
    question = next(q for q in read_jsonl(DATA / 'questions.jsonl')
                    if q['id'] == 'q_64e62d4bea7d')
    assert question['business_id'] == 'dev_006'
    asof = date.fromisoformat(question['as_of_date'])
    monday = asof - timedelta(days=asof.weekday() + 7)
    sunday = monday + timedelta(days=6)
    assert (monday.isoformat(), sunday.isoformat()) == ('2026-02-23', '2026-03-01')
    params = (monday.isoformat(), sunday.isoformat())
    with sqlite3.connect((DATA / 'data/dev_006/pos.sqlite').resolve().as_uri() + '?mode=ro', uri=True) as con:
        count, revenue = con.execute("""
            SELECT COUNT(*), SUM(TongTienHang) FROM don_hang
            WHERE date(ThoiGianLap) BETWEEN ? AND ?
              AND TinhTrang NOT IN ('cancelled','returned')
        """, params).fetchone()
        line_count, line_revenue = con.execute("""
            SELECT COUNT(DISTINCT d.MaHoaDon), SUM(l.TriGia)
            FROM (SELECT DISTINCT * FROM chi_tiet_don) l
            JOIN don_hang d USING(MaHoaDon)
            WHERE date(d.ThoiGianLap) BETWEEN ? AND ?
              AND d.TinhTrang NOT IN ('cancelled','returned')
        """, params).fetchone()
    assert (count, revenue) == (line_count, line_revenue) == (563, 57976800)
    average = (Decimal(revenue) / Decimal(count)).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
    assert average == 102978
    return {'start_date': params[0], 'end_date': params[1], 'valid_orders': count,
            'header_and_distinct_line_revenue_vnd': revenue,
            'rounded_average_vnd': int(average)}


def check_retail_threshold_source():
    """Count qualifying January bills from raw sales rows, including service fees."""
    import pandas as pd
    from common import tab, num, norm, dt
    root = DATA / 'data/dev_008'
    rows = pd.concat([tab(root / f'ChiTietHoaDon_{part}.xlsx', header=0)
                      for part in (1, 2)], ignore_index=True).drop_duplicates()
    rows = rows[rows.Bill.astype(str).str.fullmatch(r'ORD\d+')].copy()
    rows['value'] = rows['Thành tiền'].map(num)
    dates = {value: dt(value) for value in rows['Ngày xuất'].dropna().unique()}
    rows['date'] = rows['Ngày xuất'].map(dates)
    rows['status'] = rows['TT đơn'].map(norm).replace('', None)
    bills = rows.groupby('Bill').agg(date=('date', 'first'),
                                    status=('status', 'first'), value=('value', 'sum'))
    january = bills[bills.date.between('2025-01-01', '2025-01-31')]
    invalid = january.status.fillna('').str.contains(
        r'\bhuy\b|cancel|tra hang|tra lai|returned|hoan tra|doi tra|khach tra|hoan hang|hang tra ve')
    valid = january[~invalid]
    reached = int(valid.value.ge(410000).sum())
    exactly = int(valid.value.eq(410000).sum())
    assert (len(january), len(valid), reached, exactly) == (1738, 1694, 846, 3)
    return {'month': '2025-01', 'valid_bills': len(valid), 'threshold_vnd': 410000,
            'bills_at_or_above_threshold': reached, 'bills_exactly_at_threshold': exactly}


def main():
    baseline_path = HERE / 'submission_baseline0998.jsonl'
    assert hashlib.sha256(baseline_path.read_bytes()).hexdigest() == BASELINE_SHA
    baseline = {row['id']: row for row in read_jsonl(baseline_path)}
    questions = read_jsonl(DATA / 'questions.jsonl')
    target = HERE / 'submission.jsonl'
    rows = read_jsonl(target)
    assert len(rows) == len(questions) == 1000
    assert [r['id'] for r in rows] == [q['id'] for q in questions]
    assert len({r['id'] for r in rows}) == 1000
    assert set(baseline) == {q['id'] for q in questions}
    changes = []
    by_id = {q['id']: q for q in questions}
    for row in rows:
        validate_row(row)
        assert row['type'] != 'abstain'
        old = baseline[row['id']]
        if (row['type'], row.get('answer')) == (old['type'], old.get('answer')):
            continue
        qid = row['id']
        assert qid in EXPECTED_CHANGES, ('Unexpected answer change', qid, old, row)
        typ, expected, evidence = EXPECTED_CHANGES[qid]
        assert (row['type'], row.get('answer')) == (typ, expected), (qid, row)
        changes.append({'id': qid, 'business_id': by_id[qid]['business_id'],
                        'question': by_id[qid]['question'], 'old': old, 'new': row,
                        'source_calculation': evidence})
    assert {c['id'] for c in changes} == set(EXPECTED_CHANGES)
    # Preserve answers already tested on the leaderboard.
    result = {r['id']: r for r in rows}
    for qid, (typ, answer, _) in CONFIRMED_PREVIOUS_FIXES.items():
        assert (result[qid]['type'], result[qid].get('answer')) == (typ, answer), qid
    for qid in ('q_569dcd86abb4', 'q_7de8d58e315c', 'q_c38425d8aec8'):
        assert result[qid] == baseline[qid], qid
    report = {
        'run_id': 'dm-A-021-week-trc',
        'rows': len(rows), 'abstain': 0, 'changed_answers': len(changes),
        'baseline_sha256': BASELINE_SHA,
        'submission_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'independent_sql_checks': check_pos_source(),
        'independent_full_month_check': check_full_month_source(),
        'independent_order_threshold_check': check_retail_threshold_source(),
        'independent_previous_week_check': check_previous_week_source(),
        'leaderboard_score': None,
        'note': 'Source checks do not establish hidden-label accuracy. No new leaderboard submission.',
        'changes': changes,
    }
    (HERE / 'evidence_improvements.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'changes'},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
