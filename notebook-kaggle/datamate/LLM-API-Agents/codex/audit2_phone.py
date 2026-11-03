"""Focused regression audit for DataMate PA0.812 phone/repair questions.

The three cases below were traced from their wording through rulebook.md to
the released source tables. Expected values are recomputed from the loaded
source rows with explicit date bounds, independent of parse_window().
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import solve_phone_repair as solver  # noqa: E402


def q_by_id(qid: str) -> dict:
    return next(q for q in solver.QUESTIONS if q["id"] == qid)


def check(
    qid: str, explicit_window: tuple[date, date], expected: dict, evidence: dict
) -> dict:
    q = q_by_id(qid)
    parsed = solver.parse_window(q)
    actual = solver.score_question(q)
    assert parsed == explicit_window, (qid, parsed, explicit_window)
    assert actual == expected, (qid, actual, expected)
    return {
        "id": qid,
        "question": q["question"],
        "intended_window": [d.isoformat() for d in explicit_window],
        "parsed_window": [d.isoformat() for d in parsed],
        "answer": actual,
        "direct_source_evidence": evidence,
    }


def main() -> None:
    checked = []

    # `tháng 8 năm 2025`: the spelled-out year is explicit. Rulebook: paid
    # ticket average excludes warranty/zero-charge records; round VND to đồng.
    qid = "q_34a5b4766a27"
    start, end = date(2025, 8, 1), date(2025, 8, 31)
    paid = [r for r in solver.REPAIRS if start <= r["date"] <= end and r["money"] > 0]
    expected = {"type": "number", "answer": solver.rnd(sum(r["money"] for r in paid) / len(paid))}
    assert len(paid) == 441 and sum(r["money"] for r in paid) == 96_404_500
    aug25 = [r for r in solver.REPAIRS if start <= r["date"] <= end]
    aug26 = [r for r in solver.REPAIRS if date(2026, 8, 1) <= r["date"] <= date(2026, 8, 31)]
    checked.append(check(qid, (start, end), expected, {
        "rule": "mean paid ticket amount; zero-charge warranty tickets excluded",
        "correct_period_rows": len(aug25), "correct_period_paid_rows": len(paid),
        "correct_period_money_vnd": sum(r["money"] for r in paid),
        "misparsed_period_rows": len(aug26),
        "misparsed_period_paid_rows": sum(r["money"] > 0 for r in aug26),
        "misparsed_period_money_vnd": sum(r["money"] for r in aug26),
    }))

    # `8/25`: bare month/two-digit-year syntax means August 2025. Quantity is
    # the sum of used-phone sale-line units on non-return orders.
    qid = "q_32b653051a42"
    start, end = date(2025, 8, 1), date(2025, 8, 31)
    lines, orders, _ = solver.SALES["dev_014"]
    valid_keys = {o["key"] for o in orders if start <= o["date"] <= end and not o["return"]}
    qty = sum(r["qty"] for r in lines if r["key"] in valid_keys and r["kind"] == "may cu")
    expected = {"type": "number", "answer": int(qty)}
    assert qty == 59
    sep26_lines, sep26_orders, _ = solver.SALES["dev_014"]
    sep26_keys = {o["key"] for o in sep26_orders if date(2026, 9, 1) <= o["date"] <= date(2026, 9, 30) and not o["return"]}
    sep26_used = [r for r in sep26_lines if r["key"] in sep26_keys and r["kind"] == "may cu"]
    checked.append(check(qid, (start, end), expected, {
        "rule": "sum used-phone quantities on non-return sales orders",
        "correct_period_sale_lines": len([r for r in lines if r["key"] in valid_keys and r["kind"] == "may cu"]),
        "correct_period_units": qty,
        "misparsed_period_sale_lines": len(sep26_used),
        "misparsed_period_units": sum(r["qty"] for r in sep26_used),
    }))

    # This is a period, not only its first month. Rulebook: count vehicles by
    # plate, and only paid repair tickets qualify as repeat visits.
    qid = "q_c3158f13ca1c"
    start, end = date(2025, 9, 1), date(2025, 11, 30)
    paid = [r for r in solver.REPAIRS if start <= r["date"] <= end and r["money"] > 0]
    visits = Counter(r["plate"] for r in paid)
    expected = {"type": "number", "answer": sum(n >= 2 for n in visits.values())}
    assert len(paid) == 1_232 and len(visits) == 964 and expected["answer"] == 200
    sep25 = [r for r in solver.REPAIRS if date(2025, 9, 1) <= r["date"] <= date(2025, 9, 30) and r["money"] > 0]
    sep25_visits = Counter(r["plate"] for r in sep25)
    checked.append(check(qid, (start, end), expected, {
        "rule": "count plates with at least two paid repair tickets in the inclusive period",
        "correct_period_paid_ticket_rows": len(paid),
        "correct_period_distinct_plates": len(visits),
        "correct_period_repeat_plates": expected["answer"],
        "misparsed_period": ["2025-09-01", "2025-09-30"],
        "misparsed_period_paid_ticket_rows": len(sep25),
        "misparsed_period_distinct_plates": len(sep25_visits),
        "misparsed_period_repeat_plates": sum(n >= 2 for n in sep25_visits.values()),
    }))

    # `T8/2025` is an explicit month/year; verify both the current interval and
    # the previous calendar month from raw repair-ticket amounts.
    qid = "q_1205eaa6091a"
    start, end = date(2025, 8, 1), date(2025, 8, 31)
    current = [r for r in solver.REPAIRS if start <= r["date"] <= end]
    previous = [r for r in solver.REPAIRS if date(2025, 7, 1) <= r["date"] <= date(2025, 7, 31)]
    current_v = sum(r["money"] for r in current)
    previous_v = sum(r["money"] for r in previous)
    expected = {"type": "percent", "answer": solver.rnd((current_v - previous_v) / previous_v * 100, 2)}
    assert len(current) == 450 and current_v == 96_404_500
    assert len(previous) == 401 and previous_v == 83_087_000
    checked.append(check(qid, (start, end), expected, {
        "rule": "explicit T8/2025 versus the preceding full month; percent change",
        "correct_period_rows": len(current), "correct_period_revenue_vnd": current_v,
        "comparison_period_rows": len(previous), "comparison_period_revenue_vnd": previous_v,
        "misparsed_period": ["2026-08-01", "2026-08-31"],
        "misparsed_period_rows": len(aug26), "misparsed_period_revenue_vnd": sum(r["money"] for r in aug26),
        "misparsed_comparison_period": ["2026-07-01", "2026-07-31"],
        "misparsed_comparison_rows": sum(date(2026, 7, 1) <= r["date"] <= date(2026, 7, 31) for r in solver.REPAIRS),
        "misparsed_comparison_revenue_vnd": sum(r["money"] for r in solver.REPAIRS if date(2026, 7, 1) <= r["date"] <= date(2026, 7, 31)),
    }))

    # `T10/2025` is explicit, and the rulebook denominator is all phone units.
    qid = "q_792e1070baaf"
    start, end = date(2025, 10, 1), date(2025, 10, 31)
    lines, orders, _ = solver.SALES["dev_014"]
    valid_keys = {o["key"] for o in orders if start <= o["date"] <= end and not o["return"]}
    phone_rows = [r for r in lines if r["key"] in valid_keys and (r["kind"] in ("may moi", "may cu", "may") or r["imei"])]
    samsung = sum(r["qty"] for r in phone_rows if "samsung" in r["item_norm"])
    all_phone = sum(r["qty"] for r in phone_rows)
    expected = {"type": "percent", "answer": solver.rnd(samsung / all_phone * 100, 2)}
    sep26_keys = {o["key"] for o in orders if date(2026, 9, 1) <= o["date"] <= date(2026, 9, 30) and not o["return"]}
    sep26_phone = [r for r in lines if r["key"] in sep26_keys and (r["kind"] in ("may moi", "may cu", "may") or r["imei"])]
    # Rulebook §1 Line 14: exclude returned orders ('khách mang trả' in dev_014)
    assert (samsung, all_phone) == (37, 133)
    checked.append(check(qid, (start, end), expected, {
        "rule": "Samsung units / all phone units, excluding accessories and returned orders",
        "correct_period_phone_sale_lines": len(phone_rows), "correct_period_samsung_units": samsung,
        "correct_period_all_phone_units": all_phone,
        "misparsed_period": ["2026-09-01", "2026-09-30"],
        "misparsed_period_phone_sale_lines": len(sep26_phone),
        "misparsed_period_samsung_units": sum(r["qty"] for r in sep26_phone if "samsung" in r["item_norm"]),
        "misparsed_period_all_phone_units": sum(r["qty"] for r in sep26_phone),
    }))

    # `3/26` is March 2026. Rank technicians by total paid-ticket revenue.
    qid = "q_ad512cd0cb75"
    start, end = date(2026, 3, 1), date(2026, 3, 31)
    rows = [r for r in solver.REPAIRS if start <= r["date"] <= end]
    by_tech = Counter()
    for r in rows:
        by_tech[r["tech_name"]] += r["money"]
    expected = {"type": "string", "answer": sorted(by_tech, key=lambda n: (-by_tech[n], n))[0]}
    aug26 = [r for r in solver.REPAIRS if date(2026, 8, 1) <= r["date"] <= date(2026, 8, 31)]
    aug26_by_tech = Counter()
    for r in aug26:
        aug26_by_tech[r["tech_name"]] += r["money"]
    assert by_tech[expected["answer"]] == 24_247_500
    checked.append(check(qid, (start, end), expected, {
        "rule": "highest technician repair-ticket revenue in the requested month",
        "correct_period_ticket_rows": len(rows), "correct_period_revenue_by_technician_vnd": dict(by_tech),
        "misparsed_period": ["2026-08-01", "2026-08-31"],
        "misparsed_period_ticket_rows": len(aug26), "misparsed_period_revenue_by_technician_vnd": dict(aug26_by_tech),
    }))

    # `T1/2026` is explicit; total sales revenue is all valid order line totals.
    qid = "q_fd38dda74f47"
    start, end = date(2026, 1, 1), date(2026, 1, 31)
    lines, orders, _ = solver.SALES["dev_014"]
    valid = [o for o in orders if start <= o["date"] <= end and not o["return"]]
    revenue = sum(o["revenue"] for o in valid)
    expected = {"type": "number", "answer": solver.rnd(revenue)}
    sep26 = [o for o in orders if date(2026, 9, 1) <= o["date"] <= date(2026, 9, 30) and not o["return"]]
    assert (len(valid), revenue) == (433, 1_823_098_500)
    checked.append(check(qid, (start, end), expected, {
        "rule": "sum revenue (line totals) for valid orders in explicit T1/2026",
        "correct_period_valid_orders": len(valid), "correct_period_revenue_vnd": revenue,
        "misparsed_period": ["2026-09-01", "2026-09-30"],
        "misparsed_period_valid_orders": len(sep26), "misparsed_period_revenue_vnd": sum(o["revenue"] for o in sep26),
    }))

    answers = {q["id"]: solver.score_question(q) for q in solver.QUESTIONS}
    assert len(answers) == 153
    assert all(a["type"] != "abstain" for a in answers.values())
    report = {
        "scope": "dev_011/dev_012/dev_014 (153 public-test questions)",
        "focused_regressions": checked,
        "coverage": {"questions": len(answers), "abstain": sum(a["type"] == "abstain" for a in answers.values())},
        "source": "dataset/datamate/public_test/data/dev_011..dev_014 and rulebook.md",
    }
    output = HERE / "audit2_phone.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
