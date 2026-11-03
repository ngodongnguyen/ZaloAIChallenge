#!/usr/bin/env python3
"""Source-derived regressions for the DataMate retail/baby public questions."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import solve_retail_baby as solver  # noqa: E402


QIDS = (
    "q_3e36677e62a9",  # baby-category revenue share
    "q_ba2c318fc2be",  # baby in-store/tại quầy revenue share
    "q_180163673aaa",  # employee winner, abbreviated as NV
    "q_5b4680c1ccf2",  # singular best-selling product
)


def source_orders(shop: str, start: pd.Timestamp, end: pd.Timestamp):
    """Apply the rulebook's date/status gate directly to parsed release rows."""
    asof = pd.Timestamp(end).normalize()
    return [
        order
        for order in solver.DATASETS[shop]["orders"]
        if order["status"] == "valid"
        and not pd.isna(order["date"])
        and start.normalize() <= order["date"] <= end.normalize()
        and order["date"] <= asof
    ]


def expected_from_rows(qid: str, question: dict):
    shop = question["business_id"]
    ds = solver.DATASETS[shop]
    if qid == "q_3e36677e62a9":
        start, end = pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-30")
        orders = source_orders(shop, start, end)
        category_revenue = sum(
            line["amount"]
            for order in orders
            for line in order["lines"]
            if not line.get("service") and solver.norm(line.get("category")) == "sua bot"
        )
        total_revenue = sum(order["revenue"] for order in orders)
        value = solver.rnd(category_revenue / total_revenue * 100, 2)
        evidence = {
            "period": [str(start.date()), str(end.date())],
            "category_revenue_vnd": int(category_revenue),
            "shop_revenue_vnd": int(total_revenue),
            "formula": "category sales revenue / shop sales revenue * 100",
        }
        return solver.out("percent", value), evidence

    if qid == "q_ba2c318fc2be":
        start, end = pd.Timestamp("2025-11-01"), pd.Timestamp("2025-11-30")
        orders = source_orders(shop, start, end)
        in_store = sum(order["revenue"] for order in orders if order["channel"] == "tai tiem")
        total_revenue = sum(order["revenue"] for order in orders)
        value = solver.rnd(in_store / total_revenue * 100, 2)
        evidence = {
            "period": [str(start.date()), str(end.date())],
            "source_channel": "Tại tiệm (raw normalized value: tai tiem)",
            "in_store_revenue_vnd": int(in_store),
            "shop_revenue_vnd": int(total_revenue),
            "formula": "Tại tiệm revenue / shop sales revenue * 100",
        }
        return solver.out("percent", value), evidence

    if qid == "q_180163673aaa":
        start, end = pd.Timestamp("2025-11-01"), pd.Timestamp("2025-11-30")
        orders = source_orders(shop, start, end)
        totals = defaultdict(float)
        for order in orders:
            if order.get("staff"):
                totals[order["staff"]] += order["revenue"]
        winner, amount = sorted(totals.items(), key=lambda item: (-item[1], solver.norm(item[0])))[0]
        evidence = {
            "period": [str(start.date()), str(end.date())],
            "employee_revenue_vnd": {key: int(value) for key, value in sorted(totals.items())},
            "winner": winner,
            "winner_revenue_vnd": int(amount),
            "rule": "employee with greatest valid-order revenue",
        }
        return solver.out("string", solver.clean_answer_string(winner)), evidence

    if qid == "q_5b4680c1ccf2":
        asof = pd.Timestamp(question["as_of_date"]).normalize()
        start = asof - pd.Timedelta(days=asof.weekday())
        orders = source_orders(shop, start, asof)
        quantities, names, ids = defaultdict(float), {}, {}
        for order in orders:
            for line in order["lines"]:
                if not line.get("service"):
                    key = line.get("id") or line["name_norm"]
                    quantities[key] += line["qty"]
                    names[key] = line["name"]
                    ids[key] = line.get("id", "")
        ranked = sorted(quantities, key=lambda key: (-quantities[key], ids.get(key, "") or key))
        winner = ranked[0]
        evidence = {
            "period": [str(start.date()), str(asof.date())],
            "top_products_by_base_quantity": [
                {"name": names[key], "base_quantity": quantities[key], "id": ids.get(key, "")}
                for key in ranked[:5]
            ],
            "rule": "singular best-selling item is the highest base-quantity item",
        }
        return solver.out("string", solver.clean_answer_string(names[winner])), evidence

    raise KeyError(qid)


def main():
    questions = {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in (solver.DATA / "questions.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    old = {
        row["id"]: row
        for row in map(
            json.loads,
            (HERE / "submission_pa0812.jsonl").read_text(encoding="utf-8").splitlines(),
        )
    }
    cases = []
    for qid in QIDS:
        question = questions[qid]
        expected, evidence = expected_from_rows(qid, question)
        actual = solver.answer_question(question)
        archived = dict(old[qid])
        archived.pop("id", None)
        cases.append(
            {
                "id": qid,
                "business_id": question["business_id"],
                "question": question["question"],
                "archived_submission": archived,
                "solver_answer": actual,
                "source_expected": expected,
                "source_evidence": evidence,
                "pass": actual == expected,
            }
        )
    report = {"cases": cases, "passed": sum(case["pass"] for case in cases), "total": len(cases)}
    out_path = HERE / "audit2_retail.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["passed"] != report["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
