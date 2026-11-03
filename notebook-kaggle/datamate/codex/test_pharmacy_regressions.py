"""Regressions derived independently from the released POS SQLite records."""
import unittest
from datetime import date

from common import QUESTIONS
import solve_pharmacy_pos as solver


class PharmacyRegressions(unittest.TestCase):
    def answer(self, qid):
        question = next(q for q in QUESTIONS if q['id'] == qid)
        fn = solver.answer005 if question['business_id'] == 'dev_005' else solver.answer006
        return fn(question)

    def test_abbreviated_staff_filters_to_named_employee(self):
        # SQL DISTINCT detail rows and invoice headers both give 77101300.
        # Exact export duplicates inside an order count once (rulebook 8).
        self.assertEqual(self.answer('q_3b70619cf8cc'),
                         {'type': 'number', 'answer': 77101300})

    def test_repayment_includes_older_orders_in_november(self):
        # 989883 from November orders + 1304170 from earlier orders.
        self.assertEqual(self.answer('q_2d5eb3a196b7'),
                         {'type': 'number', 'answer': 2294053})

    def test_repayment_includes_older_orders_in_september(self):
        # 1434231 from September orders + 1706519 from earlier orders.
        self.assertEqual(self.answer('q_ca9e9c233325'),
                         {'type': 'number', 'answer': 3140750})

    def test_explicit_written_year_overrides_relative_month(self):
        self.assertEqual(solver.extract_explicit_months(
            'Thang 6 nam 2025 nhap ve bnhiu vien Tiffy Dey hop 25 vi x 4 vien?',
            date(2026, 8, 8)), [(2025, 6)])
        # Seven received lots: 300 + 6 * 200 basic units.
        self.assertEqual(self.answer('q_fbaf2023cfb6'),
                         {'type': 'number', 'answer': 1500})

    def test_customer_flow_keeps_customer_filter(self):
        for qid, expected in (
            ('q_0a9819711554', [8000, 4000]),
            ('q_48b77f4a1b30', [700, 233]),
            ('q_ae44d3539448', [16000, 16000]),
        ):
            with self.subTest(qid=qid):
                self.assertEqual(self.answer(qid), {'type': 'list', 'answer': expected})

    def test_cod_then_wallet_preserves_requested_order(self):
        self.assertEqual(self.answer('q_51f4751e4eb3'),
                         {'type': 'list', 'answer': [12361000, 30121300]})

    def test_non_prescription_abbreviation_remains_correct(self):
        for qid, expected in (('q_105bc863feea', 267653200),
                              ('q_107e86e608dc', 264831100)):
            with self.subTest(qid=qid):
                self.assertEqual(self.answer(qid), {'type': 'number', 'answer': expected})

    def test_explicit_month_uses_the_entire_previous_month(self):
        # Source full May = 290789000, full June = 272091800.
        # May 31 alone contributes 9505800 and must be in the denominator.
        self.assertEqual(self.answer('q_fc4f06f3006b'),
                         {'type': 'percent', 'answer': -6.43})

    def test_month_to_date_keeps_equal_elapsed_days(self):
        for qid, expected in (('q_01d1099489a0', -2.94),
                              ('q_1473ee8c46fd', -14.34),
                              ('q_4633acb03075', 15.6)):
            with self.subTest(qid=qid):
                self.assertEqual(self.answer(qid), {'type': 'percent', 'answer': expected})

    def test_abbreviated_previous_week_has_the_full_monday_sunday_window(self):
        self.assertEqual(solver.relative_window(
            'Dthu bq mỗi đơn tuần trc là bnhiu?', date(2026, 3, 3)),
            (date(2026, 2, 23), date(2026, 3, 1)))

    def test_abbreviated_previous_week_average_uses_weekly_orders(self):
        # Raw headers and deduplicated lines both give 57976800 / 563.
        # The as-of day alone is 8133100 / 81 and must not be substituted.
        self.assertEqual(self.answer('q_64e62d4bea7d'),
                         {'type': 'number', 'answer': 102978})


if __name__ == '__main__':
    unittest.main()
