"""Regression: an explicit last-year comparison overrides the previous month."""
import unittest

from common import QUESTIONS
from solve_retail_baby import answer_question


class RetailYearRegressions(unittest.TestCase):
    def test_may_compares_with_may_last_year(self):
        q = next(q for q in QUESTIONS if q['id'] == 'q_58215441e762')
        # Source invoices: May 2026 = 1285826500; May 2025 = 1078858000.
        self.assertEqual(answer_question(q), {'type': 'percent', 'answer': 19.18})

    def test_transposed_threshold_word_keeps_the_revenue_filter(self):
        q = next(q for q in QUESTIONS if q['id'] == 'q_4df64734eb4d')
        # January source: 1694 valid bills; 846 >=410000, including 3 equal.
        self.assertEqual(answer_question(q), {'type': 'number', 'answer': 846})


if __name__ == '__main__':
    unittest.main()
