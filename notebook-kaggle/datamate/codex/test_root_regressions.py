"""Regression: shipping orders count toward revenue under rulebook section 1."""
import unittest

from common import QUESTIONS
from root_data import Shop, N
from solve_root import solve


class ShippingRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shop = Shop('dev_003')

    def test_in_transit_orders_are_not_cancellations(self):
        orders = self.shop.orders
        shipping = orders[orders.status.map(N).eq('dang van chuyen')]
        self.assertEqual(len(shipping), 28)
        self.assertEqual(int(shipping.rev.sum()), 1781200)
        self.assertTrue(shipping.valid.all())

    def test_current_month_revenue_includes_shipping(self):
        cases = {'q_82bbd52c42dc': 43875000,
                 'q_cda272637d58': 67001700,
                 'q_f539cf51f360': 37387100}
        for i, question in enumerate(QUESTIONS):
            if question['id'] in cases:
                with self.subTest(qid=question['id']):
                    actual, _ = solve(self.shop, question, i)
                    self.assertEqual(actual, {'type': 'number', 'answer': cases[question['id']]})


if __name__ == '__main__':
    unittest.main()
