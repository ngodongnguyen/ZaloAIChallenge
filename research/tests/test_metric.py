import unittest

from harness import metric as M

FUZZY = {"ride_hailing.book": {"pickup", "dropoff"}}


def call(name, **args):
    return {"name": name, "arguments": args}


class MetricTests(unittest.TestCase):
    def test_empty_gold_needs_empty_submission(self):
        self.assertEqual(M.sample_score([], [], "h", "h", FUZZY), 1.0)
        self.assertEqual(M.sample_score([call("sms.send", to="a")], [], "h", "h", FUZZY), 0.0)

    def test_state_hash_mismatch_scores_zero(self):
        g = [call("sms.send", to="a", body="x")]
        self.assertEqual(M.sample_score(g, g, "h1", "h2", FUZZY), 0.0)

    def test_exact_match_scores_one(self):
        g = [call("sms.send", to="a", body="x")]
        self.assertAlmostEqual(M.sample_score(g, g, "h", "h", FUZZY), 1.0)

    def test_fuzzy_argument_uses_token_f1_threshold(self):
        gold = [call("ride_hailing.book", pickup="Blue Pearl Quận 7", dropoff="Bệnh viện Chợ Rẫy")]
        close = [call("ride_hailing.book", pickup="blue pearl quận 7", dropoff="bệnh viện chợ rẫy")]
        far = [call("ride_hailing.book", pickup="Sân bay Tân Sơn Nhất", dropoff="Bệnh viện Chợ Rẫy")]
        self.assertAlmostEqual(M.sample_score(close, gold, "h", "h", FUZZY), 1.0)
        self.assertLess(M.sample_score(far, gold, "h", "h", FUZZY), 1.0)

    def test_non_fuzzy_argument_must_be_identical(self):
        gold = [call("sms.send", to="0901", body="hi")]
        sub = [call("sms.send", to="0902", body="hi")]
        self.assertEqual(M.max_matching(sub, gold, FUZZY), 0)

    def test_each_gold_call_matches_at_most_once(self):
        gold = [call("sms.send", to="a", body="x")]
        sub = [call("sms.send", to="a", body="x"), call("sms.send", to="a", body="x")]
        self.assertEqual(M.max_matching(sub, gold, FUZZY), 1)

    def test_order_is_ignored(self):
        gold = [call("sms.send", to="a", body="x"), call("alarm.set", time="07:00")]
        sub = [call("alarm.set", time="07:00"), call("sms.send", to="a", body="x")]
        self.assertEqual(M.max_matching(sub, gold, FUZZY), 2)

    def test_extra_unmatched_call_lowers_precision(self):
        gold = [call("alarm.set", time="07:00")]
        sub = [call("alarm.set", time="07:00"), call("alarm.set", time="08:00")]
        score = M.sample_score(sub, gold, "h", "h", FUZZY)
        self.assertLess(score, 1.0)
        self.assertGreater(score, 0.0)

    def test_token_f1_handles_unicode_nfc(self):
        composed = "Quận"                    # ậ as one code point
        decomposed = "Quận"           # a + dot below + circumflex
        self.assertNotEqual(composed, decomposed)
        self.assertAlmostEqual(M.token_f1(composed, decomposed), 1.0)

    def test_mean_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            M.mean_score([])


if __name__ == "__main__":
    unittest.main()
