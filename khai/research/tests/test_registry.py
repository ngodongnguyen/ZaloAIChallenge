import json
import os
import tempfile
import time
import unittest

from harness.registry import DuplicateRun, Registry

CFG = {"hypothesis": "fp16 on one GPU keeps the same tool calls", "parent": None, "llm_dtype": "fp16"}


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.reg = Registry(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_new_run_writes_files(self):
        rid = self.reg.new_run(CFG, "cfghash1", "codehash1", run_id="r1")
        d = os.path.join(self._tmp.name, "runs", rid)
        for name in ("run.json", "config.json", "log.txt"):
            self.assertTrue(os.path.exists(os.path.join(d, name)), name)

    def test_duplicate_done_run_is_refused(self):
        rid = self.reg.new_run(CFG, "cfghash1", "codehash1", run_id="r1")
        self.reg.set_status(rid, "done")
        with self.assertRaises(DuplicateRun):
            self.reg.new_run(CFG, "cfghash1", "codehash1", run_id="r2")

    def test_same_config_different_code_is_allowed(self):
        rid = self.reg.new_run(CFG, "cfghash1", "codehash1", run_id="r1")
        self.reg.set_status(rid, "done")
        self.reg.new_run(CFG, "cfghash1", "codehash2", run_id="r2")

    def test_failed_run_does_not_block_retry(self):
        rid = self.reg.new_run(CFG, "cfghash1", "codehash1", run_id="r1")
        self.reg.set_status(rid, "failed", stop_reason="oom")
        self.reg.new_run(CFG, "cfghash1", "codehash1", run_id="r2")

    def test_stale_run_is_marked(self):
        rid = self.reg.new_run(CFG, "cfghash1", "codehash1", run_id="r1")
        self.reg.set_status(rid, "running")
        changed = self.reg.mark_stale(now=time.time() + 7 * 3600)
        self.assertEqual(changed, ["r1"])
        with open(os.path.join(self._tmp.name, "runs", "r1", "run.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["status"], "stale")

    def test_index_and_journal_are_rebuilt(self):
        rid = self.reg.new_run(CFG, "cfghash1", "codehash1", run_id="r1")
        self.reg.write_metrics(rid, {"score": 0.5})
        self.reg.set_status(rid, "done")
        index = self.reg.rebuild_index()
        self.assertEqual(index[0]["metrics"]["score"], 0.5)
        with open(os.path.join(self._tmp.name, "JOURNAL.md"), encoding="utf-8") as f:
            journal = f.read()
        self.assertIn("r1", journal)
        self.assertIn("score=0.5", journal)


if __name__ == "__main__":
    unittest.main()
