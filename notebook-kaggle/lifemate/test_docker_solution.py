"""CPU checks for the Docker entrypoint: same recipe as lm-B-045, one ToolEnv per sample, always one line per sample."""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parent
APP = ROOT / "docker/app"
DATA = ROOT.parents[1] / "dataset/lifemate/train+public_test"
sys.path.insert(0, str(APP))
if not (APP / "lmlib/agent.py").exists():
    subprocess.run([sys.executable, str(ROOT / "docker/export_lib.py")], check=True, capture_output=True)
import solution  # noqa: E402


def public_config():
    nb = json.loads((ROOT / "agent-qwen35-public/lifemate_qwen35_public.ipynb").read_text(encoding="utf-8"))
    source = next("".join(c["source"]) for c in nb["cells"] if "".join(c["source"]).startswith("CONFIG = {"))
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(source, ns)
    return ns["CONFIG"]


class FakeEnv:
    def __init__(self):
        self.calls = []

    def validate(self, name, args):
        return {"ok": args.get("time") != "bad"}

    def call(self, name, args):
        self.calls.append((name, args))
        return {"ok": True}


class RecipeTests(unittest.TestCase):
    def test_same_recipe_as_public_run_except_budget(self):
        cfg = public_config()
        self.assertEqual(dict(solution.AGENT, sample_budget_s=120.0), cfg["agent"])
        llm = {k: v for k, v in cfg["llm"].items() if k not in ("model", "revision")}
        self.assertEqual(solution.LLM, llm)
        self.assertEqual(solution.BATCH_SIZE, cfg["batch_size"])
        self.assertLess(solution.AGENT["sample_budget_s"], 30)


class NormalizingEnvTests(unittest.TestCase):
    def test_rule_edit_reaches_the_tool(self):
        env = FakeEnv()
        wrapped = solution.NormalizingEnv(env, {"id": "x"}, "r", lambda *_: ({"time": "fixed"}, [{"parameter": "time"}]))
        wrapped.call("reminders.create", {"time": "raw"})
        self.assertEqual(env.calls, [("reminders.create", {"time": "fixed"})])
        self.assertEqual((wrapped.edits, wrapped.steps[0]["arguments"]), (1, {"time": "fixed"}))

    def test_invalid_edit_or_rule_bug_keeps_the_model_call(self):
        env = FakeEnv()
        solution.NormalizingEnv(env, {"id": "x"}, "r", lambda *_: ({"time": "bad"}, [{"parameter": "time"}])).call("a.b", {"time": "raw"})

        def boom(*_):
            raise KeyError("rule")

        with contextlib.redirect_stdout(io.StringIO()):
            solution.NormalizingEnv(env, {"id": "x"}, "r", boom).call("a.b", {"time": "raw2"})
        self.assertEqual([c[1]["time"] for c in env.calls], ["raw", "raw2"])


class FindSplitTests(unittest.TestCase):
    def layout(self, root, *names):
        for rel in names:
            path = Path(root) / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")

    def test_prefers_private_and_ignores_train(self):
        with tempfile.TemporaryDirectory() as d:
            self.layout(d, "train/train_tools_catalog.json", "public_test/public_test_tools_catalog.json", "private_test/private_test_tools_catalog.json")
            self.assertEqual(solution.find_split(Path(d), None, Path(d) / "w").name, "private_test")

    def test_mount_of_the_split_itself_gets_a_named_link(self):
        with tempfile.TemporaryDirectory() as d:
            self.layout(d, "data/private_test_tools_catalog.json")
            found = solution.find_split(Path(d) / "data", None, Path(d) / "w")
            self.assertEqual(found.name, "private_test")
            self.assertTrue((found / "private_test_tools_catalog.json").exists())

    def test_nothing_mounted_is_an_error(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(FileNotFoundError):
            solution.find_split(Path(d), None, Path(d) / "w")


@unittest.skipUnless((DATA / "public_test/public_test.jsonl").exists(), "local dataset missing")
class DryRunTests(unittest.TestCase):
    def test_every_public_sample_gets_a_valid_line(self):
        with tempfile.TemporaryDirectory() as d:
            env = dict(os.environ, DATA_DIR=str(DATA), RESULT_DIR=f"{d}/result", WORK_DIR=f"{d}/work", LM_SPLIT="public_test", LM_DRY_RUN="1")
            run = subprocess.run([sys.executable, str(APP / "solution.py")], env=env, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr[-800:])
            lines = [json.loads(x) for x in Path(f"{d}/result/submission.jsonl").read_text(encoding="utf-8").splitlines()]
            summary = json.loads(Path(f"{d}/result/run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual((len(lines), summary["missing"]), (634, 0))
            self.assertEqual(summary["official_validation"], "634 lines checked, 0 schema errors")
            self.assertEqual({len(x["state_hash"]) for x in lines}, {64})


if __name__ == "__main__":
    unittest.main()
