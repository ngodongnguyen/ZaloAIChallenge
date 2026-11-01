"""CPU checks for Qwen3.5 engine configuration and notebook failure handling."""
import ast
import contextlib
import io
import json
import shlex
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch


NOTEBOOK = Path(__file__).parent / "agent-qwen35/lifemate_agent_qwen35.ipynb"


def sources():
    return ["".join(c.get("source", [])) for c in json.loads(NOTEBOOK.read_text())["cells"]]


def configuration():
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(next(s for s in sources() if s.startswith("CONFIG = {")), namespace)
    return namespace["CONFIG"]


class Qwen35NotebookTests(unittest.TestCase):
    def engine(self, cfg):
        source = next(s for s in sources() if s.startswith("%%writefile ") and "/lmlib/llm.py\n" in s)
        module = types.ModuleType("qwen35_test_llm")

        class Engine:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def get_tokenizer(self):
                return object()

        backend = types.SimpleNamespace(LLM=Engine, SamplingParams=object)
        with patch.dict(sys.modules, {module.__name__: module, "vllm": backend}):
            exec(source.split("\n", 1)[1], module.__dict__)
            return module.VLLM(**cfg).llm.kwargs

    def test_text_only_option_reaches_engine(self):
        args = self.engine(dict(model="fixture", language_model_only=True))
        self.assertTrue(args.get("language_model_only"))

    def test_qwen35_engine_disables_graphs_and_bounds_prefill(self):
        cfg = configuration()["llm"]
        args = self.engine(cfg)
        self.assertTrue(args["enforce_eager"])
        self.assertTrue(args.get("language_model_only"))
        self.assertLessEqual(args.get("max_num_seqs", 512), 4)
        self.assertLessEqual(args.get("max_num_batched_tokens", 8192), 2048)
        self.assertTrue(args.get("enable_chunked_prefill"))
        self.assertEqual(args["max_model_len"], 16384)
        self.assertLessEqual(cfg["batch_size"], 1)

    def test_setup_installs_accelerate_in_worker_venv(self):
        commands = []
        namespace = dict(CONFIG=configuration(), Path=Path,
                         glob=types.SimpleNamespace(glob=lambda *a, **kw: []),
                         sys=types.SimpleNamespace(executable="/fixture/base/python"),
                         sh=lambda command: commands.append(shlex.split(command)) or 0)
        setup = next(s for s in sources() if "def pip_install(" in s)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(setup, namespace)
        self.assertTrue(any("accelerate" in cmd and "--python" in cmd
                            and cmd[cmd.index("--python") + 1] == "/tmp/venv_vllm/bin/python"
                            for cmd in commands), "accelerate is missing from the worker venv")

    def test_worker_failure_stops_before_scoring(self):
        source = next(s for s in sources() if "def run_agent(" in s)
        function = next(n for n in ast.parse(source).body
                        if isinstance(n, ast.FunctionDef) and n.name == "run_agent")
        merged = []
        with tempfile.TemporaryDirectory() as td:
            namespace = dict(Path=Path, SD=Path(td), N_GPUS=2, LIB=Path(td),
                             LLM_PYTHON=sys.executable, time=time, json=json,
                             atomic_write_json=lambda *a: None,
                             launch=lambda *a, **kw: [1, 1],
                             merge=lambda cfg: merged.append(cfg) or {"score": 0.0})
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "workers failed"):
                    namespace["run_agent"]({}, Path(td))
            self.assertEqual(merged, [])


if __name__ == "__main__":
    unittest.main()
