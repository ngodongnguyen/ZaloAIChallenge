"""Run the notebook's SFT result handling with real CPU subprocesses; no models needed."""
import ast
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


NOTEBOOK = Path(__file__).parent / "agent-sft/lifemate_agent_sft.ipynb"


def result_handling():
    notebook = json.loads(NOTEBOOK.read_text())
    source = next("".join(c["source"]) for c in notebook["cells"]
                  if "def merge_adapter_from(" in "".join(c.get("source", [])))
    tree = ast.parse(source)
    branch = next(n for n in tree.body if isinstance(n, ast.If)
                  and ast.unparse(n.test) == "CONFIG['mode'] == 'sft'")
    start = next(i for i, n in enumerate(branch.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "t0" for t in n.targets))
    return compile(ast.Module(body=branch.body[start:], type_ignores=[]), str(NOTEBOOK), "exec")


class SftNotebookTests(unittest.TestCase):
    def run_workers(self, codes, merge_error=False):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            procs = []
            for variant, code in zip(("gold", "asr"), codes):
                folder = root / f"sft_{variant}"
                folder.mkdir()
                log = (folder / "train.log").open("w")
                script = ("from pathlib import Path; import sys; "
                          f"Path({str(folder / 'worker_finished')!r}).write_text('yes'); "
                          f"print('train exit {code}', flush=True); sys.exit({code})")
                process = subprocess.Popen([sys.executable, "-c", script], stdout=log,
                                           stderr=subprocess.STDOUT)
                procs.append((variant, process, log))
            merged = []

            def merge_adapter_from(folder, variant):
                merged.append(variant)
                if merge_error:
                    raise FileNotFoundError("no complete checkpoint")
                return str(folder / "merged")

            def atomic_write_json(path, value):
                Path(path).write_text(json.dumps(value))

            namespace = dict(procs=procs, OUT=root, MERGED={}, time=time,
                             merge_adapter_from=merge_adapter_from,
                             atomic_write_json=atomic_write_json)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(result_handling(), namespace)
            finally:
                # Every worker must be reaped and every log closed, even on failure.
                finished = [p.poll() is not None and log.closed for _, p, log in procs]
                for _, process, log in procs:
                    process.wait()
                    log.close()
                self.assertEqual(finished, [True, True])
            return merged, namespace["MERGED"]

    def test_failed_training_stops_before_merge(self):
        with self.assertRaisesRegex(RuntimeError, "gold.*1.*asr.*1"):
            self.run_workers([1, 1])

    def test_one_failed_variant_rejects_partial_success(self):
        with self.assertRaisesRegex(RuntimeError, "asr.*1"):
            self.run_workers([0, 1])

    def test_successful_workers_merge_both_variants(self):
        merged, models = self.run_workers([0, 0])
        self.assertEqual(merged, ["gold", "asr"])
        self.assertEqual(set(models), {"gold", "asr"})

    def test_missing_checkpoint_is_a_notebook_failure(self):
        with self.assertRaisesRegex(FileNotFoundError, "no complete checkpoint"):
            self.run_workers([0, 0], merge_error=True)


if __name__ == "__main__":
    unittest.main()
