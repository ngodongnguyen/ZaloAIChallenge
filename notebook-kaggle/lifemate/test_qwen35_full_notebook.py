"""CPU checks for the Qwen3.5 full-train notebook: inputs, two-session flow, same agent as lm-B-045."""
import ast
import contextlib
import io
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).parent
NOTEBOOK = ROOT / "agent-qwen35-full/lifemate_qwen35_full.ipynb"
PUBLIC = ROOT / "agent-qwen35-public/lifemate_qwen35_public.ipynb"
SFT = ROOT / "agent-qwen35-sft/lifemate_qwen35_sft.ipynb"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sources(path):
    return ["".join(c["source"]) for c in load(path)["cells"] if c["cell_type"] == "code"]


def config(path):
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(next(s for s in sources(path) if s.startswith("CONFIG = {")), ns)
    return ns


class Qwen35FullNotebookTests(unittest.TestCase):
    def test_every_code_cell_parses(self):
        for s in sources(NOTEBOOK):
            if not s.startswith("%%writefile"):
                ast.parse(s)

    def test_inputs_pinned_in_notebook(self):
        kaggle = load(NOTEBOOK)["metadata"]["kaggle"]
        ids = [d["sourceId"] for d in kaggle["dataSources"]]
        self.assertEqual(ids, [21271194, 364021094, 364801062, 363490930])
        self.assertTrue(kaggle["isGpuEnabled"] and kaggle["isInternetEnabled"])

    def test_agent_identical_to_public_run(self):
        full, public = config(NOTEBOOK)["CONFIG"], config(PUBLIC)["CONFIG"]
        for key in ("llm", "agent", "fewshot", "mode", "split", "asr_backend"):
            self.assertEqual(full[key], public[key], key)

    def test_recipe_identical_to_sft_run_except_data(self):
        full, sft = config(NOTEBOOK)["SFT"], dict(config(SFT)["SFT"])
        sft.pop("train_subset")
        self.assertEqual(full, sft)

    def test_public_cells_wait_for_finished_training(self):
        code = sources(NOTEBOOK)
        for marker in ("qwen35smoke", "launch(", "run_replay("):
            cell = next(s for s in code if marker in s and not s.startswith("%%writefile"))
            self.assertTrue(cell.startswith("if RUN_PUBLIC:"), marker)

    def test_library_cells_match_public_run(self):
        self.assertEqual([s for s in sources(NOTEBOOK) if s.startswith("%%writefile")],
                         [s for s in sources(PUBLIC) if s.startswith("%%writefile")])


if __name__ == "__main__":
    unittest.main()
