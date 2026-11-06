"""CPU checks for the synthetic warm-start notebook: same agent as lm-sft-046, warm start really loads the adapter."""
import ast
import contextlib
import io
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).parent
NOTEBOOK = ROOT / "agent-qwen35-syn/lifemate_qwen35_syn.ipynb"
FULL = ROOT / "agent-qwen35-full/lifemate_qwen35_full.ipynb"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sources(path):
    return ["".join(c["source"]) for c in load(path)["cells"] if c["cell_type"] == "code"]


def config(path):
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(next(s for s in sources(path) if s.startswith("CONFIG = {")), ns)
    return ns


class Qwen35SynNotebookTests(unittest.TestCase):
    def test_every_code_cell_parses(self):
        for s in sources(NOTEBOOK):
            if not s.startswith("%%writefile"):
                ast.parse(s)
            else:
                ast.parse(s.split("\n", 1)[1])

    def test_inputs_pinned_in_notebook(self):
        kaggle = load(NOTEBOOK)["metadata"]["kaggle"]
        ids = [d["sourceId"] for d in kaggle["dataSources"]]
        self.assertTrue({21271194, 364021094, 364801062, 363490930, 365775452} <= set(ids))
        self.assertTrue(kaggle["isGpuEnabled"] and kaggle["isInternetEnabled"])

    def test_agent_identical_to_full_run(self):
        syn, full = config(NOTEBOOK)["CONFIG"], config(FULL)["CONFIG"]
        for key in ("llm", "agent", "fewshot", "mode", "split", "asr_backend"):
            self.assertEqual(syn[key], full[key], key)

    def test_recipe_changes_only_lr_and_epochs(self):
        syn, full = dict(config(NOTEBOOK)["SFT"]), dict(config(FULL)["SFT"])
        self.assertEqual((syn.pop("lr"), syn.pop("epochs")), (5e-5, 1))
        full.pop("lr"), full.pop("epochs")
        self.assertEqual(syn, full)

    def test_only_trainer_library_cell_differs(self):
        syn = [s for s in sources(NOTEBOOK) if s.startswith("%%writefile")]
        full = [s for s in sources(FULL) if s.startswith("%%writefile")]
        self.assertEqual(len(syn), len(full))
        changed = [a.splitlines()[0] for a, b in zip(syn, full) if a != b]
        self.assertEqual(changed, ["%%writefile /kaggle/working/lib/lmlib/q35ddp.py"])

    def test_warm_start_is_loaded_and_checked(self):
        code = "\n".join(sources(NOTEBOOK))
        self.assertIn('elif cfg.get("init_adapter"):', code)
        self.assertIn("lora_B is still zero", code)
        self.assertIn("init_adapter=str(INIT_DIR)", code)
        self.assertIn("Train không nạp checkpoint 046", code)
        self.assertIn("/**/sft_q35_syn", code)

    def test_inputs_checked_by_sha(self):
        expect = config(NOTEBOOK)["EXPECT"]
        for key in ("init_adapter_sha256", "syn_sha256", "syn_catalog_sha256"):
            self.assertEqual(len(expect[key]), 64, key)

    def test_no_cached_outputs(self):
        for cell in load(NOTEBOOK)["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])

    def test_public_cells_wait_for_finished_training(self):
        code = sources(NOTEBOOK)
        for marker in ("qwen35smoke", "launch(", "run_replay("):
            cell = next(s for s in code if marker in s and not s.startswith("%%writefile"))
            self.assertTrue(cell.startswith("if RUN_PUBLIC:"), marker)


if __name__ == "__main__":
    unittest.main()
