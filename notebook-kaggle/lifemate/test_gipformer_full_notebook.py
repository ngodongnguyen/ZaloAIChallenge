"""CPU checks for the full-data Gipformer notebook: lm-A-031 recipe lr15, epoch 4, every train audio, nothing else changed."""
import ast
import contextlib
import difflib
import io
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).parent
NOTEBOOK = ROOT / "asr-gipformer-full/lifemate_gipformer_full.ipynb"
OLD = ROOT / "asr-ft-gipformer/lifemate_asr_ft_gipformer.ipynb"


def sources(path):
    return ["".join(c["source"]) for c in json.loads(path.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def config(path):
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(next(s for s in sources(path) if s.startswith("CONFIG = {")), ns)
    return ns["CONFIG"]


class GipformerFullTests(unittest.TestCase):
    def test_cells_parse(self):
        for s in sources(NOTEBOOK):
            ast.parse(s.split("\n", 1)[1] if s.startswith("%%writefile") else s)

    def test_recipe_is_lr15_epoch4(self):
        new, old = config(NOTEBOOK), config(OLD)
        self.assertEqual(list(new["variants"]), ["lr15"])
        a, b = dict(new["variants"]["lr15"]), dict(old["variants"]["lr15"])
        self.assertEqual((a.pop("epochs"), b.pop("epochs")), (4, 15))
        self.assertEqual(a.pop("gpu"), 0); b.pop("gpu")
        self.assertEqual(a, b)
        for key in ("model", "revision", "icefall_commit", "seed"):
            self.assertEqual(new[key], old[key], key)
        self.assertTrue(new["full_data"])

    def test_only_intended_trainer_lines_changed(self):
        new = next(s for s in sources(NOTEBOOK) if s.startswith("%%writefile")).splitlines()
        old = next(s for s in sources(OLD) if s.startswith("%%writefile")).splitlines()
        changed = [l for l in difflib.ndiff(old, new) if l[:2] in ("- ", "+ ")]
        self.assertEqual(sum(l.startswith("- ") for l in changed), 2)
        self.assertEqual(sum(l.startswith("+ ") for l in changed), 5)

    def test_all_train_audio_used_and_output_checked(self):
        code = "\n".join(sources(NOTEBOOK))
        self.assertIn('len(split["parts"]["train_fit"]) == 5036', code)
        self.assertIn("!= 1934", code)
        self.assertNotIn("sherpa_onnx.OfflineRecognizer", code)


if __name__ == "__main__":
    unittest.main()
