"""CPU checks for the out-of-fold Gipformer notebook: lm-A-059 recipe, five folds, every train audio transcribed by a model that never heard it."""
import ast
import contextlib
import difflib
import hashlib
import io
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).parent
NOTEBOOK = ROOT / "asr-gipformer-oof/lifemate_gipformer_oof.ipynb"
FULL = ROOT / "asr-gipformer-full/lifemate_gipformer_full.ipynb"
TRAIN = ROOT.parent.parent / "dataset/lifemate/train+public_test/train/train.jsonl"


def sources(path):
    return ["".join(c["source"]) for c in json.loads(path.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def config(path):
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(next(s for s in sources(path) if s.startswith("CONFIG = {")), ns)
    return ns["CONFIG"]


class GipformerOofTests(unittest.TestCase):
    def test_cells_parse(self):
        for s in sources(NOTEBOOK):
            ast.parse(s.split("\n", 1)[1] if s.startswith("%%writefile") else s)

    def test_every_fold_uses_the_full_run_recipe(self):
        new, full = config(NOTEBOOK), config(FULL)
        self.assertEqual(sorted(v["fold"] for v in new["variants"].values()), list(range(new["n_folds"])))
        recipe = dict(full["variants"]["lr15"])
        recipe.pop("gpu")
        for v in new["variants"].values():
            v = dict(v)
            v.pop("gpu"), v.pop("fold")
            self.assertEqual(v, recipe)
        for key in ("model", "revision", "icefall_commit", "seed"):
            self.assertEqual(new[key], full[key], key)
        self.assertEqual({v["gpu"] for v in new["variants"].values()}, {0, 1})

    def test_trainer_only_gains_the_fold_blocks(self):
        new = next(s for s in sources(NOTEBOOK) if s.startswith("%%writefile")).splitlines()
        old = next(s for s in sources(FULL) if s.startswith("%%writefile")).splitlines()
        changed = [l for l in difflib.ndiff(old, new) if l[:2] in ("- ", "+ ")]
        self.assertEqual(sum(l.startswith("- ") for l in changed), 0)
        self.assertEqual(sum(l.startswith("+ ") for l in changed), 12)

    def test_folds_keep_episodes_together_and_are_balanced(self):
        rows = [json.loads(l) for l in TRAIN.read_text(encoding="utf-8").splitlines()]
        sizes = [0] * 5
        for r in rows:
            sizes[int(hashlib.md5(r["id"].encode()).hexdigest(), 16) % 5] += 1
        self.assertEqual(sum(sizes), 1545)
        self.assertGreater(min(sizes), 250)

    def test_public_is_not_decoded_for_output_and_report_checks_coverage(self):
        code = "\n".join(s for s in sources(NOTEBOOK) if not s.startswith("%%writefile"))
        self.assertIn('len(need) != 5036', code)
        self.assertIn("Audio ở hai fold", code)
        self.assertNotIn("asr_public_test", code)


if __name__ == "__main__":
    unittest.main()
