"""CPU checks for the thinking notebook: same transcripts and agent as lm-sft-041, thinking on, nothing trained."""
import ast
import contextlib
import io
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).parent
NOTEBOOK = ROOT / "agent-qwen35-think/lifemate_qwen35_think.ipynb"
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


def libs(path):
    return {s.splitlines()[0]: s for s in sources(path) if s.startswith("%%writefile")}


class ThinkNotebookTests(unittest.TestCase):
    def test_every_code_cell_parses(self):
        for s in sources(NOTEBOOK):
            ast.parse(s.split("\n", 1)[1] if s.startswith("%%writefile") else s)

    def test_inputs_pinned(self):
        kaggle = load(NOTEBOOK)["metadata"]["kaggle"]
        self.assertEqual([d["sourceId"] for d in kaggle["dataSources"]], [21271194, 364021094, 364609062, 364738806, 363490930, 365345962])
        self.assertTrue(kaggle["isGpuEnabled"] and kaggle["isInternetEnabled"])

    def test_same_agent_as_041_and_no_resume(self):
        new, old = config(NOTEBOOK)["CONFIG"], config(SFT)["CONFIG"]
        for key in ("llm", "agent", "fewshot", "subset", "asr_backend", "split", "seed"):
            self.assertEqual(new[key], old[key], key)
        self.assertEqual(new["resume_from_dirs"], [])

    def test_only_llm_library_differs(self):
        new, old = libs(NOTEBOOK), libs(SFT)
        self.assertEqual(set(new), set(old))
        self.assertEqual([k for k in new if new[k] != old[k]], ["%%writefile /kaggle/working/lib/lmlib/llm.py"])

    def test_thought_is_removed_before_parsing(self):
        lib = libs(NOTEBOOK)["%%writefile /kaggle/working/lib/lmlib/llm.py"]
        self.assertIn('text = text.rsplit("</think>", 1)[1] if "</think>" in text else ""', lib)
        ns = {}
        code = 'def strip(text, thinking=True):\n    if thinking:\n        text = text.rsplit("</think>", 1)[1] if "</think>" in text else ""\n    return text\n'
        exec(code, ns)
        self.assertEqual(ns["strip"]('nghĩ {"name": "x"}\n</think>\n\n{"done": true}').strip(), '{"done": true}')
        self.assertEqual(ns["strip"]('đang nghĩ {"name": "x"}'), "")

    def test_two_branches_each_on_its_gpu_with_thinking(self):
        code = "\n".join(s for s in sources(NOTEBOOK) if not s.startswith("%%writefile"))
        self.assertIn("'lora_think': {'gpu': 0, 'model': str(MERGED), 'k': 0}", code)
        self.assertIn("'base_think': {'gpu': 1, 'model': Q35_LOCAL, 'k': 3}", code)
        self.assertIn("enable_thinking=True", code)
        self.assertIn("num_shards=1", code)
        self.assertIn("if True:  # merge LoRA 041 cần PEFT", code)
        self.assertNotIn("flash-linear-attention", code)
        for gone in ("dual_", "train(", "run_decode"):
            self.assertNotIn(gone, code)

    def test_cell_order(self):
        code = "\n".join(s for s in sources(NOTEBOOK) if not s.startswith("%%writefile"))
        steps = ["prepare(CONFIG, OUT)", "snapshot_download(", "'lmlib.qwen35smoke', 'merge'", "threading.Thread", "paired(REF_041"]
        for a, b in zip(steps, steps[1:]):
            self.assertLess(code.index(a), code.index(b), (a, b))

    def test_no_cached_outputs(self):
        for cell in load(NOTEBOOK)["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])


if __name__ == "__main__":
    unittest.main()
