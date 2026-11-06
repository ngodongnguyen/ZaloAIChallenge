"""CPU checks for the fused-transcript notebook: same LLM and agent as lm-B-052, only the transcript cache changes."""
import ast
import contextlib
import io
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).parent
NOTEBOOK = ROOT / "agent-qwen35-fuse/lifemate_qwen35_fuse.ipynb"
BEAM = ROOT / "agent-qwen35-beam5/lifemate_qwen35_beam5.ipynb"


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


class Qwen35FuseNotebookTests(unittest.TestCase):
    def test_every_code_cell_parses(self):
        for s in sources(NOTEBOOK):
            ast.parse(s.split("\n", 1)[1] if s.startswith("%%writefile") else s)

    def test_model_and_agent_identical_to_beam_run(self):
        new, beam = config(NOTEBOOK), config(BEAM)
        for key in ("llm", "agent", "fewshot", "mode", "split", "subset", "asr_backend", "seed", "batch_size"):
            self.assertEqual(new["CONFIG"].get(key), beam["CONFIG"].get(key), key)
        self.assertEqual(new["EXPECT"]["checkpoint_sha256"], beam["EXPECT"]["checkpoint_sha256"])
        self.assertEqual(len(new["EXPECT"]["fuse_cache_sha256"]), 64)

    def test_libraries_unchanged_and_no_decoder(self):
        new, beam = libs(NOTEBOOK), libs(BEAM)
        self.assertEqual(set(beam) - set(new), {"%%writefile /kaggle/working/lib/lmlib/whisper_decode.py"})
        self.assertEqual(set(new) - set(beam), set())
        self.assertEqual([k for k in new if new[k] != beam[k]], [])

    def test_cache_is_where_replay_reads_it(self):
        code = "\n".join(s for s in sources(NOTEBOOK) if not s.startswith("%%writefile"))
        self.assertIn("ASR_DIR = OUT / 'asr';", code)
        self.assertIn("PUBLIC_OUT = OUT / 'public'", code)
        self.assertIn("asr_cache_dirs=[str(ASR_DIR)]", code)
        self.assertIn("**/lifemate-asr-fuse/asr_public_test_whisper_full.shard0.jsonl", code)
        for gone in ("WHISPER_MODEL", "run_decode", "asr_greedy", "EXPECT['asr_"):
            self.assertNotIn(gone, code)

    def test_cell_order(self):
        code = "\n".join(s for s in sources(NOTEBOOK) if not s.startswith("%%writefile"))
        steps = ["EXPECT['fuse_cache_sha256']", "'lmlib.qwen35smoke', 'merge'", "launch(str(path)", "run_replay("]
        for a, b in zip(steps, steps[1:]):
            self.assertLess(code.index(a), code.index(b), (a, b))

    def test_expected_sha_matches_local_report_when_present(self):
        report = ROOT / "agent-qwen35-fuse/results/fuse_report.json"
        if report.is_file():
            self.assertEqual(config(NOTEBOOK)["EXPECT"]["fuse_cache_sha256"], json.loads(report.read_text())["cache_sha256"])

    def test_no_cached_outputs(self):
        for cell in load(NOTEBOOK)["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])


if __name__ == "__main__":
    unittest.main()
