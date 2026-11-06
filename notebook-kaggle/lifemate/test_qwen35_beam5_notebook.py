"""CPU checks for the beam5_prompt notebook: same model and agent as lm-sft-046, only the ASR decoding changes."""
import ast
import contextlib
import importlib.util
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parent
NOTEBOOK = ROOT / "agent-qwen35-beam5/lifemate_qwen35_beam5.ipynb"
FULL = ROOT / "agent-qwen35-full/lifemate_qwen35_full.ipynb"
PUBLIC = ROOT.parent.parent / "dataset/lifemate/train+public_test/public_test/public_test.jsonl"


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


class Qwen35Beam5NotebookTests(unittest.TestCase):
    def test_every_code_cell_parses(self):
        for s in sources(NOTEBOOK):
            ast.parse(s.split("\n", 1)[1] if s.startswith("%%writefile") else s)

    def test_inputs_pinned_in_notebook(self):
        kaggle = load(NOTEBOOK)["metadata"]["kaggle"]
        self.assertEqual([d["sourceId"] for d in kaggle["dataSources"]], [21271194, 365775452, 364801062, 363490930])
        self.assertTrue(kaggle["isGpuEnabled"] and kaggle["isInternetEnabled"])

    def test_model_and_agent_identical_to_full_run(self):
        ns, full = config(NOTEBOOK), config(FULL)
        for key in ("llm", "agent", "fewshot", "mode", "split", "subset", "asr_backend", "seed", "batch_size", "max_steps"):
            self.assertEqual(ns["CONFIG"].get(key), full["CONFIG"].get(key), key)
        self.assertEqual(ns["EXPECT"]["checkpoint_sha256"], "b0477e043f0017094990bfe862bb1e415ac4fb0b8e67cd45e6925141325b9f70")
        self.assertEqual(ns["EXPECT"]["asr_identity"], full["EXPECT"]["asr_identity"])
        self.assertEqual(ns["EXPECT"]["asr_cache_sha256"], full["EXPECT"]["asr_cache_sha256"])

    def test_only_whisper_decode_library_is_new(self):
        new, full = libs(NOTEBOOK), libs(FULL)
        self.assertEqual(set(new) - set(full), {"%%writefile /kaggle/working/lib/lmlib/whisper_decode.py"})
        changed = [k for k in full if new.get(k) != full[k]]
        self.assertEqual(changed, [])

    def test_decode_uses_same_weights_and_precision_as_cache(self):
        code = "\n".join(sources(NOTEBOOK))
        self.assertIn("**/lm_whisper_full/asr/model/model.safetensors", code)
        self.assertIn("'dtype': 'float16'", code)
        self.assertIn('torch_dtype=getattr(torch, cfg.get("dtype", "float32"))', code)
        self.assertIn("if same < CONTROL_MIN_SAME:", code)

    def test_order_decode_before_merge_before_agent(self):
        code = sources(NOTEBOOK)
        at = lambda marker: next(i for i, s in enumerate(code) if marker in s and not s.startswith("%%writefile"))
        self.assertLess(at("snapshot_download("), at("run_decode('control'"))
        self.assertLess(at("run_decode('beam'"), at("lmlib.qwen35smoke"))
        self.assertLess(at("lmlib.qwen35smoke"), at("launch(str(path)"))
        self.assertLess(at("launch(str(path)"), at("run_replay("))

    def test_agent_reads_only_the_new_cache(self):
        code = "\n".join(s for s in sources(NOTEBOOK) if not s.startswith("%%writefile"))
        self.assertEqual(len(re.findall(r"asr_cache_dirs=", code)), 1)
        self.assertIn("asr_cache_dirs=[str(ASR_BEAM)]", code)
        self.assertIn("asr_public_test_whisper_full.shard0.jsonl", code)
        self.assertIn("OUT / 'asr_greedy'", code)
        # replay_actions đọc cache ở <PUBLIC_OUT.parent>/asr; greedy nằm ở thư mục khác nên không lẫn vào.
        self.assertIn("ASR_BEAM = OUT / 'asr';", code)
        self.assertIn("PUBLIC_OUT = OUT / 'public'", code)

    def test_context_prompt_on_real_public_rows(self):
        lib = next(s for s in sources(NOTEBOOK) if s.startswith("%%writefile /kaggle/working/lib/lmlib/whisper_decode.py"))
        with tempfile.TemporaryDirectory() as d:
            pkg = Path(d) / "pkg"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")
            (pkg / "asr.py").write_text("edit_distance = norm_text = read_wave = None\n")
            (pkg / "common.py").write_text("atomic_write_json = None\n")
            (pkg / "whisper_decode.py").write_text(lib.split("\n", 1)[1], encoding="utf-8")
            sys.path.insert(0, d)
            try:
                spec = importlib.util.spec_from_file_location("pkg.whisper_decode", pkg / "whisper_decode.py", submodule_search_locations=None)
                import pkg as _pkg  # noqa: F401
                mod = importlib.util.module_from_spec(spec)
                sys.modules["pkg.whisper_decode"] = mod
                spec.loader.exec_module(mod)
            finally:
                sys.path.remove(d)
                for name in [n for n in sys.modules if n == "pkg" or n.startswith("pkg.")]:
                    sys.modules.pop(name)
        if not PUBLIC.is_file():
            self.skipTest("dataset local không có")
        rows = [json.loads(line) for line in PUBLIC.read_text(encoding="utf-8").splitlines()]
        greedy = {t["audio"]: f"GREEDY {t['audio']}" for r in rows for t in r.get("history", []) if t.get("audio")}
        with_history = next(r for r in rows if r.get("history"))
        first = with_history["history"][0]["audio"]
        self.assertNotIn("GREEDY", mod.context_prompt(with_history, first, greedy))
        final = mod.context_prompt(with_history, with_history["audio"], greedy)
        self.assertIn(f"GREEDY {first}", final)
        self.assertNotIn(with_history["audio"], final)

    def test_no_cached_outputs(self):
        for cell in load(NOTEBOOK)["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])


if __name__ == "__main__":
    unittest.main()
