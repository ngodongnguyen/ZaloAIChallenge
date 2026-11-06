"""CPU checks for the warm-start notebook: same agent and dev set as lm-sft-041; only the training text, result order and full descriptions change."""
import ast
import contextlib
import importlib
import io
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parent
NOTEBOOK = ROOT / "agent-qwen35-noise/lifemate_qwen35_noise.ipynb"
SFT = ROOT / "agent-qwen35-sft/lifemate_qwen35_sft.ipynb"
SYN = ROOT / "agent-qwen35-syn/lifemate_qwen35_syn.ipynb"
TRAIN = ROOT.parent.parent / "dataset/lifemate/train+public_test/train"
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / "agent-qwen35-noise"), str(ROOT / "agent-qwen35-fuse")]


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


def lmlib(notebook, package):
    """A notebook's agent and sft modules, importable without torch."""
    d = tempfile.mkdtemp()
    pkg = Path(d) / package
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    for head, body in libs(notebook).items():
        name = Path(head.split()[1]).name
        if name in ("agent.py", "sft.py", "asr.py", "common.py", "data.py"):
            (pkg / name).write_text(body.split("\n", 1)[1], encoding="utf-8")
    sys.path.insert(0, d)
    return importlib.import_module(f"{package}.agent"), importlib.import_module(f"{package}.sft")


class NoiseNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent, cls.sft = lmlib(NOTEBOOK, "noiselib")
        cls.old_agent, cls.old_sft = lmlib(SFT, "sftlib")
        cls.rows = [json.loads(l) for l in (TRAIN / "train.jsonl").read_text(encoding="utf-8").splitlines()]
        cls.catalog = cls.agent.Catalog(json.loads((TRAIN / "train_tools_catalog.json").read_text(encoding="utf-8")))

    def test_every_code_cell_parses(self):
        for s in sources(NOTEBOOK):
            ast.parse(s.split("\n", 1)[1] if s.startswith("%%writefile") else s)

    def test_inputs_pinned(self):
        kaggle = load(NOTEBOOK)["metadata"]["kaggle"]
        self.assertEqual([d["sourceId"] for d in kaggle["dataSources"]][:6], [21271194, 364021094, 364609062, 364738806, 363490930, 365345962])
        self.assertTrue(kaggle["isGpuEnabled"] and kaggle["isInternetEnabled"])

    def test_same_agent_and_dev_as_041(self):
        new, old = config(NOTEBOOK)["CONFIG"], config(SFT)["CONFIG"]
        for key in ("llm", "agent", "fewshot", "subset", "asr_backend", "split", "seed", "expected_samples", "expected_audio"):
            self.assertEqual(new[key], old[key], key)

    def test_recipe_changes_only_what_is_stated(self):
        new, old = dict(config(NOTEBOOK)["SFT"]), dict(config(SFT)["SFT"])
        self.assertEqual((new.pop("epochs"), old.pop("epochs")), (1, 2))
        self.assertEqual((new.pop("seed"), old.pop("seed")), (2, 0))
        self.assertEqual((new.pop("asr_prob"), old.pop("asr_prob")), (1.0, 0.5))
        self.assertEqual(new.pop("shuffle_p"), 0.5)
        self.assertEqual(new, old)

    def test_library_changes(self):
        new, old = libs(NOTEBOOK), libs(SFT)
        self.assertEqual(set(new), set(old))
        changed = sorted(k.split("/")[-1] for k in new if new[k] != old[k])
        self.assertEqual(changed, ["agent.py", "q35ddp.py", "sft.py"])
        ddp = "%%writefile /kaggle/working/lib/lmlib/q35ddp.py"
        self.assertEqual(new[ddp], libs(SYN)[ddp])
        self.assertIn("warm start from", new[ddp])
        self.assertIn('bits.append(p["description"])', new["%%writefile /kaggle/working/lib/lmlib/agent.py"])

    def test_without_new_options_examples_match_041(self):
        raw = json.loads((TRAIN / "train_tools_catalog.json").read_text(encoding="utf-8"))
        old = self.old_sft.build_examples(self.rows[:200], self.old_agent.Catalog(raw), aug=True, seed=0)
        for app in raw["apps"]:
            for tool in app["tools"]:
                for p in (tool.get("parameters") or {}).get("properties", {}).values():
                    if p.get("description"):
                        p["description"] = p["description"][:80]
        new = self.sft.build_examples(self.rows[:200], self.agent.Catalog(raw), aug=True, seed=0)
        self.assertEqual(old, new)

    def test_shuffle_keeps_targets_and_records(self):
        rows = self.rows[:200]
        a = self.sft.build_examples(rows, self.catalog, seed=2)
        b = self.sft.build_examples(rows, self.catalog, seed=2, shuffle_p=1.0)
        self.assertEqual([x["target"] for x in a], [x["target"] for x in b])
        self.assertEqual([x["id"] for x in a], [x["id"] for x in b])
        self.assertTrue(any(x["messages"] != y["messages"] for x, y in zip(a, b)))
        self.assertEqual(self.sft._shuffle_lists({"ok": True, "items": [{"id": 1}, {"id": 2}], "tags": ["a", "b"]}, random.Random(1))["tags"], ["a", "b"])
        self.assertEqual(sorted(x["id"] for x in self.sft._shuffle_lists({"items": [{"id": 1}, {"id": 2}, {"id": 3}]}, random.Random(1))["items"]), [1, 2, 3])

    def test_order_within_a_sample_stays_the_same_across_steps(self):
        r = next(r for r in self.rows if len(r["label"]["tool_calls"]) >= 3 and not self.sft._says_position(r)
                 and any(isinstance(v, list) and len(v) > 1 for x in r["label"]["tool_results"][:1] if isinstance(x, dict) for v in x.values()))
        ex = self.sft.build_examples([r], self.catalog, seed=2, shuffle_p=1.0)
        first = ex[1]["messages"][1]["content"].split("# CÁC BƯỚC ĐÃ LÀM CHO YÊU CẦU CUỐI\n")[1].split("\n2. ")[0].split("\n\nTrả về")[0]
        later = ex[2]["messages"][1]["content"]
        self.assertIn(first, later)

    def test_position_words_block_shuffling(self):
        self.assertTrue(self.sft._says_position({"audio_text": "Chọn cái đầu tiên nhé"}))
        self.assertFalse(self.sft._says_position({"audio_text": "Chuyển tiền cho mẹ"}))
        share = sum(self.sft._says_position(r) for r in self.rows) / len(self.rows)
        self.assertLess(share, 0.25, share)

    def test_full_description_reaches_the_prompt(self):
        spec = self.agent._param_spec("x", {"type": "string", "description": "a" * 200}, set())
        self.assertIn("a" * 200, spec)

    def test_noisy_text_keeps_format_and_takes_the_errors(self):
        from asr_fuse import TrigramLM
        from make_noisy_cache import noisy_text
        lm = TrigramLM(["chuyển ba mươi nghìn cho mẹ"])
        self.assertEqual(noisy_text("Chuyển 30 nghìn cho mẹ.", "CHUYỂN BA MƯƠI NGHÌN CHO MẸ", lm), "Chuyển 30 nghìn cho mẹ.")
        self.assertEqual(noisy_text("Bật đèn phòng khách nhé.", "BẬT ĐÈN PHÒNG KHÁCH NHA", lm).lower().split()[:4], ["bật", "đèn", "phòng", "khách"])
        self.assertIn("nha", noisy_text("Bật đèn phòng khách nhé.", "BẬT ĐÈN PHÒNG KHÁCH NHA", lm).lower())

    def test_warm_start_and_resume_paths_do_not_mix(self):
        code = "\n".join(s for s in sources(NOTEBOOK) if not s.startswith("%%writefile"))
        self.assertIn('INIT_DIR = one("**/lm_q35/sft_q35"', code)
        self.assertIn('SFT_DIR = OUT / "sft_q35_noise"', code)
        self.assertIn('**/sft_q35_noise"', code)
        self.assertIn('init_adapter=str(INIT_DIR)', code)
        self.assertIn('"warm start from" not in', code)
        self.assertIn('!= "ckpt-000190"', code)
        self.assertNotIn('PREPARED["init_adapter"]', code)

    def test_no_cached_outputs(self):
        for cell in load(NOTEBOOK)["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])


if __name__ == "__main__":
    unittest.main()
