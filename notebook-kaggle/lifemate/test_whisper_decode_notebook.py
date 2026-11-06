"""CPU checks for lm-A-049: prompts never see gold text, the WER gate works, every notebook cell compiles."""
import ast
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).parent
NB = json.loads((ROOT / "asr-decode/lifemate_whisper_decode.ipynb").read_text(encoding="utf-8"))
CELLS = ["".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "code"]
SOURCE = next(c for c in CELLS if c.startswith("%%writefile /kaggle/working/lib/lmlib/whisper_decode.py")).split("\n", 1)[1]


def load_module():
    """The module without lmlib: its three text helpers are stubbed with the real behaviour they need here."""
    pkg = types.ModuleType("fakepkg")
    pkg.__path__ = []
    asr = types.ModuleType("fakepkg.asr")
    asr.norm_text = lambda s: s.lower()
    asr.read_wave = None

    def edit_distance(a, b):
        prev = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            cur = [i]
            for j, y in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
            prev = cur
        return prev[-1]

    asr.edit_distance = edit_distance
    common = types.ModuleType("fakepkg.common")
    common.atomic_write_json = None
    sys.modules.update({"fakepkg": pkg, "fakepkg.asr": asr, "fakepkg.common": common})
    spec = importlib.util.spec_from_loader("fakepkg.whisper_decode", loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "fakepkg"
    sys.modules["fakepkg.whisper_decode"] = mod
    exec(compile(SOURCE, "whisper_decode.py", "exec"), mod.__dict__)
    return mod


wd = load_module()
ROW = {"context": {"user_profile": {"name": "Bác Tư"}},
       "history": [{"role": "user", "audio": "u1.wav", "audio_text": "CÂU CHUẨN MỘT"},
                   {"role": "assistant", "text": "Đã gọi Chú Hùng.", "tool_results": [{"ok": True, "id": "c_01", "name": "Chú Hùng", "at": "2026-11-05T10:00:00+07:00"}]},
                   {"role": "user", "audio": "u2.wav", "audio_text": "CÂU CHUẨN HAI"},
                   {"role": "assistant", "text": "Vâng ạ."}],
       "audio": "final.wav", "audio_text": "CÂU CHUẨN CUỐI"}
EARLIER = {"u1.wav": "nghe một", "u2.wav": "nghe hai"}


class PromptTests(unittest.TestCase):
    def test_final_turn_sees_the_whole_history_but_no_gold_text(self):
        p = wd.context_prompt(ROW, "final.wav", EARLIER)
        self.assertNotIn("CHUẨN", p)
        for piece in ("Bác Tư", "Chú Hùng", "nghe một", "nghe hai", "Vâng ạ."):
            self.assertIn(piece, p)
        self.assertTrue(p.endswith("Vâng ạ."))

    def test_history_turn_sees_only_earlier_turns(self):
        p = wd.context_prompt(ROW, "u2.wav", EARLIER)
        self.assertIn("nghe một", p)
        self.assertNotIn("nghe hai", p)
        self.assertNotIn("Vâng ạ.", p)
        self.assertEqual(wd.context_prompt(ROW, "u1.wav", EARLIER), "Bác Tư")

    def test_ids_and_dates_are_not_names(self):
        self.assertEqual(wd._names(ROW["history"][1]["tool_results"], []), ["Chú Hùng"])


class GateTests(unittest.TestCase):
    def test_wer_and_gain(self):
        gold = {f"a{i}": "một hai ba bốn" for i in range(50)}
        base = wd.word_stats(gold, {a: "một hai ba năm" for a in gold})
        same = wd.word_stats(gold, {a: "một hai ba bốn" for a in gold})
        self.assertEqual(wd.wer(base), 0.25)
        self.assertEqual(wd.gain_ci(base, same, n=50), {"gain_points": 25.0, "ci95": [25.0, 25.0]})


class NotebookTests(unittest.TestCase):
    def test_code_cells_parse(self):
        for c in CELLS:
            ast.parse("\n".join(l for l in c.splitlines() if not l.startswith(("%%", "!", "%"))) if not c.startswith("%%writefile") else c.split("\n", 1)[1])

    def test_metadata_and_gate_guard_the_paid_steps(self):
        k = NB["metadata"]["kaggle"]
        self.assertTrue(k["isGpuEnabled"] and k["isInternetEnabled"])
        self.assertEqual([d["sourceId"] for d in k["dataSources"]], [21271194, 364021094, 364609062, 364738806, 363490930, 365345962])
        for c in CELLS:
            if "lmlib.qwen35smoke', 'merge'" in c or "launch(str(cfg_path)" in c:
                self.assertTrue(c.lstrip().startswith(("if GATE_PASSED", "REPORT, cmp = None, None\nif GATE_PASSED")), c[:60])
        self.assertLess(CELLS.index(next(c for c in CELLS if c.startswith("%%writefile /kaggle/working/lib/lmlib/whisper_decode.py"))),
                        CELLS.index(next(c for c in CELLS if "run_decode('sweep'" in c)))


if __name__ == "__main__":
    unittest.main()
