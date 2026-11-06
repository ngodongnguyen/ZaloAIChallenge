"""CPU checks for the PhoWhisper-large notebook: same LLM and agent as lm-B-052, only the ASR weights change."""
import ast
import contextlib
import importlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parent
NOTEBOOK = ROOT / "agent-qwen35-large/lifemate_qwen35_large.ipynb"
BEAM = ROOT / "agent-qwen35-beam5/lifemate_qwen35_beam5.ipynb"
TRAIN = ROOT.parent.parent / "dataset/lifemate/train+public_test/train/train.jsonl"
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


class Qwen35LargeNotebookTests(unittest.TestCase):
    def test_every_code_cell_parses(self):
        for s in sources(NOTEBOOK):
            ast.parse(s.split("\n", 1)[1] if s.startswith("%%writefile") else s)

    def test_inputs_pinned_in_notebook(self):
        kaggle = load(NOTEBOOK)["metadata"]["kaggle"]
        self.assertEqual([d["sourceId"] for d in kaggle["dataSources"]], [21271194, 365775452, 364801062, 363490930])
        self.assertTrue(kaggle["isGpuEnabled"] and kaggle["isInternetEnabled"])

    def test_model_and_agent_identical_to_beam_run(self):
        new, beam = config(NOTEBOOK), config(BEAM)
        for key in ("llm", "agent", "fewshot", "mode", "split", "subset", "asr_backend", "seed", "batch_size"):
            self.assertEqual(new["CONFIG"].get(key), beam["CONFIG"].get(key), key)
        self.assertEqual(new["EXPECT"], beam["EXPECT"])
        self.assertEqual(new["VARIANT"], "beam5_prompt")

    def test_only_whisper_training_libraries_are_new(self):
        new, beam = libs(NOTEBOOK), libs(BEAM)
        self.assertEqual(set(new) - set(beam), {"%%writefile /kaggle/working/lib/lmlib/whisper_audio.py",
                                                "%%writefile /kaggle/working/lib/lmlib/whisper_full.py"})
        self.assertEqual([k for k in beam if new.get(k) != beam[k]], [])

    def test_recipe_is_the_chosen_043_variant(self):
        asr = config(NOTEBOOK)["ASR"]
        self.assertEqual((asr["model"], asr["revision"]), ("vinai/PhoWhisper-large", "b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb"))
        self.assertEqual((asr["lr"], asr["epochs"], asr["optim"]), (3e-4, 4, "adamw"))
        self.assertEqual(asr["batch_size"] * asr["world_size"] * asr["grad_accum"], 16)
        self.assertEqual(asr["lora"], {"r": 32, "alpha": 64, "dropout": 0.05,
                                       "targets": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]})

    def test_parameter_budget(self):
        ns = config(NOTEBOOK)
        self.assertEqual(ns["TEXT_PARAMS"], 4205751296)
        self.assertLess(ns["TEXT_PARAMS"] + 6173655608 // 4, 6_000_000_000)
        code = "\n".join(sources(NOTEBOOK))
        self.assertIn("if kept != TEXT_PARAMS:", code)
        self.assertIn("if SYSTEM_PARAMS >= 6_000_000_000:", code)
        self.assertIn("k.startswith('model.language_model.')", code)

    def test_fail_fast_order(self):
        code = "\n".join(s for s in sources(NOTEBOOK) if not s.startswith("%%writefile"))
        at = code.index
        steps = ["'lmlib.qwen35smoke', 'merge'", "if kept != TEXT_PARAMS:", "VLLM_TEXT_ONLY_OK", "'--stage', 'smoke'", "'--stage', 'train'",
                 "run_decode('greedy'", "run_decode('beam'", "launch(str(path)", "run_replay("]
        for a, b in zip(steps, steps[1:]):
            self.assertLess(at(a), at(b), (a, b))

    def test_replay_finds_the_cache_and_training_output_is_elsewhere(self):
        code = "\n".join(s for s in sources(NOTEBOOK) if not s.startswith("%%writefile"))
        self.assertIn("ASR_BEAM = OUT / 'asr';", code)
        self.assertIn("PUBLIC_OUT = OUT / 'public'", code)
        self.assertIn("ASR_OUT = OUT / 'whisper_large'", code)
        self.assertIn("asr_cache_dirs=[str(ASR_BEAM)]", code)
        self.assertNotIn("run_decode('control'", code)

    def test_no_cached_outputs(self):
        for cell in load(NOTEBOOK)["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])

    def test_trainer_applies_lora_before_ddp_and_merges_before_saving(self):
        lib = libs(NOTEBOOK)["%%writefile /kaggle/working/lib/lmlib/whisper_full.py"]
        self.assertLess(lib.index("model = _add_lora(model, lora)"), lib.index("wrapped = DDP("))
        self.assertLess(lib.index("trainable = [p for p"), lib.index("wrapped = DDP("))
        self.assertLess(lib.index("model.merge_and_unload()"), lib.index('save_pretrained(out / "model"'))
        self.assertNotIn("clip_grad_norm_(model.parameters()", lib)

    def test_real_inputs_prepare_the_training_plan(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            lib = root / "lib/lmlib"
            lib.mkdir(parents=True)
            for head, body in libs(NOTEBOOK).items():
                if head.endswith(("__init__.py", "common.py", "whisper_full.py")):
                    (lib / Path(head.split()[1]).name).write_text(body.split("\n", 1)[1], encoding="utf-8")
            for part, source in (("train", TRAIN), ("public_test", PUBLIC)):
                (root / "input" / part).parent.mkdir(exist_ok=True)
                (root / "input" / part).symlink_to(source.parent)
            ck = root / "input/lm_q35_full/sft_q35_full/ckpt-000610"
            ck.mkdir(parents=True)
            for name in ("COMPLETE", "adapter_state.pt", "adapter_config.json"):
                (ck / name).write_text("x")
            ns = config(NOTEBOOK)
            sys.path.insert(0, str(root / "lib"))
            try:
                sys.modules.pop("lmlib", None)
                for name in [m for m in sys.modules if m.startswith("lmlib.")]:
                    sys.modules.pop(name)
                module = importlib.import_module("lmlib.whisper_full")
                cfg = dict(ns["CONFIG"], input_roots=[str(root / "input")])
                # A symlink in the root is not followed by **, so hand the split directories as roots too.
                cfg["input_roots"] = [str(root / "input"), str(TRAIN.parent.parent)]
                with self.assertRaises(ValueError):
                    module.prepare_inputs(dict(cfg, input_roots=[str(root / "missing")]), ns["ASR"], root / "out")
                (root / "out").mkdir()
                prepared = module.prepare_inputs(cfg, ns["ASR"], root / "out")
            finally:
                sys.path.pop(0)
                shutil.rmtree(root / "lib")
                for name in [m for m in sys.modules if m == "lmlib" or m.startswith("lmlib.")]:
                    sys.modules.pop(name)
        self.assertEqual(prepared["training_plan"]["total_steps"], 1260)
        self.assertEqual(prepared["training_plan"]["effective_batch"], 16)
        self.assertEqual(len(prepared["manifest"]["train"]), 5036)
        self.assertEqual(len(prepared["manifest"]["public"]), 1933)


if __name__ == "__main__":
    unittest.main()
