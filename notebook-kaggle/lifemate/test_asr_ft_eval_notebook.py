"""CPU checks for the paired ASR fine-tune agent evaluation notebook."""
import contextlib
import copy
import importlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).parent
NOTEBOOK = HERE / "agent-asr-ft-eval/lifemate_agent_asr_ft_eval.ipynb"


class AsrFtEvalTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(NOTEBOOK.exists(), "The importable ASR FT eval notebook is missing")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.nb = json.loads(NOTEBOOK.read_text())
        lib = self.root / "lib"
        for cell in self.nb["cells"]:
            source = "".join(cell["source"])
            if source.startswith("%%writefile /kaggle/working/lib/"):
                head, body = source.split("\n", 1)
                p = lib / head.split("/kaggle/working/lib/", 1)[1]
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(body)
        saved = {k: sys.modules.pop(k) for k in list(sys.modules)
                 if k == "lmlib" or k.startswith("lmlib.")}
        sys.path.insert(0, str(lib))

        def restore():
            sys.path.remove(str(lib))
            for k in list(sys.modules):
                if k == "lmlib" or k.startswith("lmlib."):
                    del sys.modules[k]
            sys.modules.update(saved)
        self.addCleanup(restore)
        self.mod = importlib.import_module("lmlib.asr_ft_eval")
        self.run = importlib.import_module("lmlib.run")
        ns = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(next("".join(c["source"]) for c in self.nb["cells"]
                      if "".join(c["source"]).startswith("CONFIG = {")), ns)
        self.cfg = copy.deepcopy(ns["CONFIG"])
        self.cfg.update(input_roots=[str(self.root / "input")], subset=["x", "y"],
                        expected_samples=2, expected_audio=3)
        self.out = self.root / "output"
        self.ckpt = self.root / "input/agent/lm_agent/sft_ep2/ckpt-000380"
        self.ckpt.mkdir(parents=True)
        for name in ("COMPLETE", "adapter_state.pt", "adapter_config.json"):
            (self.ckpt / name).write_text("{}")
        self.rows = [
            {"id": "x", "audio": "x.wav", "audio_text": "GOLD SECRET",
             "history": [{"role": "user", "audio": "history.wav", "audio_text": "SECRET"}]},
            {"id": "y", "audio": "y.wav", "history": []},
        ]
        split = self.root / "input/dataset/train"
        split.mkdir(parents=True)
        (split / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.rows))
        self.gip = self.root / "input/gip/gip_ft/lr15/asr_train_gipft_lr15.shard0.jsonl"
        self.whisper = self.root / "input/whisper/asr_ft/medium/asr_train_medium.shard0.jsonl"
        self.write_cache(self.gip, "gipft-lr15-ep4", "GIP")
        self.write_cache(self.whisper, "asrft-medium-ep5", "WHISPER")

    def write_cache(self, path, tag, text, missing=()):
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"_header": {"run_hash": tag, "created": 1}}]
        rows += [{"id": audio, "status": "completed", "text": text + " " + audio}
                 for audio in ["x.wav", "y.wav", "history.wav", "holdout.wav"] if audio not in missing]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def test_each_variant_reads_its_own_final_and_history_transcripts(self):
        prepared = self.mod.prepare_inputs(self.cfg, self.out)
        original = copy.deepcopy(self.cfg)
        for name, text in [("gipft_lr15", "GIP"), ("whisper_medium", "WHISPER")]:
            cfg = self.mod.variant_config(self.cfg, prepared, name)
            transcripts = self.run.transcripts_for(cfg, self.rows)
            self.assertEqual(transcripts, {"x.wav": text + " x.wav", "y.wav": text + " y.wav",
                                           "history.wav": text + " history.wav"})
            self.assertEqual(cfg["agent"], original["agent"])
            self.assertEqual(cfg["llm"], original["llm"])
            self.assertEqual(cfg["mode"], "e2e")
            self.assertNotIn("audio_text", self.run.inference_row(self.rows[0], cfg["mode"]))
            self.assertNotIn("audio_text", self.run.inference_row(self.rows[0], cfg["mode"])["history"][0])
        self.assertEqual(self.cfg, original)
        self.assertEqual(Path(prepared["adapter_dir"], "ckpt-000380").resolve(), self.ckpt.resolve())

    def test_missing_history_stops_before_any_model_runs(self):
        self.write_cache(self.gip, "gipft-lr15-ep4", "GIP", missing=["history.wav"])
        with self.assertRaisesRegex(ValueError, "missing.*history.wav"):
            self.mod.prepare_inputs(self.cfg, self.out)

    def test_wrong_epoch_and_ambiguous_input_are_rejected(self):
        self.write_cache(self.gip, "gipft-lr15-ep15", "GIP")
        with self.assertRaisesRegex(ValueError, "run_hash"):
            self.mod.prepare_inputs(self.cfg, self.out)
        self.write_cache(self.gip, "gipft-lr15-ep4", "GIP")
        extra = self.root / "input/duplicate/gip_ft/lr15/asr_train_gipft_lr15.shard0.jsonl"
        self.write_cache(extra, "gipft-lr15-ep4", "OTHER")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.mod.prepare_inputs(self.cfg, self.out)

    def test_incomplete_checkpoint_and_conflicting_audio_are_rejected(self):
        (self.ckpt / "COMPLETE").unlink()
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            self.mod.prepare_inputs(self.cfg, self.out)
        (self.ckpt / "COMPLETE").write_text("{}")
        with self.gip.open("a") as f:
            f.write(json.dumps({"id": "x.wav", "status": "completed", "text": "WRONG"}) + "\n")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.mod.prepare_inputs(self.cfg, self.out)

    def test_failed_or_partial_workers_never_count_as_complete(self):
        good = {"baseline_complete": True, "n": 2, "completed": 2, "schema_errors": 0, "score": 50.0}
        self.mod.require_complete([0, 0], good, 2)
        for codes, report in [([1, 0], good), ([0, 0], dict(good, completed=1)),
                              ([0, 0], dict(good, schema_errors=1)),
                              ([0, 0], dict(good, baseline_complete=False))]:
            with self.assertRaises(RuntimeError):
                self.mod.require_complete(codes, report, 2)

    def test_paired_summary_joins_ids_and_rejects_incomplete_scores(self):
        for name, scores in [("gipft_lr15", [("y", 0.5), ("x", 1.0)]),
                             ("whisper_medium", [("x", 0.0), ("y", 1.0)])]:
            directory = self.out / ("eval_" + name)
            directory.mkdir(parents=True)
            (directory / "per_id_scores.jsonl").write_text("".join(
                json.dumps({"id": qid, "score": score, "hash_ok": score > 0, "pending": False}) + "\n"
                for qid, score in scores))
        result = self.mod.compare_scores(self.out, ["x", "y"])
        self.assertEqual(result["delta_points"], 25.0)
        self.assertEqual((result["improved"], result["worsened"]), (1, 1))
        self.assertTrue((self.out / "paired_per_id.jsonl").exists())
        (self.out / "eval_whisper_medium/per_id_scores.jsonl").write_text(
            json.dumps({"id": "x", "score": 0.0, "pending": False}) + "\n")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.mod.compare_scores(self.out, ["x", "y"])


if __name__ == "__main__":
    unittest.main()
