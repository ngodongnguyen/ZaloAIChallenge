"""CPU checks for full-train Whisper -> full agent -> public submission."""
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
NOTEBOOK = HERE / "asr-full-public/lifemate_whisper_full_public.ipynb"


class WhisperFullPublicTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(NOTEBOOK.exists(), "Full-training/public notebook is missing")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.nb = json.loads(NOTEBOOK.read_text())
        lib = self.root / "lib"
        for cell in self.nb["cells"]:
            src = "".join(cell["source"])
            if src.startswith("%%writefile /kaggle/working/lib/"):
                head, body = src.split("\n", 1)
                path = lib / head.split("/kaggle/working/lib/")[1]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(body)
        saved = {k: sys.modules.pop(k) for k in list(sys.modules)
                 if k == "lmlib" or k.startswith("lmlib.")}
        sys.path.insert(0, str(lib))

        def restore():
            sys.path.remove(str(lib))
            for key in list(sys.modules):
                if key == "lmlib" or key.startswith("lmlib."):
                    del sys.modules[key]
            sys.modules.update(saved)
        self.addCleanup(restore)
        self.mod = importlib.import_module("lmlib.whisper_full")
        ns = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(next("".join(c["source"]) for c in self.nb["cells"]
                      if "".join(c["source"]).startswith("CONFIG = {")), ns)
        self.cfg, self.asr = copy.deepcopy(ns["CONFIG"]), copy.deepcopy(ns["ASR"])
        self.train = [
            {"id": "train_a", "audio": "audio/a.wav", "audio_text": "gọi mẹ",
             "history": [{"role": "user", "audio": "audio/a_u1.wav", "audio_text": "xin chào"}]},
            {"id": "old_dev", "audio": "audio/b.wav", "audio_text": "bật đèn", "history": []},
            {"id": "old_holdout", "audio": "audio/c.wav", "audio_text": "mở nhạc", "history": []},
        ]
        self.public = [{"id": "public_a", "audio": "audio/p.wav", "audio_text": "MUST NOT TRAIN",
                        "label": {"secret": "MUST NOT READ"},
                        "history": [{"role": "user", "audio": "audio/p_u1.wav", "audio_text": "SECRET"}]}]

    def test_full_manifest_includes_old_dev_holdout_history_and_no_public_targets(self):
        manifest = self.mod.build_full_manifest(self.train, self.public)
        self.assertEqual({r["episode"] for r in manifest["train"]}, {"train_a", "old_dev", "old_holdout"})
        self.assertEqual({r["audio"] for r in manifest["train"]},
                         {"audio/a.wav", "audio/a_u1.wav", "audio/b.wav", "audio/c.wav"})
        self.assertEqual({r["text"] for r in manifest["train"]}, {"gọi mẹ", "xin chào", "bật đèn", "mở nhạc"})
        self.assertEqual({r["audio"] for r in manifest["public"]}, {"audio/p.wav", "audio/p_u1.wav"})
        self.assertTrue(all("text" not in r and "label" not in r for r in manifest["public"]))

    def test_duplicate_audio_or_cross_split_audio_is_rejected(self):
        bad = copy.deepcopy(self.train)
        bad[1]["audio"] = bad[0]["audio"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.mod.build_full_manifest(bad, self.public)
        bad = copy.deepcopy(self.public)
        bad[0]["audio"] = self.train[0]["audio"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.mod.build_full_manifest(self.train, bad)

    def test_training_plan_and_final_accumulation_window(self):
        plan = self.mod.training_plan(5036, 2, 4, 2, 5)
        self.assertEqual((plan["steps_per_epoch"], plan["total_steps"], plan["effective_batch"]), (315, 1575, 16))
        self.assertEqual([self.mod.accum_window(i, 5, 2) for i in range(5)],
                         [(2, False), (2, True), (2, False), (2, True), (1, True)])

    def test_preflight_requires_full_checkpoint_and_all_train_audio(self):
        source = self.root / "input"
        for name, rows in [("train", self.train), ("public_test", self.public)]:
            split = source / "dataset" / name
            split.mkdir(parents=True)
            (split / (name + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in rows))
            for row in rows:
                for turn in row.get("history", []) + [row]:
                    audio = split / turn["audio"]
                    audio.parent.mkdir(exist_ok=True)
                    audio.write_bytes(b"fixture")
        ck = source / "agent/lm_agent/sft_full/ckpt-000610"
        ck.mkdir(parents=True)
        for name in ["COMPLETE", "adapter_config.json", "adapter_state.pt"]:
            (ck / name).write_text("{}")
        self.cfg.update(input_roots=[str(source)], expected_train_samples=3, expected_train_audio=4,
                        expected_public_samples=1, expected_public_audio=2)
        prepared = self.mod.prepare_inputs(self.cfg, self.asr, self.root / "out")
        self.assertEqual(prepared["checkpoint"], str(ck))
        self.assertEqual(len(prepared["manifest"]["train"]), 4)
        (source / "dataset/train/audio/b.wav").unlink()
        with self.assertRaisesRegex(ValueError, "audio"):
            self.mod.prepare_inputs(self.cfg, self.asr, self.root / "out")

    def test_incomplete_training_never_allows_public_inference(self):
        out = self.root / "asr"
        model = out / "model"
        model.mkdir(parents=True)
        for name in ["config.json", "model.safetensors", "preprocessor_config.json", "tokenizer.json"]:
            (model / name).write_text("fixture")
        good = {"identity": "recipe-full", "finished_epochs": 5, "planned_epochs": 5,
                "optimizer_steps": 1575, "planned_steps": 1575, "n_train": 5036}
        (out / "training_summary.json").write_text(json.dumps(dict(good, finished_epochs=4)))
        (out / "COMPLETE").write_text("recipe-full")
        with self.assertRaises(RuntimeError):
            self.mod.require_asr_complete(out, "recipe-full", 5, 5036)
        (out / "training_summary.json").write_text(json.dumps(good))
        self.mod.require_asr_complete(out, "recipe-full", 5, 5036)
        with self.assertRaises(RuntimeError):
            self.mod.require_asr_complete(out, "another-model", 5, 5036)

    def test_public_cache_requires_own_model_and_complete_history(self):
        manifest = self.mod.build_full_manifest(self.train, self.public)
        directory = self.root / "asr"
        directory.mkdir()
        for shard in range(2):
            rows = [{"_header": {"run_hash": "recipe-full"}}]
            rows += [{"id": it["audio"], "text": "ASR " + it["audio"], "status": "completed"}
                     for it in manifest["public"] if self.mod.shard_of(it["audio"], 2) == shard]
            (directory / f"asr_public_test_whisper_full.shard{shard}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows))
        texts = self.mod.validate_public_cache(manifest, directory, "recipe-full", 2)
        self.assertEqual(set(texts), {"audio/p.wav", "audio/p_u1.wav"})
        with self.assertRaisesRegex(ValueError, "identity"):
            self.mod.validate_public_cache(manifest, directory, "old-split-model", 2)
        shard = self.mod.shard_of("audio/p_u1.wav", 2)
        path = directory / f"asr_public_test_whisper_full.shard{shard}.jsonl"
        rows = [json.loads(l) for l in path.read_text().splitlines()]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows if r.get("id") != "audio/p_u1.wav"))
        with self.assertRaisesRegex(ValueError, "missing"):
            self.mod.validate_public_cache(manifest, directory, "recipe-full", 2)

    def test_final_export_rejects_partial_reports_and_wrong_ids(self):
        source = self.root / "candidate.jsonl"
        source.write_text(json.dumps({"id": "public_a", "tool_calls": [], "state_hash": "a" * 64}) + "\n")
        report = {"baseline_complete": True, "n": 1, "completed": 1, "pending": 0,
                  "schema_errors": 0, "submission": str(source)}
        self.mod.check_submission(report, ["public_a"])
        for bad in [dict(report, completed=0), dict(report, schema_errors=1), dict(report, baseline_complete=False)]:
            with self.assertRaises(RuntimeError):
                self.mod.check_submission(bad, ["public_a"])
        with self.assertRaisesRegex(ValueError, "IDs"):
            self.mod.check_submission(report, ["public_other"])

    def test_official_cli_exit_zero_does_not_hide_schema_errors(self):
        self.mod.check_official_validation(0, "634 lines checked, 0 schema errors\n", 634)
        for code, output in [(0, "634 lines checked, 2 schema errors\n"),
                             (0, "633 lines checked, 0 schema errors\n"),
                             (0, "unexpected output\n"),
                             (1, "634 lines checked, 0 schema errors\n")]:
            with self.assertRaises(RuntimeError):
                self.mod.check_official_validation(code, output, 634)


if __name__ == "__main__":
    unittest.main()
