import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


class ReplayRequestTests(unittest.TestCase):
    def test_request_must_match_the_verified_source_asr_not_just_state_hash(self):
        directory = Path(__file__).parent / "agent-action-fix"
        sys.path.insert(0, str(directory))
        try:
            spec = importlib.util.spec_from_file_location("replay_actions", directory / "replay_actions.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            sys.path.remove(str(directory))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, asr = root / "public", root / "asr"
            baseline.mkdir(); asr.mkdir()
            (baseline / "config.json").write_text(json.dumps({"split": "public_test", "asr_backend": "whisper_full", "asr_model_identity": "model"}))
            source = [{"_header": {"run_hash": "model"}}, {"id": "p.wav", "status": "completed", "text": "tám giờ tối mai"}]
            (asr / "asr_public_test_whisper_full.shard0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in source))
            rows = {"p": {"id": "p", "audio": "p.wav"}}
            traces = {"p": {"id": "p", "request": "tám giờ tối mai"}}
            module.validate_request_source(baseline, rows, traces)
            traces["p"]["request"] = "tám giờ tối nay"
            with self.assertRaisesRegex(ValueError, "ASR"):
                module.validate_request_source(baseline, rows, traces)


if __name__ == "__main__":
    unittest.main()
