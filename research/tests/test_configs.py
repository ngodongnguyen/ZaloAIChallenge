import json
import os
import tempfile
import unittest

from harness.configs import ConfigError, code_hash, config_hash, load_config, validate_config

BASE = {"hypothesis": "fp16 on one GPU keeps the same tool calls", "parent": None, "llm_dtype": "fp16"}


class ConfigTests(unittest.TestCase):
    def test_bom_is_stripped_on_load(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.json")
            with open(p, "w", encoding="utf-8-sig") as f:  # writes a BOM
                json.dump(BASE, f)
            self.assertEqual(load_config(p)["llm_dtype"], "fp16")

    def test_unknown_key_is_rejected(self):
        with self.assertRaises(ConfigError):
            validate_config({**BASE, "llm_dtyep": "fp16"})

    def test_short_hypothesis_is_rejected(self):
        with self.assertRaises(ConfigError):
            validate_config({**BASE, "hypothesis": "short"})

    def test_parent_is_required(self):
        cfg = {k: v for k, v in BASE.items() if k != "parent"}
        with self.assertRaises(ConfigError):
            validate_config(cfg)

    def test_bad_dtype_is_rejected(self):
        with self.assertRaises(ConfigError):
            validate_config({**BASE, "llm_dtype": "int8"})

    def test_hash_ignores_metadata(self):
        a = config_hash(BASE)
        b = config_hash({**BASE, "hypothesis": "a different sentence of text", "parent": "run-1"})
        self.assertEqual(a, b)

    def test_hash_changes_with_result_inputs(self):
        self.assertNotEqual(config_hash(BASE), config_hash({**BASE, "llm_dtype": "bf16"}))

    def test_hash_ignores_key_order(self):
        reordered = dict(reversed(list(BASE.items())))
        self.assertEqual(config_hash(BASE), config_hash(reordered))

    def test_code_hash_changes_when_file_changes(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.py")
            with open(p, "w", encoding="utf-8") as f:
                f.write("x = 1\n")
            before = code_hash({"m": p})
            with open(p, "w", encoding="utf-8") as f:
                f.write("x = 2\n")
            self.assertNotEqual(before, code_hash({"m": p}))


if __name__ == "__main__":
    unittest.main()
