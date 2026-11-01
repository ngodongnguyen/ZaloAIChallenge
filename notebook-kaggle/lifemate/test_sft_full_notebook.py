"""CPU checks for lm-sft-023 (agent-sft-full: all of train, both GPUs training one model, then public_test).

Preset / cell checks need no extra package; the data-parallel test needs torch + transformers + peft (skipped
otherwise) and runs 1 process vs 2 gloo processes on a tiny random model.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

from test_sft_cont_notebooks import EP2, cells, config_ns

HERE = Path(__file__).parent
FULL = HERE / "agent-sft-full/lifemate_agent_sft_full.ipynb"

try:
    import peft  # noqa: F401
    import torch  # noqa: F401
    import transformers  # noqa: F401
except ImportError:
    torch = None


def write_lib(nb, root):
    for s in cells(nb):
        if s.startswith("%%writefile /kaggle/working/lib/"):
            head, body = s.split("\n", 1)
            p = Path(root) / head.split("/kaggle/working/lib/")[1]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body, encoding="utf-8")


class SftFullNotebookTests(unittest.TestCase):
    def test_preset(self):
        ns = config_ns(FULL)
        c, sft = ns["CONFIG"], ns["SFT"]
        self.assertEqual((ns["CONFIG_PRESET"], c["mode"], c["split"], c["experiment"]), ("sft_full", "sft", "train", "lm-sft-023-full-ddp"))
        self.assertEqual((sft["train_subset"], sft["variants"], sft["epochs"]), ("all", ["full"], 2))
        self.assertTrue(sft["ddp"] and sft["eval_public"] and sft["eval_when_finished"])
        self.assertEqual(c["resume_from_dirs"], ["/kaggle/input"])
        # same recipe as the lm-sft-021 "ep2" variant (best dev 36.47), only the data and the GPU layout differ
        ep2 = config_ns(EP2)["SFT"]
        self.assertEqual(sft["variant_specs"]["full"], ep2["variant_specs"]["ep2"])
        for k in ("lora_r", "lora_alpha", "lora_dropout", "lr", "warmup_ratio", "grad_accum", "max_len", "seed"):
            self.assertEqual(sft[k], ep2[k], k)
        self.assertEqual(c["agent"], config_ns(EP2)["CONFIG"]["agent"])

    def test_cells(self):
        src = "\n".join(cells(FULL))
        self.assertIn("ddp_selftest_launch", src)
        self.assertIn('agent=dict(CONFIG["agent"], ground_xref=True)', src)
        self.assertIn('run_asr("public_test"', src)
        # the self-test cell runs before the ASR cell
        cs = cells(FULL)
        i_test = next(i for i, s in enumerate(cs) if "DDP_SELFTEST_OK" in s)
        i_asr = next(i for i, s in enumerate(cs) if s.startswith('if not CONFIG.get("strict_dev"):\n    # Transcripts:'))
        self.assertLess(i_test, i_asr)
        meta = json.loads(FULL.read_text(encoding="utf-8"))["metadata"]["kaggle"]
        self.assertEqual([d["datasetId"] for d in meta["dataSources"]], [12872794])
        self.assertTrue(meta["isGpuEnabled"] and meta["isInternetEnabled"])

    @unittest.skipIf(torch is None, "needs torch + transformers + peft")
    def test_data_parallel_matches_one_process(self):
        with tempfile.TemporaryDirectory() as d:
            write_lib(FULL, Path(d) / "lib")
            # other tests import lmlib from other notebooks: take this notebook's copy, then put theirs back
            saved = {m: sys.modules.pop(m) for m in [m for m in sys.modules if m == "lmlib" or m.startswith("lmlib.")]}
            sys.path.insert(0, str(Path(d) / "lib"))
            try:
                from lmlib.sft import ddp_selftest_launch

                r = ddp_selftest_launch(str(Path(d) / "t"), 2, str(Path(d) / "lib"), port=29561, timeout_s=600)
            finally:
                sys.path.pop(0)
                for m in [m for m in sys.modules if m == "lmlib" or m.startswith("lmlib.")]:
                    del sys.modules[m]
                sys.modules.update(saved)
            self.assertTrue(r["ok"], r)
            self.assertTrue((Path(d) / "t/half/ckpt-000004/rng_rank1.pt").exists())


if __name__ == "__main__":
    unittest.main()
