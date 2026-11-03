"""CPU checks for lm-sft-042 (reasoning line before each tool call): SFT targets, prompt, parser, reason checker."""
import contextlib
import importlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).parent
NOTEBOOK = HERE / "agent-sft-reason/lifemate_agent_sft_reason.ipynb"
sys.path.insert(0, str(HERE / "agent-sft-reason"))
import gen_reasons  # noqa: E402


def sources():
    return ["".join(c.get("source", [])) for c in json.loads(NOTEBOOK.read_text())["cells"]]


def load_lmlib(td: Path):
    for s in sources():
        if s.startswith("%%writefile /kaggle/working/lib/"):
            first, rest = s.split("\n", 1)
            p = td / first.split()[1].replace("/kaggle/working/lib/", "")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(rest, encoding="utf-8")
    sys.path.insert(0, str(td))
    for name in [m for m in sys.modules if m == "lmlib" or m.startswith("lmlib.")]:
        del sys.modules[name]
    return importlib.import_module("lmlib.agent"), importlib.import_module("lmlib.sft")


CATALOG = {"apps": [
    {"id": "bill_payment", "name_vi": "Hoá đơn", "description": "", "tools": [
        {"name": "bill_payment.list_bills", "kind": "read", "description": "", "parameters": {"properties": {}}},
        {"name": "bill_payment.pay", "kind": "write", "description": "", "parameters": {
            "properties": {"bill_id": {"type": "string", "x-ref": "bill_payment.bills"}}, "required": ["bill_id"]}}]},
    {"id": "sys", "name_vi": "", "description": "", "tools": [
        {"name": "sys.ask_missing_info", "kind": "meta", "description": "", "parameters": {"properties": {}}}]}]}

ROW = {"id": "train_x", "audio": "audio/train_x/train_x.wav", "audio_text": "Trả hoá đơn nào dưới 500 nghìn.",
       "context": {"current_time": "2026-11-03T09:00:00+07:00", "installed_apps": {"bill_payment": {
           "tools": ["bill_payment.list_bills", "bill_payment.pay"]}}},
       "history": [],
       "label": {"tool_calls": [{"name": "bill_payment.list_bills", "arguments": {}},
                                {"name": "bill_payment.pay", "arguments": {"bill_id": "bill_02"}}],
                 "tool_results": [{"ok": True, "bills": [{"id": "bill_01", "amount": 620000}, {"id": "bill_02", "amount": 450000}]},
                                  {"ok": True}]}}


class ReasonNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.agent, cls.sft = load_lmlib(Path(cls.td.name))
        cls.catalog = cls.agent.Catalog(CATALOG)

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def test_targets_carry_reason_then_parseable_json(self):
        reasons = {"train_x": ["Cần danh sách hoá đơn để so với 500 nghìn", "bill_02 450k < 500k, bill_01 620k → trả bill_02",
                               "Đã trả hoá đơn dưới 500k → kết thúc"]}
        ex = self.sft.build_examples([ROW], self.catalog, reasons=reasons)
        self.assertEqual([e["id"] for e in ex], ["train_x#0", "train_x#1", "train_x#done"])
        self.assertTrue(all(e["target"].startswith("Suy nghĩ: ") for e in ex))
        self.assertEqual(self.agent.parse_action(ex[1]["target"]), ("call", "bill_payment.pay", {"bill_id": "bill_02"}))
        self.assertEqual(self.agent.parse_action(ex[2]["target"])[0], "done")
        self.assertIn("Suy nghĩ", ex[0]["messages"][0]["content"])

    def test_without_reasons_targets_and_prompt_unchanged(self):
        ex = self.sft.build_examples([ROW], self.catalog)
        self.assertEqual(ex[1]["target"], '{"name": "bill_payment.pay", "arguments": {"bill_id": "bill_02"}}')
        self.assertEqual(ex[0]["messages"][0]["content"], self.agent.SYSTEM)

    def test_reason_count_mismatch_fails(self):
        with self.assertRaises(ValueError):
            self.sft.build_examples([ROW], self.catalog, reasons={"train_x": ["chỉ một dòng"]})

    def test_config_turns_reasoning_on_with_room_to_generate(self):
        ns = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(next(s for s in sources() if s.startswith("CONFIG = {")), ns)
        self.assertTrue(ns["CONFIG"]["agent"]["reason"])
        self.assertGreaterEqual(ns["CONFIG"]["agent"]["max_new_tokens"], 384)
        self.agent.AgentConfig(**ns["CONFIG"]["agent"])  # every agent key is a known field
        self.assertEqual(ns["CONFIG"]["llm"]["revision"], "cdbee75f17c01a7cc42f958dc650907174af0554")

    def test_missing_reasons_input_is_attached_with_kagglehub(self):
        cell = next(s for s in sources() if "PREPARED = prepare(CONFIG, OUT)" in s)
        snippet = cell[cell.index("from lmlib.common import sha256_file"):]
        with tempfile.TemporaryDirectory() as td:
            data = Path(td) / "reasons.jsonl"
            data.write_text(json.dumps({"id": "train_x", "reasons": ["a", "b", "c"]}) + "\n", encoding="utf-8")
            sha = self.sft.sha256_file(data)
            asked = []
            hub = types.ModuleType("kagglehub")
            hub.dataset_download = lambda ref: asked.append(ref) or td
            ns = {"Path": Path, "json": json, "REASONS_SHA256": sha, "REASONS_DATASET": "owner/lifemate-sft-reasons/versions/1"}
            with patch.dict(sys.modules, {"kagglehub": hub}), contextlib.redirect_stdout(io.StringIO()):
                exec(snippet, ns)  # /kaggle/input does not exist on CPU, like an Import without this input
        self.assertEqual(asked, ["owner/lifemate-sft-reasons/versions/1"])
        self.assertEqual(ns["REASONS"], {"train_x": ["a", "b", "c"]})

    def test_metadata_has_competition_dataset(self):
        meta = json.loads(NOTEBOOK.read_text())["metadata"]["kaggle"]
        ids = [d.get("datasetId") for d in meta["dataSources"] if d["sourceType"] == "datasetVersion"]
        self.assertIn(12872794, ids)
        self.assertTrue(meta["isInternetEnabled"] and meta["isGpuEnabled"])


class CheckReasonsTests(unittest.TestCase):
    def test_good_reasons_pass(self):
        rs = ["Cần danh sách hoá đơn", "bill_02 450k < 500k → trả bill_02", "Xong → kết thúc"]
        self.assertEqual(gen_reasons.check_reasons(ROW, rs, set()), [])

    def test_hindsight_id_is_rejected(self):
        rs = ["Cần danh sách để trả bill_02", "bill_02 450k → trả", "Xong"]
        self.assertTrue(any("hindsight" in e for e in gen_reasons.check_reasons(ROW, rs, set())))

    def test_brace_and_count_are_rejected(self):
        self.assertTrue(gen_reasons.check_reasons(ROW, ["a", "b"], set()))
        self.assertTrue(any("brace" in e for e in gen_reasons.check_reasons(ROW, ["{x}", "b", "c"], set())))

    def test_ask_last_has_no_done_target(self):
        row = dict(ROW, label=dict(ROW["label"], tool_calls=[{"name": "sys.ask_missing_info", "arguments": {}}], tool_results=[{}]))
        self.assertEqual(len(gen_reasons.targets(row)), 1)


if __name__ == "__main__":
    unittest.main()
