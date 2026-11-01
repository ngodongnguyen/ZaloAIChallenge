"""CPU checks for the lm-B-015 agent guards and the lm-sft-016 transcript mix; fixture catalog, no dataset or model."""
import contextlib
import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).parent
NOTEBOOKS = [HERE / "agent-guard/lifemate_agent_guard.ipynb", HERE / "agent-sft-mix/lifemate_agent_sft_mix.ipynb"]

CATALOG = {"apps": [
    {"id": "contacts", "tools": [{"name": "contacts.search", "kind": "read", "parameters": {"properties": {"query": {"type": "string"}}}}]},
    {"id": "phone", "tools": [{"name": "phone.call", "kind": "write", "parameters": {
        "properties": {"contact_id": {"type": "string", "x-ref": "contacts.contacts"}}, "required": ["contact_id"]}}]},
    {"id": "music", "tools": [{"name": "music.play", "kind": "write", "parameters": {"properties": {"track_id": {"type": "string"}}}}]},
    {"id": "sys", "tools": [{"name": "sys.ask_missing_info", "kind": "meta", "parameters": {
        "properties": {"tool_name": {"type": "string"}, "params": {"type": "array"}}}}]},
]}
ROW = {"id": "x", "audio": "a.wav", "audio_text": "gọi cho Lan", "history": [],
       "context": {"installed_apps": {"contacts": {"tools": ["contacts.search"]}, "phone": {"tools": ["phone.call"]},
                                      "music": {"tools": ["music.play"]}}},
       "label": {"tool_calls": [{"name": "contacts.search", "arguments": {"query": "Lan"}},
                                {"name": "phone.call", "arguments": {"contact_id": "c1"}}],
                 "tool_results": [{"contacts": [{"id": "c1"}]}, {"ok": True}]}}
ASK = {"name": "sys.ask_missing_info", "arguments": {"tool_name": "phone.call", "params": ["contact_id"]}}
SEARCH = {"name": "contacts.search", "arguments": {"query": "Lan"}}


class Env:
    def __init__(self):
        self.calls = []

    def validate(self, name, args):
        return {"ok": True}

    def call(self, name, args):
        self.calls.append({"name": name, "arguments": args})
        return {"ok": True}


def lmlib(notebook: Path, root: Path):
    for c in json.loads(notebook.read_text(encoding="utf-8"))["cells"]:
        s = "".join(c["source"])
        if s.startswith("%%writefile "):
            first, body = s.split("\n", 1)
            p = root / first.split("/kaggle/working/lib/")[1]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body, encoding="utf-8")
    sys.path.insert(0, str(root))
    for m in [m for m in sys.modules if m == "lmlib" or m.startswith("lmlib.")]:
        del sys.modules[m]
    import lmlib.agent, lmlib.llm, lmlib.sft  # noqa: E401
    sys.path.remove(str(root))
    return lmlib


class GuardNotebookTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.td.cleanup()

    def run_script(self, lib, script, **cfg):
        seq = iter(script)
        llm = lib.llm.MockLLM(lambda conv: json.dumps(next(seq), ensure_ascii=False))
        e = lib.agent.Episode(ROW, ROW["audio_text"], Env(), time.time())
        lib.agent.run_episodes([e], llm, lib.agent.Catalog(CATALOG), {}, lib.agent.AgentConfig(**cfg))
        return e

    def test_guards(self):
        guards = dict(block_xref_ask=True, read_repeat_feedback=True, nudge_done_after_read=True, max_guard=3)
        for nb in NOTEBOOKS:
            with self.subTest(nb.name):
                lib = lmlib(nb, Path(self.td.name) / nb.stem)
                e = self.run_script(lib, [ASK, SEARCH, SEARCH, ASK], **guards)
                self.assertEqual((e.finished, e.guards), ("ask", {"xref_ask": 1, "repeat_read": 1}))
                self.assertEqual([c["name"] for c in e.env.calls], ["contacts.search", "sys.ask_missing_info"])
                e = self.run_script(lib, [SEARCH, {"done": True}, {"done": True}], **guards)
                self.assertEqual((e.finished, e.guards), ("done", {"nudge_done": 1}))
                e = self.run_script(lib, [ASK] * 4, **guards)  # guard budget spent -> the model's choice goes through
                self.assertEqual((e.finished, e.guards), ("ask", {"xref_ask": 3}))
                e = self.run_script(lib, [ASK])  # defaults: guards off
                self.assertEqual((e.finished, e.guards), ("ask", {}))

    def test_sft_mix_examples(self):
        nb = NOTEBOOKS[1]
        lib = lmlib(nb, Path(self.td.name) / "sft")
        cat = lib.agent.Catalog(CATALOG)
        rows = [dict(ROW, id=f"x{i}", audio=f"a{i}.wav") for i in range(200)]
        asr = {f"a{i}.wav": "ASRMARK" for i in range(200)}
        mix = lib.sft.build_examples(rows, cat, texts=asr, asr_prob=0.5, seed=0)
        share = sum("ASRMARK" in e["messages"][1]["content"] for e in mix) / len(mix)
        self.assertTrue(0.35 < share < 0.65, share)
        self.assertEqual(mix, lib.sft.build_examples(rows, cat, texts=asr, asr_prob=0.5, seed=0))
        self.assertEqual(lib.sft.build_examples(rows, cat), lib.sft.build_examples(rows, cat, texts=asr, asr_prob=0.0))
        aug = lib.sft.build_examples(rows, cat, texts=asr, asr_prob=0.5, aug=True, drop_p=1.0, seed=0)
        self.assertEqual([e["target"] for e in aug], [e["target"] for e in mix])
        self.assertTrue(all("music.play" not in e["messages"][1]["content"] for e in aug))  # unused app dropped
        self.assertTrue(all("phone.call [" in e["messages"][1]["content"] for e in aug))  # used apps kept

    def test_presets_and_dataset_metadata(self):
        for nb, preset in zip(NOTEBOOKS, ("dev_guard", "sft_mix")):
            cells = ["".join(c["source"]) for c in json.loads(nb.read_text(encoding="utf-8"))["cells"]]
            ns = {}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(next(s for s in cells if s.startswith("CONFIG = {")), ns)
            self.assertEqual(ns["CONFIG_PRESET"], preset)
            self.assertTrue(ns["CONFIG"]["agent"]["block_xref_ask"])
            self.assertEqual(ns["CONFIG"]["asr_backend"], "gipformer")
            meta = json.loads(nb.read_text(encoding="utf-8"))["metadata"]["kaggle"]
            self.assertEqual([d["datasetId"] for d in meta["dataSources"]], [12872794])


if __name__ == "__main__":
    unittest.main()
