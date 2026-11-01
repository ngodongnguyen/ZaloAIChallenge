"""CPU checks for lm-sft-018 (warm start + synthetic mix) and lm-B-017 (public with the lm-sft-016 adapter).

Presets / cell code need no extra package; the warm-start test needs torch + transformers + peft (skipped otherwise).
"""
import ast
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from test_guard_notebooks import CATALOG, ROW, SEARCH, lmlib

HERE = Path(__file__).parent
CONT = HERE / "agent-sft-cont/lifemate_agent_sft_cont.ipynb"
PUBLIC = HERE / "agent-sft-public/lifemate_agent_sft_public.ipynb"
WEIGHT = HERE / "agent-sft-weight/lifemate_agent_sft_weight.ipynb"
GROUND = HERE / "agent-sft-ground/lifemate_agent_sft_ground.ipynb"
EP2 = HERE / "agent-sft-ep2/lifemate_agent_sft_ep2.ipynb"
ORACLE = HERE / "agent-sft-oracle/lifemate_agent_sft_oracle.ipynb"
ALL = (CONT, PUBLIC, WEIGHT, GROUND, EP2, ORACLE)


class ContactsEnv:
    def __init__(self):
        self.calls = []

    def validate(self, name, args):
        return {"ok": True}

    def call(self, name, args):
        self.calls.append(name)
        return {"contacts": [{"id": "c1", "name": "Lan"}]} if name == "contacts.search" else {"ok": True}

try:
    import peft  # noqa: F401
    import torch
    import transformers  # noqa: F401
except ImportError:
    torch = None


def cells(nb):
    return ["".join(c["source"]) for c in json.loads(nb.read_text(encoding="utf-8"))["cells"]]


def config_ns(nb):
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(next(s for s in cells(nb) if s.startswith("CONFIG = {")), ns)
    return ns


class Tok:
    eos_token = ""

    def apply_chat_template(self, m, **k):
        return m[0]["content"]

    def __call__(self, s, add_special_tokens=False):
        return {"input_ids": [int(x) for x in s.split()] if s else []}


class SftContNotebookTests(unittest.TestCase):
    def test_presets_and_metadata(self):
        ns = config_ns(CONT)
        self.assertEqual((ns["CONFIG_PRESET"], ns["CONFIG"]["mode"]), ("sft_cont", "sft"))
        self.assertEqual(ns["CONFIG"]["fewshot"], {"k": 0})
        sft = ns["SFT"]
        self.assertEqual(sft["variants"], ["cont", "cont_syn"])
        self.assertEqual((sft["init_adapter"], sft["max_steps"]), ("/kaggle/input/**/lm_agent/sft_mix_aug", 150))
        a, b = sft["variant_specs"]["cont"], sft["variant_specs"]["cont_syn"]
        self.assertEqual(dict(b, syn=None), dict(a, syn=None))  # the two GPUs differ only by the synthetic rows
        self.assertTrue(b["syn"])
        # LoRA shape must match the lm-sft-016 adapter it starts from
        self.assertEqual((sft["lora_r"], sft["lora_alpha"]), (16, 32))
        ns = config_ns(PUBLIC)
        c = ns["CONFIG"]
        self.assertEqual((ns["CONFIG_PRESET"], c["mode"], c["split"]), ("public_sft", "public", "public_test"))
        self.assertEqual((c["fewshot"], c["asr_backend"], c["llm"]["adapter"]), ({"k": 0}, "gipformer", "/kaggle/input/**/lm_agent/sft_mix_aug"))
        self.assertTrue(c["agent"]["block_xref_ask"] and c["agent"]["nudge_done_after_read"])  # same agent as the 016 dev eval
        ns = config_ns(WEIGHT)
        self.assertEqual((ns["CONFIG_PRESET"], ns["CONFIG"]["experiment"]), ("sft_weight", "lm-sft-019-cont-weight"))
        w = ns["SFT"]
        self.assertEqual((w["variants"], w["init_adapter"], w["max_steps"]), (["cont", "cont_w"], sft["init_adapter"], 150))
        self.assertEqual(w["variant_specs"]["cont"], sft["variant_specs"]["cont"])  # GPU0 identical to lm-sft-018's
        self.assertEqual(dict(w["variant_specs"]["cont_w"], oversample=None), dict(a, oversample=None))
        self.assertNotIn("syn_glob", w)
        ns = config_ns(GROUND)
        c = ns["CONFIG"]
        self.assertEqual((ns["CONFIG_PRESET"], c["mode"], c["subset"], c["fewshot"]), ("dev_ground", "e2e", "dev_seen+dev_unseen", {"k": 0}))
        self.assertEqual((c["asr_backend"], c["llm"]["adapter"]), ("gipformer", "/kaggle/input/**/lm_agent/sft_mix_aug"))
        self.assertEqual(c["eval_variants"], {"base": {}, "ground": {"ground_xref": True}})
        self.assertEqual(c["agent"], config_ns(PUBLIC)["CONFIG"]["agent"])  # "base" = the 016 dev eval settings
        ground = c
        ns = config_ns(ORACLE)
        c = ns["CONFIG"]
        self.assertEqual((ns["CONFIG_PRESET"], c["experiment"]), ("dev_oracle", "lm-B-022-oracle-transcript"))
        self.assertEqual(c["eval_variants"], {"gold": {"config": {"mode": "diagnostic"}}, "asr": {}})
        self.assertEqual(c["llm"]["adapter"], "/kaggle/input/**/lm_agent/sft_ep2")
        self.assertEqual(dict(c, eval_variants=None, experiment=None, llm=None), dict(ground, eval_variants=None, experiment=None, llm=None))  # only checkpoint differs; same ASR / agent
        run_cell = next(s for s in cells(ORACLE) if s.startswith("from lmlib.run import launch, merge"))
        self.assertIn('top = over.pop("config", {})', run_cell)
        ns = config_ns(EP2)
        c, e = ns["CONFIG"], ns["SFT"]
        self.assertEqual((ns["CONFIG_PRESET"], c["mode"], c["resume_from_dirs"]), ("sft_ep2", "sft", ["/kaggle/input"]))
        self.assertEqual((e["variants"], e["epochs"], e["eval_when_finished"]), (["ep2", "ep2_drop"], 2, True))
        self.assertNotIn("init_adapter", e)  # from scratch: runs on any account
        self.assertEqual(dict(e["variant_specs"]["ep2_drop"], drop_p=0.3), e["variant_specs"]["ep2"])
        for nb in ALL:
            meta = json.loads(nb.read_text(encoding="utf-8"))["metadata"]["kaggle"]
            self.assertEqual([d["datasetId"] for d in meta["dataSources"]], [12872794])
            for c in json.loads(nb.read_text(encoding="utf-8"))["cells"]:
                s = "".join(c["source"])
                if c["cell_type"] == "code" and not s.startswith("%"):
                    ast.parse(s)

    def test_oracle_pins_complete_ep2_checkpoint_before_setup(self):
        src = next(s for s in cells(ORACLE) if 'pinned = OUT / "oracle_adapter_input"' in s)
        preflight = src.split("# Thư viện cài thêm", 1)[0]
        for mounts, succeeds in [(["sft_ep2/ckpt-000380"], True),
                                  (["sft_cont_w/ckpt-000150"], False),
                                  (["sft_ep2/ckpt-000370"], False),
                                  (["sft_ep2/ckpt-000380", "sft_ep2/ckpt-000380"], False)]:
            with self.subTest(mounts=mounts), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                inputs = root / "input"
                for i, mount in enumerate(mounts):
                    ck = inputs / str(i) / "lm_agent" / mount
                    ck.mkdir(parents=True)
                    for name in ("COMPLETE", "adapter_state.pt", "adapter_config.json"):
                        (ck / name).write_text("fixture")
                out = root / "output"
                out.mkdir()
                cfg = config_ns(ORACLE)["CONFIG"]
                ns = dict(Path=Path, json=json, CONFIG_PRESET="dev_oracle", CONFIG=cfg, OUT=out)
                code = preflight.replace('Path("/kaggle/input")', 'Path(' + repr(str(inputs)) + ')')
                if succeeds:
                    with contextlib.redirect_stdout(io.StringIO()):
                        exec(code, ns)
                    pinned = Path(cfg["llm"]["adapter"]) / "ckpt-000380"
                    self.assertEqual(pinned.resolve(), inputs / "0/lm_agent/sft_ep2/ckpt-000380")
                    self.assertEqual(json.loads((out / "oracle_checkpoint.json").read_text())["run"], "lm-sft-021")
                else:
                    with self.assertRaises(RuntimeError):
                        exec(code, ns)

    def test_eval_variant_config_override(self):
        # the cell's variant loop: "config" -> top-level keys (mode), other keys -> agent; base CONFIG untouched
        src = next(s for s in cells(ORACLE) if s.startswith("from lmlib.run import launch, merge"))
        start = src.index('    for v, over in CONFIG["eval_variants"].items():')
        loop = src[start:src.index("        reports[v] = run_agent", start)]
        CONFIG = config_ns(ORACLE)["CONFIG"]
        got = {}
        exec("for v, over in CONFIG['eval_variants'].items():\n" + "\n".join(loop.splitlines()[1:]) + "\n        got[v] = conf", {"CONFIG": CONFIG, "got": got})
        self.assertEqual((got["gold"]["mode"], got["asr"]["mode"], CONFIG["mode"]), ("diagnostic", "e2e", "e2e"))
        self.assertEqual(got["gold"]["agent"], CONFIG["agent"])  # "config" never leaks into the agent settings
        self.assertEqual(got["asr"]["experiment"], "lm-B-022-oracle-transcript-asr")

    def test_syn_rows_build_with_merged_catalog(self):
        with tempfile.TemporaryDirectory() as td:
            lib = lmlib(CONT, Path(td))
            extra = {"apps": [{"id": "stocks", "tools": [{"name": "stocks.buy", "kind": "write", "parameters": {
                "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}}]}]}
            have = {a["id"] for a in CATALOG["apps"]}
            cat = lib.agent.Catalog(dict(CATALOG, apps=CATALOG["apps"] + [a for a in extra["apps"] if a["id"] not in have]))
            syn = dict(ROW, id="syn_7001", audio="audio/syn_7001/syn_7001.wav", audio_text="mua mười cổ phiếu FPT",
                       context=dict(ROW["context"], installed_apps={"stocks": {"tools": ["stocks.buy"]}}),
                       label={"tool_calls": [{"name": "stocks.buy", "arguments": {"symbol": "FPT"}}], "tool_results": [{"ok": True}]})
            ex = lib.sft.build_examples([ROW, syn], cat, texts={"a.wav": "ASR"}, asr_prob=0.5, aug=True, seed=0)
            mine = [e for e in ex if e["id"].startswith("syn_7001")]
            self.assertEqual([e["target"] for e in mine], ['{"name": "stocks.buy", "arguments": {"symbol": "FPT"}}', '{"done": true}'])
            self.assertIn("stocks.buy [write]", mine[0]["messages"][1]["content"])
            self.assertIn("mua mười cổ phiếu FPT", mine[0]["messages"][1]["content"])  # no audio yet -> gold text

    def test_oversample(self):
        with tempfile.TemporaryDirectory() as td:
            lib = lmlib(WEIGHT, Path(td))
            ask = dict(ROW, id="ask", label={"tool_calls": [ROW["label"]["tool_calls"][0], {"name": "sys.ask_missing_info", "arguments": {}}],
                                             "tool_results": [{}, {}]})
            long = dict(ROW, id="long", label={"tool_calls": ROW["label"]["tool_calls"] * 2, "tool_results": [{}] * 4})
            out = lib.sft.oversample([ROW, ask, long], ask=3, calls4=2)
            self.assertEqual([r["id"] for r in out], ["x", "ask", "ask~1", "ask~2", "long", "long~1"])
            self.assertEqual(lib.sft.oversample([ROW, ask, long]), [ROW, ask, long])
            cat = lib.agent.Catalog(CATALOG)
            ex = lib.sft.build_examples(out, cat, texts={"a.wav": "ASR"}, asr_prob=0.5, aug=True, seed=0)
            self.assertEqual(sum(e["id"].startswith("ask") for e in ex), 3 * 2)  # 2 steps, no done after an ask

    def test_ground_xref_guard(self):
        with tempfile.TemporaryDirectory() as td:
            lib = lmlib(GROUND, Path(td))
            import time as _t

            def run(script, **cfg):
                seq = iter(script)
                llm = lib.llm.MockLLM(lambda conv: json.dumps(next(seq), ensure_ascii=False))
                e = lib.agent.Episode(ROW, ROW["audio_text"], ContactsEnv(), _t.time())
                lib.agent.run_episodes([e], llm, lib.agent.Catalog(CATALOG), {}, lib.agent.AgentConfig(**cfg))
                return e

            call = lambda cid: {"name": "phone.call", "arguments": {"contact_id": cid}}
            e = run([call("c9"), SEARCH, call("c1"), {"done": True}], ground_xref=True)
            self.assertEqual((e.finished, e.guards, e.env.calls), ("done", {"ground_xref": 1}, ["contacts.search", "phone.call"]))
            self.assertIn("contacts.search", e.steps[0]["error"])  # hint names the read tool of the referenced app
            e = run([SEARCH, call("c1"), {"done": True}], ground_xref=True)  # grounded ID: untouched
            self.assertEqual((e.finished, e.guards), ("done", {}))
            hist = dict(ROW, history=[{"role": "user", "audio_text": "tìm Lan"},
                                      {"role": "assistant", "text": "", "tool_calls": [SEARCH], "tool_results": [{"contacts": [{"id": "c1"}]}]}])
            seq = iter([call("c1"), {"done": True}])
            e = lib.agent.Episode(hist, "gọi Lan", ContactsEnv(), _t.time())
            lib.agent.run_episodes([e], lib.llm.MockLLM(lambda conv: json.dumps(next(seq))), lib.agent.Catalog(CATALOG), {},
                                   lib.agent.AgentConfig(ground_xref=True))
            self.assertEqual((e.finished, e.guards), ("done", {}))  # ID seen in a history result counts
            e = run([call("c9"), {"done": True}])  # off by default
            self.assertEqual((e.finished, e.guards, e.env.calls), ("done", {}, ["phone.call"]))
            e = run([call("c9")] * 4 + [{"done": True}], ground_xref=True, max_guard=3)  # guard budget spent -> the model's call goes through
            self.assertEqual((e.guards, e.env.calls), ({"ground_xref": 3}, ["phone.call"]))

    @unittest.skipIf(torch is None, "needs torch + transformers + peft")
    def test_warm_start_then_exact_resume(self):
        from transformers import LlamaConfig, LlamaForCausalLM

        with tempfile.TemporaryDirectory() as td:
            lib = lmlib(CONT, Path(td) / "lib")

            def fresh():
                torch.manual_seed(0)
                return LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                                    num_attention_heads=2, num_key_value_heads=2))

            ex = [{"messages": [{"role": "user", "content": " ".join(str((i * 7 + j) % 64) for j in range(8))}],
                   "target": " ".join(str((i + j) % 64) for j in range(4))} for i in range(32)]
            base = {"max_len": 32, "lora_r": 4, "lora_alpha": 8, "lora_dropout": 0.0, "target_modules": ["q_proj", "v_proj"],
                    "grad_accum": 2, "seed": 0, "ckpt_every_steps": 1000, "gradient_checkpointing": False}
            quiet = dict(tok=Tok(), log=lambda *_: None)
            lib.sft.train({**base, "lr": 1e-2, "max_steps": 6, "out_dir": f"{td}/a"}, ex, model=fresh(), **quiet)
            a = torch.load(next(Path(f"{td}/a").glob("ckpt-*")) / "adapter_state.pt")
            # lr 0: the warm-started adapter must come out exactly as loaded
            lib.sft.train({**base, "lr": 0.0, "max_steps": 2, "out_dir": f"{td}/b", "init_adapter": f"{td}/**/a"}, ex, model=fresh(), **quiet)
            b = torch.load(next(Path(f"{td}/b").glob("ckpt-*")) / "adapter_state.pt")
            self.assertEqual(a.keys(), b.keys())
            self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))
            # warm start + split session == warm start straight; a new mount path for init_adapter is not a new config
            run = {**base, "lr": 1e-2, "max_steps": 8}
            s = lib.sft.train({**run, "out_dir": f"{td}/c", "init_adapter": f"{td}/a"}, ex, model=fresh(), **quiet)
            lib.sft.train({**run, "out_dir": f"{td}/d", "init_adapter": f"{td}/a", "max_steps_this_session": 4}, ex, model=fresh(), **quiet)
            r = lib.sft.train({**run, "out_dir": f"{td}/d", "init_adapter": "/elsewhere/a"}, ex, model=fresh(), **quiet)
            self.assertEqual((s["global_step"], r["global_step"]), (8, 8))
            self.assertLess(max(abs(x - y) for x, y in zip(s["last_losses"], r["last_losses"])), 1e-4)
            with self.assertRaises(FileNotFoundError):
                lib.sft.train({**run, "out_dir": f"{td}/e", "init_adapter": f"{td}/missing"}, ex, model=fresh(), **quiet)


if __name__ == "__main__":
    unittest.main()
