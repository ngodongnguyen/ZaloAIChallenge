"""Run the agent on dev locally from fixed transcripts and score it (no Kaggle).

  python loop/local_dev_eval.py --transcripts loop/runs_work/asr_bias/ft_b150.json --field bias --tag ft_bias
Uses the research workspace's src/ (the prompt the local LLM was trained with) and models/llm_sft.
"""
import argparse
import json
import sys
import time
from pathlib import Path

RESEARCH = Path("C:/Project/zalo_research")
sys.path.insert(0, str(RESEARCH / "workspace" / "src"))
sys.path.insert(0, str(RESEARCH))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from harness.score import DATA_ROOT, load_train, score_run  # noqa: E402

sys.path.insert(0, str(DATA_ROOT / "train"))
from zaic_tools import ToolEnv  # noqa: E402
from agent import parse_action, run_agent  # noqa: E402
from config import MAX_NEW_TOKENS  # noqa: E402

OUT = RESEARCH / "loop" / "runs_work" / "local_dev"


class Gen:
    def __init__(self, path):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16).to("cuda").eval()
        self.outputs = []

    def generate(self, messages):
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        ids = ids.to("cuda")
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        self.outputs.append(text)
        return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", required=True)
    ap.add_argument("--fields", default="plain,bias")
    ap.add_argument("--model", default=str(RESEARCH / "models" / "llm_sft"))
    args = ap.parse_args()

    trans = {x["id"]: x for x in json.load(open(args.transcripts, encoding="utf-8"))}
    rows = {r["id"]: r for r in load_train(DATA_ROOT / "train" / "train.jsonl")}
    catalog = json.load(open(DATA_ROOT / "train" / "train_tools_catalog.json", encoding="utf-8-sig"))
    llm = Gen(args.model)
    OUT.mkdir(parents=True, exist_ok=True)
    for field in args.fields.split(","):
        path = OUT / f"{Path(args.transcripts).stem}_{field}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for k, i in enumerate(trans, 1):
                env = ToolEnv("train", i, root=DATA_ROOT)
                t1 = time.time()
                start = len(llm.outputs)
                err = None
                try:
                    run_agent(env, rows[i], trans[i][field], catalog, llm, t1)
                except Exception as e:
                    err = repr(e)
                outs = llm.outputs[start:]
                f.write(json.dumps({"id": i, "sec": round(time.time() - t1, 2), "err": err,
                                    "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash(),
                                    "outputs": outs,
                                    "invalid_outputs": sum(1 for o in outs if parse_action(o) is None)},
                                   ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{field} {k}/{len(trans)}] {i}", flush=True)
        m = score_run(path, RESEARCH / "splits" / "lifemate_split.json")
        m.pop("per_sample")
        print(f"RESULT {field}: {m}", flush=True)


if __name__ == "__main__":
    main()
