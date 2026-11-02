"""Ceiling check: run the team's agent loop with gpt-6.1-sol as the LLM on train samples whose labels are known.

Same prompt, run_agent and ToolEnv as a Kaggle run with the E16 settings, so a gap to the same ids of u001 comes from the
model (and from the transcript, with --text gold). Only train ids go in; public_test must never be run through this.

  python loop/ceiling_run.py --label c001-asr --text asr --n 3 --workers 1
  python loop/ceiling_run.py --label c002-gold --text gold --n 60 --workers 4

--text asr uses data/asr_train.jsonl (PhoWhisper without the a002 adapter) for the current turn and the gold text of
history turns, so it is slightly different from u001 on both counts. Results append to ceiling/<label>.jsonl and a rerun
skips finished ids. A sample where Codex ran a command is flagged: it could have read the labels from disk.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.update(MONEY_HINT="0", RESULT_DIGEST="0", TIME_BUDGET_S="100000")
RESEARCH = Path("C:/Project/zalo_research")
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(RESEARCH / "workspace" / "src"))

from harness.score import score_run  # noqa: E402
from harness.split import DATA_ROOT, load_split, load_train  # noqa: E402

sys.path.insert(0, str(DATA_ROOT / "train"))
from agent import run_agent  # noqa: E402
from zaic_tools import ToolEnv  # noqa: E402

CODEX = "C:/Users/Khai/.vscode/extensions/openai.chatgpt-26.51021.22027-win32-x64/bin/windows-x86_64/codex.exe"
MODEL = "gpt-6-luna"  # far cheaper than gpt-6.1-sol; sol stays for data generation (user, 2026-11-02)
HEAD = ("Bạn là mô hình bên trong một vòng lặp tác tử. Không chạy lệnh, không đọc file, không dùng công cụ nào của bạn. "
        "Dưới đây là toàn bộ hội thoại. Hãy viết lượt [assistant] tiếp theo theo đúng định dạng trong [system]: "
        "một đối tượng JSON duy nhất, không có gì khác.\n\n")
SPLIT = RESEARCH / "splits" / "unseen_apps.json"
BASELINE = RESEARCH / "loop" / "runs_work" / "u001-r017-unseen" / "per_sample.json"
OUT = RESEARCH / "ceiling"
lock = threading.Lock()


def last_text(audio_text):
    if isinstance(audio_text, list):
        return audio_text[-1] if audio_text else ""
    return audio_text or ""


class CodexLLM:
    def __init__(self, effort, model=MODEL):
        self.effort = effort
        self.model = model
        self.outputs, self.usage, self.commands = [], [], 0

    def generate(self, messages):
        prompt = HEAD + "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.txt"
            cmd = [CODEX, "exec", "--json", "--model", self.model, "-c", f'model_reasoning_effort="{self.effort}"',
                   "--sandbox", "read-only", "--skip-git-repo-check", "--ephemeral", "-C", tmp, "-o", str(out), "-"]
            proc = subprocess.run(cmd, input=prompt, text=True, encoding="utf-8", errors="replace", capture_output=True,
                                  check=True, timeout=1800)
            text = out.read_text(encoding="utf-8").strip()
        for line in proc.stdout.splitlines():
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("type") == "turn.completed":
                self.usage.append(ev.get("usage"))
            if ev.get("type") == "item.completed" and ev.get("item", {}).get("type") != "agent_message" \
                    and "command" in ev.get("item", {}).get("type", ""):
                self.commands += 1
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
        self.outputs.append(text)
        return text


def run_one(item, text, catalog, effort, path):
    llm = CodexLLM(effort)
    env = ToolEnv("train", item["id"], root=DATA_ROOT)
    t = time.time()
    err = None
    try:
        run_agent(env, item, text, catalog, llm, t)
    except Exception as e:  # keep the partial calls, as the kernel does
        err = repr(e)
    row = {"id": item["id"], "sec": round(time.time() - t, 1), "err": err, "tool_calls": env.get_tool_calls(),
           "state_hash": env.get_state_hash(), "outputs": llm.outputs, "usage": llm.usage, "commands": llm.commands}
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(item["id"], row["sec"], "s", "steps", len(llm.outputs), "cmd", llm.commands, "err", err, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--text", required=True, choices=["asr", "gold"])
    ap.add_argument("--n", type=int, default=187, help="evenly spaced ids of the split")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--effort", default="max")  # gpt-6-luna tops out at max (ultra is sol only)
    args = ap.parse_args()

    split = load_split(SPLIT)
    ids = sorted(split["dev"])
    ids = [ids[k * len(ids) // args.n] for k in range(args.n)]
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{args.label}.jsonl"
    rows = [json.loads(line) for line in open(path, encoding="utf-8")] if path.exists() else []
    done = {r["id"] for r in rows if not r["err"]}  # a failed sample (e.g. rate limit) runs again; scoring keeps the last row
    by_id = {r["id"]: r for r in load_train(DATA_ROOT / "train" / "train.jsonl")}
    with open(DATA_ROOT / "train" / "train_tools_catalog.json", encoding="utf-8-sig") as f:
        catalog = json.load(f)
    asr = {}
    if args.text == "asr":
        asr = {json.loads(line)["id"]: json.loads(line)["text"] for line in open(RESEARCH / "data" / "asr_train.jsonl", encoding="utf-8")}
    text = lambda i: asr[i] if args.text == "asr" else last_text(by_id[i].get("audio_text"))
    with ThreadPoolExecutor(args.workers) as pool:
        for f in [pool.submit(run_one, by_id[i], text(i), catalog, args.effort, path) for i in ids if i not in done]:
            f.result()

    sub = OUT / f"{args.label}.split.json"
    sub.write_text(json.dumps(dict(split, dev=ids), ensure_ascii=False), encoding="utf-8")
    metrics = score_run(path, sub)
    base = json.load(open(BASELINE, encoding="utf-8"))
    rows = list({r["id"]: r for r in map(json.loads, open(path, encoding="utf-8"))}.values())
    print("score", metrics["score"], "n", metrics["n"], "zeros", metrics["zero_samples"],
          "r017 on same ids", round(sum(base[i] for i in ids) / len(ids), 6),
          "samples with commands", sum(r["commands"] > 0 for r in rows), "errors", sum(bool(r["err"]) for r in rows), flush=True)


if __name__ == "__main__":
    main()
