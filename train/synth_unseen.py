"""Synthetic training trajectories for the apps that appear in public_test but never in train (ecommerce, grocery, stocks...).

Only the catalog and the initial environment states of public_test are used (no audio, no labels), all locally.
  utter: a local teacher writes one spoken request per (sample, unseen app) that fits the app's tools and its state.
  roll:  the fine-tuned student answers each request in the real environment several times (first greedy, then sampled);
         a request is kept only when every run makes exactly the same, error-free calls (self-consistency), and never when
         the answer is a question back to the user. The kept runs become SFT trajectories in build_sft.py's format.

python train/synth_unseen.py --stage utter --per-app 40
COND_HINT=1 python train/synth_unseen.py --stage roll --llm models/lifemate-llm-para-q35e3   # same hint flags as the training run
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("MONEY_HINT", "0")
os.environ.setdefault("RESULT_DIGEST", "0")
os.environ.setdefault("TIME_BUDGET_S", "120")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from build_sft import last_text, trajectory  # noqa: E402
from prompt import render_tool  # noqa: E402

SYS = (
    "Bạn tạo dữ liệu huấn luyện cho trợ lý giọng nói tiếng Việt. Viết MỘT câu người dùng nói tự nhiên (như đang nói chuyện, có thể "
    "có từ đệm, xưng hô hợp với hồ sơ) để nhờ trợ lý làm việc bằng ứng dụng dưới đây. Câu phải cụ thể (tên, số lượng, địa điểm, "
    "giờ nếu cần) và làm được với dữ liệu trong 'Trạng thái ứng dụng': dùng tên sản phẩm/địa điểm/mã cổ phiếu... đọc được trong "
    "trạng thái, không đọc ID kỹ thuật. Chỉ trả về đúng câu nói trên một dòng, không giải thích, không đặt trong ngoặc kép."
)


def unseen_apps(root, split):
    cat = lambda s: {a["id"]: a for a in json.load(open(Path(root) / s / f"{s}_tools_catalog.json", encoding="utf-8"))["apps"]}
    seen, here = cat("train"), cat(split)
    return sorted(set(here) - set(seen)), here


def utter_prompt(item, app, apps, state, styles, rng):
    ctx = item["context"]
    tools = apps[app]["tools"]
    focus = rng.sample(tools, k=min(len(tools), rng.choice([1, 1, 2])))
    others = [a for a in (ctx.get("installed_apps") or {}) if a in apps and a != app and a != "sys" and apps[a].get("description")]
    extra = ""
    if others and rng.random() < 0.35:
        o = apps[rng.choice(others)]
        extra = f"\nCó thể kết hợp thêm một việc nhỏ với ứng dụng '{o['name_vi']}' ({o['description']})."
    state_txt = json.dumps(state, ensure_ascii=False)[:1500]
    user = (
        "Ví dụ phong cách nói (của các yêu cầu khác):\n" + "\n".join(f"- {s}" for s in styles)
        + f"\n\nỨng dụng: {apps[app]['name_vi']} - {apps[app].get('description', '')}\nCông cụ:\n"
        + "\n".join(render_tool(t) for t in tools)
        + f"\n\nYêu cầu nên cần dùng: {', '.join(t['name'] for t in focus)}"
        + f"\n\nTrạng thái ứng dụng (rút gọn): {state_txt}\n\nHồ sơ người dùng: {json.dumps(ctx.get('user_profile'), ensure_ascii=False)}"
        + extra + "\n\nCâu nói:"
    )
    return [{"role": "system", "content": SYS}, {"role": "user", "content": user}]


def stage_utter(args):
    root = Path(args.root)
    unseen, apps = unseen_apps(root, args.split)
    items = [json.loads(l) for l in open(root / args.split / f"{args.split}.jsonl", encoding="utf-8")]
    styles_pool = [last_text(json.loads(l).get("audio_text")) for l in open(root / "train" / "train.jsonl", encoding="utf-8")]
    styles_pool = [s for s in styles_pool if 20 < len(s) < 200]
    done = {json.loads(l)["id"] for l in open(args.out_utter, encoding="utf-8")} if Path(args.out_utter).exists() else set()

    rng = random.Random(args.seed)
    todo = []
    for app in unseen:
        have = [it for it in items if app in (it["context"].get("installed_apps") or {})]
        rng.shuffle(have)
        for it in have[: args.per_app]:
            sid = f"synth_{it['id']}_{app}"
            if sid not in done:
                todo.append((sid, it, app))
    print(f"unseen apps: {unseen}\nrequests to generate: {len(todo)} (already done: {len(done)})")
    if not todo:
        return

    from gen_thoughts import load_model
    import torch
    tok, model = load_model(args.teacher, args.quant)
    prompts = []
    for sid, it, app in todo:
        state = json.load(open(root / args.split / f"{args.split}_environment" / it["id"] / "state.json", encoding="utf-8")).get(app, {})
        msgs = utter_prompt(it, app, apps, state, rng.sample(styles_pool, 5), rng)
        prompts.append((sid, it["id"], app, tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False)))
    prompts.sort(key=lambda p: len(p[3]))
    t0, kept = time.time(), 0
    with open(args.out_utter, "a", encoding="utf-8") as fout:
        for s in range(0, len(prompts), args.batch):
            chunk = prompts[s : s + args.batch]
            enc = tok([c[3] for c in chunk], return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=100, do_sample=True, temperature=0.8, top_p=0.95, pad_token_id=tok.pad_token_id)
            for (sid, rid, app, _), o in zip(chunk, out):
                text = tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True).strip().strip('"“”').replace("\n", " ")
                ok = 8 <= len(text) <= 300
                kept += ok
                fout.write(json.dumps({"id": sid, "item": rid, "app": app, "text": text, "ok": ok}, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"{s + len(chunk)}/{len(prompts)} kept {kept} elapsed {(time.time() - t0) / 60:.1f}m", flush=True)


class Sampled:
    """The student with a fixed temperature, so run_agent can call generate(messages) as usual."""

    def __init__(self, llm, temperature):
        self.llm, self.temperature = llm, temperature

    def generate(self, messages):
        return self.llm.generate(messages, temperature=self.temperature)


def rollout(root, split, item, text, catalog, llm, temperature):
    """One agent run in a fresh environment. Returns the executed [(name, arguments, result)], or None when something failed."""
    from agent import run_agent
    from zaic_tools import ToolEnv
    env = ToolEnv(split, item["id"], root=root)
    run_agent(env, item, text, catalog, Sampled(llm, temperature), time.time())
    log = [(e["name"], e["arguments"], e["result"]) for e in env.log]
    if not log or any(not isinstance(r, dict) or r.get("ok") is False for _, _, r in log):
        return None
    return log


def signature(log):
    return json.dumps([(n, a) for n, a, _ in log], sort_keys=True, ensure_ascii=False)


def make_row(sid, item, text, log, catalog):
    sample = dict(item, history=[], label={
        "tool_calls": [{"name": n, "arguments": a} for n, a, _ in log],
        "tool_results": [r for _, _, r in log],
    })
    return {"id": sid, "messages": trajectory(sample, text, catalog)}


def stage_roll(args):
    root = Path(args.root)
    sys.path.insert(0, str(root / args.split))  # zaic_tools of the split
    catalog = json.load(open(root / args.split / f"{args.split}_tools_catalog.json", encoding="utf-8"))
    items = {}
    for l in open(root / args.split / f"{args.split}.jsonl", encoding="utf-8"):
        it = json.loads(l)
        items[it["id"]] = dict(it, history=[])
    rows = [json.loads(l) for l in open(args.out_utter, encoding="utf-8")]
    rows = [r for r in rows if r["ok"]]
    done = set()
    if Path(args.done).exists():
        done = set(open(args.done).read().split())
    todo = [r for r in rows if r["id"] not in done][: args.limit]
    print(f"requests {len(rows)}, to roll out {len(todo)}")
    if not todo:
        return

    from llm import LLM
    llm = LLM(args.llm)
    stats = {"asked": 0, "failed": 0, "disagree": 0, "kept": 0}
    t0 = time.time()
    with open(args.out, "a", encoding="utf-8") as fout, open(args.done, "a") as fdone:
        for n, r in enumerate(todo, 1):
            item, text = items[r["item"]], r["text"]
            logs = [rollout(root, args.split, item, text, catalog, llm, None if k == 0 else args.temp) for k in range(args.runs)]
            if any(l is None for l in logs):
                stats["failed"] += 1
            elif any(name == "sys.ask_missing_info" for name, _, _ in logs[0]):
                stats["asked"] += 1
            elif len({signature(l) for l in logs}) > 1:
                stats["disagree"] += 1
            else:
                stats["kept"] += 1
                fout.write(json.dumps(make_row(r["id"], item, text, logs[0], catalog), ensure_ascii=False) + "\n")
                fout.flush()
            fdone.write(r["id"] + "\n")
            fdone.flush()
            if n % 10 == 0 or n == len(todo):
                print(f"{n}/{len(todo)} {stats} elapsed {(time.time() - t0) / 60:.1f}m", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["utter", "roll"])
    ap.add_argument("--root", default="data")
    ap.add_argument("--split", default="public_test")
    ap.add_argument("--teacher", default="Qwen/Qwen3-14B")
    ap.add_argument("--quant", default="4bit", choices=["4bit", "none"])
    ap.add_argument("--llm", default="models/lifemate-llm-para-q35e3", help="student that answers the requests (stage roll)")
    ap.add_argument("--per-app", type=int, default=40, help="requests per unseen app")
    ap.add_argument("--runs", type=int, default=3, help="agent runs per request; all must agree")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only the first N requests (smoke test)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-utter", default="train/synth_utterances.jsonl")
    ap.add_argument("--out", default="train/synth_sft.jsonl")
    ap.add_argument("--done", default="train/synth_done.txt")
    args = ap.parse_args()
    {"utter": stage_utter, "roll": stage_roll}[args.stage](args)


if __name__ == "__main__":
    main()
