"""Step-wise DPO on the model's own failed steps (TP-LLaMA / When2Call style), on top of a merged SFT model.

mine:  at every assistant step of a gold trajectory (gold prefix, so each pair isolates one decision) the model writes
       a greedy answer and N samples; any parseable answer whose action (tool + cleaned arguments, or done) differs
       from the gold step is a rejected answer for that step.
train: LoRA with the DPO loss plus a small NLL term on the chosen answer (RPO), reference = the same model with the
       adapter disabled, then merge into the SFT model.

  python train/dpo.py mine --model M --gold gold.jsonl --exclude ids.txt --out pairs.jsonl --shard 0/2 --limit 300
  python train/dpo.py train --model M --gold gold.jsonl --pairs a.jsonl b.jsonl --catalog cat.json --out OUT --fp16
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from sft import no_gqa_sdpa  # noqa: E402


def parse_action(text):
    i = text.find("{")
    if i < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[i:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def action_key(text):
    """Same tool and same arguments (placeholders dropped, as the agent does), or both done."""
    a = parse_action(text)
    if a is None:
        return None
    if a.get("done"):
        return "done"
    args = {k: v for k, v in (a.get("arguments") or {}).items() if v not in ("", None, [], {})}
    return json.dumps([a.get("name"), args], ensure_ascii=False, sort_keys=True)


def gold_rows(path, exclude):
    held = set(Path(exclude).read_text().split()) if exclude else set()
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return [r for r in rows if r["id"] not in held]


def ids_of(tok, messages, gen):
    out = tok.apply_chat_template(messages, add_generation_prompt=gen, tokenize=True)
    return list(out["input_ids"] if hasattr(out, "keys") else out)


def mine(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    no_gqa_sdpa()
    rows = gold_rows(args.gold, args.exclude)
    random.Random(0).shuffle(rows)  # the same order as loop/mine_pairs.py; --reverse takes the other end
    if args.reverse:
        rows.reverse()
    skip = set()
    for p in args.skip or []:
        if Path(p).exists():
            skip |= {json.loads(l)["id"] for l in open(p, encoding="utf-8") if l.strip()}
    rows = [r for r in rows if r["id"] not in skip][:args.limit]
    k, n = map(int, args.shard.split("/"))
    rows = rows[k::n]
    done = {json.loads(l)["id"] for l in open(args.out, encoding="utf-8")} if Path(args.out).exists() else set()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
    t0, n_pairs = time.time(), 0
    with open(args.out, "a", encoding="utf-8") as f:
        for j, r in enumerate(rows, 1):
            if r["id"] in done:
                continue
            msgs, pairs, steps = r["messages"], [], 0
            for i, m in enumerate(msgs):
                if m["role"] != "assistant":
                    continue
                steps += 1
                gold = action_key(m["content"])
                ids = torch.tensor([ids_of(tok, msgs[:i], True)], device="cuda")
                outs = []
                with torch.no_grad():
                    for sample in [False] + [True] * args.n_samples:
                        kw = dict(do_sample=True, temperature=args.temp, top_p=1.0, top_k=0) if sample else dict(do_sample=False)
                        o = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=320,
                                           pad_token_id=tok.eos_token_id, **kw)
                        outs.append(("sample" if sample else "greedy", tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).strip()))
                seen = set()
                for src, text in outs:
                    key = action_key(text)
                    if key is None or key == gold or key in seen:
                        continue
                    seen.add(key)
                    pairs.append({"step": i, "chosen": m["content"], "rejected": text, "src": src})
            n_pairs += len(pairs)
            f.write(json.dumps({"id": r["id"], "steps": steps, "pairs": pairs}, ensure_ascii=False) + "\n")
            f.flush()
            if j % 10 == 0:
                print(f"mine shard {args.shard}: {j}/{len(rows)} {round(time.time() - t0)}s pairs {n_pairs}", flush=True)
    print(f"mine shard {args.shard} done: {n_pairs} pairs", flush=True)


def optional_read_filter_only(chosen, rejected, tools):
    """True when both answers call the same read tool and differ only in optional arguments (filters): train labels
    disagree on these (e.g. smart_home.get_status is called without filters in 87% of gold calls), so such pairs would
    push the model both ways."""
    a, b = parse_action(chosen), parse_action(rejected)
    if not a or not b or a.get("done") or b.get("done") or a.get("name") != b.get("name"):
        return False
    t = tools.get(a.get("name"))
    if not t or t.get("kind") != "read":
        return False
    required = set((t.get("parameters") or {}).get("required") or [])
    ra, rb = a.get("arguments") or {}, b.get("arguments") or {}
    diff = {k for k in set(ra) | set(rb) if ra.get(k) != rb.get(k)}
    return bool(diff) and not (diff & required)


def build_pairs(args, tok):
    gold = {r["id"]: r["messages"] for r in gold_rows(args.gold, None)}
    tools = {}
    if args.catalog:
        cat = json.load(open(args.catalog, encoding="utf-8-sig"))
        tools = {t["name"]: t for a in cat["apps"] for t in a["tools"]}
    out, dropped = [], {"filter_only": 0, "too_long": 0, "no_gold": 0}
    seen = set()
    for path in args.pairs:
        if not Path(path).exists():
            continue
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r["id"] not in gold:
                dropped["no_gold"] += len(r["pairs"])
                continue
            for p in r["pairs"]:
                key = (r["id"], p["step"], action_key(p["rejected"]))
                if key in seen:
                    continue
                seen.add(key)
                if optional_read_filter_only(p["chosen"], p["rejected"], tools):
                    dropped["filter_only"] += 1
                    continue
                prefix = gold[r["id"]][:p["step"]]
                prompt = ids_of(tok, prefix, True)
                enc = []
                for text in (p["chosen"], p["rejected"]):
                    full = ids_of(tok, prefix + [{"role": "assistant", "content": text}], False)
                    enc.append(full)
                if max(len(x) for x in enc) > args.max_len:
                    dropped["too_long"] += 1
                    continue
                out.append({"prompt_len": len(prompt), "chosen": enc[0], "rejected": enc[1]})
    if args.max_pairs:
        random.Random(0).shuffle(out)
        out = out[:args.max_pairs]
    print(f"dpo pairs: {len(out)} kept, dropped {dropped}", flush=True)
    return out


def seq_logp(peft_model, ids, prompt_len):
    """Sum of log-probs of the answer tokens. Only the answer positions go through lm_head: the full 4k x 151k
    logits would not fit a T4."""
    base = peft_model.get_base_model()
    x = torch.tensor([ids], device="cuda")
    h = base.model(input_ids=x).last_hidden_state[0, prompt_len - 1:-1]
    logits = base.lm_head(h).float()
    tgt = x[0, prompt_len:]
    return torch.log_softmax(logits, -1).gather(1, tgt[:, None]).sum(), len(tgt)


def train(args):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed)
    no_gqa_sdpa()
    tok = AutoTokenizer.from_pretrained(args.model)
    data = build_pairs(args, tok)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    t0 = time.time()
    model.eval()
    with torch.no_grad(), model.disable_adapter(), torch.autocast("cuda", dtype=torch.float16):
        for d in data:
            d["ref_c"] = seq_logp(model, d["chosen"], d["prompt_len"])[0].item()
            d["ref_r"] = seq_logp(model, d["rejected"], d["prompt_len"])[0].item()
    print(f"dpo reference log-probs: {round(time.time() - t0)}s", flush=True)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    total = math.ceil(len(data) * args.epochs / args.accum)
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    step, stats = 0, {"loss": 0.0, "acc": 0.0, "margin": 0.0, "n": 0}
    order = []
    for e in range(math.ceil(args.epochs)):
        idx = list(range(len(data)))
        random.Random(args.seed + e).shuffle(idx)
        order += idx
    order = order[:int(len(data) * args.epochs)]
    for k, i in enumerate(order, 1):
        d = data[i]
        with torch.autocast("cuda", dtype=torch.float16):
            pc, nc = seq_logp(model, d["chosen"], d["prompt_len"])
            pr, _ = seq_logp(model, d["rejected"], d["prompt_len"])
        margin = (pc - d["ref_c"]) - (pr - d["ref_r"])
        loss = -F.logsigmoid(args.beta * margin) + args.alpha * (-pc / nc)
        if not torch.isfinite(loss):
            print("non-finite loss, skipped", flush=True)
            opt.zero_grad(set_to_none=True)
            continue
        scaler.scale(loss / args.accum).backward()
        stats["loss"] += loss.item()
        stats["acc"] += float(margin.item() > 0)
        stats["margin"] += margin.item()
        stats["n"] += 1
        if k % args.accum == 0 or k == len(order):
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            if step % 5 == 0 or k == len(order):
                n = max(stats["n"], 1)
                print(f"dpo step {step}/{total} loss {stats['loss'] / n:.4f} reward_acc {stats['acc'] / n:.3f} "
                      f"margin {stats['margin'] / n:.3f} lr {sched.get_last_lr()[0]:.2e} {round(time.time() - t0)}s", flush=True)
                stats = {"loss": 0.0, "acc": 0.0, "margin": 0.0, "n": 0}
    model.save_pretrained(args.out + "_adapter")
    merged = model.merge_and_unload()
    for p in merged.parameters():
        p.data = p.data.half()
    merged.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("dpo done", args.out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mine")
    m.add_argument("--model", required=True)
    m.add_argument("--gold", required=True)
    m.add_argument("--exclude")
    m.add_argument("--out", required=True)
    m.add_argument("--skip", nargs="*", help="pair files whose ids are already mined")
    m.add_argument("--reverse", action="store_true")
    m.add_argument("--shard", default="0/1")
    m.add_argument("--limit", type=int, default=100000)
    m.add_argument("--n-samples", type=int, default=2)
    m.add_argument("--temp", type=float, default=1.0)
    t = sub.add_parser("train")
    t.add_argument("--model", required=True)
    t.add_argument("--gold", required=True)
    t.add_argument("--pairs", nargs="+", required=True)
    t.add_argument("--catalog")
    t.add_argument("--out", required=True)
    t.add_argument("--beta", type=float, default=0.1)
    t.add_argument("--alpha", type=float, default=0.2, help="weight of the NLL term on the chosen answer")
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--epochs", type=float, default=1.0)
    t.add_argument("--accum", type=int, default=8)
    t.add_argument("--rank", type=int, default=16)
    t.add_argument("--max-len", type=int, default=6144)
    t.add_argument("--max-pairs", type=int, default=0)
    t.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    mine(args) if args.cmd == "mine" else train(args)


if __name__ == "__main__":
    main()
