"""Generate one-sentence reasoning ("thought") for every step of the gold trajectories with a local open model.

The teacher sees the same prompt as the student plus the gold next action and explains why it is right, citing the
values it used (numbers, IDs, conditions). Runs fully locally (no data leaves the machine). Resumable.

python train/gen_thoughts.py --root data --model Qwen/Qwen3-14B --quant 4bit --out train/thoughts.jsonl [--limit 20]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from build_sft import last_text  # noqa: E402
from prompt import initial_messages, installed_tool_names, truncate  # noqa: E402

TEACHER_SYS = (
    "Bạn viết lập luận ngắn cho một trợ lý giọng nói tiếng Việt gọi tool. Bạn được xem yêu cầu của người dùng, "
    "các bước đã làm cùng kết quả tool, và HÀNH ĐỘNG TIẾP THEO ĐÚNG. Hãy viết suy nghĩ của chính trợ lý ngay trước khi "
    "thực hiện hành động đó: 1-2 câu, tối đa 40 từ, nêu cụ thể điều kiện/số liệu/ID dựa trên lời người dùng và kết quả "
    "tool (ví dụ so sánh số, vì sao chọn ID này, giờ hoặc ngày tính ra sao, còn việc nào chưa làm). "
    "Nếu hành động là {\"done\": true}: nếu chưa gọi tool nào thì nêu lý do cụ thể dựa trên danh sách ứng dụng đã cài "
    "(không có ứng dụng nào xử lý được yêu cầu này, hoặc yêu cầu không phải hành động); nếu đã làm rồi thì tóm tắt các việc "
    "đã xong và xác nhận không còn việc nào. "
    "Không nhắc đến 'hành động đúng' hay 'đáp án'. Chỉ trả về đúng câu suy nghĩ, không JSON, không markdown."
)


def steps(item, catalog):
    """Yield (step_index, teacher_user_text, action_json) for each assistant turn of the gold trajectory."""
    label = item["label"]
    calls, results = label["tool_calls"], label.get("tool_results") or []
    transcript = last_text(item.get("audio_text"))
    full = initial_messages(item, transcript, catalog)[1]["content"]
    app_ids, _ = installed_tool_names(item)
    # the full tool list is long and not needed to explain an action; the installed apps are (out-of-scope requests)
    base = f"## Ứng dụng đã cài\n{', '.join(sorted(app_ids))}\n\n" + full[full.index("## Ngữ cảnh"):]
    done_so_far, n = [], 0
    for i, c in enumerate(calls):
        action = {"name": c["name"], "arguments": c.get("arguments", {})}
        yield i, base, done_so_far, action
        n = i + 1
        done_so_far = done_so_far + [(action, results[i] if i < len(results) else {"ok": True})]
        if c["name"] == "sys.ask_missing_info":
            break
    yield n, base, done_so_far, {"done": True}


def teacher_text(base, done_so_far, action):
    parts = [base, "\n\n## Các bước đã làm"]
    for a, r in done_so_far:
        parts.append(f"- {json.dumps(a, ensure_ascii=False)}\n  KẾT QUẢ: {truncate(r)}")
    if not done_so_far:
        parts.append("(chưa có)")
    parts.append("\n## HÀNH ĐỘNG TIẾP THEO ĐÚNG\n" + json.dumps(action, ensure_ascii=False))
    parts.append("\nViết suy nghĩ của trợ lý trước hành động này.")
    return "\n".join(parts)


def load_model(name, quant, offload_gb=0):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    kw = {"dtype": torch.bfloat16, "device_map": {"": 0}}
    if offload_gb:  # model larger than VRAM: keep what fits on the GPU, the rest in CPU RAM (much slower)
        kw["device_map"] = "auto"
        kw["max_memory"] = {0: "14GiB", "cpu": f"{offload_gb}GiB"}
    if quant == "4bit":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=bool(offload_gb),
        )
    model = AutoModelForCausalLM.from_pretrained(name, **kw).eval()
    return tok, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--quant", default="4bit", choices=["4bit", "none"])
    ap.add_argument("--out", default="train/thoughts.jsonl")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=100)
    ap.add_argument("--offload-gb", type=int, default=0, help="CPU RAM (GiB) to spill layers into, for models above 16GB VRAM")
    ap.add_argument("--ids", nargs="*", default=None, help="only these sample ids (smoke test on hard cases)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N samples (smoke test)")
    args = ap.parse_args()

    import time

    import torch

    base = Path(args.root) / args.split
    catalog = json.load(open(base / f"{args.split}_tools_catalog.json", encoding="utf-8"))
    done = set()
    if Path(args.out).exists():
        for line in open(args.out, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["id"], r["step"]))

    todo = []
    for n, line in enumerate(open(base / f"{args.split}.jsonl", encoding="utf-8")):
        if args.limit is not None and n >= args.limit:
            break
        item = json.loads(line)
        if args.ids and item["id"] not in args.ids:
            continue
        for i, b, d, action in steps(item, catalog):
            if (item["id"], i) not in done:
                todo.append((item["id"], i, teacher_text(b, d, action)))
    print(f"steps to generate: {len(todo)} (already done: {len(done)})")

    tok, model = load_model(args.model, args.quant, args.offload_gb)

    def prompt_ids(text):
        msgs = [{"role": "system", "content": TEACHER_SYS}, {"role": "user", "content": text}]
        return tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False)

    prompts = [(sid, i, prompt_ids(t)) for sid, i, t in todo]
    prompts.sort(key=lambda x: len(x[2]))  # batch similar lengths together
    t0 = time.time()
    with open(args.out, "a", encoding="utf-8") as fout:
        for s in range(0, len(prompts), args.batch):
            chunk = prompts[s : s + args.batch]
            enc = tok([c[2] for c in chunk], return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
            for (sid, i, _), o in zip(chunk, out):
                thought = tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True).strip().replace("\n", " ")
                fout.write(json.dumps({"id": sid, "step": i, "thought": thought}, ensure_ascii=False) + "\n")
            fout.flush()
            k = s + len(chunk)
            el = (time.time() - t0) / 60
            print(f"{k}/{len(prompts)}  elapsed {el:.1f}m  eta {el / k * (len(prompts) - k):.1f}m", flush=True)


if __name__ == "__main__":
    main()
