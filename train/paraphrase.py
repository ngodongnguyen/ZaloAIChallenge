"""Rewrite each train utterance in other words with a local open model; the gold tool_calls stay the same.

Only the wording changes. A rewrite is kept when it still carries every number, time/day word, condition word and
proper name of the original (and adds no new number), so the old label is still correct for it. Runs fully locally.
Resumable by (id, k).

python train/paraphrase.py --root data --model Qwen/Qwen3-14B --out train/paraphrases.jsonl [--n 2] [--limit 20]
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from build_sft import last_text  # noqa: E402
from gen_thoughts import load_model  # noqa: E402

SYS = (
    "Bạn viết lại câu nói của một người dùng với trợ lý giọng nói tiếng Việt. Giữ nguyên ý, mọi con số, giờ, ngày, tên người, "
    "địa điểm, ứng dụng, điều kiện (nếu/thì/trên/dưới/không thì) và thứ tự các việc cần làm; không thêm hay bớt yêu cầu nào. "
    "Chỉ thay cách diễn đạt. Phong cách: {style}. "
    "Chỉ trả về đúng câu mới trên một dòng, không giải thích, không đặt trong dấu ngoặc kép."
)
STYLES = [
    "nói tự nhiên đời thường, có từ đệm như 'ờ', 'à', 'ừm', 'nhỉ', xưng hô khác câu gốc",
    "ngắn gọn, thân mật, bỏ bớt từ thừa nhưng vẫn đủ ý",
    "lịch sự, dài hơn, có lời mở đầu hoặc lời nhờ vả nhẹ nhàng",
    "như người đang vội, nói nhanh, câu cộc, không dấu chấm câu, vài chỗ viết thường",
]

# words whose presence in the original must survive: numbers, units, time and day words, conditions, comparisons
KEEP = {
    "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín", "mười", "mươi", "trăm", "nghìn", "ngàn", "triệu", "tỷ", "rưỡi",
    "giờ", "phút", "sáng", "trưa", "chiều", "tối", "đêm", "mai", "nay", "hôm", "tuần", "tháng", "ngày", "mốt", "qua",
    "nếu", "thì", "không", "trên", "dưới", "hơn", "ít", "nhiều", "trừ", "ngoại", "tất", "hết", "cả", "mỗi",
    "tắt", "bật", "mở", "đóng", "khóa", "khoá", "gọi", "nhắn", "chuyển", "đặt", "hủy", "huỷ", "xóa", "xoá", "thêm", "báo",
}
DIGITS = re.compile(r"\d+(?:[.,]\d+)?")
WORD = re.compile(r"\w+", re.UNICODE)


def names(text):
    """Capitalised words that are not the first word of the sentence (people, places, brands)."""
    ws = WORD.findall(text)
    return {w for w in ws[1:] if w[0].isupper()}


def keeps(orig, new):
    if set(DIGITS.findall(orig)) != set(DIGITS.findall(new)):
        return False
    lo, ln = {w.lower() for w in WORD.findall(orig)}, {w.lower() for w in WORD.findall(new)}
    if any(k in lo and k not in ln for k in KEEP):
        return False
    if any(n.lower() not in ln for n in names(orig)):
        return False
    r = len(new) / max(1, len(orig))
    return 0.5 <= r <= 2.5 and new.strip().lower() != orig.strip().lower() and "\n" not in new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--quant", default="4bit", choices=["4bit", "none"])
    ap.add_argument("--out", default="train/paraphrases.jsonl")
    ap.add_argument("--n", type=int, default=2, help="rewrites per sample, each in a different style")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None, help="only the first N samples (smoke test)")
    args = ap.parse_args()

    import time

    import torch

    base = Path(args.root) / args.split
    done = set()
    if Path(args.out).exists():
        for line in open(args.out, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["id"], r["k"]))

    todo = []
    for n, line in enumerate(open(base / f"{args.split}.jsonl", encoding="utf-8")):
        if args.limit is not None and n >= args.limit:
            break
        item = json.loads(line)
        text = last_text(item.get("audio_text"))
        if not text.strip():
            continue
        for k in range(args.n):
            if (item["id"], k) not in done:
                todo.append((item["id"], k, text, STYLES[(n + k) % len(STYLES)]))
    print(f"rewrites to generate: {len(todo)} (already done: {len(done)})")
    if not todo:
        return

    tok, model = load_model(args.model, args.quant)

    def prompt(text, style):
        msgs = [{"role": "system", "content": SYS.format(style=style)}, {"role": "user", "content": f"Câu gốc: {text}"}]
        return tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False)

    prompts = sorted(((sid, k, text, prompt(text, st)) for sid, k, text, st in todo), key=lambda x: len(x[3]))
    t0, kept = time.time(), 0
    with open(args.out, "a", encoding="utf-8") as fout:
        for s in range(0, len(prompts), args.batch):
            chunk = prompts[s : s + args.batch]
            enc = tok([c[3] for c in chunk], return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
            for (sid, k, text, _), o in zip(chunk, out):
                new = tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True).strip().strip('"“”').replace("\n", " ")
                ok = keeps(text, new)
                kept += ok
                # rejected ones are written too (ok=false) so a re-run does not redo them
                fout.write(json.dumps({"id": sid, "k": k, "text": new, "ok": ok}, ensure_ascii=False) + "\n")
            fout.flush()
            k_ = s + len(chunk)
            el = (time.time() - t0) / 60
            print(f"{k_}/{len(prompts)}  kept {kept}  elapsed {el:.1f}m  eta {el / k_ * (len(prompts) - k_):.1f}m", flush=True)


if __name__ == "__main__":
    main()
