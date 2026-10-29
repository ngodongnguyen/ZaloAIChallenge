"""Turn train.jsonl labels into multi-turn SFT trajectories (same prompt as inference).

Each sample becomes: system/user prompt -> [assistant call, user result]* -> assistant {"done": true}.
With --asr-cache, each sample is emitted twice (gold transcript and ASR transcript) to be robust to ASR noise.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from prompt import call_msg, done_msg, initial_messages, result_msg  # noqa: E402
from augment import augment  # noqa: E402


def last_text(audio_text):
    if isinstance(audio_text, list):
        return audio_text[-1] if audio_text else ""
    return audio_text or ""


def trajectory(item, transcript, catalog):
    label = item["label"]
    calls = label["tool_calls"]
    results = label.get("tool_results") or []
    msgs = initial_messages(item, transcript, catalog)
    for i, c in enumerate(calls):
        msgs.append(call_msg(c["name"], c.get("arguments", {})))
        if c["name"] == "sys.ask_missing_info":
            break
        res = results[i] if i < len(results) else {"ok": True}
        msgs.append(result_msg(res))
    msgs.append(done_msg())
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--split", default="train")
    ap.add_argument("--asr-cache", default=None)
    ap.add_argument("--out", default="train/sft.jsonl")
    ap.add_argument("--aug-frac", type=float, default=0.0, help="fraction of trajectories whose app/tool names are renamed")
    ap.add_argument("--seed", default="aug1")
    args = ap.parse_args()

    base = Path(args.root) / args.split
    catalog = json.load(open(base / f"{args.split}_tools_catalog.json", encoding="utf-8"))
    asr = {}
    if args.asr_cache:
        for line in open(args.asr_cache, encoding="utf-8"):
            r = json.loads(line)
            asr[r["id"]] = r["text"]

    n = n_aug = 0
    with open(base / f"{args.split}.jsonl", encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            item = json.loads(line)
            texts = [last_text(item.get("audio_text"))]
            if item["id"] in asr and asr[item["id"]] != texts[0]:
                texts.append(asr[item["id"]])
            for k, t in enumerate(texts):
                it, cat = item, catalog
                if random.Random(f"{args.seed}-{item['id']}-{k}-pick").random() < args.aug_frac:
                    it, cat = augment(item, catalog, f"{args.seed}-{item['id']}-{k}")
                    n_aug += 1
                fout.write(json.dumps({"id": item["id"], "messages": trajectory(it, t, cat)}, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} trajectories ({n_aug} renamed) to {args.out}")


if __name__ == "__main__":
    main()
