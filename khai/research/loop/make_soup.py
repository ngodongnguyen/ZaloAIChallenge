"""Write a model soup of LoRA adapters as one LoRA adapter, so any kernel that merges an adapter can run it.

  python loop/make_soup.py --runs r018-e16-final r020-e16-final-seed7 --out soup-r018-r020

The mean of the deltas s_i * B_i @ A_i (s_i = alpha_i / r_i) is exactly s * B @ A with A = [A_1; ...; A_n] stacked
along the rank and B = [B_1, ..., B_n] * (s_i / s) / n, rank sum(r_i) and alpha = s * rank. Written to
loop/runs_work/<out>/output/out/lora, where infer_run.py --llm-run <out> finds it.
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

RUNS = Path("C:/Project/zalo_research/loop/runs_work")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dirs = [RUNS / r / "output" / "out" / "lora" for r in args.runs]
    cfgs = [json.load(open(d / "adapter_config.json")) for d in dirs]
    ws = [load_file(str(d / "adapter_model.safetensors")) for d in dirs]
    scales = [c["lora_alpha"] / c["r"] for c in cfgs]
    s, n = scales[0], len(dirs)
    rank = sum(c["r"] for c in cfgs)
    out = {}
    for k in ws[0]:
        if ".lora_A." not in k:
            continue
        kb = k.replace(".lora_A.", ".lora_B.")
        if not all(k in w and kb in w for w in ws):
            raise SystemExit(f"{k} is missing from some adapter")
        out[k] = torch.cat([w[k].float() for w in ws], 0).contiguous()
        out[kb] = torch.cat([w[kb].float() * (si / s) / n for w, si in zip(ws, scales)], 1).contiguous()
    dest = RUNS / args.out / "output" / "out" / "lora"
    dest.mkdir(parents=True, exist_ok=True)
    save_file(out, str(dest / "adapter_model.safetensors"))
    cfg = dict(cfgs[0], r=rank, lora_alpha=s * rank)
    cfg.pop("rank_pattern", None)
    cfg.pop("alpha_pattern", None)
    json.dump(cfg, open(dest / "adapter_config.json", "w"), indent=2)
    (dest / "SOUP.txt").write_text("mean of the deltas of: " + ", ".join(args.runs) + "\n")
    # exactness check on one matrix
    k = next(x for x in out if ".lora_A." in x)
    kb = k.replace(".lora_A.", ".lora_B.")
    want = sum(si * (w[kb].float() @ w[k].float()) for w, si in zip(ws, scales)) / n
    got = cfg["lora_alpha"] / cfg["r"] * (out[kb] @ out[k])
    print(f"soup of {n} adapters, rank {rank}, {len(out) // 2} modules, max abs error {float((want - got).abs().max()):.2e} -> {dest}")


if __name__ == "__main__":
    main()
