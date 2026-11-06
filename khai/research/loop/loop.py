"""Research loop: Codex proposes a change to src/, the harness evaluates it on the dev split on Kaggle,
and the change is kept only if the dev score rises by at least GATE. Otherwise the workspace is reset.

Stop conditions (first to hit wins): budget line, MAX_ROUNDS, NO_IMPROVE_LIMIT rounds without a keep,
ERROR_LIMIT consecutive errors. Every stop reason is written to loop/log.txt.
"""
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

RESEARCH = Path("C:/Project/zalo_research")
sys.path.insert(0, str(RESEARCH))

from harness.budget import Budget
from harness.evalrun import evaluate
from harness.registry import Registry
from harness.split import load_split, load_train, DATA_ROOT, group_of, sha_check
from harness.loop_config import SFT_CONFIG, eval_params

WORKSPACE = RESEARCH / "workspace"
SPLIT = RESEARCH / "splits" / "lifemate_split.json"
LOOP = RESEARCH / "loop"
CODEX = Path("C:/Users/Khai/.vscode/extensions/openai.chatgpt-26.1021.22027-win32-x64/bin/windows-x86_64/codex.exe")
CODEX_MODEL = "gpt-6-luna"
ALLOWED_EDITS = {"src/agent.py", "src/config.py", "train/sft.py", "train/build_sft.py"}  # prompt.py is tuned later
GATE = 0.02            # keep a change only if dev score rises by at least this much
MAX_ROUNDS = 10        # per invocation
NO_IMPROVE_LIMIT = 3
ERROR_LIMIT = 2
CODEX_TIMEOUT_S = 45 * 60
EVAL_EST_S = 5400      # SFT (about 1 hour) plus dev eval (about 35 minutes)
BASELINE_LABEL = "r001-sft"
ACCOUNT = sys.argv[2] if len(sys.argv) > 2 else "acc2"  # Kaggle account for this loop


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOOP / "log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def git(*args, check=True):
    return subprocess.run(["git", "-C", str(WORKSPACE), *args], check=check,
                          capture_output=True, text=True).stdout.strip()


def load_rows(label):
    got = {}
    for line in (LOOP / "runs_work" / label / "output.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            got[r["id"]] = r
    return got


def gold_calls():
    return {r["id"]: r["label"]["tool_calls"] for r in load_train(DATA_ROOT / "train" / "train.jsonl")}


def error_summary(label, split):
    """Counts of error types over the dev split, plus mean score per app group."""
    per = json.loads((LOOP / "runs_work" / label / "per_sample.json").read_text(encoding="utf-8"))
    got, gold = load_rows(label), gold_calls()
    gh = split["gold_state_hash"]
    cnt = Counter()
    groups = defaultdict(list)
    for sid in split["dev"]:
        r, g = got[sid], gold[sid]
        sub = r["tool_calls"]
        gn = {c["name"] for c in g}
        sn = {c["name"] for c in sub}
        if g and not sub:
            cnt["gold has calls, submitted none"] += 1
        if not g and sub:
            cnt["gold empty, submitted calls"] += 1
        if "sys.ask_missing_info" in sn and "sys.ask_missing_info" not in gn:
            cnt["asked for info, gold did not"] += 1
        if "sys.ask_missing_info" in gn and "sys.ask_missing_info" not in sn:
            cnt["gold asked for info, submitted did not"] += 1
        if g and sub and gn - sn:
            cnt["some gold tool never called"] += 1
        if r["invalid_outputs"]:
            cnt["sample has an unparseable model output"] += 1
        if r["state_hash"] != gh[sid]:
            cnt["final state differs from gold"] += 1
        groups[group_of({"label": {"tool_calls": g}})].append(per[sid])
    n = len(split["dev"])
    lines = [f"- {k}: {v} trên {n} mẫu" for k, v in cnt.most_common()]
    lines.append("Điểm trung bình theo nhóm app (nhóm = app của tool call gold đầu tiên):")
    lines += [f"- {g}: {len(v)} mẫu, điểm TB {sum(v) / len(v):.3f}" for g, v in sorted(groups.items())]
    return "\n".join(lines)


def worst_report(label, split, k=12):
    per = json.loads((LOOP / "runs_work" / label / "per_sample.json").read_text(encoding="utf-8"))
    got, gold, gh = load_rows(label), gold_calls(), split["gold_state_hash"]
    worst = sorted(((sid, per[sid]) for sid in split["dev"]), key=lambda kv: (kv[1], kv[0]))[:k]
    blocks = []
    for sid, s in worst:
        r = got[sid]
        g = [(c["name"], c["arguments"]) for c in gold[sid]]
        sub = [(c["name"], c["arguments"]) for c in r["tool_calls"]]
        raw = [o[:400] for o in r["outputs"][:3]]
        blocks.append(
            f"- {sid} điểm={s:.2f}, trạng thái khớp gold={r['state_hash'] == gh[sid]}\n"
            f"  gold: {json.dumps(g, ensure_ascii=False)[:900]}\n"
            f"  đã gọi: {json.dumps(sub, ensure_ascii=False)[:900]}\n"
            f"  output thô của model (tối đa 3): {json.dumps(raw, ensure_ascii=False)}")
    return "\n".join(blocks)


def history_text(n=8):
    rows = []
    p = LOOP / "summary.jsonl"
    if p.exists():
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return "- (chưa có vòng nào sau baseline)"
    return "\n".join(f"- vòng {r['round']}: {r['hypothesis']} → thay đổi điểm {r['gain']:+.4f}, "
                     f"{'giữ' if r['kept'] else 'bỏ'}" for r in rows[-n:])


def codex_prompt(score, best_label, split):
    readme = (WORKSPACE / "README.md").read_text(encoding="utf-8")[:1500]
    return f"""Bạn là nhà nghiên cứu tự động cho bài LifeMate (agent giọng nói tiếng Việt → tool call).
Workspace là repo mã nguồn. Model được LoRA SFT trên dữ liệu train (train/build_sft.py tạo quỹ đạo, train/sft.py huấn luyện)
trước khi đánh giá. Được sửa: src/agent.py, src/config.py, train/sft.py, train/build_sft.py. Không sửa src/prompt.py
(prompt tune riêng sau) và file khác. Dev và holdout đã bị loại khỏi dữ liệu SFT tự động, đừng cố đưa lại vào.
Khi sửa SFT, cân nhắc cả dữ liệu (chọn mẫu, cách biểu diễn trajectory) lẫn siêu tham số (số mẫu, epoch, lr, độ dài).

Mục tiêu: tăng điểm trung bình trên tập dev. Điểm hiện tại: {score:.4f} (run {best_label}).

Tóm tắt README:
{readme}

Thống kê lỗi trên tập dev hiện tại:
{error_summary(best_label, split)}

Các mẫu điểm thấp nhất:
{worst_report(best_label, split)}

Các vòng trước đã thử (không lặp lại giả thuyết đã bỏ):
{history_text()}

Yêu cầu:
1. Phân tích lỗi trước: loại lỗi nào chiếm nhiều nhất, nhóm app nào sai, và vì sao theo code trong src/agent.py.
   Đọc được src/prompt.py để hiểu, nhưng không sửa.
2. Nêu một cơ chế nghi ngờ cụ thể, dựa trên số liệu và mẫu ở trên.
3. Suy ra MỘT thay đổi nhỏ có thể làm, nhắm đúng loại lỗi đã quan sát, và dự đoán nó cải thiện loại lỗi nào, bao nhiêu mẫu.
4. Sửa code trong workspace. Không chạy huấn luyện, không chạy đánh giá trên Kaggle, không đọc thư mục splits/.
5. Kết thúc bằng một khối ```json có các khóa: "hypothesis" (ít nhất 10 ký tự), "target_error", "predicted_fix".
"""


def run_codex(prompt, last_msg_path):
    LOOP.mkdir(exist_ok=True)
    proc = subprocess.run(
        [str(CODEX), "exec", "-m", CODEX_MODEL, "-s", "workspace-write", "-C", str(WORKSPACE),
         "-o", str(last_msg_path), "-"],
        input=prompt, capture_output=True, text=True, encoding="utf-8",
        timeout=CODEX_TIMEOUT_S)
    if proc.returncode != 0:
        raise RuntimeError(f"codex exec exit {proc.returncode}: {proc.stderr[-500:]}")
    return last_msg_path.read_text(encoding="utf-8")


def parse_proposal(text):
    blocks = re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    if not blocks:
        raise ValueError("no ```json proposal block in Codex reply")
    p = json.loads(blocks[-1])
    if not isinstance(p.get("hypothesis"), str) or len(p["hypothesis"].strip()) < 10:
        raise ValueError("proposal hypothesis missing or too short")
    return p


def changed_files():
    # Not `git status --porcelain`: git() strips its output, which eats the leading space of the
    # first status line and shifts every path by one character.
    tracked = git("diff", "--name-only", "HEAD").splitlines()
    untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
    return sorted({p for p in tracked + untracked if p.strip()})


def commit_round(message):
    git("add", "-A", "src")
    git("-c", "user.name=research", "-c", "user.email=research@local", "commit", "-m", message)
    return git("rev-parse", "HEAD")


def main(max_rounds=MAX_ROUNDS):
    split_hash_before = sha_check(SPLIT)
    reg = Registry(RESEARCH)
    budget = Budget(RESEARCH / ("budget.json" if ACCOUNT == "acc1" else f"budget_{ACCOUNT}.json"))
    split = load_split(SPLIT)
    dev = split["dev"]

    summary_path = LOOP / "summary.jsonl"
    prev_rounds = 0
    if summary_path.exists():
        prev_rounds = sum(1 for _ in summary_path.open(encoding="utf-8"))

    best_label = BASELINE_LABEL
    best_metrics = json.loads((reg.runs_dir / best_label / "metrics.json").read_text(encoding="utf-8"))
    best_commit = git("rev-parse", "HEAD")
    parent = best_label
    no_improve = errors = 0

    for n in range(1, max_rounds + 1):
        round_no = prev_rounds + n
        label = f"r{round_no:03d}"
        if not budget.can_run(EVAL_EST_S):
            log("STOP budget line reached")
            break
        if no_improve >= NO_IMPROVE_LIMIT:
            log(f"STOP {no_improve} rounds without a keep")
            break
        if errors >= ERROR_LIMIT:
            log(f"STOP {errors} consecutive errors")
            break
        if sha_check(SPLIT) != split_hash_before:
            log("STOP split file changed; refusing to continue")
            break

        log(f"round {round_no}: asking Codex (best={best_label} {best_metrics['score']})")
        try:
            reply = run_codex(codex_prompt(best_metrics["score"], best_label, split),
                              LOOP / f"codex_{label}.md")
            proposal = parse_proposal(reply)
        except Exception as e:
            errors += 1
            log(f"round {round_no} codex error: {e!r}")
            git("reset", "--hard", best_commit)
            continue

        bad = [p for p in changed_files() if p not in ALLOWED_EDITS]
        if bad or not changed_files():
            if bad:
                errors += 1
            log(f"round {round_no} rejected: changed={changed_files()} (allowed: {sorted(ALLOWED_EDITS)})")
            git("reset", "--hard", best_commit)
            git("clean", "-fd", "src")
            no_improve += 1
            continue

        head = commit_round(f"{label}: {proposal['hypothesis'][:60]}")
        params = eval_params(split, label)
        cfg = {"hypothesis": proposal["hypothesis"], "parent": parent, "llm_dtype": "fp16",
               "llm_device": "1gpu", "n_samples": len(dev), "split": "dev", "sft": SFT_CONFIG}
        try:
            run_id, metrics = evaluate(label, params, SPLIT, WORKSPACE, reg, budget,
                                       est_s=EVAL_EST_S, cfg=cfg, account=ACCOUNT)
            errors = 0
        except Exception as e:
            errors += 1
            log(f"round {round_no} eval error: {e!r}")
            git("reset", "--hard", best_commit)
            continue

        gain = metrics["score"] - best_metrics["score"]
        kept = gain >= GATE
        if kept:
            best_label, best_metrics, best_commit, parent = run_id, metrics, head, run_id
            no_improve = 0
        else:
            git("reset", "--hard", best_commit)
            no_improve += 1
        reg.rebuild_index()
        row = {"round": round_no, "run_id": run_id, "score": metrics["score"],
               "gain": round(gain, 6), "kept": kept, "hypothesis": proposal["hypothesis"],
               "target_error": proposal.get("target_error"), "predicted_fix": proposal.get("predicted_fix")}
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        log(f"round {round_no}: score={metrics['score']} gain={gain:+.4f} kept={kept}")

    reg.rebuild_index()
    log(f"loop finished; best={best_label} score={best_metrics['score']}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else MAX_ROUNDS)
