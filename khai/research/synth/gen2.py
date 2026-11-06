"""Second round of synthetic data from gpt-6.1-sol: more apps, and scenarios aimed at the errors r021 still makes.

Resumable by design, since the Codex quota can run out at any point:
- plan.json fixes every batch (apps, focus, seed) once; a rerun follows the same plan.
- Each finished call is written to its own file (temp file, then rename), so a crash never leaves a half file.
- A rerun skips every call whose file exists; failed calls have no file and simply run again.
- A quota or rate-limit message, or 3 failures in a row, stops all workers instead of burning retries.

  python synth/gen2.py apps        # new apps into synth/v2/apps/*.json
  python synth/gen2.py plan --batches 140
  python synth/gen2.py scenarios --workers 4 [--limit 2]
  python synth/gen2.py status
"""
import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gen import APPS_PROMPT, CODEX, MODEL, SCEN_PROMPT  # noqa: E402
from validate import check_app, real_app_ids  # noqa: E402

HERE = Path(__file__).parent
V2 = HERE / "v2"
QUOTA = re.compile(r"usage limit|rate.?limit|quota|429|too many requests|exceeded|credits", re.I)
BUSY = re.compile(r"at capacity|overloaded|temporarily unavailable|503", re.I)  # transient: wait and retry

# From the first-divergence analysis of r021 on unseen apps; "general" keeps the round-1 mix.
FOCUS = {
    "general": "",
    "conditional": "Mọi tình huống là yêu cầu có điều kiện phải đọc kết quả rồi mới quyết định (so sánh số tiền, số lượng, "
                   "ngày giờ, trạng thái). Đúng một nửa số tình huống điều kiện KHÔNG thỏa và trợ lý làm nhánh còn lại "
                   "hoặc dừng sau khi đọc. Có cả số lớn đọc bằng chữ (một trăm ba mươi mốt triệu, triệu rưỡi).",
    "dates": "Mọi tình huống cần tính ngày giờ tương đối từ current_time: thứ hai tuần sau, cuối tuần này, trước một sự kiện "
             "một ngày, hai tiếng trước cuộc họp, tối mai, sáng mốt, mười giờ rưỡi đêm nay, sau bốn mươi lăm phút. "
             "Có tình huống cần đổi ra số phút hoặc số ngày. Tính thật cẩn thận.",
    "pick_id": "Mọi tình huống phải gọi tool read trả về NHIỀU mục giống nhau (3 đến 6 mục), rồi chọn đúng một hoặc vài "
               "mục theo thuộc tính người dùng nói (tên gần giống, phòng, giờ, trạng thái, giá rẻ nhất, gần nhất). "
               "ID gọi tiếp phải là ID của đúng mục đó.",
    "ask_or_act": "Một nửa tình huống thiếu thông tin bắt buộc mà không tool nào tra được (giờ mới, số tiền, người nhận "
                  "không rõ) nên kết thúc bằng sys.ask_missing_info; nửa còn lại trông giống vậy nhưng thông tin tra được "
                  "bằng tool read hoặc có trong history, nên trợ lý tự tra rồi làm luôn, không hỏi lại.",
    "self_correct": "Mọi tình huống người dùng tự sửa giữa câu hoặc thu hẹp yêu cầu: \"à khoan\", \"mà thôi\", \"chỉ cái ... "
                    "thôi\", \"không, ý là\", đổi giờ hay đổi người giữa chừng. Trợ lý làm theo ý cuối cùng.",
    "multi_step": "Mọi tình huống cần 3 đến 5 tool call trên 2 hoặc 3 app, thứ tự có ý nghĩa (dừng rồi mới chạy lại, tra "
                  "rồi mới ghi). Chỉ gọi tool read khi thật sự cần ID hoặc dữ liệu cho bước sau, không đọc thừa.",
    "history": "Mọi tình huống có history 2 đến 4 cặp lượt và yêu cầu hiện tại chỉ hiểu được nhờ history (\"cái đó\", "
               "\"người lúc nãy\", \"đơn hồi nãy\", \"đổi lại giờ khác\"). ID cần dùng lấy từ tool_results trong history, "
               "không gọi lại tool read nếu đã có.",
    "exact_text": "Mọi tình huống có tham số chữ lấy nguyên văn từ lời người dùng (tiêu đề, ghi chú, nội dung tin nhắn, tên "
                  "món) và tham số tùy chọn chỉ truyền khi người dùng nói tới. Giữ đúng chữ người dùng nói, không diễn "
                  "giải lại.",
    # private test may hold situations seen nowhere in train; only conventions the real labels show are used
    "odd": "Mọi tình huống là cách nói lạ nhưng có thật: nói vòng vo hay kể lể trước khi vào ý, nhiều ý trong một câu, "
           "nói nhầm rồi sửa, từ địa phương ba miền (hông, nghen, mô, răng, bữa ni), trộn tiếng Anh, hoặc yêu cầu mà "
           "không app nào trong catalog làm được (tool_calls rỗng). Không làm tình huống chỉ làm được một phần.",
}
# The pilot used ids like pol_binh_an_01; real ids carry no meaning, so the student must read results to find them.
OPAQUE_IDS = ("- ID trong tool_results phải vô nghĩa như hệ thống thật: tiền tố ngắn cộng số hoặc mã ngẫu nhiên "
              "(pol_4812, site_207, GX58201937, a3f9k2), không chứa tên, địa danh hay nội dung của mục.\n")
WEIGHTS = {"general": 5, "conditional": 2, "dates": 2, "pick_id": 1, "ask_or_act": 1, "self_correct": 1,
           "multi_step": 1, "history": 1, "exact_text": 1, "odd": 1}

stop = threading.Event()
fails = [0]
lock = threading.Lock()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def codex(prompt, effort):
    """Returns parsed JSON. Raises QuotaError when the CLI reports a limit, ValueError on a bad reply."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.txt"
        cmd = [CODEX, "exec", "--model", MODEL, "-c", f'model_reasoning_effort="{effort}"', "--sandbox", "read-only",
               "--skip-git-repo-check", "--ephemeral", "-C", tmp, "-o", str(out), "-"]
        p = subprocess.run(cmd, input=prompt, text=True, encoding="utf-8", errors="replace", capture_output=True,
                           timeout=1800)
        text = out.read_text(encoding="utf-8") if out.exists() else ""
    if p.returncode != 0 or not text.strip():
        msg = (p.stderr or "")[-600:] + (p.stdout or "")[-300:]
        if QUOTA.search(msg):
            raise QuotaError(msg.strip()[-300:])
        raise ValueError(f"codex exit {p.returncode}: {msg.strip()[-300:]}")
    return json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip()))


class QuotaError(RuntimeError):
    pass


def all_apps():
    apps = json.loads((HERE / "apps.json").read_text(encoding="utf-8"))["apps"]
    for f in sorted((V2 / "apps").glob("*.json")):
        apps += json.loads(f.read_text(encoding="utf-8"))
    return apps


def step_apps(args):
    real = real_app_ids()
    for k in range(args.calls):
        path = V2 / "apps" / f"apps_{k:02d}.json"
        if path.exists():
            continue
        have = {a["id"] for a in all_apps()}
        prompt = APPS_PROMPT.format(n=args.n) + (
            f"\n- Không trùng lĩnh vực hay id với các app đã có: {', '.join(sorted(have))}.\n")
        try:
            apps = codex(prompt, args.effort)["apps"]
        except QuotaError as e:
            print("quota reached, stopping:", e, flush=True)
            return
        keep = [a for a in apps if a["id"] not in have and not check_app(a, real)]
        save(path, keep)
        print(f"apps call {k}: kept {len(keep)}/{len(apps)}: {[a['id'] for a in keep]}", flush=True)


def step_plan(args):
    path = V2 / "plan.json"
    if path.exists():
        print("plan exists, keeping it:", path)
        return
    apps = [a["id"] for a in all_apps()]
    rng = random.Random(7)
    bag = [f for f, w in WEIGHTS.items() for _ in range(w)]
    plan = [{"id": f"b{b:04d}", "focus": bag[b % len(bag)], "apps": rng.sample(apps, 4), "n": args.per_batch}
            for b in range(args.batches)]
    save(path, plan)
    print("plan", len(plan), "batches,", sum(p["n"] for p in plan), "scenarios")


def run_batch(b, apps, effort):
    if stop.is_set():
        return
    chosen = [apps[i] for i in b["apps"]]
    prompt = SCEN_PROMPT.format(catalog=json.dumps(chosen, ensure_ascii=False), n=b["n"]) + OPAQUE_IDS
    if FOCUS[b["focus"]]:
        prompt += f"\nTrọng tâm của lô này (ưu tiên hơn các phân bố ở trên nếu mâu thuẫn): {FOCUS[b['focus']]}\n"
    try:
        for wait in (60, 180, 0):
            try:
                rows = codex(prompt, effort)["scenarios"]
                break
            except ValueError as e:
                if not (wait and BUSY.search(str(e))) or stop.is_set():
                    raise
                print(f"{b['id']}: model busy, retrying in {wait}s", flush=True)
                stop.wait(wait)
    except QuotaError as e:
        print(f"{b['id']}: quota reached, stopping all workers: {e}", flush=True)
        stop.set()
        return
    except Exception as e:
        with lock:
            fails[0] += 1
            n = fails[0]
        print(f"{b['id']}: failed ({n} in a row): {e!r}"[:300], flush=True)
        if n >= 3:
            print("3 failures in a row, stopping all workers", flush=True)
            stop.set()
        return
    with lock:
        fails[0] = 0
    for r in rows:
        r["apps"], r["focus"], r["batch"] = b["apps"], b["focus"], b["id"]
    save(V2 / "batches" / f"{b['id']}.json", rows)
    print(f"{b['id']} {b['focus']}: {len(rows)} scenarios", flush=True)


def step_scenarios(args):
    plan = json.loads((V2 / "plan.json").read_text(encoding="utf-8"))
    apps = {a["id"]: a for a in all_apps()}
    todo = [b for b in plan if not (V2 / "batches" / f"{b['id']}.json").exists()]
    done = len(plan) - len(todo)
    if args.only:
        todo = [b for b in todo if b["id"] in args.only.split(",")]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} batches to run, {done} already done", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(lambda b: run_batch(b, apps, args.effort), todo))
    step_status(args)


def step_status(args):
    plan = json.loads((V2 / "plan.json").read_text(encoding="utf-8")) if (V2 / "plan.json").exists() else []
    done = sorted((V2 / "batches").glob("*.json")) if (V2 / "batches").exists() else []
    n = sum(len(json.loads(f.read_text(encoding="utf-8"))) for f in done)
    print(f"status: {len(done)}/{len(plan)} batches done, {n} scenarios", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["apps", "plan", "scenarios", "status"])
    ap.add_argument("--calls", type=int, default=4, help="apps: Codex calls")
    ap.add_argument("--n", type=int, default=10, help="apps: apps per call")
    ap.add_argument("--batches", type=int, default=140)
    ap.add_argument("--per-batch", type=int, default=14)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="scenarios: run at most this many batches (pilot)")
    ap.add_argument("--only", default="", help="scenarios: comma-separated batch ids (pilot)")
    ap.add_argument("--effort", default="high")
    args = ap.parse_args()
    {"apps": step_apps, "plan": step_plan, "scenarios": step_scenarios, "status": step_status}[args.step](args)


if __name__ == "__main__":
    main()
