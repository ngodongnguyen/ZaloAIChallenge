"""Synthetic apps and scenarios from gpt-6.1-sol (Codex CLI), for SFT only, never at inference.

Codex runs read-only in an empty folder. Prompts hold no train or public_test content; overlaps with real apps are
filtered here, so no real app list leaves this machine. Every scenario must pass validate.py to be kept.

  python synth/gen.py apps --out synth/apps_pilot.json --n 12
  python synth/gen.py scenarios --apps synth/apps_pilot.json --out synth/scen_pilot.jsonl --per-batch 12 --batches 1
"""
import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
CODEX = "C:/Users/Khai/.vscode/extensions/openai.chatgpt-26.51021.22027-win32-x64/bin/windows-x86_64/codex.exe"
MODEL = "gpt-6.1-sol"

APPS_PROMPT = """Hãy tự nghĩ ra {n} ứng dụng di động hư cấu cho một trợ lý giọng nói tiếng Việt, dùng ở Việt Nam. Mỗi ứng dụng là một lĩnh vực khác hẳn nhau. Khoảng 3/4 là ứng dụng phổ thông nhiều người Việt thực sự dùng hằng ngày (mua sắm, đi lại, sức khỏe, học tập, giải trí, tài chính cá nhân, ăn uống, thể thao, tiện ích công cộng, công việc văn phòng...), còn lại là ứng dụng ngách. Không làm chuyên ngành hẹp như thiết bị phòng thí nghiệm. Không dùng các lĩnh vực: nhà thông minh, thiết bị gia dụng, khóa cửa, robot hút bụi, giá vàng, theo dõi kiện hàng.

Trả về DUY NHẤT một JSON object {{"apps": [...]}}, không giải thích. Mỗi app:
{{"id": "snake_case_tieng_anh", "name_vi": "...", "group": "...", "description": "...", "tools": [...]}}
Mỗi app có 3 đến 6 tool. Mỗi tool:
{{"name": "<app_id>.<ten_tool>", "kind": "read" hoặc "write", "description": "mô tả ngắn tiếng Việt",
 "parameters": {{"type": "object", "properties": {{"<ten>": {{"type": "string|integer|number|boolean|array", "description": "...", "enum": [...]}}}}, "required": [...]}},
 "returns": "ví dụ orders[]"}}
Quy tắc:
- Mỗi app có ít nhất 1 tool read và 1 tool write.
- Tham số là ID lấy từ kết quả tool read thì thêm "x-ref": "<app_id>.<tên danh sách>", ví dụ "x-ref": "pet_clinic.pets", và mô tả ghi rõ lấy từ tool nào.
- Tham số thời gian có "format": "date-time" và mô tả "ISO-8601 đầy đủ giây và offset múi giờ, ví dụ 2026-11-11T05:30:00+07:00".
- Tham số tự do như ghi chú, tên món có "x-match": "fuzzy".
- Tên tham số và mô tả rõ ràng, thực tế như API thật. Tên tool đa dạng, không chỉ get/list/create.
"""

SCEN_PROMPT = """Dưới đây là catalog các ứng dụng hư cấu của một trợ lý giọng nói tiếng Việt:
{catalog}

Hãy viết {n} tình huống huấn luyện KHÁC NHAU. Trả về DUY NHẤT một JSON object {{"scenarios": [...]}}, không giải thích. Mỗi tình huống:
{{"current_time": "<ISO 8601, +07:00>",
 "user_profile": {{"name": "...", "home_address": "...", "work_address": "...", "city": "..."}},
 "history": [],
 "utterance": "câu người dùng nói",
 "tool_calls": [{{"name": "<app.tool>", "arguments": {{...}}}}],
 "tool_results": [{{"ok": true, ...}}]}}

Phân bố cần bám theo, tính trên {n} tình huống:
- Số tool call: 0 gọi 7%, 1 gọi 24%, 2 gọi 28%, 3 gọi 23%, 4 gọi 10%, 5 gọi trở lên 8%.
- current_time mỗi tình huống khác nhau: ngày bất kỳ trong tháng 11, 12/2026 và 1/2027, mọi khung giờ trong ngày.
- Độ dài utterance: một nửa dưới 80 ký tự, 3/4 dưới 100 ký tự, chỉ khoảng 5% trên 150 ký tự; đa số là một câu ngắn như người ta nói thật. Yêu cầu nhiều bước vẫn nói gọn.
- Khoảng 14% thiếu thông tin bắt buộc mà không tra được: gọi tool read nếu có ích rồi kết thúc bằng {{"name": "sys.ask_missing_info", "arguments": {{"tool_name": "<tool định gọi>", "params": ["<tham số thiếu>"]}}}}; call này là call cuối và không có tool_result.
- Khoảng 7% không app nào làm được yêu cầu: tool_calls rỗng.
- Khoảng 45% dùng 2 app trở lên trong một yêu cầu, có cả yêu cầu có điều kiện ("nếu ... thì ...") phải đọc kết quả rồi mới quyết định.
- Khoảng 65% có history: 1 đến 4 cặp lượt trước. Lượt người dùng {{"role": "user", "text": "..."}}; lượt trợ lý {{"role": "assistant", "text": "...", "tool_calls": [...], "tool_results": [...]}}. Yêu cầu hiện tại thường dựa vào history ("cái đó", "người lúc nãy", "hủy luôn đi").

Quy tắc bắt buộc:
- utterance là lời nói tự nhiên, đời thường của người Việt, có từ đệm, xưng hô đa dạng (anh, chị, em, cô, chú, con, mình, tôi), không nêu tên tool và không đọc ID. Số và giờ có thể đọc bằng chữ. Không viết như văn bản.
- tool_results có đúng một phần tử cho mỗi call đã chạy, theo thứ tự. Tool read trả dữ liệu thực tế có ID, ví dụ {{"ok": true, "pets": [{{"id": "pet_01", "name": "Mực"}}]}}. Tool write trả {{"ok": true, "id": "...", "item": {{...}}}} hoặc tương tự.
- Tham số có x-ref phải là ID đã xuất hiện trong tool_results trước đó của tình huống hoặc trong history; chưa có thì phải gọi tool read trước.
- Chỉ truyền tham số người dùng thực sự nói tới hoặc bắt buộc, không truyền tham số tùy chọn thừa, không dùng "" hay null. Tham số phải đúng tên, đúng kiểu, đúng enum trong catalog.
- Tính ngày giờ đúng theo current_time, định dạng ISO 8601 đầy đủ, múi giờ +07:00.
- Không gọi tool thừa. Mỗi tình huống dùng dữ liệu, tên người và địa danh khác nhau.
"""


def codex(prompt, effort):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.txt"
        cmd = [CODEX, "exec", "--model", MODEL, "-c", f'model_reasoning_effort="{effort}"', "--sandbox", "read-only",
               "--skip-git-repo-check", "--ephemeral", "-C", tmp, "-o", str(out), "-"]
        subprocess.run(cmd, input=prompt, text=True, encoding="utf-8", check=True, stdout=subprocess.DEVNULL, timeout=1800)
        text = out.read_text(encoding="utf-8")
    return json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["apps", "scenarios"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--apps", default=None)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--per-batch", type=int, default=12)
    ap.add_argument("--batches", type=int, default=1)
    ap.add_argument("--apps-per-batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--effort", default="high")
    args = ap.parse_args()

    if args.step == "apps":
        apps = codex(APPS_PROMPT.format(n=args.n), args.effort)["apps"]
        Path(args.out).write_text(json.dumps({"apps": apps}, ensure_ascii=False, indent=1), encoding="utf-8")
        print("apps", len(apps), [a["id"] for a in apps])
        return

    apps = json.loads(Path(args.apps).read_text(encoding="utf-8"))["apps"]
    rng = random.Random(args.seed)
    with open(args.out, "a", encoding="utf-8") as f:
        for b in range(args.batches):
            chosen = rng.sample(apps, min(args.apps_per_batch, len(apps)))
            cat = json.dumps(chosen, ensure_ascii=False)
            try:
                rows = codex(SCEN_PROMPT.format(catalog=cat, n=args.per_batch), args.effort)["scenarios"]
            except Exception as e:  # one bad reply or a timeout must not lose the other batches
                print(f"batch {b + 1} failed: {e!r}"[:300], flush=True)
                continue
            for r in rows:
                r["apps"] = [a["id"] for a in chosen]
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            print(f"batch {b + 1}/{args.batches}: {len(rows)} scenarios on {[a['id'] for a in chosen]}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
