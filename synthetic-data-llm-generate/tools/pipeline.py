"""Local, text-only LifeMate authoring and review. Never used at inference."""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import re
import sys
import tempfile
import unicodedata
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "train"
HOLDOUT = {"fitness", "flights", "music", "tv"}
FAMILIES = {"missing_info", "filter", "time", "entity", "history", "multi", "read_only", "empty"}
ID_RE = re.compile(r"syn_\d{4,}")
GROUP_RE = re.compile(r"g[1-9]\d*")


class PipelineError(ValueError):
    pass


def norm(value):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFC", str(value)).lower()))


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise PipelineError(f"{path}: {e}") from e


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_json(path, value):
    atomic_bytes(path, json.dumps(value, ensure_ascii=False, indent=2).encode() + b"\n")


def jsonl(path):
    rows = []
    try:
        for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise PipelineError(f"{path}:{n}: cần JSON object")
                rows.append(row)
    except (OSError, ValueError) as e:
        raise PipelineError(f"{path}: {e}") from e
    return rows


def scalars(value):
    if isinstance(value, dict):
        return set().union(*(scalars(v) for v in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(scalars(v) for v in value)) if value else set()
    return {encoded(value)}


def canonical_arguments(value):
    """Official string normalization only; keep argument keys, numbers and list order."""
    if isinstance(value, str):
        return ' '.join(unicodedata.normalize('NFC', value).lower().split())
    if isinstance(value, dict):
        return {k: canonical_arguments(v) for k, v in value.items()}
    if isinstance(value, list):
        return [canonical_arguments(v) for v in value]
    return value


class Pipeline:
    REVIEW_CHECKS = ("intent", "grounding", "minimality", "ambiguity", "history", "goals")

    def __init__(self, out=None):
        self.out = Path(out or os.environ.get("SYN_OUT", DEFAULT_OUT)).resolve()
        self.catalog = read_json(self.out / "train_tools_catalog.json")
        self.tools = {t["name"]: t for a in self.catalog["apps"] for t in a["tools"]}
        runtime = self.out if (self.out / "zaic_tools").is_dir() else DEFAULT_OUT
        sys.path.insert(0, str(runtime))
        from zaic_tools.executor import Session
        from zaic_tools.utils import state_hash
        self.Session, self.state_hash = Session, state_hash
        # Hash the actually imported simulator, including transitive app code.
        import zaic_tools
        runtime_dir = Path(zaic_tools.__file__).parent
        self.runtime_hash = digest({str(p.relative_to(runtime_dir)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(runtime_dir.rglob("*.py"))})

    @contextmanager
    def lock(self):
        self.out.mkdir(parents=True, exist_ok=True)
        with (self.out / ".pipeline.lock").open("a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def checked_id(self, sid):
        if not isinstance(sid, str) or not ID_RE.fullmatch(sid):
            raise PipelineError("id phải dạng syn_NNNN")
        return sid

    def group_path(self, group):
        if not isinstance(group, str) or not GROUP_RE.fullmatch(group):
            raise PipelineError("group phải dạng gN")
        return self.out / "parts" / f"{group}.jsonl"

    def work(self, sid):
        return self.out / "work" / self.checked_id(sid)

    def rows(self):
        return [r for p in sorted((self.out / "parts").glob("g*.jsonl")) for r in jsonl(p)]

    def rejected_ids(self):
        path = self.out.parent / 'rejected/rejected.jsonl'
        return {r['id'] for r in jsonl(path)} if path.exists() else set()

    def sample(self, sid):
        self.checked_id(sid)
        matches = [r for r in self.rows() if r.get("id") == sid]
        if len(matches) != 1:
            raise PipelineError(f"{sid}: cần đúng một dòng, tìm thấy {len(matches)}")
        return matches[0]

    def entry(self, sid):
        return read_json(self.out / "train_environment" / self.checked_id(sid) / "state.json")

    def spec(self, sid):
        return read_json(self.work(sid) / "spec.json")

    def fingerprint(self, row):
        sid = row["id"]
        return digest({"row": row, "state": self.entry(sid), "initial": read_json(self.work(sid) / "initial_state.json"), "spec": self.spec(sid), "catalog": self.catalog, "runtime": self.runtime_hash})

    def _calls(self, session, calls, visible, text):
        if not isinstance(calls, list):
            raise PipelineError("tool_calls phải là list")
        results = []
        for i, call in enumerate(calls):
            if not isinstance(call, dict) or set(call) != {"name", "arguments"} or not isinstance(call["arguments"], dict):
                raise PipelineError("call cần name và arguments object")
            name, args = call["name"], call["arguments"]
            valid = session.validate(name, args)
            if not valid.get("ok"):
                raise PipelineError(f"{name}: {valid.get('error')}")
            if name == "sys.ask_missing_info":
                if i != len(calls) - 1:
                    raise PipelineError("ask_missing_info phải kết thúc lượt")
                target, params = args["tool_name"], args["params"]
                if target not in self.tools or target == name or target not in session.allowed:
                    raise PipelineError(f"ask: tool_name không khả dụng {target}")
                props = self.tools[target]["parameters"].get("properties", {})
                if not params or len(set(params)) != len(params) or any(p not in props for p in params):
                    raise PipelineError(f"ask: params sai schema {params}")
            for key, value in args.items():
                prop = self.tools[name]["parameters"].get("properties", {}).get(key, {})
                if "x-ref" in prop and not scalars(value) <= visible and not re.search(r'(?<!\w)' + re.escape(str(value)) + r'(?!\w)', text):
                    raise PipelineError(f"{name}.{key}: x-ref chưa được tra cứu hay nói rõ")
                if key in {"origin", "destination", "pickup", "dropoff"} and norm(value) in {"nhà", "công ty"}:
                    raise PipelineError(f"{name}.{key}: cần địa chỉ đầy đủ trong profile")
            result = session.call(name, args)
            if result.get("ok") is not True or result.get("done") is False:
                raise PipelineError(f"{name}: call thất bại/noop: {result}")
            results.append(result)
            visible.update(scalars(result))
        return results

    def _history(self, session, history, sid):
        if not isinstance(history, list) or len(history) % 2:
            raise PipelineError("history cần các cặp user/assistant")
        visible, text = set(), ""
        for i, turn in enumerate(history):
            if not isinstance(turn, dict) or turn.get("role") != ("user" if i % 2 == 0 else "assistant"):
                raise PipelineError("history phải xen kẽ user/assistant")
            allowed = {'role', 'audio', 'audio_text'} if i % 2 == 0 else {'role', 'text', 'tool_calls', 'tool_results'}
            if not set(turn) <= allowed:
                raise PipelineError('history có trường thừa có thể làm lộ nhãn')
            if i % 2 == 0:
                if not isinstance(turn.get("audio_text"), str) or not turn["audio_text"].strip():
                    raise PipelineError("history user thiếu audio_text")
                turn["audio"] = f"audio/{sid}/{sid}_u{i // 2 + 1}.wav"
                text += " " + turn["audio_text"]
            else:
                if not isinstance(turn.get("text"), str) or not turn["text"].strip():
                    raise PipelineError("history assistant thiếu text")
                if turn.get("tool_calls"):
                    turn["tool_results"] = self._calls(session, turn["tool_calls"], visible, text)
                else:
                    turn.pop("tool_calls", None)
                    turn.pop("tool_results", None)
        return visible, text

    def _path(self, state, path):
        if not isinstance(path, list) or not path:
            raise PipelineError("goal.path cần list không rỗng")
        value = state
        for key in path:
            if isinstance(key, dict):
                found = [v for v in value if isinstance(v, dict) and all(v.get(k) == x for k, x in key.items())]
                if len(found) != 1:
                    raise PipelineError(f"goal.path chọn {len(found)} đối tượng")
                value = found[0]
            else:
                try:
                    value = value[key]
                except (KeyError, IndexError, TypeError) as e:
                    raise PipelineError(f"goal.path không tồn tại: {path}") from e
        return value

    def check_goals(self, goals, before, after, calls):
        if not isinstance(goals, list) or not goals:
            raise PipelineError("spec.goals cần điều kiện thành công độc lập")
        for goal in goals:
            op = goal.get("op")
            if op == "call_count":
                actual = sum(c["name"] == goal["name"] for c in calls) if "name" in goal else len(calls)
                good = actual == goal["value"]
            else:
                actual = self._path(after, goal["path"])
                if op == "eq":
                    good = actual == goal["value"]
                elif op == "unchanged":
                    good = actual == self._path(before, goal["path"])
                elif op == "delta":
                    good = actual - self._path(before, goal["path"]) == goal["value"]
                else:
                    raise PipelineError(f"goal.op không hỗ trợ: {op}")
            if not good:
                raise PipelineError(f"goal không đạt: {goal}")

    def validate(self, row):
        warnings = []
        try:
            sid = self.checked_id(row["id"])
            if set(row) != {"id", "audio", "audio_text", "context", "history", "label"}:
                raise PipelineError("mẫu cần đúng các khoá id/audio/audio_text/context/history/label")
            if not isinstance(row["audio_text"], str) or not row["audio_text"].strip():
                raise PipelineError("audio_text rỗng")
            if row["audio"] != f"audio/{sid}/{sid}.wav":
                raise PipelineError("audio sai đường dẫn")
            ctx, state, spec = row["context"], self.entry(sid), self.spec(sid)
            if set(ctx) != {'current_time', 'timezone', 'location', 'user_profile', 'installed_apps'}:
                raise PipelineError('context có trường thiếu/thừa')
            if set(ctx['location']) != {'lat', 'lng'} or set(ctx['user_profile']) != {'name', 'home_address', 'work_address', 'city'}:
                raise PipelineError('location/user_profile có trường thiếu/thừa')
            stamp = datetime.fromisoformat(ctx["current_time"])
            if stamp.utcoffset() is None or stamp.utcoffset().total_seconds() != 25200 or ctx["timezone"] != "Asia/Ho_Chi_Minh":
                raise PipelineError("thời gian cần offset +07:00 và timezone Asia/Ho_Chi_Minh")
            if not isinstance(ctx["location"], dict) or not (-90 <= ctx["location"]["lat"] <= 90 and -180 <= ctx["location"]["lng"] <= 180):
                raise PipelineError("location không hợp lệ")
            if not all(isinstance(ctx["user_profile"].get(k), str) and ctx["user_profile"][k].strip() for k in ("name", "home_address", "work_address", "city")):
                raise PipelineError("user_profile thiếu trường")
            if not isinstance(spec.get("author"), str) or not spec["author"].strip() or spec.get("family") not in FAMILIES or not str(spec.get("intent", "")).strip():
                raise PipelineError("spec cần author, family và intent")
            installed = ctx["installed_apps"]
            apps = {a["id"] for a in self.catalog["apps"]} - {"sys"}
            if not installed or not set(installed) <= apps or set(state) - {"_meta"} != set(installed):
                raise PipelineError("installed_apps khác state hoặc có app không hợp lệ")
            allowed = sorted(t for a in installed.values() for t in a["tools"])
            for app, info in installed.items():
                if set(info) != {'tools'} or not isinstance(info['tools'], list) or len(set(info['tools'])) != len(info['tools']):
                    raise PipelineError(f'installed_apps.{app} cần danh sách tools duy nhất')
                known = {t["name"] for t in self.tools.values() if t["app"] == app}
                if not set(info["tools"]) <= known:
                    raise PipelineError(f"tool không thuộc app {app}")
                if set(info["tools"]) != known:
                    warnings.append(f"app {app} chỉ cài một phần tool")
            meta = {"question_id": sid, "current_time": ctx["current_time"], "tools": allowed}
            if any(state.get("_meta", {}).get(k) != v for k, v in meta.items()):
                raise PipelineError("_meta khác context/installed_apps")
            initial = read_json(self.work(sid) / "initial_state.json")
            if initial.get("_meta") != state["_meta"] or set(initial) != set(state):
                raise PipelineError("initial_state khác apps/_meta của state.json")
            session = self.Session(sid, initial, self.catalog)
            history = copy.deepcopy(row["history"])
            visible, text = self._history(session, history, sid)
            if history != row["history"] or session.get_state() != {k: v for k, v in state.items() if k != "_meta"}:
                raise PipelineError("history replay khác tool_results hoặc state.json")
            if history and norm(history[-2]["audio_text"]) == norm(row["audio_text"]):
                raise PipelineError("lượt cuối lặp lại history")
            before = session.get_state()
            calls = row["label"]["tool_calls"]
            results = self._calls(session, calls, visible, text + " " + row["audio_text"])
            after = session.get_state()
            changed = {a: v for a, v in after.items() if v != before.get(a)}
            if row["label"] != {"tool_calls": calls, "tool_results": results, "final_state": changed}:
                raise PipelineError("label replay không khớp tool_results/final_state")
            self.check_goals(spec.get("goals"), before, after, calls)
            if not 6 <= len(installed) <= 14:
                warnings.append(f"cài {len(installed)} app; mục tiêu thông thường 6–14")
            if spec.get("pair"):
                pair = spec["pair"]
                if not isinstance(pair, dict) or any(not isinstance(pair.get(k), str) or not pair[k].strip() for k in ("id", "dimension", "variant")):
                    raise PipelineError("spec.pair cần id/dimension/variant")
            return [], warnings
        except (PipelineError, KeyError, TypeError, ValueError, AttributeError, IndexError) as e:
            return [str(e)], warnings

    def stage(self, group, proposal, replace=False):
        """Publish one complete draft atomically. Simulator owns all result/state fields."""
        path = self.group_path(group)
        sid = self.checked_id(proposal["id"])
        if int(sid[4:]) // 1000 != int(group[1:]):
            raise PipelineError("id nằm ngoài dải group")
        ctx, history = copy.deepcopy(proposal["context"]), copy.deepcopy(proposal.get("history", []))
        meta = {"question_id": sid, "current_time": ctx["current_time"], "tools": sorted(t for a in ctx["installed_apps"].values() for t in a["tools"])}
        initial = {**copy.deepcopy(proposal["initial_state"]), "_meta": meta}
        session = self.Session(sid, initial, self.catalog)
        visible, text = self._history(session, history, sid)
        before = session.get_state()
        calls = copy.deepcopy(proposal["tool_calls"])
        results = self._calls(session, calls, visible, text + " " + proposal["audio_text"])
        after = session.get_state()
        self.check_goals(proposal["spec"].get("goals"), before, after, calls)
        row = {"id": sid, "audio": f"audio/{sid}/{sid}.wav", "audio_text": proposal["audio_text"], "context": ctx, "history": history, "label": {"tool_calls": calls, "tool_results": results, "final_state": {a: v for a, v in after.items() if v != before.get(a)}}}
        with self.lock():
            claim_path = path.with_suffix('.claim.json')
            if claim_path.exists():
                claim = read_json(claim_path)
                if 'ID_FIRST' in claim and not int(claim['ID_FIRST'][4:]) <= int(sid[4:]) <= int(claim['ID_LAST'][4:]):
                    raise PipelineError('id nằm ngoài dải claim')
            parts = list((self.out / "parts").glob("g*.jsonl"))
            for part in parts:
                if any(r.get("id") == sid for r in jsonl(part)) and (part != path or not replace):
                    raise PipelineError(f"id trùng {sid}; sửa cùng group phải dùng --replace")
            rows = jsonl(path) if path.exists() else []
            # Validate in memory via temporary sidecar reads, then restore on failure.
            paths = [self.work(sid) / "initial_state.json", self.work(sid) / "spec.json", self.out / "train_environment" / sid / "state.json"]
            previous = [p.read_bytes() if p.exists() else None for p in paths]
            try:
                for p, value in zip(paths, [initial, proposal["spec"], {**before, "_meta": meta}]):
                    atomic_json(p, value)
                errors, _ = self.validate(row)
                if errors:
                    raise PipelineError("; ".join(errors))
            except Exception:
                for p, data in zip(paths, previous):
                    if data is None:
                        p.unlink(missing_ok=True)
                    else:
                        atomic_bytes(p, data)
                raise
            path.with_suffix(".complete.json").unlink(missing_ok=True)
            rows = [r for r in rows if r.get("id") != sid] + [row]
            atomic_bytes(path, b"".join(encoded(r) + b"\n" for r in rows))
        return row

    def seal(self, group):
        with self.lock():
            path = self.group_path(group)
            rows = [r for r in jsonl(path) if r.get('id') not in self.rejected_ids()]
            if not rows:
                raise PipelineError("group rỗng")
            claim_path = path.with_suffix('.claim.json')
            claim = read_json(claim_path) if claim_path.exists() else {}
            if 'N' in claim and len(rows) != claim['N']:
                raise PipelineError(f"số mẫu {len(rows)} khác N={claim['N']}")
            all_rows = self.rows()
            ids = [r.get("id") for r in all_rows]
            if len(ids) != len(set(ids)):
                raise PipelineError("id trùng giữa các group")
            other_texts = {norm(r.get('audio_text', '')) for r in all_rows if r.get('id') not in {x.get('id') for x in rows}}
            real_train = Path(os.environ.get('LM_TRAIN_DIR', ROOT.parent / 'dataset/lifemate/train+public_test/train')) / 'train.jsonl'
            real_texts = {norm(r.get('audio_text', '')) for r in jsonl(real_train)} if real_train.exists() else set()
            pairs, texts = {}, {}
            for row in rows:
                errors, _ = self.validate(row)
                if errors:
                    raise PipelineError(f"{row.get('id')}: {errors}")
                pair = self.spec(row["id"]).get("pair")
                if pair:
                    pairs.setdefault(pair["id"], []).append(row)
                key = norm(row["audio_text"])
                if key in other_texts or key in real_texts:
                    raise PipelineError('audio_text trùng nhóm khác hoặc train gốc')
                if key in texts:
                    prev = self.spec(texts[key]).get("pair")
                    if not pair or not prev or pair["id"] != prev["id"]:
                        raise PipelineError("câu trùng ngoài counterfactual pair")
                texts[key] = row["id"]
            for key, members in pairs.items():
                variants = [self.spec(r["id"])["pair"]["variant"] for r in members]
                if len(members) < 2 or len(set(variants)) != len(variants) or len({digest(r["label"]["tool_calls"]) for r in members}) < 2:
                    raise PipelineError(f"pair {key}: cần >=2 biến thể với hành vi khác nhau")
            if len(pairs) < claim.get('COUNTERFACTUAL_PAIRS', 0):
                raise PipelineError('chưa đủ counterfactual pairs theo claim')
            atomic_json(path.with_suffix(".complete.json"), {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "n": len(rows)})
            return {"group": group, "n": len(rows)}

    def packet(self, sid):
        row = self.sample(sid)
        allowed = {t for app in row["context"]["installed_apps"].values() for t in app["tools"]} | {"sys.ask_missing_info"}
        return {"sample": {k: row[k] for k in ("id", "audio_text", "context", "history")}, "tools": [t for name, t in self.tools.items() if name in allowed]}

    def _actor(self, actor):
        if not isinstance(actor, str) or not actor.strip() or len(actor) > 128:
            raise PipelineError("actor cần tên phiên agent riêng, 1–128 ký tự")

    def _solver_path(self, sid, actor):
        self._actor(actor)
        return self.work(sid) / "solvers" / f"{digest(actor)}.json"

    def start_solver(self, sid, actor):
        self._actor(actor)
        with self.lock():
            row = self.sample(sid)
            errors, _ = self.validate(row)
            if errors:
                raise PipelineError(f"mẫu chưa hợp lệ: {errors}")
            if actor == self.spec(sid)["author"]:
                raise PipelineError("solver phải khác author")
            path = self._solver_path(sid, actor)
            if path.exists():
                raise PipelineError("solver đã bắt đầu; dùng actor phiên mới nếu cần làm lại")
            atomic_json(path, {"actor": actor, "fingerprint": self.fingerprint(row), "calls": [], "finished": False})
        return self.packet(sid)

    def _attempt(self, sid, actor):
        attempt = read_json(self._solver_path(sid, actor))
        row = self.sample(sid)
        if attempt.get("fingerprint") != self.fingerprint(row):
            raise PipelineError("mẫu đã đổi; cần phiên solver mới")
        return attempt, row

    def solver_call(self, sid, actor, call):
        with self.lock():
            attempt, row = self._attempt(sid, actor)
            if attempt["finished"] or any(c["name"] == "sys.ask_missing_info" for c in attempt["calls"]):
                raise PipelineError("phiên đã kết thúc")
            s = self.Session(sid, self.entry(sid), self.catalog)
            visible = scalars([t.get("tool_results", []) for t in row["history"]])
            text = " ".join([t.get("audio_text", "") for t in row["history"]] + [row["audio_text"]])
            self._calls(s, attempt["calls"], visible, text)
            result = self._calls(s, [call], visible, text)[0]
            attempt["calls"].append(call)
            atomic_json(self._solver_path(sid, actor), attempt)
        return result

    def finish_solver(self, sid, actor):
        with self.lock():
            attempt, row = self._attempt(sid, actor)
            if attempt["finished"]:
                raise PipelineError("solver đã finish")
            attempt["finished"] = True
            attempt["agrees"] = self.agreement(row, attempt['calls'])
            attempt['agreement_policy'] = 'nfc-lower-whitespace+state-hash-v2'
            atomic_json(self._solver_path(sid, actor), attempt)
        return {"finished": True, "agrees": attempt["agrees"], "n_calls": len(attempt["calls"])}

    def agreement(self, row, calls):
        reference = row['label']['tool_calls']
        def canon_calls(items):
            return [{'name': c['name'], 'arguments': canonical_arguments(c['arguments'])} for c in items]
        if canon_calls(calls) != canon_calls(reference):
            return False
        # Normalized text must also produce the same canonical final state.
        hashes = []
        for trace in (reference, calls):
            session = self.Session(row['id'], self.entry(row['id']), self.catalog)
            for call in trace:
                result = session.call(call['name'], call['arguments'])
                if result.get('ok') is not True or result.get('done') is False:
                    return False
            hashes.append(session.get_state_hash())
        return hashes[0] == hashes[1]

    def recheck_solver(self, sid, actor):
        """Recompute stored trace validation after a checker fix; never sample a new answer."""
        with self.lock():
            attempt, row = self._attempt(sid, actor)
            if not attempt.get('finished') or attempt.get('excluded_reason'):
                raise PipelineError('chỉ recheck trace hoàn tất, không bị loại')
            attempt.setdefault('agreement_history', []).append({'agrees': attempt.get('agrees'), 'policy': attempt.get('agreement_policy', 'exact-v1')})
            attempt['agrees'] = self.agreement(row, attempt['calls'])
            attempt['agreement_policy'] = 'nfc-lower-whitespace+state-hash-v2'
            atomic_json(self._solver_path(sid, actor), attempt)
        return {'agrees': attempt['agrees'], 'n_calls': len(attempt['calls'])}

    def exclude_solver(self, sid, actor, reason):
        """Coordinator invalidates procedural evidence, retaining the original calls/verdict."""
        if not isinstance(reason, str) or len(reason.strip()) < 20:
            raise PipelineError('cần lý do cụ thể để loại bằng chứng solver')
        with self.lock():
            attempt = read_json(self._solver_path(sid, actor))
            attempt['excluded_reason'] = reason
            attempt['excluded_by'] = 'coordinator'
            atomic_json(self._solver_path(sid, actor), attempt)

    def approve(self, sid, actor, report):
        self._actor(actor)
        if not isinstance(report, dict):
            raise PipelineError('report cần JSON object')
        with self.lock():
            row, spec = self.sample(sid), self.spec(sid)
            fingerprint = self.fingerprint(row)
            if report.get('fingerprint') != fingerprint:
                raise PipelineError('report.fingerprint khác snapshot được duyệt; lấy review-packet mới')
            solvers = [read_json(p) for p in (self.work(sid) / "solvers").glob("*.json")]
            eligible = [s for s in solvers if not s.get('excluded_reason') and s.get("finished") and s.get("agrees") and s.get("fingerprint") == fingerprint and s.get("actor") != spec["author"] and self.agreement(row, s.get('calls', []))]
            if actor == spec["author"] or any(s.get("actor") == actor for s in solvers):
                raise PipelineError("reviewer phải khác author và solver")
            if not eligible:
                raise PipelineError("chưa có solver độc lập đồng ý với nhãn hiện tại")
            if report.get("verdict") != "approve" or not isinstance(report.get('checks'), dict) or set(report["checks"]) != set(self.REVIEW_CHECKS):
                raise PipelineError("report cần verdict approve và đủ sáu checks")
            for check in report["checks"].values():
                if not isinstance(check, dict) or check.get("passed") is not True or not isinstance(check.get("evidence"), str) or len(check["evidence"].strip()) < 20:
                    raise PipelineError("mỗi check cần passed:true và evidence cụ thể >=20 ký tự")
            errors, _ = self.validate(row)
            if errors:
                raise PipelineError(str(errors))
            solver = eligible[0]
            atomic_json(self.work(sid) / "review.json", {"fingerprint": fingerprint, "reviewer": actor, "solver": solver["actor"], "solver_digest": digest(solver), "report": report})

    def ready(self, row):
        try:
            sid, fingerprint = row["id"], self.fingerprint(row)
            review = read_json(self.work(sid) / "review.json")
            solver = read_json(self._solver_path(sid, review["solver"]))
            if review["fingerprint"] != fingerprint or solver["fingerprint"] != fingerprint or review["solver_digest"] != digest(solver):
                return False
            if len({self.spec(sid)["author"], review["reviewer"], review["solver"]}) != 3:
                return False
            checks = review["report"]["checks"]
            return (review["report"].get('fingerprint') == fingerprint and review["report"]["verdict"] == "approve" and set(checks) == set(self.REVIEW_CHECKS)
                    and all(isinstance(c, dict) and c.get("passed") is True and isinstance(c.get("evidence"), str) and len(c["evidence"].strip()) >= 20 for c in checks.values())
                    and not solver.get('excluded_reason') and solver["finished"] is True and solver["agrees"] is True and self.agreement(row, solver['calls']))
        except (PipelineError, KeyError, TypeError, ValueError, AttributeError):
            return False

    def build(self, include_holdout=False):
        with self.lock():
            approved, seen, skipped, pairs = [], set(), [], {}
            rejected = self.rejected_ids()
            for path in sorted((self.out / "parts").glob("g*.jsonl")):
                marker = path.with_suffix(".complete.json")
                if not marker.exists():
                    skipped.append({"group": path.stem, "reason": "unsealed"})
                    continue
                if read_json(marker).get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
                    raise PipelineError(f"{path.stem}: group đã thay đổi sau seal")
                for row in jsonl(path):
                    sid = row.get("id")
                    if sid in seen:
                        raise PipelineError(f"id trùng {sid}")
                    seen.add(sid)
                    if sid in rejected:
                        continue
                    errors, _ = self.validate(row)
                    if errors:
                        raise PipelineError(f"{sid}: {errors}")
                    pair = self.spec(sid).get('pair')
                    if pair:
                        pairs.setdefault((path.stem, pair['id']), set()).add(sid)
                    calls = row["label"]["tool_calls"] + [c for t in row["history"] for c in t.get("tool_calls", [])]
                    if not include_holdout and any((c['arguments']['tool_name'] if c['name'] == 'sys.ask_missing_info' else c['name']).split('.')[0] in HOLDOUT for c in calls):
                        skipped.append({"id": sid, "reason": "holdout_app"})
                    elif not self.ready(row):
                        skipped.append({"id": sid, "reason": "pending_or_stale_review"})
                    else:
                        approved.append(row)
            approved_ids = {r['id'] for r in approved}
            partial_pair_ids = set().union(*(ids & approved_ids for ids in pairs.values() if len(ids) < 2 or not ids <= approved_ids)) if pairs else set()
            for sid in sorted(partial_pair_ids):
                skipped.append({'id': sid, 'reason': 'incomplete_counterfactual_pair'})
            approved = [r for r in approved if r['id'] not in partial_pair_ids]
            if not approved:
                raise PipelineError("0 mẫu đủ bằng chứng; giữ nguyên release cũ. Hoàn tất solver/reviewer/seal trước.")
            approved.sort(key=lambda r: r["id"])
            data = b"".join(encoded(r) + b"\n" for r in approved)
            evidence = {r["id"]: read_json(self.work(r["id"]) / "review.json") for r in approved}
            content_id = digest({"data_sha256": hashlib.sha256(data).hexdigest(), "evidence": evidence, "include_holdout": include_holdout, 'skipped': skipped})
            release_dir = self.out / "releases" / content_id
            manifest = {"n": len(approved), "data_path": str(release_dir / "train.jsonl"), "sha256": hashlib.sha256(data).hexdigest(), "include_holdout_apps": include_holdout, "evidence": evidence, "skipped": skipped, "distribution": self.distribution(approved)}
            # Readers follow one atomic pointer only after both immutable files are complete.
            atomic_bytes(release_dir / "train.jsonl", data)
            atomic_json(release_dir / "manifest.json", manifest)
            atomic_json(self.out / ("verified_with_holdout.json" if include_holdout else "verified.json"), manifest)
            return manifest

    def distribution(self, rows):
        return {"n": len(rows), "calls": dict(sorted(Counter(len(r["label"]["tool_calls"]) for r in rows).items())), "history_pairs": dict(sorted(Counter(len(r["history"]) // 2 for r in rows).items())), "families": dict(Counter(self.spec(r["id"])["family"] for r in rows)), "ask": sum(any(c["name"] == "sys.ask_missing_info" for c in r["label"]["tool_calls"]) for r in rows)}
