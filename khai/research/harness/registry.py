"""Run registry: one directory per run, index.json and JOURNAL.md regenerated from run directories.

A run directory holds run.json (status, hashes, env), config.json, metrics.json, log.txt.
index.json and JOURNAL.md are never edited by hand.
"""
import json
import time
from pathlib import Path

RUN_STATUSES = {"queued", "running", "done", "failed", "stale"}
STALE_AFTER_S = 6 * 3600  # a run still "running" after this long is marked stale


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _read_json(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


class Registry:
    def __init__(self, root):
        self.root = Path(root)
        self.runs_dir = self.root / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def new_run(self, cfg, cfg_hash, code_h, run_id=None):
        """Create a run directory. Returns the run id. Refuses a duplicate of a done run."""
        existing = self.find_done(cfg_hash, code_h)
        if existing is not None:
            raise DuplicateRun(existing)
        run_id = run_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{cfg_hash[:8]}"
        run_dir = self.runs_dir / run_id
        if run_dir.exists():
            raise FileExistsError(run_dir)
        run_dir.mkdir(parents=True)
        _write_json(run_dir / "config.json", cfg)
        _write_json(run_dir / "run.json", {
            "run_id": run_id, "status": "queued", "config_hash": cfg_hash, "code_hash": code_h,
            "created": time.time(), "updated": time.time(), "stop_reason": None,
        })
        (run_dir / "log.txt").touch()
        return run_id

    def set_status(self, run_id, status, stop_reason=None, extra=None):
        if status not in RUN_STATUSES:
            raise ValueError(f"unknown status {status}")
        path = self.runs_dir / run_id / "run.json"
        info = _read_json(path)
        info.update({"status": status, "updated": time.time(), "stop_reason": stop_reason})
        if extra:
            info.update(extra)
        _write_json(path, info)

    def write_metrics(self, run_id, metrics):
        _write_json(self.runs_dir / run_id / "metrics.json", metrics)

    def mark_stale(self, now=None):
        """Mark runs that have been 'running' longer than STALE_AFTER_S. Returns the ids changed."""
        now = now or time.time()
        changed = []
        for info in self.all_runs():
            if info["status"] == "running" and now - info["updated"] > STALE_AFTER_S:
                self.set_status(info["run_id"], "stale", stop_reason="no update for too long")
                changed.append(info["run_id"])
        return changed

    def all_runs(self):
        out = []
        for d in sorted(self.runs_dir.iterdir()):
            p = d / "run.json"
            if p.exists():
                out.append(_read_json(p))
        return out

    def find_done(self, cfg_hash, code_h):
        for info in self.all_runs():
            if (info["config_hash"] == cfg_hash and info["code_hash"] == code_h
                    and info["status"] == "done"):
                return info["run_id"]
        return None

    def rebuild_index(self):
        """Regenerate index.json and JOURNAL.md from run directories."""
        runs = self.all_runs()
        index = []
        for info in runs:
            rid = info["run_id"]
            metrics = {}
            mp = self.runs_dir / rid / "metrics.json"
            if mp.exists():
                metrics = _read_json(mp)
            index.append({"run_id": rid, "status": info["status"], "config_hash": info["config_hash"],
                          "code_hash": info["code_hash"], "metrics": metrics,
                          "stop_reason": info.get("stop_reason")})
        _write_json(self.root / "index.json", index)
        lines = ["# JOURNAL (sinh tự động từ runs/, không sửa tay)", ""]
        for row in index:
            score = row["metrics"].get("score", "-")
            lines.append(f"- `{row['run_id']}` — {row['status']} — score={score} — "
                         f"config={row['config_hash']} code={row['code_hash']}"
                         + (f" — dừng: {row['stop_reason']}" if row["stop_reason"] else ""))
        (self.root / "JOURNAL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return index


class DuplicateRun(Exception):
    def __init__(self, run_id):
        super().__init__(f"same config and code already done as {run_id}")
        self.run_id = run_id
