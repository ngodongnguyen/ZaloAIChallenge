"""Kaggle GPU budget ledger. Counts wall-clock seconds of each kernel run since the weekly reset.

The user allows using the full weekly quota of one account (30 GPU hours). The default stop line
keeps 1 hour of margin. Edit BUDGET_LIMIT_S if the quota changes.
"""
import json
import time
from pathlib import Path

BUDGET_LIMIT_S = 29 * 3600


class Budget:
    def __init__(self, path):
        self.path = Path(path)
        self.entries = []
        if self.path.exists():
            with open(self.path, encoding="utf-8-sig") as f:
                self.entries = json.load(f)

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=2)
            f.write("\n")

    def used_last_week(self, now=None):
        """Seconds used since the last quota reset (Saturday 00:00 UTC, as reported by the user; not documented)."""
        now = now or time.time()
        days_since_saturday = (time.gmtime(now).tm_wday - 5) % 7
        reset = now - now % 86400 - days_since_saturday * 86400
        return sum(e["seconds"] for e in self.entries if e["end"] >= reset)

    def can_run(self, estimate_s, now=None):
        return self.used_last_week(now) + estimate_s <= BUDGET_LIMIT_S

    def record(self, run_id, seconds, start, end):
        self.entries.append({"run_id": run_id, "seconds": seconds, "start": start, "end": end})
        self._save()
