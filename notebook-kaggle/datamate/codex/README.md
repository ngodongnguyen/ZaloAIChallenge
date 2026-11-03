# DataMate — source audit

Local CPU calculations using the released DataMate public data. This directory
preserves the previous solver separately from `LLM-API-Agents/codex` and exports
the new file as `submission.jsonl`. It does not submit to the competition portal.

The previous audit reached a user-confirmed leaderboard score of 0.998. That
file is preserved as `submission_baseline0998.jsonl`. The current change recognizes
`tuần trc` as the previous full Monday–Sunday week. For the affected question,
563 valid orders from February 23 through March 1, 2026 have 57976800 VND revenue,
giving a rounded average of 102978 VND. The previous parser used only March 3.
A new score requires leaderboard evaluation.

Run from this directory with the dependencies in `requirements.txt`:

```bash
python -m unittest discover -p 'test_*_regressions.py' -v
python run.py
python verify_submission.py
```

`DATAMATE_DATA` may point to another local copy of the same public split. By default,
the scripts read `dataset/datamate/public_test` from the repository root. Data
files are read-only; parsed spreadsheet caches are stored under `/tmp`.

Generated answers, evidence, reports, logs, and submission files are local
artifacts excluded from Git. `evidence_improvements.json` records the exact
changes and source calculations; `report.json` records schema and source
reconciliation checks. Neither report measures hidden-label accuracy.
