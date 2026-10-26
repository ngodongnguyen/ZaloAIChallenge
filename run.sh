#!/usr/bin/env bash
# Container entrypoint: reads the mounted split at /data, writes result/submission.jsonl
set -euo pipefail
mkdir -p /app/result
python /app/src/infer.py --data-dir /data --out /app/result/submission.jsonl
# organisers collect result/submission.jsonl; also copy next to the mount if writable
mkdir -p /data/result 2>/dev/null && cp /app/result/submission.jsonl /data/result/submission.jsonl || true
