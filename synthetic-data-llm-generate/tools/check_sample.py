"""Validate text/state drafts; publish only independently reviewed, sealed samples."""
from __future__ import annotations
import argparse
import json
import sys
from pipeline import Pipeline, PipelineError, ROOT, jsonl, read_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--id')
    group.add_argument('--group')
    group.add_argument('--all', action='store_true')
    group.add_argument('--verified-path', action='store_true', help='print validated current release data path')
    ap.add_argument('--build', action='store_true')
    ap.add_argument('--include-holdout-apps', action='store_true', help='separate transfer experiment; never overwrite clean release pointer')
    ap.add_argument('-q', action='store_true')
    a = ap.parse_args()
    p = Pipeline()
    if a.build and not a.all:
        raise PipelineError('--build cần --all')
    if a.verified_path:
        import hashlib
        manifest = read_json(p.out / ('verified_with_holdout.json' if a.include_holdout_apps else 'verified.json'))
        from pathlib import Path
        if hashlib.sha256(Path(manifest['data_path']).read_bytes()).hexdigest() != manifest['sha256']:
            raise PipelineError('release checksum không khớp')
        print(manifest['data_path'])
        return
    # Build reads only sealed groups, so incomplete drafts cannot block publication.
    if a.build:
        manifest = p.build(include_holdout=a.include_holdout_apps)
        print(json.dumps({k: manifest[k] for k in ('n', 'data_path', 'distribution', 'skipped')}, ensure_ascii=False, indent=2))
        return
    rows = [p.sample(a.id)] if a.id else jsonl(p.group_path(a.group)) if a.group else p.rows()
    rejected_file = ROOT / 'rejected/rejected.jsonl'
    rejected = {r['id'] for r in jsonl(rejected_file)} if rejected_file.exists() else set()
    errors = warnings = ready = checked = 0
    seen = set()
    with p.lock():
        for row in rows:
            sid = row.get('id')
            if sid in rejected:
                continue
            checked += 1
            e, w = p.validate(row)
            if sid in seen:
                e.append('id trùng')
            seen.add(sid)
            errors += bool(e)
            warnings += bool(w)
            is_ready = not e and p.ready(row)
            ready += bool(is_ready)
            if e or not a.q:
                print(('ERR' if e else 'READY' if is_ready else 'DRAFT'), sid)
                for msg in e:
                    print('  ERR', msg)
                if not a.q:
                    for msg in w:
                        print('  WARN', msg)
    print(f'{checked} mẫu: {errors} ERR, {warnings} WARN, {ready} đã duyệt; không yêu cầu TTS/audio')
    sys.exit(1 if errors else 0)


if __name__ == '__main__':
    try:
        main()
    except (PipelineError, OSError, KeyError, TypeError, ValueError) as e:
        sys.exit(f'ERR: {e}')
