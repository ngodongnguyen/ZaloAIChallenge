#!/usr/bin/env python3
"""Generate comprehensive audit manifest for Codex v7 public test evaluation."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path('/run/media/kietnehi/Windows/Users/ADMIN/Desktop/ZaloAIChallenge')
ROOT = REPO_ROOT / 'dataset/lifemate/train+public_test'
sys.path.insert(0, str(ROOT / 'public_test'))
from zaic_tools import ToolEnv

PUBLIC_TEST_FILE = ROOT / 'public_test/public_test.jsonl'
V7_SUBMISSION = HERE / 'results/codex/submission.jsonl'
ASR_DIR = HERE / 'results/asr'
OUTPUT_FILE = HERE / 'audit_manifest_v7.json'

rows = [json.loads(l) for l in PUBLIC_TEST_FILE.read_text().splitlines() if l.strip()]
v7_rows = {json.loads(l)['id']: json.loads(l) for l in V7_SUBMISSION.read_text().splitlines() if l.strip()}

manifest = []
for idx, r in enumerate(rows):
    sid = r['id']
    asr_path = ASR_DIR / f'{sid}.json'
    asr = json.loads(asr_path.read_text())['transcript'] if asr_path.exists() else ''
    
    # Process history
    history = r.get('history', [])
    clean_history = []
    for h in history:
        entry = {
            'role': h.get('role'),
            'text': h.get('text', ''),
        }
        if h.get('tool_calls'):
            entry['tool_calls'] = h['tool_calls']
        if h.get('tool_results'):
            entry['tool_results'] = h['tool_results']
        clean_history.append(entry)

    v7_entry = v7_rows.get(sid, {'tool_calls': [], 'state_hash': ''})
    v7_calls = v7_entry['tool_calls']
    v7_hash = v7_entry['state_hash']
    
    # Replay in ToolEnv to get actual execution feedback
    env = ToolEnv('public_test', sid, root=ROOT)
    replayed_results = []
    for c in v7_calls:
        res = env.call(c['name'], c.get('arguments', {}))
        replayed_results.append({
            'name': c['name'],
            'arguments': c.get('arguments', {}),
            'ok': res.get('ok'),
            'done': res.get('done'),
            'message': res.get('message'),
            'keys': list(res.keys()),
            'sample_res': {k: v for k, v in res.items() if k not in ['state', 'profile', 'context']}
        })
    replayed_hash = env.get_state_hash()
    
    manifest.append({
        'index': idx,
        'id': sid,
        'transcript': asr,
        'current_time': r['context'].get('current_time'),
        'user_profile': r['context'].get('user_profile'),
        'installed_apps': list(r['context'].get('installed_apps', {}).keys()),
        'history': clean_history,
        'v7_calls': v7_calls,
        'v7_hash': v7_hash,
        'replayed_hash': replayed_hash,
        'hash_match': v7_hash == replayed_hash,
        'execution_results': replayed_results
    })

OUTPUT_FILE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
print(f"Generated {len(manifest)} manifest records in {OUTPUT_FILE}")
