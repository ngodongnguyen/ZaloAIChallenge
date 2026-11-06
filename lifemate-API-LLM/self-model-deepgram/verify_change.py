#!/usr/bin/env python3
"""Helper utility to verify candidate trajectory modifications for public test samples."""
import json
import sys
from pathlib import Path

REPO_ROOT = Path('/run/media/kietnehi/Windows/Users/ADMIN/Desktop/ZaloAIChallenge')
ROOT = REPO_ROOT / 'dataset/lifemate/train+public_test'
sys.path.insert(0, str(ROOT / 'public_test'))
from zaic_tools import ToolEnv, schema_validate

CATALOG = json.loads((ROOT / 'public_test/public_test_tools_catalog.json').read_text())

def test_sample(sample_id: str, candidate_calls: list):
    """Validate schema and replay candidate calls against ToolEnv."""
    errs = [e for e in schema_validate(candidate_calls, CATALOG) if not e.get('ok', True)]
    if errs:
        return {'ok': False, 'schema_errors': errs}
        
    env = ToolEnv('public_test', sample_id, root=ROOT)
    results = []
    for c in candidate_calls:
        res = env.call(c['name'], c.get('arguments', {}))
        results.append({
            'name': c['name'],
            'arguments': c.get('arguments', {}),
            'result': res
        })
    final_hash = env.get_state_hash()
    return {
        'ok': True,
        'sample_id': sample_id,
        'state_hash': final_hash,
        'results': results
    }

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 verify_change.py <sample_id> '<calls_json>'")
        sys.exit(1)
    sid = sys.argv[1]
    calls = json.loads(sys.argv[2])
    res = test_sample(sid, calls)
    print(json.dumps(res, ensure_ascii=False, indent=2))
