"""Test and verify tool calls on fresh ToolEnv sessions."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / 'dataset/lifemate/train+public_test'
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'public_test'))
from zaic_tools import ToolEnv, schema_validate

CATALOG = json.loads((ROOT / 'public_test/public_test_tools_catalog.json').read_text())
ROWS = [json.loads(l) for l in (ROOT / 'public_test/public_test.jsonl').read_text().splitlines() if l.strip()]
ITEMS = {r['id']: r for r in ROWS}


def test_trajectory(sample_id, calls):
    """Run candidate calls on a fresh ToolEnv, return ok, calls, state_hash, errors."""
    assert sample_id in ITEMS, sample_id
    item = ITEMS[sample_id]
    allowed = {t for a in item['context']['installed_apps'].values() for t in a['tools']} | {'sys.ask_missing_info'}
    
    # Schema check
    schema_res = schema_validate(calls, CATALOG, allowed)
    if not all(c['ok'] for c in schema_res):
        return {'ok': False, 'stage': 'schema', 'errors': schema_res}
    
    env = ToolEnv('public_test', sample_id, root=ROOT)
    results = []
    for c in calls:
        res = env.call(c['name'], c.get('arguments', {}))
        results.append(res)
        if not res.get('ok'):
            return {'ok': False, 'stage': 'execution_ok', 'call': c, 'result': res}
        if res.get('done') is False:
            return {'ok': False, 'stage': 'execution_done', 'call': c, 'result': res}
    
    return {
        'ok': True,
        'sample_id': sample_id,
        'tool_calls': env.get_tool_calls(),
        'state_hash': env.get_state_hash(),
        'results': results
    }


if __name__ == '__main__':
    # Simple CLI test
    if len(sys.argv) > 2:
        sid = sys.argv[1]
        calls = json.loads(sys.argv[2])
        out = test_trajectory(sid, calls)
        print(json.dumps(out, ensure_ascii=False, indent=2))
