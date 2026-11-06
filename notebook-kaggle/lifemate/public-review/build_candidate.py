"""Execute a frozen reviewed plan once per sample, retaining every tool call.

This local research artifact uses public-test predictions, never training labels.
No environment state files are read by this script. Output must be a new directory.
"""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def strings(value):
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        return set().union(*(strings(v) for v in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(strings(v) for v in value)) if value else set()
    return set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--patches', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sessions', type=Path,
                        help='Trusted local ToolEnv checkpoints from this same review run; resume without resetting')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; refusing to overwrite a previous run')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.data_root / 'public_test'))
    from zaic_tools import ToolEnv, schema_validate

    rows = read_jsonl(args.base)
    items = {r['id']: r for r in read_jsonl(args.data_root / 'public_test/public_test.jsonl')}
    catalog = json.loads((args.data_root / 'public_test/public_test_tools_catalog.json').read_text())
    schemas = {tool['name']: tool for app in catalog['apps'] for tool in app['tools']}
    patches = json.loads(args.patches.read_text())
    assert len(rows) == len(items) == len({r['id'] for r in rows})
    assert {r['id'] for r in rows} == set(items)
    assert set(patches) <= set(items)
    seeded = {}
    if args.sessions:
        for path in sorted(args.sessions.glob('*.pkl')):
            sid = path.stem
            assert sid in items, ('unknown checkpoint ID', sid)
            observation = json.loads(path.with_suffix('.json').read_text())
            assert observation['id'] == sid
            env = pickle.loads(path.read_bytes())
            assert env.get_tool_calls() == observation['tool_calls'], (sid, 'checkpoint log mismatch')
            assert len(observation['tool_calls']) == len(observation['tool_results']), sid
            seeded[sid] = (env, observation)
    plans = []
    for row in rows:
        sid = row['id']
        calls = patches.get(sid, {}).get('calls', row['tool_calls'])
        allowed = {name for app in items[sid]['context']['installed_apps'].values()
                   for name in app['tools']} | {'sys.ask_missing_info'}
        checks = schema_validate(calls, catalog, allowed)
        assert all(check.get('ok') for check in checks), (sid, checks)
        assert all(c['name'] != 'sys.ask_missing_info' for c in calls[:-1]), sid
        if sid in seeded:
            prefix = seeded[sid][1]['tool_calls']
            assert calls[:len(prefix)] == prefix, (sid, 'cannot remove or change executed calls')
        plans.append((row, calls))

    # All decisions are frozen before execution; no probing or retrying plans.
    args.output.mkdir(parents=True)
    (args.output / 'patches.json').write_text(json.dumps(patches, ensure_ascii=False, indent=2) + '\n')
    changes, failures, output = [], [], []
    with (args.output / 'trace.jsonl').open('w') as trace:
        for row, calls in plans:
            sid = row['id']
            if sid in seeded:
                env, observation = seeded[sid]
                results = list(observation['tool_results'])
                prefix_size = len(observation['tool_calls'])
            else:
                env = ToolEnv('public_test', sid, root=args.data_root)
                results = []
                prefix_size = 0
            known = strings(items[sid]['history'])
            for result in results:
                known.update(strings(result))
            for call in calls[prefix_size:]:
                properties = schemas[call['name']]['parameters'].get('properties', {})
                for key, value in call['arguments'].items():
                    if properties.get(key, {}).get('x-ref'):
                        assert isinstance(value, str) and value in known, (sid, 'ungrounded reference', key, value)
                if call['name'] == 'food_delivery.place_order':
                    assert all(item['item_id'] in known for item in call['arguments']['items']), (sid, 'ungrounded menu item')
                check = env.validate(call['name'], call['arguments'])
                assert check.get('ok'), (sid, check)
                result = env.call(call['name'], call['arguments'])
                results.append(result)
                known.update(strings(result))
                if not result.get('ok') or result.get('done') is False:
                    failures.append({'id': sid, 'call': call, 'result': result})
            actual = env.get_tool_calls()
            assert actual == calls, (sid, 'execution log differs from frozen plan')
            result_row = {'id': sid, 'tool_calls': actual, 'state_hash': env.get_state_hash()}
            output.append(result_row)
            trace.write(json.dumps({'id': sid, 'tool_calls': actual,
                                    'tool_results': results, 'state_hash': result_row['state_hash']},
                                   ensure_ascii=False) + '\n')
            if result_row != row:
                changes.append({'id': sid, 'before': row, 'after': result_row,
                                'hash_changed': result_row['state_hash'] != row['state_hash'],
                                'reason': patches.get(sid, {}).get('reason', '')})
    submission = args.output / 'submission.jsonl'
    submission.write_text(''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n'
                                  for row in output))
    verification = {
        'rows': len(output), 'tool_calls': sum(len(r['tool_calls']) for r in output),
        'changed_rows': len(changes), 'hash_changes': sum(c['hash_changed'] for c in changes),
        'schema_errors': 0, 'execution_failures': failures,
        'environment_instances': len(output), 'all_executed_calls_retained': True,
        'resumed_environment_instances': len(seeded),
        'new_environment_instances': len(output) - len(seeded),
        'all_xref_and_food_menu_ids_grounded_before_use': True,
        'base_sha256': hashlib.sha256(args.base.read_bytes()).hexdigest(),
        'submission_sha256': hashlib.sha256(submission.read_bytes()).hexdigest(),
        'score': None, 'score_status': 'not submitted; correctness versus hidden labels unknown',
        'purpose': 'external-agent public-test research; not for model training or fine-tuning',
    }
    (args.output / 'verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2) + '\n')
    (args.output / 'changes.json').write_text(json.dumps(changes, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(verification, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit('candidate has execution failures; inspect complete trace')


if __name__ == '__main__':
    main()
