"""Execute reviewed v2 plans once per sample, then merge with unchanged v1 rows.

Explicit public IDs only. Existing v1 artifacts and simulator sessions are never
modified. A v2 session is never reset or rerun with a different plan.
"""
import argparse
import hashlib
import json
import pickle
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2] / 'dataset/lifemate/train+public_test'
OUT = HERE/'results/codex_v2'
BASE = HERE/'results/codex/submission.jsonl'
DELIVER = HERE/'results/codex/v2'
sys.dont_write_bytecode = True
sys.path.insert(0,str(ROOT/'public_test'))
from zaic_tools import ToolEnv, schema_validate


def jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['read','apply','export'])
    parser.add_argument('--plan',type=Path)
    args = parser.parse_args()
    inputs = jsonl(ROOT/'public_test/public_test.jsonl')
    byid = {r['id']:r for r in inputs}
    cat = json.loads((ROOT/'public_test/public_test_tools_catalog.json').read_text())
    cases = OUT/'cases'
    cases.mkdir(parents=True,exist_ok=True)
    if args.command in ('read','apply'):
        plans = json.loads(args.plan.read_text())
        toolspecs = {t['name']:t for app in cat['apps'] for t in app['tools']}
        assert len(plans) == len({p['id'] for p in plans}), 'duplicate plan IDs'
        for plan in plans:
            sid = plan['id']
            assert isinstance(sid,str) and re.fullmatch(r'public_\d{4}',sid) and sid in byid, sid
            assert plan.get('reason') and isinstance(plan.get('tool_calls'),list)
            allowed = {n for a in byid[sid]['context']['installed_apps'].values() for n in a['tools']}|{'sys.ask_missing_info'}
            errors = [r for r in schema_validate(plan['tool_calls'],cat,allowed) if not r['ok']]
            assert not errors,(sid,errors)
            if args.command == 'read':
                assert all(toolspecs[c['name']]['kind']=='read' for c in plan['tool_calls'])
            # Entire file is preflighted before any simulator is opened.
            target = cases/(sid+'.json')
            if target.exists():
                prior = json.loads(target.read_text())
                calls = prior['prediction']['tool_calls']
                if prior.get('finalized'):
                    assert calls == plan['tool_calls'], f'{sid}: refusing a different completed v2 plan'
                else:
                    assert plan['tool_calls'][:len(calls)] == calls, f'{sid}: existing calls must be an exact prefix; never reset'
        for plan in plans:
            sid = plan['id']
            target = cases/(sid+'.json')
            if target.exists():
                record = json.loads(target.read_text())
                if record.get('finalized'):
                    print(sid,'already executed; unchanged')
                    continue
                env = pickle.loads(target.with_suffix('.pkl').read_bytes())
                trace = record['trace']
            else:
                env = ToolEnv('public_test',sid,root=ROOT)
                trace = []
                target.with_suffix('.pkl').write_bytes(pickle.dumps(env))
            for call in plan['tool_calls'][len(trace):]:
                result = env.call(call['name'],call['arguments'])
                trace.append({**call,'result':result})
                target.with_suffix('.pkl').write_bytes(pickle.dumps(env))
                if args.command == 'read':
                    print(json.dumps([sid,call['name'],result],ensure_ascii=False))
            record = {'plan':plan,'prediction':{'id':sid,'tool_calls':env.get_tool_calls(),'state_hash':env.get_state_hash()},'trace':trace,'finalized':args.command=='apply'}
            target.with_suffix('.tmp').write_text(json.dumps(record,ensure_ascii=False,indent=2))
            target.with_suffix('.tmp').replace(target)
            print(sid,'calls',len(trace),'noops',sum(t['result'].get('done') is False for t in trace))
    else:
        base_bytes = BASE.read_bytes()
        base = jsonl(BASE)
        assert [r['id'] for r in base] == [r['id'] for r in inputs]
        revised = {p.stem:json.loads(p.read_text()) for p in cases.glob('public_*.json')}
        assert all(p.get('finalized') for p in revised.values()),'Unfinished v2 sessions'
        predictions = []
        changes = []
        for source,old in zip(inputs,base):
            sid = source['id']
            new = revised[sid]['prediction'] if sid in revised else old
            allowed = {n for a in source['context']['installed_apps'].values() for n in a['tools']}|{'sys.ask_missing_info'}
            assert all(r['ok'] for r in schema_validate(new['tool_calls'],cat,allowed)),sid
            assert re.fullmatch('[0-9a-f]{64}',new['state_hash']),sid
            if sid in revised:
                rec = revised[sid]
                assert rec['plan']['tool_calls'] == new['tool_calls']
                assert all(t['result'].get('ok') is True for t in rec['trace']),sid
                if new != old:
                    changes.append({'id':sid,'reason':rec['plan']['reason'],'old':old,'new':new,'hash_changed':new['state_hash']!=old['state_hash']})
            predictions.append(new)
        assert len(predictions)==634
        data=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in predictions)
        DELIVER.mkdir(parents=True,exist_ok=True)
        (DELIVER/'submission.jsonl').write_text(data)
        (DELIVER/'changes.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2))
        summary={'samples':len(predictions),'changed_samples':len(changes),'changed_state_hashes':sum(r['hash_changed'] for r in changes),'schema_errors':0,'baseline_sha256':hashlib.sha256(base_bytes).hexdigest(),'submission_sha256':hashlib.sha256(data.encode()).hexdigest(),'leaderboard_score':None,'leaderboard_submitted':False}
        (DELIVER/'verification.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
