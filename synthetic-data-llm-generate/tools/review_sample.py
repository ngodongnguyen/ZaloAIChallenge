"""Blind solver tool gateway and separate semantic review. Development data only."""
import argparse
import json
import sys
from pipeline import Pipeline, PipelineError, read_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['packet', 'start', 'call', 'finish', 'review-packet', 'approve', 'exclude', 'recheck'])
    ap.add_argument('--id', required=True)
    ap.add_argument('--actor', help='unique identity of this agent session; never impersonate another role')
    ap.add_argument('--call', help='JSON object {"name": ..., "arguments": ...}')
    ap.add_argument('--report', help='semantic review report JSON')
    ap.add_argument('--reason', help='coordinator reason for excluding a procedurally invalid solver attempt')
    a = ap.parse_args()
    p = Pipeline()
    if a.action == 'packet':
        result = p.packet(a.id)
    elif a.action == 'start':
        result = p.start_solver(a.id, a.actor)
    elif a.action == 'call':
        if not a.call:
            ap.error('call cần --call')
        result = p.solver_call(a.id, a.actor, json.loads(a.call))
    elif a.action == 'finish':
        result = p.finish_solver(a.id, a.actor)
    elif a.action == 'exclude':
        p.exclude_solver(a.id, a.actor, a.reason)
        result = {'id': a.id, 'actor': a.actor, 'excluded': True}
    elif a.action == 'recheck':
        result = p.recheck_solver(a.id, a.actor)
    elif a.action == 'review-packet':
        if not a.actor:
            ap.error('review-packet cần --actor')
        if a.actor == p.spec(a.id)['author']:
            raise PipelineError('reviewer phải khác author')
        result = {'fingerprint': p.fingerprint(p.sample(a.id)), 'sample': p.sample(a.id), 'spec': p.spec(a.id),
                  'solvers': [read_json(f) for f in (p.work(a.id) / 'solvers').glob('*.json')],
                  'initial_state': read_json(p.work(a.id) / 'initial_state.json'), 'entry_state': p.entry(a.id)}
    else:
        if not a.report:
            ap.error('approve cần --report')
        p.approve(a.id, a.actor, read_json(a.report))
        result = {'id': a.id, 'status': 'reviewed'}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (PipelineError, OSError, KeyError, TypeError, ValueError) as e:
        sys.exit(f'ERR: {e}')
