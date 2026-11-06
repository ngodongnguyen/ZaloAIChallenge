"""Export a legacy row as an editable proposal. Never infer pre-write history state."""
import argparse
import copy
import json
import sys
from pipeline import Pipeline, PipelineError, atomic_json, read_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--id', required=True)
    ap.add_argument('--spec', required=True, help='human/agent-authored spec JSON with author/family/intent/goals')
    ap.add_argument('--initial', help='explicit state BEFORE history, mandatory if history has any writes')
    ap.add_argument('--output', required=True, help='editable proposal path outside parts/')
    a = ap.parse_args()
    p = Pipeline()
    row = p.sample(a.id)
    hist_calls = [c for t in row['history'] for c in t.get('tool_calls', [])]
    if a.initial:
        initial = read_json(a.initial)
    else:
        if any(p.tools.get(c['name'], {}).get('kind') == 'write' for c in hist_calls):
            raise PipelineError('history có write: phải cung cấp --initial trước history; không suy ngược tự động')
        initial = p.entry(a.id)
    initial = {k: v for k, v in initial.items() if k != '_meta'}
    history = copy.deepcopy(row['history'])
    for turn in history:
        turn.pop('tool_results', None)
    proposal = {k: row[k] for k in ('id', 'audio_text', 'context')}
    proposal.update(history=history, initial_state=initial, tool_calls=row['label']['tool_calls'], spec=read_json(a.spec))
    atomic_json(a.output, proposal)
    print(json.dumps({'proposal': a.output, 'status': 'editable; correct intent/calls/goals before stage --replace'}, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except (PipelineError, OSError, KeyError, TypeError, ValueError) as e:
        sys.exit(f'ERR: {e}')
