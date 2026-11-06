"""File-backed official tool sessions for direct inference in the current chat.

No language model API, training predictions, hidden-state inspection, or replay.
Sessions are instantiated once, then serialized to resume the same evolving state.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2] / 'dataset/lifemate/train+public_test'
OUT = HERE / 'results'
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'public_test'))
from zaic_tools import ToolEnv

ROWS = [json.loads(l) for l in (ROOT / 'public_test/public_test.jsonl').read_text().splitlines() if l.strip()]
ITEMS = {r['id']: r for r in ROWS}


def sid(value):
    return f'public_{int(value):04d}' if str(value).isdigit() else value


def load_session(sample_id):
    path = OUT / 'sessions' / (sample_id + '.pkl')
    if path.exists():
        return pickle.loads(path.read_bytes())
    session = ToolEnv('public_test', sample_id, root=ROOT)
    save_session(sample_id, session)
    return session


def save_session(sample_id, session):
    path = OUT / 'sessions' / (sample_id + '.pkl')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix('.tmp').write_bytes(pickle.dumps(session))
    path.with_suffix('.tmp').replace(path)


def transcript(audio):
    path = OUT / 'asr' / (Path(audio).stem + '.json')
    return json.loads(path.read_text())['transcript'] if path.exists() else '[ASR PENDING]'


def show(args):
    selected = ROWS[args.start:args.start+args.count] if not args.ids else [ITEMS[sid(i)] for i in args.ids]
    for row in selected:
        ctx = row['context']
        history = []
        for h in row['history']:
            history.append({'user': transcript(h['audio'])} if h.get('audio') else {k: v for k, v in h.items() if k != 'role'})
        if args.full:
            value = {'id': row['id'], 'now': ctx['current_time'], 'profile': ctx['user_profile'], 'apps': list(ctx['installed_apps']), 'history': history, 'user': transcript(row['audio'])}
        else:
            history = [h.get('text', '') for h in history if 'text' in h] if args.brief else [h.get('user', h.get('text', '')) for h in history]
            value = [int(row['id'].split('_')[1]), ctx['current_time'], ctx['user_profile'], ','.join(ctx['installed_apps']), history, transcript(row['audio'])]
        print(json.dumps(value, ensure_ascii=False, separators=(',', ':')))


def act(args):
    actions = json.load(sys.stdin)
    for item in actions:
        sample_id = sid(item[0])
        assert sample_id in ITEMS, sample_id
        env = load_session(sample_id)
        for name, arguments in item[1:]:
            valid = env.validate(name, arguments)
            if not valid['ok']:
                save_session(sample_id, env)
                raise ValueError((sample_id, name, arguments, valid))
            result = env.call(name, arguments)
            save_session(sample_id, env)
            with (OUT / 'trace.jsonl').open('a') as f:
                f.write(json.dumps({'id': sample_id, 'name': name, 'arguments': arguments, 'result': result}, ensure_ascii=False) + '\n')
            if not args.quiet:
                print(json.dumps([sample_id, name, result], ensure_ascii=False, separators=(',', ':')))
        if len(item) == 1:
            save_session(sample_id, env)
        if args.done:
            mark(sample_id)
    if args.quiet:
        print('Processed', len(actions), 'samples;', sum(len(x)-1 for x in actions), 'calls.')


def mark(sample_id):
    path = OUT / 'done' / sample_id
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def finish(args):
    missing = [r['id'] for r in ROWS if not (OUT / 'done' / r['id']).exists()]
    if missing:
        print('Unfinished:', len(missing), ','.join(missing))
        return
    output = OUT / 'submission.jsonl'
    with output.open('w') as f:
        for row in ROWS:
            env = load_session(row['id'])
            f.write(json.dumps({'id': row['id'], 'tool_calls': env.get_tool_calls(), 'state_hash': env.get_state_hash()}, ensure_ascii=False) + '\n')
    print('Wrote', len(ROWS), 'samples to', output)


parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest='command', required=True)
p = sub.add_parser('show')
p.add_argument('start', type=int, nargs='?', default=0)
p.add_argument('count', type=int, nargs='?', default=20)
p.add_argument('--ids', nargs='+')
p.add_argument('--full', action='store_true')
p.add_argument('--brief', action='store_true')
p.set_defaults(run=show)
p = sub.add_parser('act')
p.add_argument('--done', action='store_true')
p.add_argument('--quiet', action='store_true')
p.set_defaults(run=act)
p = sub.add_parser('done')
p.add_argument('ids', nargs='+')
p.set_defaults(run=lambda a: [mark(sid(i)) for i in a.ids])
p = sub.add_parser('finish')
p.set_defaults(run=finish)
args = parser.parse_args()
args.run(args)
