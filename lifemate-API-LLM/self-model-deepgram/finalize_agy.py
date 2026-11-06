"""Verify complete direct-inference artifacts for AGY without replaying tool sessions."""
import collections
import hashlib
import json
import pickle
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2] / 'dataset/lifemate/train+public_test/public_test'
OUT = HERE / 'results/agy'
ASR_DIR = HERE / 'results/asr'

sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from zaic_tools import schema_validate


def main():
    rows = [json.loads(l) for l in (ROOT / 'public_test.jsonl').read_text().splitlines() if l.strip()]
    catalog = json.loads((ROOT / 'public_test_tools_catalog.json').read_text())
    expected = {r['audio'] for r in rows} | {h['audio'] for r in rows for h in r['history'] if h.get('audio')}
    asr = {}
    for audio in sorted(expected):
        record = json.loads((ASR_DIR / (Path(audio).stem + '.json')).read_text())
        assert record['audio'] == audio
        assert hashlib.sha256((ROOT / audio).read_bytes()).hexdigest() == record['audio_sha256'], audio
        asr[audio] = record
    submission_path = OUT / 'submission.jsonl'
    if not submission_path.exists():
        print('Error: submission.jsonl does not exist yet.')
        sys.exit(1)
    submission = [json.loads(l) for l in submission_path.read_text().splitlines() if l.strip()]
    assert [r['id'] for r in rows] == [r['id'] for r in submission], 'IDs missing, duplicate, or reordered'
    traces = collections.defaultdict(list)
    noops = []
    if (OUT / 'trace.jsonl').exists():
        for line in (OUT / 'trace.jsonl').read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            traces[rec['id']].append(rec)
            assert rec['result'].get('ok') is True, rec
            if rec['result'].get('done') is False:
                noops.append(rec)
    call_count = 0
    empty_count = 0
    ask_count = 0
    per_tool = collections.Counter()
    for source, prediction in zip(rows, submission):
        sample_id = source['id']
        assert set(prediction) == {'id', 'tool_calls', 'state_hash'}
        assert (OUT / 'done' / sample_id).exists(), sample_id
        session = pickle.loads((OUT / 'sessions' / (sample_id + '.pkl')).read_bytes())
        assert prediction['tool_calls'] == session.get_tool_calls(), sample_id
        assert prediction['state_hash'] == session.get_state_hash(), sample_id
        assert re.fullmatch('[0-9a-f]{64}', prediction['state_hash'])
        assert prediction['tool_calls'] == [{'name': t['name'], 'arguments': t['arguments']} for t in traces[sample_id]], sample_id
        allowed = {name for app in source['context']['installed_apps'].values() for name in app['tools']} | {'sys.ask_missing_info'}
        checks = schema_validate(prediction['tool_calls'], catalog, allowed)
        assert all(c['ok'] for c in checks), (sample_id, checks)
        call_count += len(prediction['tool_calls'])
        empty_count += not prediction['tool_calls']
        ask_count += any(c['name'] == 'sys.ask_missing_info' for c in prediction['tool_calls'])
        per_tool.update(c['name'] for c in prediction['tool_calls'])
    with (OUT / 'transcripts.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps({'id': row['id'], 'audio': row['audio'], 'text': asr[row['audio']]['transcript'], 'confidence': asr[row['audio']].get('confidence'), 'history': [{'audio': h['audio'], 'text': asr[h['audio']]['transcript'], 'confidence': asr[h['audio']].get('confidence')} for h in row['history'] if h.get('audio')]}, ensure_ascii=False) + '\n')
    with (OUT / 'asr_transcripts.jsonl').open('w') as f:
        for record in asr.values():
            f.write(json.dumps({k: v for k, v in record.items() if k != 'words'}, ensure_ascii=False) + '\n')
    summary = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'split': 'public_test',
        'samples': len(rows),
        'audio_files': len(asr),
        'asr_model': 'Deepgram nova-3; language=vi; smart_format=true; punctuate=true',
        'inference': 'Direct inference with AGY multi-subagent verification and execution',
        'reused_experiments': False,
        'schema_errors': 0,
        'trace_session_mismatches': 0,
        'tool_calls': call_count,
        'empty_samples': empty_count,
        'ask_samples': ask_count,
        'noop_results': len(noops),
        'empty_asr': sum(not x['transcript'].strip() for x in asr.values()),
        'submission_sha256': hashlib.sha256(submission_path.read_bytes()).hexdigest(),
        'leaderboard_submitted': False,
        'leaderboard_score': None,
        'per_tool': dict(per_tool)
    }
    (OUT / 'verification.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'noop_results.json').write_text(json.dumps(noops, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
