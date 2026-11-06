"""Export structured audit manifest of all 634 public test samples for subagents."""
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2] / 'dataset/lifemate/train+public_test'
OUT = HERE / 'results'
sys_rows = [json.loads(l) for l in (ROOT / 'public_test/public_test.jsonl').read_text().splitlines() if l.strip()]

traces = [json.loads(l) for l in (OUT / 'trace.jsonl').read_text().splitlines() if l.strip()]
traces_by_id = defaultdict(list)
for t in traces:
    traces_by_id[t['id']].append(t)

manifest = []
for idx, r in enumerate(sys_rows):
    sid = r['id']
    asr_path = OUT / 'asr' / f'{sid}.json'
    asr = json.loads(asr_path.read_text())['transcript'] if asr_path.exists() else ''
    history_calls = []
    history_asst_text = []
    for h in r.get('history', []):
        if h.get('tool_calls'):
            history_calls.extend([c['name'] for c in h['tool_calls']])
        if h.get('text'):
            history_asst_text.append(h['text'])
    
    c_calls = traces_by_id.get(sid, [])
    call_summaries = []
    for c in c_calls:
        res = c['result']
        call_summaries.append({
            'name': c['name'],
            'arguments': c['arguments'],
            'ok': res.get('ok'),
            'done': res.get('done'),
            'keys': list(res.keys())
        })
    
    manifest.append({
        'index': idx,
        'id': sid,
        'transcript': asr,
        'now': r['context'].get('current_time'),
        'profile': r['context'].get('user_profile'),
        'apps': list(r['context'].get('installed_apps', {}).keys()),
        'history_calls': history_calls,
        'history_text': history_asst_text,
        'codex_calls': call_summaries
    })

out_path = HERE / 'audit_manifest.json'
out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
print(f'Wrote {len(manifest)} samples to {out_path}')
