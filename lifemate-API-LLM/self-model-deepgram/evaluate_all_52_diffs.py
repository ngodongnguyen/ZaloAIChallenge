import json
import sys
from pathlib import Path

ROOT = Path('/run/media/kietnehi/Windows/Users/ADMIN/Desktop/ZaloAIChallenge/dataset/lifemate/train+public_test')
sys.path.insert(0, str(ROOT / 'public_test'))
from zaic_tools import ToolEnv

with open('audit_manifest_v9.json') as f:
    manifest = json.load(f)

with open('results/agy/submission.jsonl') as f:
    agy_lines = {json.loads(l)['id']: json.loads(l) for l in f if l.strip()}

shard2 = [d for d in manifest if d['index'] >= 316]

diff_samples = []
for d in shard2:
    sid = d['id']
    v9 = d.get('v9_calls', [])
    agy = agy_lines.get(sid, {}).get('tool_calls', [])
    if v9 != agy:
        diff_samples.append((d, v9, agy))

print(f"Total diff samples: {len(diff_samples)}")
for idx, (d, v9, agy) in enumerate(diff_samples):
    sid = d['id']
    env_v9 = ToolEnv('public_test', sid, root=ROOT)
    for c in v9:
        env_v9.call(c['name'], c.get('arguments', {}))
    h_v9 = env_v9.get_state_hash()

    env_agy = ToolEnv('public_test', sid, root=ROOT)
    for c in agy:
        env_agy.call(c['name'], c.get('arguments', {}))
    h_agy = env_agy.get_state_hash()

    if h_v9 != h_agy:
        print(f"[{idx+1}] {sid} (idx={d['index']}) HASH DIFF! v9_hash={h_v9[:8]} agy_hash={h_agy[:8]}")
        print(f"    Transcript: {d['transcript']}")
        print(f"    v9 : {json.dumps(v9, ensure_ascii=False)}")
        print(f"    agy: {json.dumps(agy, ensure_ascii=False)}")

