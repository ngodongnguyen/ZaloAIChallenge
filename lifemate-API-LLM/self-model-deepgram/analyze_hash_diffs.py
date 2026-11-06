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
claude_dir = Path('results/claude/logs')

discrepancies = []

for d in shard2:
    sid = d['id']
    v9 = d.get('v9_calls', [])
    agy = agy_lines.get(sid, {}).get('tool_calls', [])
    
    cl_path = claude_dir / f"{sid}.json"
    cl_calls = []
    cl_note = ""
    if cl_path.exists():
        cl_data = json.loads(cl_path.read_text())
        cl_calls = cl_data.get('calls', [])
        cl_note = cl_data.get('note', '')

    # Run in env to get hashes
    env = ToolEnv('public_test', sid, root=ROOT)
    for c in v9:
        env.call(c['name'], c.get('arguments', {}))
    h_v9 = env.get_state_hash()
    
    env_agy = ToolEnv('public_test', sid, root=ROOT)
    for c in agy:
        env_agy.call(c['name'], c.get('arguments', {}))
    h_agy = env_agy.get_state_hash()

    env_cl = ToolEnv('public_test', sid, root=ROOT)
    for c in cl_calls:
        env_cl.call(c['name'], c.get('arguments', {}))
    h_cl = env_cl.get_state_hash()

    # If v9 hash != agy hash, or v9 hash != cl hash
    if h_v9 != h_agy or h_v9 != h_cl or v9 != agy:
        discrepancies.append({
            'id': sid,
            'index': d['index'],
            'transcript': d['transcript'],
            'h_v9': h_v9,
            'h_agy': h_agy,
            'h_cl': h_cl,
            'v9': v9,
            'agy': agy,
            'cl': cl_calls,
            'note': cl_note
        })

print(f"Total discrepancies: {len(discrepancies)}")
# Find samples where AGY and Claude AGREE on hash, but Codex v9 has a DIFFERENT hash!
consensus_diff = [x for x in discrepancies if x['h_agy'] == x['h_cl'] and x['h_v9'] != x['h_agy']]
print(f"Samples where AGY and Claude AGREE on hash, but Codex v9 differs: {len(consensus_diff)}")
for x in consensus_diff:
    print("="*70)
    print(f"[{x['id']}] (idx={x['index']})")
    print(f"TRANSCRIPT: {x['transcript']}")
    print(f"NOTE: {x['note']}")
    print(f"v9    : {json.dumps(x['v9'], ensure_ascii=False)}")
    print(f"agy   : {json.dumps(x['agy'], ensure_ascii=False)}")
    print(f"claude: {json.dumps(x['cl'], ensure_ascii=False)}")

