import json
import sys
from pathlib import Path

ROOT = Path('/run/media/kietnehi/Windows/Users/ADMIN/Desktop/ZaloAIChallenge/dataset/lifemate/train+public_test')
sys.path.insert(0, str(ROOT / 'public_test'))
from zaic_tools import ToolEnv

with open('audit_manifest_v9.json') as f:
    manifest = json.load(f)

shard2 = [d for d in manifest if d['index'] >= 316]
claude_dir = Path('results/claude/logs')

hash_diffs = []
for d in shard2:
    sid = d['id']
    v9 = d.get('v9_calls', [])
    cl_path = claude_dir / f"{sid}.json"
    if not cl_path.exists():
        continue
    cl_data = json.loads(cl_path.read_text())
    cl_calls = cl_data.get('calls', [])
    
    # Run v9 in env
    env_v9 = ToolEnv('public_test', sid, root=ROOT)
    for c in v9:
        env_v9.call(c['name'], c.get('arguments', {}))
    h_v9 = env_v9.get_state_hash()
    
    # Run claude in env
    env_cl = ToolEnv('public_test', sid, root=ROOT)
    for c in cl_calls:
        env_cl.call(c['name'], c.get('arguments', {}))
    h_cl = env_cl.get_state_hash()
    
    if h_v9 != h_cl:
        hash_diffs.append({
            'id': sid,
            'index': d['index'],
            'transcript': d['transcript'],
            'h_v9': h_v9,
            'h_cl': h_cl,
            'v9': v9,
            'cl': cl_calls,
            'note': cl_data.get('note', '')
        })

print(f"Total state hash diffs between Codex v9 and Claude in Shard 2: {len(hash_diffs)}")
for idx, item in enumerate(hash_diffs):
    print("="*70)
    print(f"[{idx+1}/{len(hash_diffs)}] {item['id']} (idx={item['index']})")
    print(f"TRANSCRIPT: {item['transcript']}")
    print(f"CLAUDE NOTE: {item['note']}")
    print(f"v9    : {json.dumps(item['v9'], ensure_ascii=False)}")
    print(f"claude: {json.dumps(item['cl'], ensure_ascii=False)}")

