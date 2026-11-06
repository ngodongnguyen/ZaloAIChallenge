import json
from pathlib import Path

ROOT = Path('/run/media/kietnehi/Windows/Users/ADMIN/Desktop/ZaloAIChallenge/dataset/lifemate/train+public_test')
with open('audit_manifest_v9.json') as f:
    manifest = json.load(f)

shard2 = [d for d in manifest if d['index'] >= 316]

diffs = []
claude_dir = Path('results/claude/logs')

for d in shard2:
    sid = d['id']
    v9 = d.get('v9_calls', [])
    
    cl_path = claude_dir / f"{sid}.json"
    if not cl_path.exists():
        continue
    cl_data = json.loads(cl_path.read_text())
    cl_calls = cl_data.get('calls', [])
    cl_note = cl_data.get('note', '')
    
    if v9 != cl_calls:
        diffs.append({
            'id': sid,
            'index': d['index'],
            'transcript': d['transcript'],
            'v9': v9,
            'claude': cl_calls,
            'note': cl_note
        })

print(f"Total diffs between Codex v9 and Claude logs in Shard 2: {len(diffs)}")
for idx, item in enumerate(diffs):
    print("="*70)
    print(f"[{idx+1}/{len(diffs)}] {item['id']} (idx={item['index']})")
    print(f"TRANSCRIPT: {item['transcript']}")
    print(f"NOTE: {item['note']}")
    print(f"v9    : {json.dumps(item['v9'], ensure_ascii=False)}")
    print(f"claude: {json.dumps(item['claude'], ensure_ascii=False)}")

