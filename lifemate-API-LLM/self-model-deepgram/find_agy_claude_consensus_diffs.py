import json
from pathlib import Path

with open('audit_manifest_v9.json') as f:
    manifest = json.load(f)

with open('results/agy/submission.jsonl') as f:
    agy_lines = {json.loads(l)['id']: json.loads(l) for l in f if l.strip()}

shard2 = [d for d in manifest if d['index'] >= 316]
claude_dir = Path('results/claude/logs')

consensus_hits = []

for d in shard2:
    sid = d['id']
    v9 = d.get('v9_calls', [])
    agy = agy_lines.get(sid, {}).get('tool_calls', [])
    
    cl_path = claude_dir / f"{sid}.json"
    if not cl_path.exists():
        continue
    cl_data = json.loads(cl_path.read_text())
    cl_calls = cl_data.get('calls', [])
    cl_note = cl_data.get('note', '')

    # Check if agy and claude are very close or identical, but v9 differs
    if agy == cl_calls and v9 != agy:
        consensus_hits.append((sid, d['index'], d['transcript'], v9, agy, cl_note))

print(f"Total samples where AGY and Claude are IDENTICAL, but Codex v9 differs: {len(consensus_hits)}")
for item in consensus_hits:
    print("="*70)
    print(f"[{item[0]}] (idx={item[1]}): {item[2]}")
    print(f"NOTE: {item[5]}")
    print(f"v9 : {json.dumps(item[3], ensure_ascii=False)}")
    print(f"AGY/Claude: {json.dumps(item[4], ensure_ascii=False)}")

