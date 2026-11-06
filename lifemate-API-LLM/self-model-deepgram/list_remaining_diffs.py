import json

with open('audit_manifest_v9.json') as f:
    manifest = json.load(f)

with open('results/agy/submission.jsonl') as f:
    agy_lines = {json.loads(l)['id']: json.loads(l) for l in f if l.strip()}

shard2 = [d for d in manifest if d['index'] >= 316]

already_checked = {
    'public_0352', 'public_0356', 'public_0357', 'public_0358', 'public_0360',
    'public_0366', 'public_0380', 'public_0383', 'public_0396', 'public_0410',
    'public_0411', 'public_0413', 'public_0419', 'public_0420', 'public_0422',
    'public_0427', 'public_0442', 'public_0453', 'public_0475', 'public_0478',
    'public_0489', 'public_0510', 'public_0516', 'public_0522', 'public_0527',
    'public_0528', 'public_0549', 'public_0569', 'public_0592', 'public_0596',
    'public_0610', 'public_0638', 'public_0640', 'public_0649', 'public_0664',
    'public_0671', 'public_0678', 'public_0679', 'public_0680', 'public_0683',
    'public_0688', 'public_0699'
}

remaining_diffs = []
for d in shard2:
    sid = d['id']
    v9 = d.get('v9_calls', [])
    agy = agy_lines.get(sid, {}).get('tool_calls', [])
    if v9 != agy and sid not in already_checked:
        remaining_diffs.append((d, v9, agy))

print(f"Remaining diffs to inspect: {len(remaining_diffs)}")
for idx, (d, v9, agy) in enumerate(remaining_diffs):
    print("="*70)
    print(f"[{idx+1}/{len(remaining_diffs)}] {d['id']} (idx={d['index']})")
    print(f"TRANSCRIPT: {d['transcript']}")
    print(f"v9 : {json.dumps(v9, ensure_ascii=False)}")
    print(f"agy: {json.dumps(agy, ensure_ascii=False)}")

