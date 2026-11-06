import json

with open('audit_manifest_v9.json') as f:
    manifest = json.load(f)
by_id = {d['id']: d for d in manifest}

with open('shard2_audit_result.json') as f:
    s2 = json.load(f)

print(f"Checking {len(s2)} entries from shard2_audit_result.json against audit_manifest_v9.json:")
matched = 0
diffs = []
for sid, item in s2.items():
    if sid not in by_id:
        print(f"  {sid} not in manifest!")
        continue
    v9_calls = by_id[sid].get('v9_calls', [])
    audit_calls = item.get('calls', [])
    if v9_calls == audit_calls:
        matched += 1
    else:
        diffs.append((sid, v9_calls, audit_calls, item.get('reason')))

print(f"Exact match with v9_calls: {matched}/{len(s2)}")
print(f"Diffs with v9_calls: {len(diffs)}")
for sid, v9_calls, audit_calls, reason in diffs:
    print("="*60)
    print(f"[{sid}] Reason: {reason}")
    print(f"  v9   : {json.dumps(v9_calls, ensure_ascii=False)}")
    print(f"  audit: {json.dumps(audit_calls, ensure_ascii=False)}")

