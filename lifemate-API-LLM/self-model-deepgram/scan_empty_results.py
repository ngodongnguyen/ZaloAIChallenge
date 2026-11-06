import json

data = json.load(open('shard1_manifest.json'))

empty_findings = []

for s in data:
    sid = s['id']
    transcript = s.get('transcript', '')
    calls = s.get('v7_calls', [])
    results = s.get('execution_results', [])
    
    for c_idx, (c, r) in enumerate(zip(calls, results)):
        c_name = c['name']
        c_args = c.get('arguments', {})
        res = r.get('sample_res', {})
        
        # Check if any list field is empty []
        if isinstance(res, dict):
            for k, v in res.items():
                if isinstance(v, list) and len(v) == 0 and k not in ['programs']:
                    # Special check: is this an empty search?
                    empty_findings.append({
                        'sid': sid,
                        'call_idx': c_idx,
                        'name': c_name,
                        'args': c_args,
                        'field': k,
                        'transcript': transcript,
                        'total_calls': len(calls)
                    })

print(f"Total calls with empty list results: {len(empty_findings)}")
for f in empty_findings:
    print(f"[{f['sid']}] Call {f['call_idx']}: {f['name']}({f['args']}) -> {f['field']}=[]")
    print(f"   Transcript: {f['transcript']}")
    print('---')

