import json

data = json.load(open('shard1_manifest.json'))

for s in data:
    sid = s['id']
    calls = s.get('v7_calls', [])
    ask_calls = [c for c in calls if c.get('name') == 'sys.ask_missing_info']
    if not ask_calls:
        continue
    
    print(f"\n{'='*70}\n[{sid}] Transcript: {s.get('transcript')}")
    print(f"User Profile: {s.get('user_profile')}")
    print(f"Calls: {json.dumps(calls, ensure_ascii=False)}")
    
    # Check results before the ask call
    for r in s.get('execution_results', []):
        name = r.get('name')
        args = r.get('arguments')
        res = r.get('sample_res')
        if name != 'sys.ask_missing_info':
            print(f"  Result of {name}({args}):")
            # print summarized result
            if isinstance(res, dict):
                for k, v in res.items():
                    if k != 'ok':
                        print(f"    {k}: {str(v)[:150]}")
            else:
                print(f"    {str(res)[:150]}")
    
    # Check history
    hist = s.get('history', [])
    if hist:
        print(f"  History ({len(hist)} items):")
        for h in hist:
            if h.get('tool_calls'):
                print(f"    Hist calls: {h.get('tool_calls')}")
            if h.get('tool_results'):
                print(f"    Hist results: {str(h.get('tool_results'))[:120]}")

