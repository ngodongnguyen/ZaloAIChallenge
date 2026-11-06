import json
import re

data = json.load(open('shard1_manifest.json'))

cond_words = [
    'nếu', 'trường hợp', 'thời tiết', 'mưa', 'nắng', 'nhiệt độ', 'độ', 'giá vàng',
    'xăng', 'số dư', 'tài khoản', 'tiền', 'trùng', 'bận', 'rảnh', 'lịch'
]

checked_samples = []

for s in data:
    sid = s['id']
    transcript = s.get('transcript', '')
    calls = s.get('v7_calls', [])
    results = s.get('execution_results', [])
    
    # Check if transcript contains conditional logic: e.g. "nếu ... thì ...", "mưa thì ... còn ...", etc.
    t_lower = transcript.lower()
    if 'nếu' in t_lower or ('thì' in t_lower and any(w in t_lower for w in ['mưa', 'nắng', 'giá', 'dư', 'lớn hơn', 'nhỏ hơn', 'cao hơn', 'thấp hơn', 'trên', 'dưới'])):
        checked_samples.append(s)

print(f"Total conditional samples detected: {len(checked_samples)}")

for s in checked_samples:
    sid = s['id']
    transcript = s.get('transcript', '')
    calls = s.get('v7_calls', [])
    results = s.get('execution_results', [])
    
    print(f"\n{'='*70}\n[{sid}] Transcript: {transcript}")
    call_summary = []
    for c, r in zip(calls, results):
        c_name = c['name']
        c_args = c.get('arguments', {})
        res = r.get('sample_res', {})
        # summarize result
        res_summary = ""
        if 'forecast' in res:
            res_summary = f"forecast: rain={res['forecast'][0].get('rain')}, cond={res['forecast'][0].get('condition')}, temp={res['forecast'][0].get('temp_max')}"
        elif 'current' in res:
            res_summary = f"current: temp={res['current'][0].get('temperature_c')}, cond={res['current'][0].get('condition')}"
        elif 'prices' in res:
            res_summary = f"price: {res['prices']}"
        elif 'balance' in res:
            res_summary = f"balance: {res['balance']}"
        elif 'accounts' in res:
            res_summary = f"accounts: {res['accounts']}"
        elif 'events' in res:
            res_summary = f"events: {[e.get('title') for e in res['events']]}"
        elif 'done' in res:
            res_summary = f"done={res.get('done')}"
        elif 'ok' in res:
            res_summary = f"ok={res.get('ok')}"
            
        call_summary.append(f"{c_name}({c_args}) -> {res_summary}")
    
    for cs in call_summary:
        print("  ", cs)

