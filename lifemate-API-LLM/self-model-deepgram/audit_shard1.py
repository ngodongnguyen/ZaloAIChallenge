import json
import re
from pathlib import Path

data = json.load(open('shard1_manifest.json'))
print(f"Total samples: {len(data)}")

ID_KEYS = ['recipe_id', 'flight_id', 'booking_id', 'trip_id', 'task_id', 'event_id', 'note_id', 
           'item_id', 'restaurant_id', 'contact_id', 'account_id', 'bill_id', 'lot_id', 'alarm_id']

# Categories of findings
error_samples = []
ask_samples = []
id_mismatch_samples = []
conditional_samples = []
empty_calls_samples = []

for s in data:
    sid = s['id']
    calls = s.get('v7_calls', [])
    results = s.get('execution_results', [])
    transcript = s.get('transcript', '')
    history = s.get('history', [])
    profile = s.get('user_profile', {})
    
    # 0. Empty calls
    if not calls:
        empty_calls_samples.append((sid, transcript))
        
    # 1. Execution errors / done=False / error messages
    has_error = False
    error_details = []
    for r in results:
        res = r.get('result', {})
        if isinstance(res, dict):
            if res.get('done') is False:
                has_error = True
                error_details.append(f"done=False in {r.get('name')}: {res.get('message', '')}")
            if res.get('status') in ['failed', 'error']:
                has_error = True
                error_details.append(f"status={res.get('status')} in {r.get('name')}: {res.get('message', '')}")
            msg = str(res.get('message', '')).lower()
            if any(w in msg for w in ['không tìm thấy', 'thất bại', 'không tồn tại', 'lỗi', 'không hợp lệ']):
                has_error = True
                error_details.append(f"error msg in {r.get('name')}: {msg}")
        elif isinstance(res, str) and any(w in res.lower() for w in ['error', 'không tìm thấy', 'thất bại']):
            has_error = True
            error_details.append(f"error string in {r.get('name')}: {res}")
            
    if has_error:
        error_samples.append((sid, transcript, calls, error_details))
        
    # 2. sys.ask_missing_info
    ask_calls = [c for c in calls if c.get('name') == 'sys.ask_missing_info']
    if ask_calls:
        ask_samples.append((sid, transcript, ask_calls))
        
    # 3. ID mismatch with history
    history_ids = set()
    for h in history:
        for res in h.get('tool_results', []):
            def extract_ids(obj):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k in ID_KEYS or k.endswith('_id') or k == 'id':
                            if isinstance(v, str):
                                history_ids.add(v)
                        extract_ids(v)
                elif isinstance(obj, list):
                    for elem in obj:
                        extract_ids(elem)
            extract_ids(res)
            
    # Also extract IDs returned in current turn execution_results (earlier steps in multi-step)
    turn_ids = set(history_ids)
    for r in results:
        res = r.get('result', {})
        def extract_turn_ids(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k in ID_KEYS or k.endswith('_id') or k == 'id':
                        if isinstance(v, str):
                            turn_ids.add(v)
                    extract_turn_ids(v)
            elif isinstance(obj, list):
                for elem in obj:
                    extract_turn_ids(elem)
        extract_turn_ids(res)

    for c in calls:
        c_name = c['name']
        args = c.get('arguments', {})
        for k, v in args.items():
            if (k in ID_KEYS or k.endswith('_id')) and isinstance(v, str):
                # If it's a specific ID key and not in history or previous turn results
                # Note: some IDs might be newly created or standard, but let's check
                if history_ids and v not in turn_ids:
                    id_mismatch_samples.append((sid, transcript, c_name, k, v, list(history_ids)[:5]))
                    
    # 4. Conditional keywords in transcript
    cond_patterns = [r'\bnếu\b', r'\btrường hợp\b', r'\btrời mưa\b', r'\bmưa\b', r'\bkhông mưa\b', 
                     r'\bgiá vàng\b', r'\bvàng\b', r'\bsố dư\b', r'\bđủ tiền\b', r'\bkhông đủ\b',
                     r'\bhơn\b', r'\bdưới\b', r'\bcao hơn\b', r'\bthấp hơn\b', r'\blớn hơn\b', r'\bnhỏ hơn\b']
    if any(re.search(p, transcript.lower()) for p in cond_patterns):
        conditional_samples.append((sid, transcript, [c['name'] for c in calls]))

print(f"\n--- Summary of findings ---")
print(f"Empty calls: {len(empty_calls_samples)}")
print(f"Execution error/no-op samples: {len(error_samples)}")
print(f"sys.ask_missing_info samples: {len(ask_samples)}")
print(f"Potential ID mismatch samples: {len(id_mismatch_samples)}")
print(f"Potential conditional samples: {len(conditional_samples)}")

