import json
import sys
from pathlib import Path

ROOT = Path('/run/media/kietnehi/Windows/Users/ADMIN/Desktop/ZaloAIChallenge/dataset/lifemate/train+public_test')
sys.path.insert(0, str(ROOT / 'public_test'))
from zaic_tools import ToolEnv

with open('audit_manifest_v9.json') as f:
    manifest = json.load(f)

shard2 = [d for d in manifest if d['index'] >= 316]

ID_KEYS = [
    'recipe_id', 'flight_id', 'booking_id', 'trip_id', 'task_id', 'event_id', 'note_id',
    'item_id', 'restaurant_id', 'contact_id', 'account_id', 'bill_id', 'lot_id', 'alarm_id',
    'product_id', 'order_id', 'beneficiary_id', 'place_id', 'device_id', 'session_id', 'card_id',
    'merchant_id', 'ride_id', 'habit_id', 'chat_id'
]

unprovenanced = []

for d in shard2:
    sid = d['id']
    idx = d['index']
    t = d['transcript']
    v9 = d.get('v9_calls', [])
    hist = d.get('history', [])
    
    # Extract all IDs from history
    known_ids = set()
    for h in hist:
        for tr in h.get('tool_results', []):
            def extract_ids(obj):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k in ID_KEYS or k.endswith('_id') or k == 'id':
                            if isinstance(v, str):
                                known_ids.add(v)
                        extract_ids(v)
                elif isinstance(obj, list):
                    for elem in obj:
                        extract_ids(elem)
            extract_ids(tr)
            
    # Also check user profile
    prof = d.get('user_profile', {})
    for k, v in prof.items():
        if isinstance(v, str):
            known_ids.add(v)
            
    # Now simulate turn calls
    turn_ids = set(known_ids)
    env = ToolEnv('public_test', sid, root=ROOT)
    for c in v9:
        cname = c['name']
        args = c.get('arguments', {})
        for k, v in args.items():
            if k in ID_KEYS and isinstance(v, str):
                if v not in turn_ids and v not in t:
                    # ID not in history, not in prompt, not in turn results yet!
                    unprovenanced.append((sid, idx, t, cname, k, v, list(known_ids)[:10]))
                    
        res = env.call(cname, args)
        def extract_res_ids(obj):
            if isinstance(obj, dict):
                for k, val in obj.items():
                    if k in ID_KEYS or k.endswith('_id') or k == 'id':
                        if isinstance(val, str):
                            turn_ids.add(val)
                    extract_res_ids(val)
            elif isinstance(obj, list):
                for elem in obj:
                    extract_res_ids(elem)
        extract_res_ids(res)

print(f"Total unprovenanced ID usages in shard2: {len(unprovenanced)}")
for item in unprovenanced:
    print("="*60)
    print(f"[{item[0]}] (idx={item[1]}): {item[2]}")
    print(f"  Call: {item[3]} uses {item[4]}='{item[5]}'")
    print(f"  Known IDs from history: {item[6]}")

