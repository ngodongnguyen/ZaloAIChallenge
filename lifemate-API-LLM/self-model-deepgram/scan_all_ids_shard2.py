import json
import sys
from pathlib import Path

ROOT = Path('/run/media/kietnehi/Windows/Users/ADMIN/Desktop/ZaloAIChallenge/dataset/lifemate/train+public_test')
sys.path.insert(0, str(ROOT / 'public_test'))
from zaic_tools import ToolEnv

with open('audit_manifest_v9.json') as f:
    manifest = json.load(f)

shard2 = [d for d in manifest if d['index'] >= 316]

ID_PARAM_MAP = {
    'recipe_id': 'recipes',
    'product_id': 'grocery', # or ecommerce
    'booking_id': 'flights', # or train_bus_tickets
    'trip_id': 'train_bus_tickets',
    'lot_id': 'parking',
    'session_id': 'parking',
    'event_id': 'event_tickets',
    'place_id': 'maps',
    'alarm_id': 'alarm',
    'task_id': 'tasks',
    'note_id': 'notes',
    'item_id': 'pantry', # or shopping_list
    'bill_id': 'bill_payment',
    'account_id': 'banking',
    'beneficiary_id': 'banking',
    'order_id': 'ecommerce', # or food_delivery
    'ride_id': 'ride_hailing',
    'device_id': 'smart_home',
    'chat_id': 'chat_ott',
    'contact_id': 'contacts',
    'habit_id': 'habit_tracker'
}

invalid_id_calls = []

for d in shard2:
    sid = d['id']
    v9 = d.get('v9_calls', [])
    if not v9:
        continue
        
    env = ToolEnv('public_test', sid, root=ROOT)
    for c_idx, c in enumerate(v9):
        cname = c['name']
        args = c.get('arguments', {})
        res = env.call(cname, args)
        
        # Check if call failed or reported not found
        msg = str(res.get('message', '') or '')
        sres = str(res)
        if not res.get('ok', True) or res.get('done') is False:
            invalid_id_calls.append((sid, d['index'], d['transcript'], cname, args, res))
        elif any(w in msg.lower() for w in ['không tìm thấy', 'không tồn tại', 'invalid', 'lỗi']):
            invalid_id_calls.append((sid, d['index'], d['transcript'], cname, args, res))

print(f"Total calls with execution failure or not found in shard2: {len(invalid_id_calls)}")
for item in invalid_id_calls:
    print("="*60)
    print(f"[{item[0]}] (idx={item[1]}): {item[2]}")
    print(f"  Call: {item[3]} args: {item[4]}")
    print(f"  Result: {item[5]}")

