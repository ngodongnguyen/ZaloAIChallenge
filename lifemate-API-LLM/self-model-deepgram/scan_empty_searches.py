import json
import sys
from pathlib import Path

ROOT = Path('/run/media/kietnehi/Windows/Users/ADMIN/Desktop/ZaloAIChallenge/dataset/lifemate/train+public_test')
sys.path.insert(0, str(ROOT / 'public_test'))
from zaic_tools import ToolEnv

with open('audit_manifest_v9.json') as f:
    manifest = json.load(f)

shard2 = [d for d in manifest if d['index'] >= 316]

empty_searches = []
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
        
        # Check if it was a search/find/list call that returned an empty list
        for key in ['events', 'recipes', 'products', 'items', 'places', 'lots', 'flights', 'trips', 'contacts', 'orders', 'sessions']:
            if key in res and isinstance(res[key], list) and len(res[key]) == 0:
                empty_searches.append((sid, d['index'], d['transcript'], cname, args, key, c_idx, len(v9)))

print(f"Total search/list calls returning empty list in shard2: {len(empty_searches)}")
for item in empty_searches:
    print("="*60)
    print(f"[{item[0]}] (idx={item[1]}): {item[2]}")
    print(f"  Call {item[6]+1}/{item[7]}: {item[3]} args: {item[4]}")
    print(f"  Empty key: {item[5]}")

