import json
import re
import sys
from pathlib import Path

ROOT = Path('/run/media/kietnehi/Windows/Users/ADMIN/Desktop/ZaloAIChallenge/dataset/lifemate/train+public_test')
sys.path.insert(0, str(ROOT / 'public_test'))
from zaic_tools import ToolEnv

with open('audit_manifest_v9.json') as f:
    manifest = json.load(f)

with open('results/agy/submission.jsonl') as f:
    agy_lines = {json.loads(l)['id']: json.loads(l) for l in f if l.strip()}

shard2 = [d for d in manifest if d['index'] >= 316]

# Check 1: Empty calls
empty_samples = [d for d in shard2 if len(d.get('v9_calls', [])) == 0]
print(f"Empty v9_calls samples in shard2: {len(empty_samples)}")
for d in empty_samples:
    sid = d['id']
    agy_c = agy_lines.get(sid, {}).get('tool_calls', [])
    print(f"  {sid} (idx={d['index']}): '{d['transcript']}' -> agy has {len(agy_c)} calls: {[c['name'] for c in agy_c]}")

# Check 2: Hash unchanged samples
unchanged_samples = [d for d in shard2 if d.get('v9_hash') == d.get('init_hash') and len(d.get('v9_calls', [])) > 0]
print(f"\nUnchanged hash samples with calls in shard2: {len(unchanged_samples)}")
for d in unchanged_samples:
    sid = d['id']
    calls = [c['name'] for c in d.get('v9_calls', [])]
    # Check if there are any write calls
    write_calls = [c['name'] for c in d.get('v9_calls', []) if not any(c['name'].endswith(s) for s in ['.get', '.list', '.search', '.find', '.status', '.guide', '.read', '.get_current', '.get_forecast', '.get_quote', '.get_price', '.find_routes', '.next_departures', '.get_balance', '.list_transactions', '.list_beneficiaries', '.get_portfolio', '.order_status', '.order_history', '.my_bookings', '.get_status', '.get_ride', '.get_quotes', '.translate', '.ask_missing_info'])]
    if write_calls:
        print(f"  WARNING: {sid} has write calls {write_calls} but hash unchanged! Calls: {calls}")

