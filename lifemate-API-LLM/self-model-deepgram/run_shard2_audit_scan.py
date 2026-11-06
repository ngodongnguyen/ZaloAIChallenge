import json
import re
import sys
from pathlib import Path

ROOT = Path('/run/media/kietnehi/Windows/Users/ADMIN/Desktop/ZaloAIChallenge/dataset/lifemate/train+public_test')
sys.path.insert(0, str(ROOT / 'public_test'))
from zaic_tools import ToolEnv, schema_validate

CATALOG = json.loads((ROOT / 'public_test/public_test_tools_catalog.json').read_text())

with open('audit_manifest_v9.json') as f:
    manifest = json.load(f)

with open('results/agy/submission.jsonl') as f:
    agy_lines = {json.loads(l)['id']: json.loads(l) for l in f if l.strip()}

# Shard 2: index >= 316 (covers public_0351 to public_0699)
shard2 = [d for d in manifest if d['index'] >= 316]
print(f"Loaded {len(shard2)} samples for Shard 2 (indices {shard2[0]['index']} to {shard2[-1]['index']}, IDs {shard2[0]['id']} to {shard2[-1]['id']})")

findings = []

ID_FIELDS = [
    'recipe_id', 'flight_id', 'booking_id', 'trip_id', 'task_id', 'event_id', 'note_id',
    'item_id', 'restaurant_id', 'contact_id', 'account_id', 'bill_id', 'lot_id', 'alarm_id',
    'product_id', 'order_id', 'beneficiary_id', 'place_id', 'device_id', 'session_id', 'card_id'
]

COND_KEYWORDS = [
    r'\bnếu\b', r'\btrường hợp\b', r'\bmưa\b', r'\bkhông mưa\b', r'\btăng\b', r'\bgiảm\b',
    r'\bcao hơn\b', r'\bthấp hơn\b', r'\blớn hơn\b', r'\bnhỏ hơn\b', r'\btrên\b', r'\bdưới\b',
    r'\bđủ\b', r'\bkhông đủ\b', r'\bhết\b', r'\bcòn\b', r'\bthì\b', r'\bkhông thì\b'
]

for d in shard2:
    sid = d['id']
    idx = d['index']
    transcript = d['transcript']
    v9 = d.get('v9_calls', [])
    agy = agy_lines.get(sid, {}).get('tool_calls', [])
    history = d.get('history', [])
    now = d.get('current_time')
    profile = d.get('user_profile')
    apps = d.get('installed_apps', [])

    # Check 1: Conditions in transcript
    is_conditional = any(re.search(pat, transcript.lower()) for pat in COND_KEYWORDS)

    # Check 2: Hallucinated / unmatched IDs in v9
    # Gather all IDs seen in history
    hist_ids = set()
    for h in history:
        for tr in h.get('tool_results', []):
            def extract_ids(obj):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k in ID_FIELDS or k.endswith('_id') or k == 'id':
                            if isinstance(v, str):
                                hist_ids.add(v)
                        extract_ids(v)
                elif isinstance(obj, list):
                    for elem in obj:
                        extract_ids(elem)
            extract_ids(tr)

    # Check 3: Check execution results in v9
    v9_exec = d.get('execution_results', [])
    
    # Check 4: sys.ask_missing_info
    has_ask = any(c['name'] == 'sys.ask_missing_info' for c in v9)

    # Check 5: Discrepancy with AGY
    has_diff = (v9 != agy)

    issues = []
    if is_conditional:
        issues.append('conditional')
    if has_ask:
        issues.append('ask_missing_info')
    if has_diff:
        issues.append('diff_with_agy')

    # Check ID validity in ToolEnv
    # For each call in v9, test in fresh ToolEnv
    env = ToolEnv('public_test', sid, root=ROOT)
    for c in v9:
        cname = c['name']
        args = c.get('arguments', {})
        res = env.call(cname, args)
        if not res.get('ok', True) or res.get('done') is False:
            issues.append(f"v9_call_failed: {cname} -> {res}")

    if issues:
        findings.append({
            'id': sid,
            'index': idx,
            'transcript': transcript,
            'issues': issues,
            'v9': v9,
            'agy': agy
        })

print(f"Total samples with potential audit focus: {len(findings)}")
