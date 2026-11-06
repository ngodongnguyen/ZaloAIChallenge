#!/usr/bin/env python3
"""Deep scanner for potential anomalies and improvement opportunities across all 634 public test samples."""
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path('/run/media/kietnehi/Windows/Users/ADMIN/Desktop/ZaloAIChallenge')
ROOT = REPO_ROOT / 'dataset/lifemate/train+public_test'
sys.path.insert(0, str(ROOT / 'public_test'))
from zaic_tools import ToolEnv

V7_PATH = Path('/home/kietnehi/Desktop/API-LLM/self-model-deepgram/results/codex/v7/submission.jsonl')
MANIFEST_PATH = Path('/home/kietnehi/Desktop/API-LLM/self-model-deepgram/audit_manifest_v7.json')

v7_rows = {json.loads(l)['id']: json.loads(l) for l in V7_PATH.read_text().splitlines() if l.strip()}
manifest = {x['id']: x for x in json.load(MANIFEST_PATH.open())}

anomalies = []

# 1. Check for ID mismatches
ID_KEYS = ['recipe_id', 'flight_id', 'booking_id', 'trip_id', 'task_id', 'event_id', 'note_id', 
           'item_id', 'restaurant_id', 'contact_id', 'account_id', 'bill_id', 'lot_id', 'alarm_id']

for sid, m in manifest.items():
    v7_calls = v7_rows[sid]['tool_calls']
    
    # Collect all IDs in history
    history_ids = set()
    for h in m.get('history', []):
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
            
    # Check calls in v7
    for c in v7_calls:
        args = c.get('arguments', {})
        for k, v in args.items():
            if (k in ID_KEYS or k.endswith('_id')) and isinstance(v, str):
                # If there is history and this ID is used in a mutation or get, check if it exists in history
                if history_ids and v not in history_ids:
                    # check if it starts with standard prefixes like c_01, acc_01
                    anomalies.append({
                        'id': sid,
                        'type': 'id_not_in_history',
                        'call': c['name'],
                        'arg_key': k,
                        'arg_val': v,
                        'history_ids': list(history_ids)[:10],
                        'transcript': m['transcript'],
                        'calls': v7_calls
                    })

print(f"Total ID anomalies: {len(anomalies)}")
for a in anomalies:
    print(f"[{a['id']}] {a['call']} {a['arg_key']}={a['arg_val']} (history had {a['history_ids']})")
    print(f"   Transcript: {a['transcript']}")
