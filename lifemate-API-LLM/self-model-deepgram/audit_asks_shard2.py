import json
import sys
from pathlib import Path

ROOT = Path('/run/media/kietnehi/Windows/Users/ADMIN/Desktop/ZaloAIChallenge/dataset/lifemate/train+public_test')
sys.path.insert(0, str(ROOT / 'public_test'))
from zaic_tools import ToolEnv

with open('audit_manifest_v9.json') as f:
    manifest = json.load(f)

shard2 = [d for d in manifest if d['index'] >= 316]

for d in shard2:
    sid = d['id']
    calls = d.get('v9_calls', [])
    asks = [c for c in calls if c['name'] == 'sys.ask_missing_info']
    if not asks:
        continue
        
    print("="*70)
    print(f"[{sid}] (idx={d['index']}) now={d.get('current_time')}")
    print(f"TRANSCRIPT: {d.get('transcript')}")
    print(f"APPS: {d.get('installed_apps')}")
    print(f"V9 CALLS: {json.dumps(calls, ensure_ascii=False)}")
    hist = d.get('history', [])
    for h_idx, h in enumerate(hist):
        print(f"  Hist {h_idx} asst: {h.get('text')}")
        if h.get('tool_calls'):
            print(f"    calls: {json.dumps(h.get('tool_calls'), ensure_ascii=False)}")
        if h.get('tool_results'):
            print(f"    results: {json.dumps(h.get('tool_results'), ensure_ascii=False)[:300]}")
