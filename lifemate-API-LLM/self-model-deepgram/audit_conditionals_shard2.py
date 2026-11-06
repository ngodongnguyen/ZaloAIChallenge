import json
import re
import sys
from pathlib import Path

ROOT = Path('/run/media/kietnehi/Windows/Users/ADMIN/Desktop/ZaloAIChallenge/dataset/lifemate/train+public_test')
sys.path.insert(0, str(ROOT / 'public_test'))
from zaic_tools import ToolEnv

with open('audit_manifest_v9.json') as f:
    manifest = json.load(f)

shard2 = [d for d in manifest if d['index'] >= 316]

cond_patterns = [
    r'\bnếu\b', r'\btrường hợp\b', r'\bmưa\b', r'\bkhông mưa\b',
    r'\bcao hơn\b', r'\bthấp hơn\b', r'\blớn hơn\b', r'\bnhỏ hơn\b',
    r'\bđủ\b', r'\bkhông đủ\b', r'\btrên\b', r'\bdưới\b', r'\btăng\b', r'\bgiảm\b'
]

cond_samples = []
for d in shard2:
    t = d['transcript'].lower()
    if any(re.search(p, t) for p in cond_patterns):
        cond_samples.append(d)

print(f"Total conditional candidates in shard2: {len(cond_samples)}")

for d in cond_samples:
    sid = d['id']
    idx = d['index']
    t = d['transcript']
    v9 = d.get('v9_calls', [])
    now = d.get('current_time')
    
    # Let's filter for truly conditional statements ("nếu ... thì", "hoặc", "không thì", "mưa thì", v.v.)
    if any(k in t.lower() for k in ['nếu', 'không thì', 'mưa thì', 'nắng thì', 'hơn thì', 'kém thì', 'đủ thì', 'tăng thì', 'giảm thì', 'trên thì', 'dưới thì']):
        print("="*70)
        print(f"[{sid}] (idx={idx}) now={now}")
        print(f"TRANSCRIPT: {t}")
        print(f"V9 CALLS: {json.dumps(v9, ensure_ascii=False)}")
