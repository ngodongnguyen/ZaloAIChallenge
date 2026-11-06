import json
import re
from datetime import datetime
from pathlib import Path

with open('audit_manifest_v9.json') as f:
    data = json.load(f)

shard2 = [d for d in data if d['index'] >= 316]

date_keywords = ['tuần sau', 'tuần tới', 'thứ', 'chủ nhật', 'hôm nay', 'ngày mai', 'sáng mai', 'tối mai', 'chiều mai', 'mốt', 'hôm qua']

anomalies = []
for s in shard2:
    sid = s['id']
    idx = s['index']
    transcript = s.get('transcript', '')
    curr_time_str = s.get('current_time', '')
    calls = s.get('v9_calls', [])
    
    t_lower = transcript.lower()
    if any(k in t_lower for k in date_keywords):
        curr_dt = datetime.fromisoformat(curr_time_str)
        curr_weekday = curr_dt.weekday() # 0 = Monday, 6 = Sunday
        
        for c in calls:
            for k, v in c.get('arguments', {}).items():
                if isinstance(v, str) and re.match(r'^\d{4}-\d{2}-\d{2}', v):
                    call_dt = datetime.fromisoformat(v[:19])
                    call_weekday = call_dt.weekday()
                    
                    # Check "ngày mai"
                    if 'ngày mai' in t_lower or 'sáng mai' in t_lower or 'tối mai' in t_lower or 'chiều mai' in t_lower:
                        diff_days = (call_dt.date() - curr_dt.date()).days
                        # If diff_days != 1 and not also mentioning another date
                        if diff_days != 1 and not any(w in t_lower for w in ['thứ', 'tuần sau', 'tuần tới']):
                            anomalies.append((sid, idx, transcript, curr_time_str, c['name'], k, v, f"Expected +1 day but got +{diff_days} days"))
                    
                    # Check "hôm nay"
                    if 'hôm nay' in t_lower or 'tối nay' in t_lower or 'chiều nay' in t_lower or 'trưa nay' in t_lower:
                        diff_days = (call_dt.date() - curr_dt.date()).days
                        if diff_days != 0 and not any(w in t_lower for w in ['ngày mai', 'sáng mai', 'thứ', 'tuần sau']):
                            anomalies.append((sid, idx, transcript, curr_time_str, c['name'], k, v, f"Expected same day but got +{diff_days} days"))

print(f"Total date anomalies found in shard2: {len(anomalies)}")
for a in anomalies:
    print("="*60)
    print(f"[{a[0]}] (idx={a[1]}): {a[2]}")
    print(f"  now: {a[3]} | Call: {a[4]} {a[5]}={a[6]}")
    print(f"  Reason: {a[7]}")

