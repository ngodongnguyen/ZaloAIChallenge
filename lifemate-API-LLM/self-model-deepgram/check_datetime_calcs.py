import json
import re
from datetime import datetime, timedelta

data = json.load(open('shard1_manifest.json'))

date_keywords = ['tuần sau', 'tuần tới', 'thứ', 'chủ nhật', 'hôm nay', 'ngày mai', 'sáng mai', 'tối mai', 'chiều mai']

for s in data:
    sid = s['id']
    transcript = s.get('transcript', '')
    curr_time_str = s.get('current_time', '')
    calls = s.get('v7_calls', [])
    
    t_lower = transcript.lower()
    if any(k in t_lower for k in date_keywords):
        # Extract dates in calls
        call_dates = []
        for c in calls:
            for k, v in c.get('arguments', {}).items():
                if isinstance(v, str) and re.match(r'^\d{4}-\d{2}-\d{2}', v):
                    call_dates.append((c['name'], k, v))
        if call_dates:
            print(f"[{sid}] curr_time={curr_time_str}")
            print(f"   transcript: {transcript}")
            for cd in call_dates:
                print(f"   call: {cd[0]} {cd[1]}={cd[2]}")
            print('---')

