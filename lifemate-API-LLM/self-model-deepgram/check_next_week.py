import json
import re
from datetime import datetime

data = json.load(open('shard1_manifest.json'))

for s in data:
    sid = s['id']
    transcript = s.get('transcript', '')
    curr_time_str = s.get('current_time', '')
    calls = s.get('v7_calls', [])
    
    t_lower = transcript.lower()
    if 'tuần sau' in t_lower or 'tuần tới' in t_lower:
        print(f"[{sid}] curr_time={curr_time_str}")
        print(f"   transcript: {transcript}")
        for c in calls:
            args = c.get('arguments', {})
            date_args = {k: v for k, v in args.items() if isinstance(v, str) and re.search(r'\d{4}-\d{2}-\d{2}', v)}
            if date_args:
                print(f"   call: {c['name']} -> {date_args}")
        print('---')

