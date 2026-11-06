import json
import sys
from pathlib import Path

ROOT = Path('/run/media/kietnehi/Windows/Users/ADMIN/Desktop/ZaloAIChallenge/dataset/lifemate/train+public_test')
sys.path.insert(0, str(ROOT / 'public_test'))
from zaic_tools import ToolEnv

with open('audit_manifest_v9.json') as f:
    data = json.load(f)

shard2 = [d for d in data if d['index'] >= 316]

for s in shard2:
    sid = s['id']
    idx = s['index']
    t = s.get('transcript', '')
    calls = s.get('v9_calls', [])
    results = s.get('execution_results', [])
    
    for c_idx, c in enumerate(calls):
        cname = c['name']
        args = c.get('arguments', {})
        
        # Check flights.book
        if cname == 'flights.book':
            fid = args.get('flight_id')
            print(f"[{sid}] (idx={idx}) FLIGHT BOOK: '{t}' -> {fid}")
            
        # Check train_bus_tickets.book
        elif cname == 'train_bus_tickets.book':
            tid = args.get('trip_id')
            print(f"[{sid}] (idx={idx}) TRAIN/BUS BOOK: '{t}' -> {tid} seats={args.get('seats')}")
            
        # Check parking.reserve
        elif cname == 'parking.reserve':
            lid = args.get('lot_id')
            print(f"[{sid}] (idx={idx}) PARKING RESERVE: '{t}' -> {lid} from={args.get('from_time')} hours={args.get('hours')}")

        # Check event_tickets.book
        elif cname == 'event_tickets.book':
            eid = args.get('event_id')
            print(f"[{sid}] (idx={idx}) EVENT BOOK: '{t}' -> {eid} qty={args.get('quantity')}")

