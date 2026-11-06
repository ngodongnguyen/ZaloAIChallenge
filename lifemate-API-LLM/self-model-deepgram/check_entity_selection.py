import json

data = json.load(open('shard1_manifest.json'))

for s in data:
    sid = s['id']
    transcript = s.get('transcript', '')
    calls = s.get('v7_calls', [])
    results = s.get('execution_results', [])
    
    # Check flight booking
    for c_idx, c in enumerate(calls):
        if c['name'] == 'flights.book':
            booked_fid = c.get('arguments', {}).get('flight_id')
            # Look at flights.search results
            search_res = None
            for r in results:
                if r.get('name') == 'flights.search':
                    search_res = r.get('sample_res', {}).get('flights', [])
            print(f"[{sid}] FLIGHT: '{transcript}'")
            print(f"   Booked: {booked_fid}")
            if search_res:
                for f in search_res:
                    print(f"     Option: {f.get('id')} {f.get('airline')} depart={f.get('depart')} arrive={f.get('arrive')} price={f.get('price')}")
            else:
                print("   No search_res in current turn")

        if c['name'] == 'train_bus_tickets.book':
            booked_tid = c.get('arguments', {}).get('trip_id')
            search_res = None
            for r in results:
                if r.get('name') == 'train_bus_tickets.search':
                    search_res = r.get('sample_res', {}).get('trips', [])
            print(f"[{sid}] TRAIN/BUS: '{transcript}'")
            print(f"   Booked: {booked_tid}")
            if search_res:
                for t in search_res:
                    print(f"     Option: {t.get('id')} {t.get('brand')} depart={t.get('depart')} arrive={t.get('arrive')} price={t.get('price')}")

