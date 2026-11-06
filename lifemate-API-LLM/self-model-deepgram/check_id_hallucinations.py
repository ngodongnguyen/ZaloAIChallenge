import json

data = json.load(open('shard1_manifest.json'))

ID_KEYS = ['recipe_id', 'flight_id', 'booking_id', 'trip_id', 'task_id', 'event_id', 'note_id', 
           'item_id', 'restaurant_id', 'contact_id', 'account_id', 'bill_id', 'lot_id', 'alarm_id', 'device_id']

for s in data:
    sid = s['id']
    transcript = s.get('transcript', '')
    history = s.get('history', [])
    calls = s.get('v7_calls', [])
    results = s.get('execution_results', [])
    
    # Map of ID to its details in history
    hist_entities = {}
    for h_idx, h in enumerate(history):
        for res in h.get('tool_results', []):
            def extract(obj):
                if isinstance(obj, dict):
                    # Check if this object is an entity with an id
                    eid = obj.get('id')
                    if eid and isinstance(eid, str):
                        hist_entities[eid] = {
                            'id': eid,
                            'name': obj.get('name') or obj.get('title') or obj.get('query'),
                            'type': obj.get('type'),
                            'raw': obj
                        }
                    for k, v in obj.items():
                        if k.endswith('_id') and isinstance(v, str):
                            if v not in hist_entities:
                                hist_entities[v] = {'id': v, 'key': k}
                        extract(v)
                elif isinstance(obj, list):
                    for elem in obj: extract(elem)
            extract(res)
            
    # Also entities in current turn results
    curr_entities = dict(hist_entities)
    for r in results:
        res = r.get('sample_res', {})
        def extract_curr(obj):
            if isinstance(obj, dict):
                eid = obj.get('id')
                if eid and isinstance(eid, str):
                    curr_entities[eid] = {
                        'id': eid,
                        'name': obj.get('name') or obj.get('title'),
                        'type': obj.get('type'),
                        'raw': obj
                    }
                for k, v in obj.items():
                    if k.endswith('_id') and isinstance(v, str):
                        if v not in curr_entities:
                            curr_entities[v] = {'id': v, 'key': k}
                    extract_curr(v)
            elif isinstance(obj, list):
                for elem in obj: extract_curr(elem)
        extract_curr(res)

    # Now check calls
    for c in calls:
        for k, v in c.get('arguments', {}).items():
            if k in ID_KEYS or k.endswith('_id'):
                if isinstance(v, str):
                    # Check if v is known
                    if v not in curr_entities:
                        print(f"[{sid}] Call {c['name']} has unknown ID: {k}='{v}'")
                        print(f"   Transcript: {transcript}")
                        print(f"   History entities: {[(k2, v2.get('name')) for k2, v2 in hist_entities.items()]}")

