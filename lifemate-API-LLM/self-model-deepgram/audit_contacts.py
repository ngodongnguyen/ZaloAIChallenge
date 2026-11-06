import json

data = json.load(open('shard1_manifest.json'))

for s in data:
    sid = s['id']
    transcript = s.get('transcript', '')
    calls = s.get('v7_calls', [])
    results = s.get('execution_results', [])
    
    contact_searches = [c for c in calls if c['name'] == 'contacts.search']
    transfers = [c for c in calls if 'transfer' in c['name']]
    messages = [c for c in calls if c['name'] in ['sms.send', 'phone.call', 'chat_ott.send_message']]
    
    for c in contact_searches:
        q = c.get('arguments', {}).get('query', '').lower()
        if q and q not in transcript.lower():
            # Check if query words appear in transcript
            words = q.split()
            if not any(w in transcript.lower() for w in words):
                print(f"[{sid}] contacts.search query '{q}' NOT in transcript: '{transcript}'")

