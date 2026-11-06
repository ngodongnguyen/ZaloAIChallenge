import json

data = json.load(open('shard1_manifest.json'))

for s in data:
    sid = s['id']
    transcript = s.get('transcript', '')
    calls = s.get('v7_calls', [])
    results = s.get('execution_results', [])
    
    pay_calls = [c for c in calls if c['name'] == 'bill_payment.pay']
    if pay_calls:
        # Check bills list
        bills_found = []
        for r in results:
            if r.get('name') == 'bill_payment.list_bills':
                bills_found = r.get('sample_res', {}).get('bills', [])
        print(f"[{sid}] Transcript: {transcript}")
        print(f"   Available bills: {[(b.get('id'), b.get('type'), b.get('amount'), b.get('due_date')) for b in bills_found]}")
        for c in pay_calls:
            print(f"   Paid: {c.get('arguments')}")
        print('---')

