import json

data = json.load(open('shard1_manifest.json'))

needs_standardization = []

for s in data:
    sid = s['id']
    calls = s.get('v7_calls', [])
    reasons = []
    
    for c in calls:
        name = c['name']
        args = c.get('arguments', {})
        
        if name == 'bill_payment.list_bills' and 'type' in args:
            reasons.append("bill_payment.list_bills has 'type'")
        if name == 'flights.search' and 'passengers' in args:
            reasons.append("flights.search has 'passengers'")
        if name == 'ride_hailing.get_quotes' and 'vehicle' in args:
            reasons.append("ride_hailing.get_quotes has 'vehicle'")
        if name == 'reminders.create' and args.get('repeat') == 'once':
            reasons.append("reminders.create has 'repeat: once'")
        if name == 'banking.list_transactions' and 'limit' in args:
            reasons.append("banking.list_transactions has 'limit'")
        if name == 'chat_ott.read_chat' and 'limit' in args:
            reasons.append("chat_ott.read_chat has 'limit'")
        if name == 'fuel_price.get_price':
            val = str(args.get('fuel_type') or args.get('type') or '').lower()
            if val not in ['ron95', 'e5', 'diesel']:
                reasons.append(f"fuel_price.get_price has non-standard type '{val}'")
            if 'fuel_type' in args:
                reasons.append("fuel_price.get_price has 'fuel_type' key instead of 'type'")
        if name == 'calendar.list_events' and 'date' in args:
            reasons.append("calendar.list_events has 'date' instead of date_from/date_to")
            
    if reasons:
        needs_standardization.append((sid, reasons))

print(f"Total samples needing argument standardization: {len(needs_standardization)}")
for sid, r in needs_standardization:
    print(f"[{sid}]: {', '.join(r)}")

