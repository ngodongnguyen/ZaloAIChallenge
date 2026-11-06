import json
import re

data = json.load(open('shard1_manifest.json'))

KEYWORDS = {
    'alarm': [r'\bbáo thức\b', r'\bđặt thức\b', r'\bthức dậy\b', r'\bhẹn giờ\b', r'\bchuông\b', r'\bđặt bỏ thức\b', r'\bbảo thức\b'],
    'reminders': [r'\bnhắc\b', r'\bdặn\b', r'\bnhắc nhở\b'],
    'calendar': [r'\blịch\b', r'\bcuộc hẹn\b', r'\bhọp\b', r'\bhẹn\b', r'\bhoãn\b', r'\bhủy lịch\b'],
    'smart_home': [r'\bđèn\b', r'\bquạt\b', r'\bđiều hòa\b', r'\bmáy lạnh\b', r'\bbật\b', r'\btắt\b', r'\bphòng khách\b', r'\bphòng ngủ\b', r'\bphòng bếp\b'],
    'banking': [r'\bngân hàng\b', r'\btài khoản thanh toán\b', r'\btiết kiệm\b', r'\bchuyển khoản\b'],
    'ewallet': [r'\bví\b', r'\bví điện tử\b', r'\bnạp tiền\b'],
    'bill_payment': [r'\bhóa đơn\b', r'\btiền điện\b', r'\btiền nước\b', r'\btiền mạng\b', r'\btiền internet\b', r'\btruyền hình\b', r'\bđóng tiền\b'],
    'phone': [r'\bgọi điện\b', r'\bgọi cho\b', r'\bnháy máy\b'],
    'sms': [r'\bnhắn tin\b', r'\bnhắn sms\b', r'\bgửi tin\b', r'\bnhắn cho\b', r'\bnhận sms\b'],
    'chat_ott': [r'\bzalo\b', r'\bchat\b', r'\bnhắn zalo\b'],
    'ride_hailing': [r'\bđặt xe\b', r'\bxe máy\b', r'\bô tô\b', r'\bgrab\b', r'\bxe 4 chỗ\b', r'\bxe 7 chỗ\b', r'\bđón\b'],
    'flights': [r'\bvé máy bay\b', r'\bchuyến bay\b', r'\bbay\b', r'\bhạ cánh\b'],
    'train_bus_tickets': [r'\bvé tàu\b', r'\btàu hỏa\b', r'\bchuyến tàu\b', r'\bxe khách\b'],
    'shopping_list': [r'\bdanh sách mua\b', r'\bdanh sách đi chợ\b', r'\bghi vào danh sách\b'],
    'notes': [r'\bghi chú\b', r'\blưu lại\b', r'\bviết lại\b', r'\bnote\b'],
    'tasks': [r'\bviệc cần làm\b', r'\btodo\b', r'\btask\b', r'\btạo việc\b'],
    'weather': [r'\bthời tiết\b', r'\bmưa\b', r'\bnắng\b', r'\bnhiệt độ\b', r'\bdự báo\b'],
    'gold_price': [r'\bgiá vàng\b', r'\bvàng sjc\b', r'\bvàng nhẫn\b'],
    'fuel_price': [r'\bgiá xăng\b', r'\bxăng ron\b', r'\bxăng e5\b'],
    'translator': [r'\bdịch\b', r'\bsang tiếng anh\b', r'\btiếng việt sang\b']
}

omitted_candidates = []

for s in data:
    sid = s['id']
    transcript = s.get('transcript', '')
    calls = s.get('v7_calls', [])
    called_apps = set(c['name'].split('.')[0] for c in calls)
    
    t_lower = transcript.lower()
    
    missing = []
    for app, patterns in KEYWORDS.items():
        if app in s.get('installed_apps', []):
            if any(re.search(pat, t_lower) for pat in patterns):
                # Check if this app was called or if sys.ask was called for it
                asked_tools = [c.get('arguments', {}).get('tool_name', '') for c in calls if c['name'] == 'sys.ask_missing_info']
                asked_apps = [t.split('.')[0] for t in asked_tools if t]
                if app not in called_apps and app not in asked_apps:
                    missing.append(app)
                    
    if missing:
        omitted_candidates.append({
            'sid': sid,
            'transcript': transcript,
            'called': list(called_apps),
            'missing': missing,
            'calls': calls
        })

print(f"Total potential omitted intent samples: {len(omitted_candidates)}")
for c in omitted_candidates:
    print(f"[{c['sid']}] Missing potential apps: {c['missing']}")
    print(f"   Transcript: {c['transcript']}")
    print(f"   Called apps: {c['called']}")
    print('---')

