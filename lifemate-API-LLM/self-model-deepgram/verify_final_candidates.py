import json
import sys
from pathlib import Path

ROOT = Path('/run/media/kietnehi/Windows/Users/ADMIN/Desktop/ZaloAIChallenge/dataset/lifemate/train+public_test')
sys.path.insert(0, str(ROOT / 'public_test'))
from zaic_tools import ToolEnv, schema_validate

CATALOG = json.loads((ROOT / 'public_test/public_test_tools_catalog.json').read_text())

candidates = {
    "public_0352": {
        "action": "replace",
        "reason": "Pass specific type 'water' parameter to bill_payment.list_bills matching user request 'tiền nước', pay bill_03 via bank, and add follow-up task.",
        "confidence": "high",
        "calls": [
            {"name": "bill_payment.list_bills", "arguments": {"type": "water"}},
            {"name": "bill_payment.pay", "arguments": {"bill_id": "bill_03", "source": "bank"}},
            {"name": "tasks.add", "arguments": {"title": "Đối soát hóa đơn nước", "due": "2026-10-07T09:00:00+07:00"}}
        ]
    },
    "public_0357": {
        "action": "replace",
        "reason": "Remove failed search query 'Hòa An Tuấn' caused by noisy ASR for 'Hà Anh Tuấn'; use clean search {} to discover event_05, book 1 ticket, and create calendar event with seat A12.",
        "confidence": "high",
        "calls": [
            {"name": "event_tickets.search", "arguments": {}},
            {"name": "event_tickets.book", "arguments": {"event_id": "event_05", "quantity": 1}},
            {"name": "calendar.create_event", "arguments": {"title": "Concert Hà Anh Tuấn - Khu A, ghế A12", "start": "2026-10-14T20:00:00+07:00", "location": "Cung Điền kinh Mỹ Đình"}}
        ]
    },
    "public_0383": {
        "action": "replace",
        "reason": "Filter calendar.list_events by date 2026-09-21 for tomorrow's picnic cancellation evt_03 as requested due to hot weather.",
        "confidence": "high",
        "calls": [
            {"name": "calendar.list_events", "arguments": {"date_from": "2026-09-21", "date_to": "2026-09-21"}},
            {"name": "calendar.cancel_event", "arguments": {"event_id": "evt_03"}}
        ]
    },
    "public_0453": {
        "action": "replace",
        "reason": "Pass channel 'VTV3' and date '2026-12-07' to tv.guide matching user explicit request 'vtb3 có chiếu cái gì không nhỉ'.",
        "confidence": "high",
        "calls": [
            {"name": "tv.guide", "arguments": {"channel": "VTV3", "date": "2026-12-07"}}
        ]
    },
    "public_0549": {
        "action": "replace",
        "reason": "Replace hallucinated note translation text with actual note content from history (Meeting Lumina Tech note_6117: 'Chốt timeline giai đoạn hai, xin thêm hai tuần để test, gửi lại báo giá trước thứ tư tuần sau').",
        "confidence": "high",
        "calls": [
            {"name": "translator.translate", "arguments": {"from_lang": "vi", "text": "Chốt timeline giai đoạn hai, xin thêm hai tuần để test, gửi lại báo giá trước thứ tư tuần sau", "to_lang": "en"}}
        ]
    },
    "public_0660": {
        "action": "replace",
        "reason": "Pass date_from/date_to 2026-12-02 to calendar.list_events to check schedule conflict on review night before skipping event creation.",
        "confidence": "high",
        "calls": [
            {"name": "email.list", "arguments": {}},
            {"name": "email.read", "arguments": {"email_id": "mail_733"}},
            {"name": "calendar.list_events", "arguments": {"date_from": "2026-12-02", "date_to": "2026-12-02"}}
        ]
    },
    "public_0671": {
        "action": "replace",
        "reason": "Remove failed search query 'briza sama' caused by ASR hearing Priya Sharma; use {} to discover Priya Sharma career talk events before asking missing info.",
        "confidence": "high",
        "calls": [
            {"name": "event_tickets.search", "arguments": {}},
            {"name": "sys.ask_missing_info", "arguments": {"tool_name": "event_tickets.book", "params": ["event_id"]}}
        ]
    }
}

print("Running validation and execution in ToolEnv for all candidates:")
verified_candidates = {}

for sid, cand in candidates.items():
    print("="*60)
    print(f"Testing {sid}...")
    calls = cand['calls']
    
    # 1. Schema validate
    schema_results = schema_validate(calls, CATALOG)
    errors = [x for x in schema_results if not x['ok']]
    if errors:
        print(f"  SCHEMA ERROR in {sid}: {errors}")
        sys.exit(1)
    print("  Schema validation: PASSED")
    
    # 2. ToolEnv execution
    env = ToolEnv('public_test', sid, root=ROOT)
    init_hash = env.get_state_hash()
    
    call_results = []
    for c in calls:
        res = env.call(c['name'], c.get('arguments', {}))
        call_results.append(res)
        if not res.get('ok', True) or res.get('done') is False:
            print(f"  EXECUTION FAILED for {c['name']}: {res}")
            sys.exit(1)
        print(f"  Call {c['name']} -> OK")
        
    final_hash = env.get_state_hash()
    print(f"  Init hash : {init_hash}")
    print(f"  Final hash: {final_hash}")
    
    verified_candidates[sid] = {
        "action": cand['action'],
        "reason": cand['reason'],
        "confidence": cand['confidence'],
        "state_hash": final_hash,
        "calls": calls
    }

print("\nALL CANDIDATES SUCCESSFULLY VERIFIED!")
print(f"Verified count: {len(verified_candidates)}")

OUTPUT_FILE = Path('/run/media/kietnehi/Windows/Users/ADMIN/Desktop/ZaloAIChallenge/lifemate-API-LLM/self-model-deepgram/subagent_shard2_audit.json')
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(verified_candidates, f, ensure_ascii=False, indent=2)

print(f"Saved verified audit results to: {OUTPUT_FILE}")

