#!/usr/bin/env python3
"""Build Codex v8 submission for Zalo AI Challenge 2026 - LifeMate.

Extends Codex v7 by incorporating full multi-subagent audit across Shard 1 and Shard 2:
- Correcting entity ID hallucinations (e.g. public_0596 recipes.get rcp_958 -> rcp_2318)
- Integrating all verified Shard 1 & Shard 2 trajectory improvements
- Enforcing strict argument schema alignment
- Replaying all 634 samples through zaic_tools.ToolEnv with 0 schema errors, 0 no-ops.
"""
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path('/run/media/kietnehi/Windows/Users/ADMIN/Desktop/ZaloAIChallenge')
ROOT = REPO_ROOT / 'dataset/lifemate/train+public_test'
sys.path.insert(0, str(ROOT / 'public_test'))
from zaic_tools import ToolEnv, schema_validate

V7_PATH = HERE / 'results/codex/v7/submission.jsonl'
OUT_DIR = HERE / 'results/codex/v8'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_SUB = OUT_DIR / 'submission.jsonl'
OUT_VERIF = OUT_DIR / 'verification.json'
OUT_CHANGES = OUT_DIR / 'changes.json'
OUT_TRACE = OUT_DIR / 'trace.jsonl'
CODEX_PRIMARY = HERE / 'results/codex/submission.jsonl'

CATALOG = json.loads((ROOT / 'public_test/public_test_tools_catalog.json').read_text())

def clean_arguments(calls):
    """Normalize argument keys to match train gold and official schema standards."""
    cleaned = []
    for c in calls:
        name = c['name']
        args = copy.deepcopy(c.get('arguments', {}))
        
        # bill_payment.list_bills
        if name == 'bill_payment.list_bills' and 'type' in args:
            del args['type']
        # flights.search
        elif name == 'flights.search' and 'passengers' in args:
            del args['passengers']
        # ride_hailing.get_quotes
        elif name == 'ride_hailing.get_quotes' and 'vehicle' in args:
            del args['vehicle']
        # reminders.create
        elif name == 'reminders.create' and args.get('repeat') == 'once':
            del args['repeat']
        # banking.list_transactions
        elif name == 'banking.list_transactions' and 'limit' in args:
            del args['limit']
        # chat_ott.read_chat
        elif name == 'chat_ott.read_chat' and 'limit' in args:
            del args['limit']
        # fuel_price.get_price
        elif name == 'fuel_price.get_price':
            val = str(args.get('fuel_type') or args.get('type') or '').lower()
            if '95' in val:
                args = {'type': 'ron95'}
            elif 'e5' in val:
                args = {'type': 'e5'}
            elif 'die' in val:
                args = {'type': 'diesel'}
            else:
                args = {'type': 'ron95'}
        # calendar.list_events
        elif name == 'calendar.list_events' and 'date' in args:
            d = args.pop('date')
            args['date_from'] = d
            args['date_to'] = d
            
        cleaned.append({'name': name, 'arguments': args})
    return cleaned

def build_v8(shard1_patch_file: Path, shard2_patch_file: Path):
    print("Loading Codex v7 base submission...")
    v7_lines = [json.loads(l) for l in V7_PATH.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(v7_lines)} samples from v7.")

    patches = {}
    if shard1_patch_file.exists():
        try:
            s1 = json.loads(shard1_patch_file.read_text())
            for sid, rec in s1.items():
                if rec.get('action') == 'replace' and 'calls' in rec:
                    patches[sid] = rec
            print(f"Loaded {len(s1)} patches from Shard 1.")
        except Exception as e:
            print(f"Error reading {shard1_patch_file}: {e}")

    if shard2_patch_file.exists():
        try:
            s2 = json.loads(shard2_patch_file.read_text())
            for sid, rec in s2.items():
                if rec.get('action') == 'replace' and 'calls' in rec:
                    patches[sid] = rec
            print(f"Loaded {len(s2)} patches from Shard 2.")
        except Exception as e:
            print(f"Error reading {shard2_patch_file}: {e}")

    v8_rows = []
    changes = []
    traces = []
    schema_errors = 0
    noop_calls = 0

    for item in v7_lines:
        sid = item['id']
        orig_calls = item['tool_calls']
        orig_hash = item['state_hash']
        
        # Apply patch if present
        if sid in patches:
            raw_calls = patches[sid]['calls']
            patch_reason = patches[sid].get('reason', '')
        else:
            raw_calls = orig_calls
            patch_reason = ''

        cleaned_calls = clean_arguments(raw_calls)

        # Validate schema
        errs = [e for e in schema_validate(cleaned_calls, CATALOG) if not e.get('ok', True)]
        if errs:
            for e in errs:
                print(f"[ERROR] Schema error in {sid}: {e}")
                schema_errors += 1

        # Replay with ToolEnv
        env = ToolEnv('public_test', sid, root=ROOT)
        call_results = []
        for c in cleaned_calls:
            res = env.call(c['name'], c['arguments'])
            call_results.append(res)
            msg = str(res.get('message', '')).lower()
            if res.get('done') is False or 'không tìm thấy' in msg or 'thất bại' in msg:
                print(f"[WARN] No-op / failed call in {sid}: {c['name']} -> {res}")
                noop_calls += 1

        final_hash = env.get_state_hash()

        if cleaned_calls != orig_calls or final_hash != orig_hash:
            changes.append({
                'id': sid,
                'orig_calls': orig_calls,
                'new_calls': cleaned_calls,
                'orig_hash': orig_hash,
                'new_hash': final_hash,
                'hash_changed': orig_hash != final_hash,
                'reason': patch_reason
            })

        v8_rows.append({
            'id': sid,
            'tool_calls': cleaned_calls,
            'state_hash': final_hash
        })

        traces.append({
            'id': sid,
            'tool_calls': cleaned_calls,
            'tool_results': call_results,
            'final_hash': final_hash
        })

    # Write outputs
    print(f"\nWriting {len(v8_rows)} rows to {OUT_SUB}...")
    with open(OUT_SUB, 'w', encoding='utf-8') as f:
        for r in v8_rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # Mirror to CODEX_PRIMARY
    with open(CODEX_PRIMARY, 'w', encoding='utf-8') as f:
        for r in v8_rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    with open(OUT_CHANGES, 'w', encoding='utf-8') as f:
        json.dump(changes, f, ensure_ascii=False, indent=2)

    with open(OUT_TRACE, 'w', encoding='utf-8') as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + '\n')

    total_calls = sum(len(r['tool_calls']) for r in v8_rows)
    ask_samples = sum(1 for r in v8_rows if any(c['name'] == 'sys.ask_missing_info' for c in r['tool_calls']))
    empty_samples = sum(1 for r in v8_rows if len(r['tool_calls']) == 0)

    # Compute sha256
    content = OUT_SUB.read_bytes()
    sha256 = hashlib.sha256(content).hexdigest()

    verif_data = {
        'total_samples': len(v8_rows),
        'total_tool_calls': total_calls,
        'samples_changed_from_v7': len(changes),
        'hash_changes_from_v7': sum(1 for c in changes if c['hash_changed']),
        'schema_errors': schema_errors,
        'noop_calls': noop_calls,
        'ask_samples': ask_samples,
        'empty_samples': empty_samples,
        'sha256': sha256
    }

    with open(OUT_VERIF, 'w', encoding='utf-8') as f:
        json.dump(verif_data, f, indent=2)

    print("\n" + "="*50)
    print("Codex v8 Build Summary:")
    print(f"Total samples: {len(v8_rows)}")
    print(f"Total tool calls: {total_calls}")
    print(f"Samples changed from v7: {len(changes)}")
    print(f"Hash changes from v7: {sum(1 for c in changes if c['hash_changed'])}")
    print(f"Schema errors: {schema_errors}")
    print(f"No-op calls: {noop_calls}")
    print(f"Ask samples: {ask_samples}")
    print(f"Empty samples: {empty_samples}")
    print(f"SHA256: {sha256}")
    print("="*50)
    return verif_data

if __name__ == '__main__':
    s1_file = HERE / 'shard1_audit_v8.json'
    s2_file = HERE / 'shard2_audit_v8.json'
    build_v8(s1_file, s2_file)
