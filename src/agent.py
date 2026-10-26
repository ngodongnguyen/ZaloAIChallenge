import json
import time

from config import MAX_STEPS, TIME_BUDGET_S
from prompt import allowed_names, call_msg, initial_messages, result_msg


def parse_action(text):
    """Return a dict from the first JSON object in the model output, or None."""
    i = text.find("{")
    if i < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[i:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def run_agent(env, item, transcript, catalog, llm, t0=None):
    """Drive the LLM/tool loop. Every executed call is logged by env itself."""
    t0 = t0 or time.time()
    allowed = allowed_names(item, catalog)
    messages = initial_messages(item, transcript, catalog)
    invalid = 0
    seen = set()

    for _ in range(MAX_STEPS):
        if time.time() - t0 > TIME_BUDGET_S:
            break
        action = parse_action(llm.generate(messages))
        if action is None:
            break
        if action.get("done"):
            break
        name, args = action.get("name"), action.get("arguments", {})
        if name not in allowed or not isinstance(args, dict):
            invalid += 1
            if invalid > 1:
                break
            messages.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False)})
            messages.append(result_msg({"ok": False, "error": f"tool không khả dụng: {name}"}))
            continue
        key = (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
        if key in seen:  # avoid looping on identical calls
            break
        seen.add(key)

        check = env.validate(name, args)
        if not check.get("ok", False):  # validate executes nothing and logs nothing
            invalid += 1
            if invalid > 1:
                break
            messages.append(call_msg(name, args))
            messages.append(result_msg(check))
            continue

        result = env.call(name, args)
        messages.append(call_msg(name, args))
        messages.append(result_msg(result))
        if name == "sys.ask_missing_info":
            break
    return env
