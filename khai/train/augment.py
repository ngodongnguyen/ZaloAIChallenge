"""Schema-level augmentation: rename apps/tools consistently so the model must read descriptions.

The same mapping is applied to the catalog, the sample context/history and the gold label, so the
trajectory stays self-consistent. Inference always sees the real names; only SFT data is renamed.
"""
import copy
import random
import re
import string

SYNONYMS = {
    "get": ["fetch", "read", "lookup"],
    "list": ["show", "browse", "all"],
    "set": ["update", "change", "apply"],
    "create": ["add", "new", "make"],
    "add": ["create", "append", "insert"],
    "delete": ["remove", "erase", "drop"],
    "remove": ["delete", "discard"],
    "search": ["find", "query", "lookup"],
    "send": ["post", "deliver", "dispatch"],
    "book": ["reserve", "order"],
    "cancel": ["abort", "revoke", "void"],
    "start": ["begin", "launch", "run"],
    "stop": ["halt", "end", "pause"],
    "update": ["modify", "edit", "set"],
    "track": ["trace", "locate"],
    "play": ["start", "run"],
    "check": ["verify", "inspect"],
    "pay": ["settle", "charge"],
    "read": ["open", "view"],
    "status": ["state", "info"],
    "price": ["rate", "quote"],
    "events": ["entries", "items"],
    "history": ["log", "records"],
}
APP_AFFIX = ["my", "smart", "pro", "plus", "go", "one", "hub", "app", "lite", "x"]


def _opaque(rng):
    return "svc_" + "".join(rng.choices(string.ascii_lowercase, k=3))


def build_mapping(catalog, rng):
    """Return (app_map, tool_map) for the apps of the catalog (the 'sys' meta app is never renamed)."""
    app_map, tool_map, used_apps = {}, {}, set()
    for app in catalog["apps"]:
        old = app["id"]
        if old == "sys":
            continue
        for _ in range(20):
            r = rng.random()
            if r < 0.4:
                new = _opaque(rng)
            elif r < 0.8:
                new = f"{rng.choice(APP_AFFIX)}_{old}" if rng.random() < 0.5 else f"{old}_{rng.choice(APP_AFFIX)}"
            else:
                new = old
            if new not in used_apps:
                break
        used_apps.add(new)
        app_map[old] = new
        used_suffix = set()
        for t in app["tools"]:
            suffix = t["name"].split(".", 1)[1]
            toks = suffix.split("_")
            new_toks = [rng.choice(SYNONYMS[x]) if x in SYNONYMS and rng.random() < 0.7 else x for x in toks]
            cand = "_".join(new_toks)
            if cand in used_suffix:
                cand = suffix
            used_suffix.add(cand)
            tool_map[t["name"]] = f"{new}.{cand}"
    return app_map, tool_map


def make_mapper(app_map, tool_map):
    pat = re.compile(r"(?<![\w.@])(" + "|".join(sorted(map(re.escape, tool_map), key=len, reverse=True)) + r")(?!\w)")

    def sub_str(s):
        if s in tool_map:
            return tool_map[s]
        return pat.sub(lambda m: tool_map[m.group(1)], s)

    def sub_ref(s):  # x-ref like "smart_home.devices": only the app prefix changes
        app, _, rest = s.partition(".")
        return f"{app_map[app]}.{rest}" if app in app_map and rest else s

    def walk(o, parent=None):
        """Bare app ids are renamed only where they denote an app (never in free text or result fields)."""
        if isinstance(o, str):
            return sub_str(o)
        if isinstance(o, list):
            if parent == "installed_apps":
                return [app_map.get(x, x) if isinstance(x, str) else walk(x) for x in o]
            return [walk(x, parent) for x in o]
        if isinstance(o, dict):
            out = {}
            is_app = "tools" in o and "name_vi" in o
            for k, v in o.items():
                nk = app_map.get(k, k) if parent in ("installed_apps", "final_state") else k
                if k == "x-ref" and isinstance(v, str):
                    out[nk] = sub_ref(v)
                elif (k == "app" or (k == "id" and is_app)) and isinstance(v, str):
                    out[nk] = app_map.get(v, v)
                else:
                    out[nk] = walk(v, k)
            return out
        return o

    return walk


def augment(item, catalog, seed_key, shuffle_tools=True):
    """Return renamed copies (item, catalog). Deterministic given seed_key."""
    rng = random.Random(seed_key)
    app_map, tool_map = build_mapping(catalog, rng)
    walk = make_mapper(app_map, tool_map)
    new_item = walk(copy.deepcopy(item))
    new_cat = walk(copy.deepcopy(catalog))
    if shuffle_tools:
        for app in new_cat["apps"]:
            rng.shuffle(app["tools"])
    return new_item, new_cat
