"""Approximate per-sample score for LifeMate, built from the rules in the competition description.

Known from the rules:
- Gold list empty: score 1 only if the submitted list is also empty, else 0.
- Otherwise the state_hash must equal the gold hash, else score 0.
- A submitted call matches a gold call when the tool name is equal, every non-fuzzy argument is
  identical, and every fuzzy argument has token-F1 > 0.6 (lowercased, Unicode NFC).
- Each call matches at most one gold call; order is ignored.
- Precision P and recall R are built from matched count M, gold count |G|, and unmatched submitted
  count U.

Assumed (NOT confirmed): the score combines P and R as F1 = 2PR/(P+R). The official formula is an
image and was not available. Treat results as an approximation until the official formula is known.
"""
import unicodedata


def normalize_text(s):
    return unicodedata.normalize("NFC", str(s)).lower()


def token_f1(a, b):
    ta = normalize_text(a).split()
    tb = normalize_text(b).split()
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    common = {}
    for t in ta:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in tb:
        if common.get(t, 0) > 0:
            common[t] -= 1
            overlap += 1
    if overlap == 0:
        return 0.0
    p = overlap / len(tb)
    r = overlap / len(ta)
    return 2 * p * r / (p + r)


def calls_match(sub, gold, fuzzy_args):
    """`fuzzy_args` is the set of argument names the catalog marks as x-match: fuzzy for this tool."""
    if sub.get("name") != gold.get("name"):
        return False
    sa = sub.get("arguments", {}) or {}
    ga = gold.get("arguments", {}) or {}
    if set(sa) != set(ga):
        return False
    for key in ga:
        if key in fuzzy_args:
            if not (isinstance(sa[key], str) and isinstance(ga[key], str)):
                if sa[key] != ga[key]:
                    return False
            elif token_f1(sa[key], ga[key]) <= 0.6:
                return False
        elif sa[key] != ga[key]:
            return False
    return True


def max_matching(sub_calls, gold_calls, fuzzy_by_tool):
    """Maximum bipartite matching (augmenting paths). Each call matches at most one partner."""
    adj = []
    for s in sub_calls:
        fz = fuzzy_by_tool.get(s.get("name"), set())
        adj.append([j for j, g in enumerate(gold_calls) if calls_match(s, g, fz)])
    owner = [-1] * len(gold_calls)

    def try_assign(i, seen):
        for j in adj[i]:
            if j in seen:
                continue
            seen.add(j)
            if owner[j] == -1 or try_assign(owner[j], seen):
                owner[j] = i
                return True
        return False

    matched = 0
    for i in range(len(sub_calls)):
        if try_assign(i, set()):
            matched += 1
    return matched


def sample_score(sub_calls, gold_calls, sub_hash, gold_hash, fuzzy_by_tool):
    """Score one sample in [0, 1] under the assumed F1 combination."""
    if not gold_calls:
        return 1.0 if not sub_calls else 0.0
    if sub_hash != gold_hash:
        return 0.0
    m = max_matching(sub_calls, gold_calls, fuzzy_by_tool)
    if m == 0:
        return 0.0
    u = len(sub_calls) - m
    p = m / (m + u)
    r = m / len(gold_calls)
    return 2 * p * r / (p + r)


def mean_score(per_sample):
    """Average over N samples, as in the formula. Empty input is an error, not a zero."""
    if not per_sample:
        raise ValueError("no samples to average")
    return sum(per_sample) / len(per_sample)
