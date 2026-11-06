"""lm-A-054: word-level fusion of two ASR hypotheses with a count-based language model.

No neural parameters: trigram counts from training text only, plus the words the agent already has for this sample
(profile, assistant turns, tool results). The primary hypothesis keeps its own formatting unless a word really differs.
"""
from __future__ import annotations

import difflib
import math
import re
import unicodedata
from collections import Counter

DIGITS = "không một hai ba bốn năm sáu bảy tám chín".split()
UNITS = [(10**9, "tỷ"), (10**6, "triệu"), (1000, "nghìn")]


def _below_thousand(n: int, full: bool) -> list[str]:
    out = []
    hundreds, rest = n // 100, n % 100
    if hundreds or full:
        out += [DIGITS[hundreds], "trăm"]
    tens, ones = rest // 10, rest % 10
    if tens == 0:
        if ones:
            out += (["linh"] if out else []) + [DIGITS[ones]]
    elif tens == 1:
        out.append("mười")
        if ones:
            out.append("lăm" if ones == 5 else DIGITS[ones])
    else:
        out += [DIGITS[tens], "mươi"]
        if ones:
            out.append({1: "mốt", 4: "tư", 5: "lăm"}.get(ones, DIGITS[ones]))
    return out


def number_words(digits: str) -> list[str]:
    """How a digit string is read aloud. Leading zeros and long strings (phone numbers) are read digit by digit."""
    if (digits.startswith("0") and len(digits) > 1) or len(digits) > 12:
        return [DIGITS[int(d)] for d in digits]
    n = int(digits)
    if n == 0:
        return ["không"]
    out = []
    for value, name in UNITS:
        if n >= value:
            head = n // value
            out += (number_words(str(head)) if head >= 1000 else _below_thousand(head, False)) + [name]
            n %= value
    if n:
        out += _below_thousand(n, bool(out))
    return out


def spoken(token: str) -> list[str]:
    """One raw token as spoken words: lower case, no punctuation, digits read out. '100.000' is one number."""
    token = unicodedata.normalize("NFC", token).lower()
    token = re.sub(r"(?<=\d)[.,](?=\d{3}(?:\D|$))", "", token)
    out = []
    for part in re.findall(r"\w+", token):
        if part.isdigit():
            out += number_words(part)
        else:
            out.append(part)
    return out


def spoken_text(text: str) -> list[str]:
    return [w for token in text.split() for w in spoken(token)]


class TrigramLM:
    def __init__(self, texts):
        self.uni, self.bi, self.tri = Counter(), Counter(), Counter()
        for text in texts:
            w = ["<s>", "<s>"] + spoken_text(text) + ["</s>"]
            self.uni.update(w)
            self.bi.update(zip(w, w[1:]))
            self.tri.update(zip(w, w[1:], w[2:]))
        self.total, self.vocab = sum(self.uni.values()), len(self.uni)

    def logp(self, a: str, b: str, c: str) -> float:
        if self.tri[(a, b, c)]:
            return math.log(self.tri[(a, b, c)] / self.bi[(a, b)])
        if self.bi[(b, c)]:
            return math.log(.4) + math.log(self.bi[(b, c)] / self.uni[b])
        return 2 * math.log(.4) + math.log((self.uni[c] + .5) / (self.total + .5 * self.vocab + .5))

    def score(self, left: list[str], words: list[str], right: list[str]) -> float:
        seq = left[-2:] + words + right[:2]
        return sum(self.logp(seq[i - 2], seq[i - 1], seq[i]) for i in range(2, len(seq)))


def span_features(lm: TrigramLM, left: list[str], a: list[str], b: list[str], right: list[str], context: set[str], app_words: set[str]) -> list[float]:
    """Primary words `a` against secondary words `b`, as differences; the last value is the intercept."""
    sa, sb = lm.score(left, a, right), lm.score(left, b, right)
    rarity = lambda words: sum(math.log(1 + lm.uni[w]) for w in words) / max(len(words), 1)
    count = lambda words, vocab: sum(w in vocab for w in words)
    return [sa - sb, sa / (len(a) + 2) - sb / (len(b) + 2), len(a) - len(b), count(a, context) - count(b, context),
            count(a, app_words) - count(b, app_words), sum(lm.uni[w] == 0 for w in a) - sum(lm.uni[w] == 0 for w in b),
            rarity(a) - rarity(b), float(not a), float(not b), 1.0]


class SpanModel:
    """Logistic choice for one disagreeing span, fitted on out-of-fold transcripts of train audio only."""

    def __init__(self, weights, mean, scale):
        self.weights, self.mean, self.scale = list(weights), list(mean), list(scale)

    def prefers_secondary(self, features: list[float]) -> bool:
        return sum(w * (f - m) / s for w, f, m, s in zip(self.weights, features, self.mean, self.scale)) > 0


def _match_case(word: str, model: str) -> str:
    return word.capitalize() if model[:1].isupper() else word


def fuse(primary: str, secondary: str, lm: TrigramLM, context: set[str], bias: float = -2., context_weight: float = 2.,
         length_penalty: float = 1., model: SpanModel | None = None, app_words: frozenset | set = frozenset()) -> str:
    """Primary text with only the disagreeing spans replaced when the language model prefers the secondary words."""
    tokens = primary.split()
    flat, owner = [], []
    for i, token in enumerate(tokens):
        words = spoken(token)
        flat += words
        owner += [i] * len(words)
    other = spoken_text(secondary)
    if not flat or not other:
        return primary
    ops = difflib.SequenceMatcher(None, flat, other, autojunk=False).get_opcodes()
    out, left = [], ["<s>", "<s>"]
    done = 0  # next primary token not yet emitted
    for op, i, j, u, v in ops:
        if op == "equal":
            for t in range(done, owner[j - 1] + 1):
                out.append(tokens[t])
            done = max(done, owner[j - 1] + 1)
            left += flat[i:j]
            continue
        a, b = flat[i:j], other[u:v]
        # A span cutting through one primary token ("118" read as several words) cannot be replaced cleanly.
        inside = 0 < i < len(flat) and owner[i - 1] == owner[i] or (a and j < len(flat) and owner[j - 1] == owner[j])
        right = flat[j:j + 2] + (["</s>"] if j + 2 > len(flat) else [])
        def total(words):
            return lm.score(left, words, right) + context_weight * sum(w in context for w in words) - length_penalty * len(words)
        keep = (not model.prefers_secondary(span_features(lm, left, a, b, right, context, app_words))) if model else total(a) + bias >= total(b)
        if inside or keep:
            if a:
                for t in range(done, owner[j - 1] + 1):
                    out.append(tokens[t])
                done = max(done, owner[j - 1] + 1)
            left += a
            continue
        shape = tokens[owner[i]] if a else ""
        if a:
            for t in range(done, owner[i]):
                out.append(tokens[t])
        out += [_match_case(w, shape if k == 0 else "") for k, w in enumerate(b)]
        if a:
            done = max(done, owner[j - 1] + 1)
        left += b
    out += tokens[done:]
    return " ".join(out)
