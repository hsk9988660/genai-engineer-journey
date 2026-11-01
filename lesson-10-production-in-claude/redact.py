"""PII redaction for anything that leaves the request path: traces, logs, eval datasets.

Regex heuristics: they find emails, SSNs, Luhn-valid card numbers and some phone formats. They miss names, addresses and
free-form identifiers. Treat this as a floor, and use a real PII service in production.
"""

import re
from collections import Counter

_PATTERNS = [  # more specific shapes first so a card number is not read as a phone number
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("PHONE", re.compile(r"(?<![\w-])(?:\+?\d{1,3}[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?![\w-])")),
]


def luhn_ok(digits: str) -> bool:
    nums = [int(d) for d in digits][::-1]
    return (sum(nums[0::2]) + sum(sum(divmod(2 * d, 10)) for d in nums[1::2])) % 10 == 0


def redact_text(text: str) -> tuple[str, Counter]:
    taken: list[tuple[int, int]] = []
    spans = []
    for kind, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            if kind == "CARD":
                digits = re.sub(r"\D", "", m.group())
                if not (13 <= len(digits) <= 19 and luhn_ok(digits)):
                    continue
            if any(m.start() < e and m.end() > s for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            spans.append((m.start(), m.end(), kind))
    found: Counter = Counter()
    out, pos = [], 0
    for start, end, kind in sorted(spans):
        out.append(text[pos:start])
        out.append(f"[{kind}]")
        found[kind] += 1
        pos = end
    out.append(text[pos:])
    return "".join(out), found


def redact_obj(obj):
    """Redact every string inside nested dicts and lists."""
    if isinstance(obj, str):
        return redact_text(obj)[0]
    if isinstance(obj, dict):
        return {k: redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact_obj(v) for v in obj]
    return obj
