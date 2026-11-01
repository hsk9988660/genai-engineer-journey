"""PII handling: redact for logs, tokenize for the model.

Two different jobs:
  * `redact_text`: irreversible. Use for audit logs and traces, which are themselves sensitive.
  * `PiiVault`: reversible, per run. The model sees placeholders like [EMAIL_1] and never the real value. Real
    values are substituted back only at the boundary of tools that are explicitly marked `allow_pii_args`.
    A fetch_webpage call whose URL contains [EMAIL_1] is NOT expanded, so the model cannot use a tool to smuggle
    a real address out.

These are regex heuristics. They miss names, addresses and free-form identifiers. For production use a proper
PII detection service and treat this as a floor, not a guarantee.
"""

import re
from collections import Counter

_PATTERNS = [  # order matters: more specific shapes first so a card number is not read as a phone number
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("PHONE", re.compile(r"(?<![\w-])(?:\+?\d{1,3}[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?![\w-])")),
]
_TOKEN = re.compile(r"\[(EMAIL|SSN|CARD|PHONE)_(\d+)\]")


def luhn_ok(digits: str) -> bool:
    nums = [int(d) for d in digits][::-1]
    total = sum(nums[0::2]) + sum(sum(divmod(2 * d, 10)) for d in nums[1::2])
    return total % 10 == 0


def _matches(text: str):
    """Yield (start, end, kind, value) for non-overlapping PII spans."""
    taken: list[tuple[int, int]] = []
    for kind, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            if kind == "CARD":
                digits = re.sub(r"\D", "", m.group())
                if not (13 <= len(digits) <= 19 and luhn_ok(digits)):
                    continue
            if any(m.start() < e and m.end() > s for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            yield m.start(), m.end(), kind, m.group()


def redact_text(text: str) -> tuple[str, Counter]:
    found: Counter = Counter()
    spans = sorted(_matches(text))
    out, pos = [], 0
    for start, end, kind, _ in spans:
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


class PiiVault:
    """Per-run, reversible placeholders. Create a new vault for every run."""

    def __init__(self):
        self._by_value: dict[str, str] = {}
        self._by_token: dict[str, str] = {}
        self._counts: Counter = Counter()

    def tokenize(self, text: str) -> str:
        out, pos = [], 0
        for start, end, kind, value in sorted(_matches(text)):
            token = self._by_value.get(value)
            if token is None:
                self._counts[kind] += 1
                token = f"[{kind}_{self._counts[kind]}]"
                self._by_value[value], self._by_token[token] = token, value
            out.extend((text[pos:start], token))
            pos = end
        out.append(text[pos:])
        return "".join(out)

    def detokenize(self, obj):
        """Replace known placeholders with real values (recursively). Unknown placeholders are left alone."""
        if isinstance(obj, str):
            return _TOKEN.sub(lambda m: self._by_token.get(m.group(0), m.group(0)), obj)
        if isinstance(obj, dict):
            return {k: self.detokenize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self.detokenize(v) for v in obj]
        return obj
