"""Canonical JSON: the same data always becomes the same bytes. A deliberately RESTRICTED subset of RFC 8785 (JCS).

A signature covers bytes, not data. If two programs serialise the same receipt differently (key order, spacing, escapes), a valid signature fails to verify, or worse, a different
document verifies. So receipts are signed over a canonical form: no whitespace, object keys sorted by UTF-16 code units, strings escaped the one way JCS prescribes, UTF-8 output.

What this accepts: objects (string keys), arrays, strings, integers within +/- (2^53 - 1), booleans and null.
What it REFUSES, on purpose:
  * floats. JCS defines their text form through ECMAScript number formatting, which is easy to get subtly wrong. Money and quantities should be integers (cents) or strings anyway
  * integers outside the range a JSON number can carry exactly
  * strings that cannot be encoded as UTF-8 (lone surrogates)
  * non-string object keys, unknown types, deep nesting (32 levels), and anything over 1 MB
This is NOT a conformant RFC 8785 implementation (no float support). It is checked against the examples in the tests, and nothing here has been cross-validated with another library.
"""

import hashlib

MAX_SAFE_INT = 2 ** 53 - 1
MAX_DEPTH = 32
MAX_BYTES = 1_000_000
_ESC = {0x08: "\\b", 0x09: "\\t", 0x0A: "\\n", 0x0C: "\\f", 0x0D: "\\r", 0x22: '\\"', 0x5C: "\\\\"}


class CanonError(ValueError):
    pass


def _string(s: str) -> str:
    try:
        s.encode("utf-8")
    except UnicodeEncodeError:
        raise CanonError("a string contains a lone surrogate and cannot be encoded as UTF-8") from None
    out = ['"']
    for ch in s:
        o = ord(ch)
        if o in _ESC:
            out.append(_ESC[o])
        elif o < 0x20:
            out.append("\\u%04x" % o)
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _encode(v, depth: int) -> str:
    if depth > MAX_DEPTH:
        raise CanonError(f"nesting deeper than {MAX_DEPTH} levels")
    if v is None:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, int):
        if abs(v) > MAX_SAFE_INT:
            raise CanonError("an integer is outside the exactly-representable JSON range")
        return str(v)
    if isinstance(v, float):
        raise CanonError("floats are not allowed: use integers (for example cents) or strings")
    if isinstance(v, str):
        return _string(v)
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_encode(x, depth + 1) for x in v) + "]"
    if isinstance(v, dict):
        if any(not isinstance(k, str) for k in v):
            raise CanonError("object keys must be strings")
        for k in v:
            _string(k)
        keys = sorted(v, key=lambda k: k.encode("utf-16-be"))
        return "{" + ",".join(_string(k) + ":" + _encode(v[k], depth + 1) for k in keys) + "}"
    raise CanonError(f"cannot canonicalise a value of type {type(v).__name__}")


def canonicalize(value) -> bytes:
    out = _encode(value, 0).encode("utf-8")
    if len(out) > MAX_BYTES:
        raise CanonError(f"canonical form is larger than {MAX_BYTES} bytes")
    return out


def digest(value) -> str:
    """'sha256:' plus the hex SHA-256 of the canonical bytes."""
    return "sha256:" + hashlib.sha256(canonicalize(value)).hexdigest()
