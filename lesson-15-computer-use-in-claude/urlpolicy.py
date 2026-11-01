"""Which URLs a browser agent may visit. Checked on EVERY hop: the first request, each link, and each redirect.

The lesson's README says to "limit it to the domains required for the task", and the notebook does not. URL checking is easy to get subtly wrong, so this parses
rather than pattern-matching and refuses the classic tricks:
  https://stays.example.evil.example   (allowed name used as a prefix: the real host is evil.example)
  https://evilstays.example            (allowed name used as a suffix without a dot)
  https://stays.example@evil.example   (userinfo: the real host is after the @)
  https://stays.example\\@evil.example  (a backslash that browsers read as a slash)
  http://, javascript:, data:, file:   (only https unless you opt in)
  https://127.0.0.1, https://[::1]     (IP literals, which skip name-based checks)
  https://xn--...  or non-ASCII names  (look-alike characters)
"""

import re
from urllib.parse import urljoin, urlsplit

_BAD_CHARS = re.compile(r"[\x00-\x20\\\x7f]")
_IP = re.compile(r"^[\d.]+$|:")


class BlockedURL(Exception):
    pass


class DomainPolicy:
    def __init__(self, allowed_hosts, allow_http: bool = False):
        self.allowed = tuple(h.lower().strip(".") for h in allowed_hosts)
        if not self.allowed:
            raise ValueError("at least one allowed host is required")
        self.allow_http = allow_http

    def check(self, url: str) -> str:
        """Returns the URL if it is allowed, else raises `BlockedURL` with the reason."""
        if not isinstance(url, str) or not url or len(url) > 2000:
            raise BlockedURL("empty or oversized URL")
        if _BAD_CHARS.search(url):
            raise BlockedURL("URL contains whitespace, a backslash or a control character")
        try:
            parts = urlsplit(url)
            host, port = parts.hostname, parts.port
        except ValueError as exc:
            raise BlockedURL(f"unparseable URL: {exc}") from exc
        if parts.scheme not in (("https", "http") if self.allow_http else ("https",)):
            raise BlockedURL(f"scheme {parts.scheme!r} is not allowed")
        if parts.username is not None or parts.password is not None:
            raise BlockedURL("URL contains credentials or an '@' before the host")
        if not host:
            raise BlockedURL("URL has no host")
        host = host.rstrip(".").lower()
        if not host.isascii() or any(label.startswith("xn--") for label in host.split(".")):
            raise BlockedURL("non-ASCII or punycode host names are not allowed")
        if _IP.search(host):
            raise BlockedURL("IP addresses are not allowed")
        if port not in (None, 443) and not (self.allow_http and port == 80):
            raise BlockedURL(f"port {port} is not allowed")
        if not any(host == a or host.endswith("." + a) for a in self.allowed):
            raise BlockedURL(f"host {host!r} is not on the allow-list")
        return url

    def allowed_url(self, url: str) -> bool:
        try:
            self.check(url)
            return True
        except BlockedURL:
            return False


def resolve(base: str, href: str) -> str:
    """Resolve a link against the page it appears on. The RESULT is what must be checked, never the raw href."""
    return urljoin(base, href)


def safe_display(url: str) -> str:
    """Host and path only. Query strings and fragments can carry tokens or stolen data, so they are not logged or shown."""
    try:
        p = urlsplit(url)
        return f"{p.scheme}://{p.hostname}{p.path}" + ("?…" if p.query else "")
    except ValueError:
        return "(unparseable url)"
