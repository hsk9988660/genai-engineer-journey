"""What the agent may DO. Default-deny: only a short list of low-risk actions run without a human, and the policy is code the model cannot argue with.

  allow      run it
  approve    a person must approve the exact action first (the summary is built by CODE from the real page, never from the model's description)
  deny       refuse and tell the model why
  handover   stop: this needs the user themselves (a password or card field)

The rules:
  * navigation targets must pass the domain policy, and sensitive paths (/book, /login, ...) need approval
  * links are resolved, then checked, so a relative link cannot smuggle the agent off-site
  * a button runs without approval only if its visible label is on a short, task-specific allow-list. Everything else (Reserve, Confirm and pay) needs approval
  * any action on a sensitive page needs approval
  * the model can never type into a password or card field. The user must
  * the model can only refer to elements it was shown. Hidden ones have no number, so they cannot be clicked

KNOWN LIMIT (a test documents it): the allow-list trusts the button's VISIBLE LABEL, which the page controls. A hostile page can label a purchase button "Next".
The sensitive-path rule is the second line of defense there. Neither is a complete answer. A real deployment adds network-level limits and a sandboxed profile.
"""

import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from observe import Observation
from runtrace import digest
from urlpolicy import BlockedURL, DomainPolicy, resolve, safe_display

SAFE_BUTTONS = frozenset({"search", "accept cookies", "next", "show more", "close"})
SENSITIVE_PATH = re.compile(r"^/(book|checkout|payment|pay|login|signin|account|settings|messages|inbox)(/|$)", re.I)
BLOCKED_INPUT = frozenset({"password", "card"})
MAX_TYPED = 200
_AMOUNT = re.compile(r"\d[\d,.\s]*\s?(?:SEK|USD|EUR|kr|€|\$)|(?:SEK|USD|EUR|kr|€|\$)\s?\d[\d,.]*")


@dataclass(frozen=True)
class Navigate:
    url: str


@dataclass(frozen=True)
class Click:
    ref: int


@dataclass(frozen=True)
class Type:
    ref: int
    text: str


@dataclass(frozen=True)
class Done:
    answer: str = ""


@dataclass
class Decision:
    kind: str  # allow | approve | deny | handover
    reason: str = ""
    summary: dict = field(default_factory=dict)
    target_key: str | None = None
    target_url: str | None = None
    fingerprint: str = ""


def norm(s: str) -> str:
    return " ".join(s.lower().split())


def amounts_on(page_elements) -> list[str]:
    """Every price-looking string visible on the page. Part of what an approval covers, so a price that changes after approval is noticed."""
    return sorted({norm(m.group(0)) for e in page_elements if e.visible for m in _AMOUNT.finditer(e.text)})


def fingerprint(url: str, el, page_elements, full: bool) -> str:
    base = {"path": urlsplit(url).path, "role": el.role, "text": norm(el.text), "href": el.href, "type": el.input_type}
    if full:
        base["amounts"] = amounts_on(page_elements)
    return digest(base)


class ActionPolicy:
    def __init__(self, domain: DomainPolicy, safe_buttons=SAFE_BUTTONS, sensitive_path=SENSITIVE_PATH):
        self.domain, self.safe_buttons, self.sensitive = domain, frozenset(norm(b) for b in safe_buttons), sensitive_path

    def _sensitive(self, url: str) -> bool:
        return bool(self.sensitive.match(urlsplit(url).path))

    def _summary(self, action: str, obs: Observation, el=None, target: str | None = None) -> dict:
        s = {"action": action, "page": safe_display(obs.url), "page_amounts": amounts_on(obs.elements)}
        if el is not None:
            s["element"] = {"role": el.role, "text": " ".join(el.text.split())[:80]}
        if target:
            s["goes_to"] = safe_display(target)
        return s

    def decide(self, action, obs: Observation) -> Decision:
        if isinstance(action, Done):
            return Decision("allow")
        if isinstance(action, Navigate):
            try:
                url = self.domain.check(resolve(obs.url, action.url))
            except BlockedURL as exc:
                return Decision("deny", f"navigation refused: {exc}")
            if self._sensitive(url):
                return Decision("approve", "sensitive page", self._summary("navigate", obs, target=url), None, url, digest(["nav", url]))
            return Decision("allow", target_url=url)
        el = obs.refs.get(getattr(action, "ref", None))
        if el is None:
            return Decision("deny", "no such element on this page (only elements that were shown can be used)")
        sensitive_page = self._sensitive(obs.url)
        key = el.key
        if isinstance(action, Type):
            if el.role != "textbox":
                return Decision("deny", "that element is not a text field")
            if el.input_type in BLOCKED_INPUT:
                return Decision("handover", f"the user must fill in this {el.input_type} field themselves")
            if len(action.text) > MAX_TYPED:
                return Decision("deny", f"text longer than {MAX_TYPED} characters")
            fp = fingerprint(obs.url, el, obs.elements, True)
            if sensitive_page:
                return Decision("approve", "typing on a sensitive page", self._summary("type", obs, el), key, None, fp)
            return Decision("allow", target_key=key)
        if isinstance(action, Click):
            if el.href:
                try:
                    url = self.domain.check(resolve(obs.url, el.href))
                except BlockedURL as exc:
                    return Decision("deny", f"that link leaves the allowed sites: {exc}")
                if self._sensitive(url) or sensitive_page:
                    return Decision("approve", "sensitive page", self._summary("click", obs, el, url), key, url, fingerprint(obs.url, el, obs.elements, True))
                return Decision("allow", target_key=key, target_url=url)
            if el.role == "button":
                if not sensitive_page and norm(el.text) in self.safe_buttons:
                    return Decision("allow", target_key=key)
                return Decision("approve", "this button is not on the low-risk list", self._summary("click", obs, el), key, None, fingerprint(obs.url, el, obs.elements, True))
            if el.role in ("textbox",):
                return Decision("allow", target_key=key)
            return Decision("deny", f"a {el.role} element is not something to click")
        return Decision("deny", "unknown action")
