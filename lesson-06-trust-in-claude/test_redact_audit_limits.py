import json

import pytest

from audit import AuditLog, verify_entries
from limits import Budget, KillSwitch, RateLimiter
from redact import PiiVault, luhn_ok, redact_obj, redact_text


# --- redaction ------------------------------------------------------------------------------
def test_redacts_common_pii():
    text, found = redact_text("Mail ana@example.com, SSN 123-45-6789, card 4111 1111 1111 1111, call (415) 555-0132.")
    assert "ana@example.com" not in text and "123-45-6789" not in text and "4111" not in text and "555-0132" not in text
    assert found == {"EMAIL": 1, "SSN": 1, "CARD": 1, "PHONE": 1}


def test_card_numbers_are_luhn_checked_and_not_mistaken_for_phones():
    assert luhn_ok("4111111111111111") and not luhn_ok("4111111111111112")
    text, found = redact_text("order 4111 1111 1111 1112")  # fails Luhn, so it is not a card
    assert "CARD" not in found
    text, found = redact_text("card 4111-1111-1111-1111")
    assert found["CARD"] == 1 and "PHONE" not in found


@pytest.mark.parametrize("harmless", ["Total $40,000 due", "Sold on 2025-01-15", "Flight AF1081 departs 09:15",
                                      "Booking BK1001", "Pay 10,000 miles"])
def test_ordinary_numbers_are_left_alone(harmless):
    assert redact_text(harmless)[0] == harmless


def test_redact_obj_walks_nested_structures():
    out = redact_obj({"a": ["x@y.co", {"b": "call 415-555-0132"}], "n": 5})
    assert out == {"a": ["[EMAIL]", {"b": "call [PHONE]"}], "n": 5}


def test_known_limit_names_and_free_form_identifiers_are_not_detected():
    assert redact_text("Customer Ana Ruiz lives at 12 Main Street")[0] == "Customer Ana Ruiz lives at 12 Main Street"


# --- vault ----------------------------------------------------------------------------------
def test_vault_round_trip_and_stable_tokens():
    v = PiiVault()
    masked = v.tokenize("a@b.co wrote to a@b.co and c@d.co")
    assert masked == "[EMAIL_1] wrote to [EMAIL_1] and [EMAIL_2]"
    assert v.detokenize(masked) == "a@b.co wrote to a@b.co and c@d.co"
    assert v.detokenize({"to": ["[EMAIL_2]"]}) == {"to": ["c@d.co"]}


def test_tokens_from_one_vault_mean_nothing_in_another():
    a, b = PiiVault(), PiiVault()
    a.tokenize("secret@x.co")
    assert b.detokenize("[EMAIL_1]") == "[EMAIL_1]"


# --- audit chain ----------------------------------------------------------------------------
def make_log(n=4):
    log = AuditLog(clock=lambda: 1.0)
    for i in range(n):
        log.append("event", {"i": i})
    return log


def test_intact_chain_verifies():
    assert make_log().verify() == (True, None)
    assert AuditLog().verify() == (True, None)


def test_editing_an_entry_is_detected_at_that_entry():
    log = make_log()
    log.entries[2]["data"]["i"] = 999
    assert log.verify() == (False, 2)


def test_deleting_or_reordering_entries_is_detected():
    log = make_log()
    entries = [e for i, e in enumerate(log.entries) if i != 1]
    assert verify_entries(entries) == (False, 1)
    swapped = [log.entries[1], log.entries[0], *log.entries[2:]]
    assert verify_entries(swapped)[0] is False


def test_truncating_the_end_is_not_detectable_without_an_external_anchor():
    log = make_log()
    anchor = log.head()
    assert verify_entries(log.entries[:-1]) == (True, None)  # looks fine on its own...
    assert log.entries[-2]["hash"] != anchor  # ...but the anchored head no longer matches


def test_audit_entries_never_contain_raw_pii():
    log = AuditLog()
    log.append("decision", {"args": {"to": "ana@example.com", "note": "card 4111 1111 1111 1111"}})
    blob = json.dumps(log.entries)
    assert "ana@example.com" not in blob and "4111" not in blob and "[EMAIL]" in blob
    assert log.verify()[0]


def test_file_persistence_and_tamper_detection(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path=path, clock=lambda: 1.0)
    for i in range(3):
        log.append("e", {"i": i})
    assert AuditLog.verify_file(path) == (True, None)
    lines = path.read_text().splitlines()
    lines[1] = lines[1].replace('"i": 1', '"i": 7')
    path.write_text("\n".join(lines) + "\n")
    assert AuditLog.verify_file(path) == (False, 1)


# --- limits ---------------------------------------------------------------------------------
def test_budget_exhaustion_by_steps_and_by_spend():
    b = Budget(max_steps=2, max_usd=1.0)
    assert b.exhausted() is None
    b.charge_step(1000, 1000)
    b.charge_step(1000, 1000)
    assert b.exhausted() == "step limit reached"
    s = Budget(max_steps=99, max_usd=0.01, usd_per_mtok_in=1000.0, usd_per_mtok_out=1000.0)
    s.charge_step(10_000, 0)
    assert s.exhausted() == "spend limit reached"


def test_budget_blocks_a_tool_that_would_overspend_or_exceed_call_count():
    b = Budget(max_usd=10.0, max_tool_calls=2)
    assert not b.would_exceed(9.0) and b.would_exceed(11.0)
    b.charge_tool(1.0)
    b.charge_tool(1.0)
    assert b.would_exceed(0.0)


def test_rate_limiter_sliding_window_per_user():
    now = [0.0]
    r = RateLimiter(max_calls=2, window_s=10, clock=lambda: now[0])
    assert r.allow("a") and r.allow("a") and not r.allow("a")
    assert r.allow("b")  # another user is unaffected
    now[0] = 10.0
    assert r.allow("a")


def test_kill_switch():
    k = KillSwitch()
    assert not k.tripped
    k.trip("incident 42")
    assert k.tripped and k.reason == "incident 42"
