import base64
from decimal import Decimal

import pytest

from expense import (Policy, ReceiptExtraction, build_claim, policy_flags, process_receipt, reconcile, render_email, to_money)
from fakes import FakeClient, Clock, submit
from telemetry import Price, PriceTable, Telemetry, TelemetryClient


def receipt(items=None, tax="0", tip="0", total="67.50", merchant="Cafe Roma"):
    items = items if items is not None else [("Taxi", "45.00", "Transportation"), ("Lunch", "22.50", "Meals")]
    return {"merchant": merchant, "tax": tax, "tip": tip, "printed_total": total,
            "items": [{"date": "2026-03-07", "description": d, "amount": a, "category": c} for d, a, c in items]}


def extract(payload):
    return ReceiptExtraction.model_validate(payload)


# --- money is Decimal ------------------------------------------------------------------------------------
def test_money_parsing_handles_strings_floats_symbols_and_commas():
    assert to_money("12.10") == Decimal("12.10") and to_money(12.1) == Decimal("12.1") and to_money("$1,234.50") == Decimal("1234.50")
    with pytest.raises(ValueError):
        to_money("twelve")


def test_float_noise_never_reaches_a_claim():
    x = extract(receipt([("a", 0.1, "Meals"), ("b", 0.2, "Meals")], total="0.30"))
    assert sum((i.amount for i in x.items), Decimal("0")) == Decimal("0.3")  # exactly, unlike 0.1 + 0.2 in floats
    assert reconcile(x) == []


# --- reconciliation: the free, deterministic check the notebook never made -------------------------------
def test_a_receipt_whose_items_match_its_total_reconciles():
    assert reconcile(extract(receipt())) == []
    assert reconcile(extract(receipt(tax="4.20", tip="8.00", total="79.70"))) == []


def test_a_missing_line_item_is_caught_and_the_gap_is_reported():
    problems = reconcile(extract(receipt(total="80.00")))
    assert len(problems) == 1 and "$67.50" in problems[0] and "$80.00" in problems[0] and "$12.50" in problems[0]


def test_a_difference_of_one_cent_is_tolerated_two_cents_is_not():
    assert reconcile(extract(receipt(total="67.51"))) == []
    assert reconcile(extract(receipt(total="67.53")))


def test_duplicates_non_positive_amounts_and_empty_receipts_are_flagged():
    dup = extract(receipt([("Taxi", "10.00", "Transportation"), ("Taxi", "10.00", "Transportation")], total="20.00"))
    assert any("duplicate" in p.lower() for p in reconcile(dup))
    neg = extract(receipt([("Refund", "-5.00", "Miscellaneous"), ("Lunch", "25.00", "Meals")], total="20.00"))
    assert any("non-positive" in p for p in reconcile(neg))
    assert any("No line items" in p for p in reconcile(extract(receipt([], total="0"))))


# --- policy is code, not a sentence in a prompt ----------------------------------------------------------------------
def test_tips_are_included_or_excluded_by_policy_in_code():
    x = extract(receipt(tip="8.00", total="75.50"))
    assert build_claim(x, Policy(reimburse_tips=True)).total == Decimal("75.50")
    assert build_claim(x, Policy(reimburse_tips=False)).total == Decimal("67.50")


def test_meal_cap_and_single_item_limits_are_enforced():
    x = extract(receipt([("Dinner", "80.00", "Meals")], total="80.00"))
    assert any("daily cap" in f for f in policy_flags(x, Policy()))
    big = extract(receipt([("Flight", "6000.00", "Transportation")], total="6000.00"))
    assert any("single-item limit" in f for f in policy_flags(big, Policy()))
    assert policy_flags(extract(receipt()), Policy()) == []


def test_a_clean_claim_is_ready_and_a_problem_makes_it_needs_review():
    assert build_claim(extract(receipt())).status == "ready"
    c = build_claim(extract(receipt(total="99.00")))
    assert c.status == "needs_review" and c.problems


# --- the email ----------------------------------------------------------------------------------------------------------
def test_the_email_is_deterministic_has_a_real_name_and_no_placeholder():
    claim = build_claim(extract(receipt()))
    subject, body = render_email(claim, "Cafe Roma", "Ana Ruiz")
    assert (subject, body) == render_email(claim, "Cafe Roma", "Ana Ruiz")
    assert "Ana Ruiz" in body and "[Your Name]" not in body and "$67.50" in subject and "Taxi" in body and "$45.00" in body


# --- the whole flow with a scripted model ------------------------------------------------------------------------
IMG = b"\xff\xd8\xff-fake-jpeg-bytes"


def run(payloads, approver=lambda s: "approve"):
    sent = []
    client = FakeClient([submit(p) for p in payloads])
    out = process_receipt(client, IMG, "image/jpeg", "Ana Ruiz", approver, lambda s, b: sent.append((s, b)))
    return out, sent, client


def test_a_clean_receipt_is_sent_after_explicit_approval():
    out, sent, client = run([receipt()])
    assert out.status == "sent" and len(sent) == 1 and "Ana Ruiz" in sent[0][1]


def test_the_receipt_image_is_sent_as_a_real_image_block():
    out, sent, client = run([receipt()])
    first = client.calls[0]["messages"][0]["content"]
    image = next(b for b in first if b["type"] == "image")
    assert image["source"]["media_type"] == "image/jpeg" and base64.b64decode(image["source"]["data"]) == IMG


def test_a_flagged_claim_is_never_sent_and_the_human_is_not_even_asked():
    asked = []
    out, sent, _ = run([receipt(total="99.00")], approver=lambda s: asked.append(s) or "approve")
    assert out.status == "needs_review" and sent == [] and asked == [] and "$31.50" in out.detail


def test_a_declined_or_crashing_approver_means_nothing_is_sent():
    assert run([receipt()], approver=lambda s: "no")[1] == []
    out, sent, _ = run([receipt()], approver=lambda s: 1 / 0)
    assert out.status == "declined" and sent == []


def test_an_unreadable_extraction_is_reported_not_silently_dropped():
    bad = {"merchant": "x", "items": "not a list", "printed_total": "10"}
    out, sent, _ = run([bad, bad])
    assert out.status == "extraction_failed" and sent == [] and "invalid output" in out.detail


def test_a_garbled_amount_fails_loudly_instead_of_being_skipped():
    garbled = receipt([("Taxi", "forty five", "Transportation")])
    out, sent, _ = run([garbled, garbled])
    assert out.status == "extraction_failed" and sent == []


def test_it_works_through_telemetry_and_an_unpriced_model_does_not_break_the_claim():
    clock = Clock()
    tel = Telemetry(PriceTable([Price("other-model", "2026-01-01", 1.0, 1.0)]), clock=clock)
    run_ = tel.start_run("ana")
    sent = []
    client = TelemetryClient(FakeClient([submit(receipt())], clock, 0.5), run_)
    out = process_receipt(client, IMG, "image/jpeg", "Ana Ruiz", lambda s: "approve", lambda s, b: sent.append(b))
    rec = run_.finish("success")
    assert out.status == "sent" and rec.steps == 1 and rec.unpriced_calls == 1 and rec.latency_ms == 500.0
