import json

import pytest

from fakes import Clock, FakeClient, reply
from sampling import SamplingPolicy
from telemetry import (JsonlExporter, MemoryExporter, Price, PriceTable, Telemetry, TelemetryClient, UnknownModel)

PRICES = PriceTable([Price("m", "2026-01-01", 3.0, 15.0, 0.3), Price("m", "2026-06-01", 6.0, 30.0, 0.6),
                     Price("nocache", "2026-01-01", 2.0, 10.0)], today=lambda: "2026-07-01")


def make(**kw):
    clock = Clock()
    summaries, traces = MemoryExporter(), MemoryExporter()
    t = Telemetry(PRICES, salt=b"salt", clock=clock, summary_sinks=[summaries], trace_exporters=[traces],
                  id_factory=iter(f"run{i}" for i in range(100)).__next__, **kw)
    return t, clock, summaries, traces


# --- price table --------------------------------------------------------------------------------
def test_the_price_in_effect_on_the_date_is_used():
    assert PRICES.cost("m", 1000, 500, on="2026-03-01") == pytest.approx((1000 * 3 + 500 * 15) / 1e6)
    assert PRICES.cost("m", 1000, 500) == pytest.approx((1000 * 6 + 500 * 30) / 1e6)  # today is 2026-07-01: the newer price


def test_cached_tokens_are_billed_at_the_cached_rate_or_the_input_rate_if_none_is_set():
    assert PRICES.cost("m", 1000, 500, cached_tokens=2000) == pytest.approx((1000 * 6 + 2000 * 0.6 + 500 * 30) / 1e6)
    assert PRICES.cost("nocache", 1000, 0, cached_tokens=1000) == pytest.approx(2000 * 2 / 1e6)


def test_an_unknown_model_or_a_date_before_any_price_raises_in_the_table():
    with pytest.raises(UnknownModel):
        PRICES.cost("mystery", 1, 1)
    with pytest.raises(UnknownModel):
        PRICES.cost("m", 1, 1, on="2025-01-01")


# --- recording ----------------------------------------------------------------------------------
def test_model_calls_become_spans_with_tokens_cost_latency_and_finish_reason():
    t, clock, summaries, traces = make()
    run = t.start_run("alice", {"prompt": "v3", "model": "m"})
    client = TelemetryClient(FakeClient([reply(1000, 500, cache_read=2000, stop="end_turn")], clock, 0.25), run)
    client.messages.create(model="m", messages=[])
    rec = run.finish("success")
    assert (rec.steps, rec.input_tokens, rec.output_tokens, rec.cached_tokens) == (1, 1000, 500, 2000)
    assert rec.cost_usd == pytest.approx((1000 * 6 + 2000 * 0.6 + 500 * 30) / 1e6)
    assert rec.spans[0]["finish_reason"] == "end_turn" and rec.spans[0]["latency_ms"] == 250.0
    assert rec.latency_ms == 250.0 and rec.versions == {"prompt": "v3", "model": "m"}


def test_a_failing_model_call_is_recorded_and_the_error_still_reaches_the_caller():
    t, clock, _, _ = make()
    run = t.start_run("alice")
    client = TelemetryClient(FakeClient([TimeoutError("upstream timed out")]), run)
    with pytest.raises(TimeoutError):
        client.messages.create(model="m", messages=[])
    rec = run.finish("error")
    assert rec.steps == 1 and "TimeoutError" in rec.errors[0] and rec.spans[0]["error"]


def test_an_unpriced_model_never_crashes_the_request_it_is_flagged():
    t, _, summaries, _ = make()
    run = t.start_run("alice")
    resp = TelemetryClient(FakeClient([reply()]), run).messages.create(model="brand-new-model", messages=[])
    assert resp.content  # the application got its answer
    rec = run.finish("success")
    assert rec.unpriced_calls == 1 and rec.cost_usd == 0.0 and rec.spans[0]["cost_usd"] is None


def test_costs_and_tokens_add_up_across_calls_and_tools_are_counted_once_each():
    t, _, _, _ = make()
    run = t.start_run("alice")
    c = TelemetryClient(FakeClient([reply(1000, 500), reply(2000, 100)]), run)
    c.messages.create(model="m", messages=[])
    run.record_tool_call("search", {"q": "x"}, {"r": 1}, 12.0)
    run.record_tool_call("search", {"q": "y"}, {"r": 2}, 8.0)
    run.record_tool_call("book", {"id": 1}, {"error": "x"}, 5.0, ok=False)
    c.messages.create(model="m", messages=[])
    rec = run.finish("success")
    assert rec.steps == 2 and rec.input_tokens == 3000 and rec.output_tokens == 600
    assert rec.cost_usd == pytest.approx((3000 * 6 + 600 * 30) / 1e6)
    assert rec.tools_used == ["book", "search"] and rec.tool_calls == 3 and rec.tool_failures == 1


# --- privacy ------------------------------------------------------------------------------------
def test_tool_arguments_results_and_errors_are_redacted_before_they_are_stored():
    t, _, _, traces = make()
    run = t.start_run("alice")
    run.record_tool_call("send", {"to": "ana.ruiz@example.com", "card": "4111 1111 1111 1111"}, {"phone": "415-555-0132"})
    run.record_model_call("m", 1, 1, error="failed for ana.ruiz@example.com")
    rec = run.finish("error")
    blob = json.dumps(rec.spans) + json.dumps(rec.errors)
    for secret in ("ana.ruiz@example.com", "4111", "415-555-0132"):
        assert secret not in blob
    assert "[EMAIL]" in blob and "[CARD]" in blob and "[PHONE]" in blob
    assert "ana.ruiz" not in json.dumps(traces.items)


def test_the_user_id_is_a_salted_stable_hash_never_the_raw_id():
    t, _, summaries, _ = make()
    assert t.pseudonymize("alice") == t.pseudonymize("alice") != t.pseudonymize("bob")
    other = Telemetry(PRICES, salt=b"different")
    assert other.pseudonymize("alice") != t.pseudonymize("alice")
    t.start_run("alice").finish("success")
    assert "alice" not in json.dumps(summaries.items)


# --- export -------------------------------------------------------------------------------------
def test_summaries_are_kept_for_every_run_but_traces_only_for_sampled_ones():
    t, _, summaries, traces = make(sampling=SamplingPolicy(rate=0.0))
    ok = t.start_run("a")
    ok.finish("success")
    bad = t.start_run("b")
    bad.finish("error")
    assert len(summaries.items) == 2 and all("spans" not in s for s in summaries.items)  # metrics are never sampled
    assert len(traces.items) == 1 and traces.items[0]["outcome"] == "error" and "outcome=error" in traces.items[0]["sample_reason"]


def test_a_failing_exporter_never_breaks_a_request():
    class Broken:
        def export(self, item):
            raise OSError("disk full")

    t = Telemetry(PRICES, summary_sinks=[Broken()], trace_exporters=[Broken()])
    t.start_run("a").finish("success")  # must not raise
    assert len(t.export_errors) == 2 and "disk full" in t.export_errors[0]


def test_the_jsonl_exporter_writes_one_parseable_line_per_run(tmp_path):
    path = tmp_path / "traces.jsonl"
    t = Telemetry(PRICES, trace_exporters=[JsonlExporter(path)])
    for i in range(3):
        t.start_run(f"u{i}").finish("success")
    lines = path.read_text().splitlines()
    assert len(lines) == 3 and all(json.loads(l)["outcome"] == "success" for l in lines)
