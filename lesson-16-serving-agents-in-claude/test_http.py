import http.client
import json
import threading

import pytest

from admission import Admission
from agentdef import AgentResult
from harness import Provider, make_service
from http_app import MAX_BODY, Authenticator, make_server

TOKENS = {"token-alice": "alice", "token-bob": "bob"}
ADMIN = "admin-secret"


class Running:
    def __init__(self, service):
        self.service = service
        self.server = make_server(service, Authenticator(TOKENS), ADMIN)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def request(self, method, path, body=None, headers=None, raw=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        data = raw if raw is not None else (json.dumps(body) if body is not None else None)
        conn.request(method, path, body=data, headers=headers or {})
        r = conn.getresponse()
        text = r.read().decode()
        out = (r.status, dict(r.getheaders()), text)
        conn.close()
        return out

    def chat(self, body, token="token-alice", **kw):
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"} if token else {"Content-Type": "application/json"}
        status, hdrs, text = self.request("POST", "/v1/chat", body, headers, **kw)
        return status, hdrs, json.loads(text)

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def app():
    svc = make_service(Provider(responder=lambda call: AgentResult(f"hello {call.user_id}", (), 10, 5)), sleep=lambda s: None)
    a = Running(svc)
    yield a
    a.stop()


def test_a_signed_in_user_gets_an_answer_a_trace_header_and_a_thread(app):
    status, headers, body = app.chat({"text": "hi"})
    assert status == 200 and body["status"] == "ok" and body["text"] == "hello alice" and body["thread_id"] and headers["X-Trace-Id"] == body["trace_id"]


def test_who_the_caller_is_comes_from_the_token_and_a_user_id_in_the_body_is_ignored():
    svc = make_service(Provider(responder=lambda call: AgentResult(f"hello {call.user_id}", (), 10, 5)), sleep=lambda s: None)
    a = Running(svc)
    try:
        _, _, body = a.chat({"text": "hi", "user_id": "bob"}, token="token-alice")
        assert body["text"] == "hello alice"
        _, _, body = a.chat({"text": "hi again", "user_id": "alice"}, token="token-bob")
        assert body["text"] == "hello bob"
    finally:
        a.stop()


def test_a_missing_wrong_or_malformed_token_is_401_and_the_model_is_never_called(app):
    for headers in ({}, {"Authorization": "Bearer nope"}, {"Authorization": "token-alice"}, {"Authorization": "Basic abc"}):
        status, hdrs, text = app.request("POST", "/v1/chat", {"text": "hi"}, headers)
        assert status == 401 and hdrs.get("WWW-Authenticate") == "Bearer"
    assert app.service.agent_fn.calls == 0


def test_bad_json_and_non_object_bodies_are_400(app):
    h = {"Authorization": "Bearer token-alice"}
    assert app.request("POST", "/v1/chat", raw="{not json", headers=h)[0] == 400
    assert app.request("POST", "/v1/chat", raw="[1, 2]", headers=h)[0] == 400


def test_invalid_fields_come_back_as_400_with_a_short_reason(app):
    status, _, body = app.chat({"text": ""})
    assert status == 400 and body["status"] == "invalid" and "detail" in body


def test_an_oversized_body_is_413_after_a_bounded_drain_and_the_connection_closes(app):
    h = {"Authorization": "Bearer token-alice", "Content-Type": "application/json"}
    status, hdrs, text = app.request("POST", "/v1/chat", raw="x" * (MAX_BODY + 1), headers=h)
    assert status == 413 and hdrs.get("Connection") == "close" and app.service.agent_fn.calls == 0


def test_a_request_without_a_length_is_411():
    svc = make_service(Provider(), sleep=lambda s: None)
    a = Running(svc)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", a.port, timeout=5)
        conn.putrequest("POST", "/v1/chat")
        conn.putheader("Authorization", "Bearer token-alice")
        conn.endheaders()
        assert conn.getresponse().status == 411
    finally:
        a.stop()


def test_a_thread_can_be_continued_over_http_and_belongs_to_its_owner(app):
    _, _, first = app.chat({"text": "one"})
    status, _, second = app.chat({"text": "two", "thread_id": first["thread_id"]})
    assert status == 200 and second["thread_id"] == first["thread_id"]
    status, _, stolen = app.chat({"text": "peek", "thread_id": first["thread_id"]}, token="token-bob")
    assert status == 404 and stolen["status"] == "not_found"


def test_an_overloaded_server_answers_429_with_retry_after():
    gate, started = threading.Event(), threading.Event()

    def slow(call):
        started.set()
        gate.wait(3)
        return AgentResult("x", (), 1, 1)
    svc = make_service(Provider(responder=slow), admission=Admission(max_concurrent=1, max_queue=0, per_user_max=5), sleep=lambda s: None)
    a = Running(svc)
    try:
        t = threading.Thread(target=lambda: a.chat({"text": "first"}, token="token-alice"))
        t.start()
        started.wait(2)
        status, headers, body = a.chat({"text": "second"}, token="token-bob")
        assert status == 429 and int(headers["Retry-After"]) >= 1 and body["detail"] == "queue_full"
        gate.set()
        t.join(3)
    finally:
        gate.set()
        a.stop()


def test_probes_healthz_is_alive_readyz_follows_readiness(app):
    assert app.request("GET", "/healthz")[:1] == (200,) and app.request("GET", "/readyz")[0] == 200
    app.service.drain(1)
    status, _, text = app.request("GET", "/readyz")
    assert status == 503 and json.loads(text)["reasons"] == ["draining"]
    assert app.request("GET", "/healthz")[0] == 200  # still alive while draining
    status, headers, _ = app.request("POST", "/v1/chat", {"text": "late"}, {"Authorization": "Bearer token-alice"})
    assert status == 503 and "Retry-After" in headers


def test_metrics_need_the_admin_token(app):
    app.chat({"text": "hi"})
    assert app.request("GET", "/metrics")[0] == 403 and app.request("GET", "/metrics", headers={"X-Admin-Token": "wrong"})[0] == 403
    status, _, text = app.request("GET", "/metrics", headers={"X-Admin-Token": ADMIN})
    assert status == 200 and "requests{status=ok} 1" in text


def test_unknown_paths_are_404_and_error_bodies_never_contain_a_traceback_or_user_text():
    svc = make_service(Provider(responder=lambda call: (_ for _ in ()).throw(RuntimeError("SECRET internal failure"))), sleep=lambda s: None)
    a = Running(svc)
    try:
        assert a.request("GET", "/nope")[0] == 404 and a.request("POST", "/other", {"text": "x"})[0] == 404
        status, _, text = a.request("POST", "/v1/chat", {"text": "SECRET-USER-TEXT"}, {"Authorization": "Bearer token-alice"})
        assert status == 500 and "SECRET" not in text and "Traceback" not in text
    finally:
        a.stop()


def test_the_authenticator_holds_only_hashes_of_tokens():
    auth = Authenticator({"super-secret-token": "carol"})
    assert "super-secret-token" not in repr(auth.__dict__) and auth.user_for("Bearer super-secret-token") == "carol" and auth.user_for("Bearer super-secret-token2") is None


def test_a_non_numeric_deadline_falls_back_to_the_default_instead_of_crashing(app):
    status, _, body = app.chat({"text": "hi", "deadline_s": "soon"})
    assert status == 200 and body["status"] == "ok"


def test_if_the_service_itself_blows_up_the_front_end_still_answers_500_instead_of_dropping_the_connection(monkeypatch):
    svc = make_service(Provider(), sleep=lambda s: None)
    monkeypatch.setattr(svc, "handle", lambda req: (_ for _ in ()).throw(RuntimeError("SECRET")))
    a = Running(svc)
    try:
        status, _, text = a.request("POST", "/v1/chat", {"text": "hi"}, {"Authorization": "Bearer token-alice"})
        assert status == 500 and "SECRET" not in text
    finally:
        a.stop()


def test_response_bodies_contain_exactly_the_documented_fields(app):
    _, _, ok = app.chat({"text": "hi"})
    assert set(ok) == {"status", "text", "thread_id", "version", "tier", "cached", "trace_id"}
    _, _, bad = app.chat({"text": ""})
    assert set(bad) == {"status", "text", "thread_id", "version", "tier", "cached", "trace_id", "detail"}
