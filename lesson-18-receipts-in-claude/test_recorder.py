import json
import threading

import pytest

from harness import NOW, SimulatedCrash, setup
from keys import KeyRegistry, Signer
from receipt import INTENT, RESULT, open_commit, receipt_hash
from recorder import ChainFileError, ChainWriter, Indeterminate, SaltStore, ToolRecorder, read_chain
from verify import ChainVerifier


@pytest.fixture
def env(tmp_path):
    agent, ckpt, reg, creg, writer, rec = setup(str(tmp_path))
    return agent, reg, writer, rec


def lines(path):
    return [ln for ln in open(path, "rb").read().splitlines() if ln.strip()]


# --- the writer -------------------------------------------------------------------------------------------------------------------------
def test_the_first_receipt_has_sequence_zero_and_no_predecessor_and_each_next_links_to_the_last(env):
    _, _, w, rec = env
    a = w.append({"type": INTENT, "tool": "t", "args_commit": "c", "policy_id": "p", "policy_hash": "h", "approval_ref": None, "action_digest": None})
    b = w.append({"type": INTENT, "tool": "t", "args_commit": "c", "policy_id": "p", "policy_hash": "h", "approval_ref": None, "action_digest": None})
    assert (a["seq"], a["prev"], b["seq"], b["prev"]) == (0, None, 1, receipt_hash(a)) and w.length == 2 and w.head == receipt_hash(b)


def test_every_receipt_names_the_agent_run_version_and_model_and_uses_the_clock(env):
    _, _, w, rec = env
    rec.call("t", lambda **a: 1, {})
    for r in read_chain(w.path):
        assert (r["agent_id"], r["run_id"], r["agent_version"], r["model_id"], r["iat"]) == ("support-bot", "run-1", "v-3f9a2c", "claude-sonnet-5-5", NOW)
    assert len({r["receipt_id"] for r in read_chain(w.path)}) == 2


def test_a_receipt_is_on_disk_when_append_returns(env):
    _, _, w, rec = env
    r = w.append({"type": INTENT, "tool": "t", "args_commit": "c", "policy_id": "p", "policy_hash": "h", "approval_ref": None, "action_digest": None})
    assert json.loads(lines(w.path)[-1]) == r and len(lines(w.path)) == 1


def test_a_chain_needs_an_agent_key(tmp_path):
    with pytest.raises(ValueError):
        ChainWriter(tmp_path / "c.jsonl", Signer.generate("approver"), agent_id="a", run_id="r", agent_version="v", model_id="m")


def test_a_writer_resumes_an_existing_file_where_it_left_off_and_the_whole_chain_still_verifies(tmp_path):
    agent, reg = Signer.generate("agent"), KeyRegistry()
    reg.add_signer(agent)
    path = tmp_path / "c.jsonl"
    kw = dict(agent_id="a", run_id="r", agent_version="v", model_id="m", clock=lambda: NOW, fsync=False)
    ToolRecorder(ChainWriter(path, agent, **kw), "p", "h").call("t", lambda **a: 1, {})
    w2 = ChainWriter(path, agent, **kw)
    assert w2.length == 2
    ToolRecorder(w2, "p", "h").call("t", lambda **a: 2, {})
    rep = ChainVerifier(reg).verify(read_chain(path))
    assert rep.valid and rep.complete and rep.length == 4


def test_an_unreadable_last_line_is_refused_and_the_file_is_not_touched(tmp_path):
    agent = Signer.generate("agent")
    path = tmp_path / "c.jsonl"
    path.write_bytes(b'{"seq": 0}\nthis is not json\n')
    before = path.read_bytes()
    for bad in (b'this is not json\n', b'[1,2]\n', b'{"seq": "x"}\n'):
        path.write_bytes(bad)
        with pytest.raises(ChainFileError):
            ChainWriter(path, agent, agent_id="a", run_id="r", agent_version="v", model_id="m")
        assert path.read_bytes() == bad
    path.write_bytes(b"")
    assert ChainWriter(path, agent, agent_id="a", run_id="r", agent_version="v", model_id="m").length == 0


def test_threads_never_get_the_same_sequence_number_and_the_chain_stays_valid(env):
    agent, reg, w, rec = env
    barrier = threading.Barrier(8)

    def work(k):
        barrier.wait()
        for i in range(20):
            rec.call("t", lambda **a: a["i"], {"k": k, "i": i})
    threads = [threading.Thread(target=work, args=(k,)) for k in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    chain = read_chain(w.path)
    assert [r["seq"] for r in chain] == list(range(len(chain))) and len(chain) == 320
    assert ChainVerifier(reg).verify(chain).complete


def test_reading_a_chain_keeps_unreadable_lines_visible_and_skips_blank_ones(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_bytes(b'{"a": 1}\n\n   \nnot json at all\n{"b": 2}\n')
    got = read_chain(p)
    assert got[0] == {"a": 1} and "unreadable" in got[1] and got[2] == {"b": 2} and len(got) == 3


# --- recording tool calls -------------------------------------------------------------------------------------------------------------
def test_a_successful_call_writes_an_intent_then_a_result_that_points_at_it(env):
    _, _, w, rec = env
    assert rec.call("add", lambda x, y: x + y, {"x": 2, "y": 3}) == 5
    intent, result = read_chain(w.path)
    assert intent["type"] == INTENT and result["type"] == RESULT and result["intent_ref"] == receipt_hash(intent) and result["outcome"] == "success" and result["tool"] == intent["tool"] == "add"
    assert intent["policy_id"] == "refund-policy" and intent["policy_hash"] == "sha256:policyhash"


def test_arguments_and_results_are_committed_and_the_salt_opens_them(env):
    _, _, w, rec = env
    rec.call("add", lambda x, y: {"sum": x + y}, {"x": 2, "y": 3})
    intent, result = read_chain(w.path)
    assert open_commit({"x": 2, "y": 3}, rec.salts.get(intent["receipt_id"], "args"), intent["args_commit"])
    assert open_commit({"sum": 5}, rec.salts.get(result["receipt_id"], "result"), result["result_commit"])
    assert not open_commit({"x": 2, "y": 4}, rec.salts.get(intent["receipt_id"], "args"), intent["args_commit"])


def test_neither_the_arguments_nor_the_result_nor_a_salt_appears_in_the_chain_file(env):
    _, _, w, rec = env
    rec.call("lookup", lambda **a: {"card": "4111-SECRET-RESULT"}, {"name": "ALICE-SECRET-ARG"})
    text = open(w.path, encoding="utf8").read()
    salts = [rec.salts.get(r["receipt_id"], f) for r in read_chain(w.path) for f in ("args", "result") if rec.salts.get(r["receipt_id"], f)]
    assert "ALICE-SECRET-ARG" not in text and "4111-SECRET-RESULT" not in text and salts and all(s.hex() not in text for s in salts)


def test_the_same_arguments_get_a_fresh_salt_each_time_so_commitments_do_not_match(env):
    _, _, w, rec = env
    rec.call("t", lambda **a: 1, {"x": 1})
    rec.call("t", lambda **a: 1, {"x": 1})
    intents = [r for r in read_chain(w.path) if r["type"] == INTENT]
    assert intents[0]["args_commit"] != intents[1]["args_commit"]


def test_a_tool_that_raises_gets_a_failure_receipt_and_the_error_reaches_the_caller(env):
    _, _, w, rec = env

    def boom(**a):
        raise KeyError("nope")
    with pytest.raises(KeyError):
        rec.call("t", boom, {})
    intent, result = read_chain(w.path)
    assert result["outcome"] == "failure" and result["error_type"] == "KeyError" and result["result_commit"] is None and result["intent_ref"] == receipt_hash(intent)


def test_an_indeterminate_outcome_is_recorded_as_such_and_re_raised(env):
    _, _, w, rec = env

    def unsure(**a):
        raise Indeterminate("timed out after sending")
    with pytest.raises(Indeterminate):
        rec.call("pay", unsure, {})
    assert read_chain(w.path)[-1]["outcome"] == "indeterminate"


def test_if_the_process_dies_after_the_effect_the_intent_is_there_and_no_result_is_written(env):
    agent, reg, w, rec = env
    effects = []

    def effect(**a):
        effects.append(1)
        raise SimulatedCrash()
    with pytest.raises(SimulatedCrash):
        rec.call("book", effect, {"flight": "QF11"})
    chain = read_chain(w.path)
    rep = ChainVerifier(reg).verify(chain)
    assert effects == [1] and len(chain) == 1 and chain[0]["type"] == INTENT and rep.valid and not rep.complete and rep.unresolved == [0]


def test_a_result_that_cannot_be_committed_is_still_recorded_as_a_success(env):
    _, _, w, rec = env
    rec.call("t", lambda **a: object(), {})
    result = read_chain(w.path)[-1]
    assert result["outcome"] == "success" and result["result_commit"] is None and result["error_type"] == "UncommittableResult"


def test_arguments_with_floats_and_decimals_are_recorded_without_error(env):
    from decimal import Decimal
    _, _, w, rec = env
    rec.call("t", lambda **a: Decimal("1.5"), {"price": 18.5, "tax": Decimal("1.10")})
    intent, result = read_chain(w.path)
    assert open_commit({"price": 18.5, "tax": Decimal("1.10")}, rec.salts.get(intent["receipt_id"], "args"), intent["args_commit"]) and result["result_commit"]


def test_an_approval_reference_and_action_digest_are_carried_on_the_intent(env):
    _, _, w, rec = env
    rec.call("refund", lambda **a: 1, {"amount": 5}, approval="sha256:aaaa", action_digest="sha256:bbbb")
    intent = read_chain(w.path)[0]
    assert intent["approval_ref"] == "sha256:aaaa" and intent["action_digest"] == "sha256:bbbb"


def test_the_salt_store_keeps_salts_apart_by_receipt_and_field():
    s = SaltStore()
    s.put("r1", "args", b"A" * 16)
    s.put("r1", "result", b"B" * 16)
    assert s.get("r1", "args") == b"A" * 16 and s.get("r1", "result") == b"B" * 16 and s.get("r2", "args") is None
