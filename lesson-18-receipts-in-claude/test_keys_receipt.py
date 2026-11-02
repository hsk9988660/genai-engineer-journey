import copy
from decimal import Decimal

import pytest

from canon import canonicalize
from keys import KeyProblem, KeyRegistry, ROLES, Signer, kid_for
from receipt import (APPROVAL, CHECKPOINT, INTENT, RESULT, ROLE_OF, b64u, b64u_decode, commit, new_salt, open_commit, receipt_hash, shape_problems, sign, signed_bytes, to_tree,
                     verify_signature)

AGENT = Signer.from_seed(b"a" * 32, "agent")
OTHER = Signer.from_seed(b"b" * 32, "agent")


def reg(*signers):
    r = KeyRegistry()
    for s in signers:
        r.add_signer(s)
    return r


def intent(**over):
    p = {"type": INTENT, "iat": 100, "receipt_id": "x1", "agent_id": "bot", "run_id": "r1", "agent_version": "v1", "model_id": "m1", "seq": 0, "prev": None, "tool": "t", "args_commit": "c1:aa",
         "policy_id": "p", "policy_hash": "h", "approval_ref": None, "action_digest": None}
    p.update(over)
    return p


# --- signers and the registry --------------------------------------------------------------------------------------------------------
def test_a_signer_has_a_role_a_stable_key_id_and_a_32_byte_public_key():
    assert AGENT.role == "agent" and len(AGENT.public_bytes) == 32 and AGENT.kid == kid_for(AGENT.public_bytes) and AGENT.kid.startswith("k_") and len(AGENT.kid) == 18


def test_a_seeded_signer_is_deterministic_and_generated_ones_are_not():
    assert Signer.from_seed(b"a" * 32, "agent").kid == AGENT.kid
    assert Signer.generate("agent").kid != Signer.generate("agent").kid


def test_only_known_roles_are_allowed():
    with pytest.raises(ValueError):
        Signer.generate("admin")
    with pytest.raises(ValueError):
        KeyRegistry().add(AGENT.public_bytes, "admin")
    assert ROLES == ("agent", "approver", "checkpoint")


def test_a_registered_key_resolves_in_its_role_inside_its_window():
    r = KeyRegistry()
    kid = r.add_signer(AGENT, valid_from=100, valid_to=200)
    assert r.resolve(kid, "agent", 100).kid == kid and r.resolve(kid, "agent", 199).kid == kid


@pytest.mark.parametrize("kid,role,at,reason", [("k_nope", "agent", 150, "unknown_key"), (None, "agent", 150, "unknown_key"), (5, "agent", 150, "unknown_key"),
                                                ("KID", "approver", 150, "wrong_role"), ("KID", "agent", 99, "key_not_yet_valid"), ("KID", "agent", 200, "key_expired"), ("KID", "agent", 500, "key_expired")])
def test_a_key_is_refused_with_the_specific_reason(kid, role, at, reason):
    r = KeyRegistry()
    real = r.add_signer(AGENT, valid_from=100, valid_to=200)
    with pytest.raises(KeyProblem) as e:
        r.resolve(real if kid == "KID" else kid, role, at)
    assert e.value.reason == reason


def test_a_revoked_key_is_refused_from_the_revocation_time_on_but_not_before():
    r = KeyRegistry()
    kid = r.add_signer(AGENT)
    r.revoke(kid, 500)
    assert r.resolve(kid, "agent", 499)
    with pytest.raises(KeyProblem) as e:
        r.resolve(kid, "agent", 500)
    assert e.value.reason == "key_revoked"
    with pytest.raises(KeyError):
        r.revoke("k_nope", 1)


def test_registration_mistakes_are_rejected():
    r = KeyRegistry()
    r.add_signer(AGENT)
    for call in (lambda: r.add_signer(AGENT), lambda: r.add(b"short", "agent"), lambda: r.add("x" * 32, "agent"), lambda: r.add(OTHER.public_bytes, "agent", 100, 100), lambda: r.add(OTHER.public_bytes, "agent", 100, 50)):
        with pytest.raises(ValueError):
            call()


def test_verify_accepts_a_good_signature_and_names_a_bad_one():
    r = reg(AGENT)
    sig = AGENT.sign(b"message")
    assert r.verify(AGENT.kid, "agent", 1, b"message", sig).kid == AGENT.kid
    for msg, s in ((b"other", sig), (b"message", OTHER.sign(b"message")), (b"message", b"\x00" * 64)):
        with pytest.raises(KeyProblem) as e:
            r.verify(AGENT.kid, "agent", 1, msg, s)
        assert e.value.reason == "bad_signature"


# --- encodings ------------------------------------------------------------------------------------------------------------------------
def test_base64url_round_trips_without_padding():
    for data in (b"", b"a", b"ab", b"abc", bytes(range(256))):
        s = b64u(data)
        assert "=" not in s and "+" not in s and "/" not in s and b64u_decode(s) == data


@pytest.mark.parametrize("bad", ["YQ==", "YQ+/", "YQ\n", "Y Q", " YQ", "YR", "YQ.", 5, None, b"YQ"])
def test_only_the_one_canonical_spelling_decodes(bad):
    with pytest.raises((ValueError, TypeError)):
        b64u_decode(bad)


# --- turning values into canonical trees ---------------------------------------------------------------------------------------------
def test_floats_decimals_bytes_and_big_ints_become_tagged_strings_instead_of_being_lost():
    assert to_tree(1.5) == "float:1.5" and to_tree(1.0) == "float:1.0" and to_tree(Decimal("1.10")) == "decimal:1.10" and to_tree(b"\x00\xff") == "bytes:" + b64u(b"\x00\xff")
    assert to_tree(2 ** 60) == f"int:{2 ** 60}" and to_tree(7) == 7 and to_tree(True) is True and to_tree(None) is None


def test_containers_are_converted_recursively_and_keys_become_strings():
    assert to_tree({1: (1.5, [b"x"])}) == {"1": ["float:1.5", ["bytes:" + b64u(b"x")]]}


def test_nan_infinity_unknown_types_and_extreme_depth_cannot_be_recorded():
    for bad in (float("nan"), float("inf"), object(), {1, 2}):
        with pytest.raises(TypeError):
            to_tree(bad)
    deep = 1
    for _ in range(40):
        deep = [deep]
    with pytest.raises(TypeError, match="deeply"):
        to_tree(deep)


def test_an_integer_one_and_a_float_one_are_recorded_differently():
    assert canonicalize(to_tree(1)) != canonicalize(to_tree(1.0))


# --- commitments ------------------------------------------------------------------------------------------------------------------------
def test_a_commitment_is_deterministic_for_the_same_value_and_salt_and_ignores_key_order():
    s = new_salt()
    assert commit({"a": 1, "b": 2}, s) == commit({"b": 2, "a": 1}, s) and commit("x", s).startswith("c1:")


def test_a_different_salt_or_value_gives_a_different_commitment():
    s1, s2 = new_salt(), new_salt()
    assert commit("x", s1) != commit("x", s2) and commit("x", s1) != commit("y", s1)


def test_the_commitment_does_not_contain_the_value():
    assert "SECRET-VALUE" not in commit({"k": "SECRET-VALUE"}, new_salt())


def test_a_salt_must_be_long_enough():
    for bad in (b"", b"short", "string" * 4, None):
        with pytest.raises(ValueError):
            commit("x", bad)


def test_the_owner_can_open_a_commitment_and_nobody_else_can_with_the_wrong_salt_or_value():
    s = new_salt()
    c = commit({"order": "A1"}, s)
    assert open_commit({"order": "A1"}, s, c) is True
    assert open_commit({"order": "A2"}, s, c) is False and open_commit({"order": "A1"}, new_salt(), c) is False
    assert open_commit({"order": "A1"}, b"short", c) is False and open_commit(object(), s, c) is False and open_commit({"order": "A1"}, s, "c1:00") is False


# --- signing and verifying ------------------------------------------------------------------------------------------------------------
def test_a_signed_receipt_carries_a_key_id_and_an_algorithm_inside_the_signed_bytes_and_no_public_key():
    r = sign(intent(), AGENT)
    assert r["kid"] == AGENT.kid and r["alg"] == "EdDSA" and "public_key" not in r and "sig" in r
    assert b'"kid"' in signed_bytes(r) and b'"alg"' in signed_bytes(r) and b"sig" not in signed_bytes(r).replace(b"signed", b"")


def test_a_genuine_receipt_verifies_against_the_pinned_registry():
    assert verify_signature(sign(intent(), AGENT), reg(AGENT)).kid == AGENT.kid


def test_changing_any_signed_field_breaks_the_signature():
    base = sign(intent(), AGENT)
    for field, value in (("tool", "other"), ("seq", 1), ("prev", "sha256:00"), ("iat", 101), ("model_id", "m2"), ("agent_version", "v2"), ("policy_hash", "h2"), ("args_commit", "c1:bb"), ("kid", OTHER.kid)):
        forged = {**base, field: value}
        with pytest.raises(KeyProblem) as e:
            verify_signature(forged, reg(AGENT, OTHER))
        assert e.value.reason == "bad_signature", field


def test_adding_a_field_after_signing_breaks_the_signature_so_an_embedded_public_key_cannot_be_smuggled_in():
    forged = {**sign(intent(), AGENT), "public_key": "AAAA"}
    with pytest.raises(KeyProblem) as e:
        verify_signature(forged, reg(AGENT))
    assert e.value.reason == "bad_signature"


def test_a_receipt_signed_with_a_key_that_is_not_registered_is_refused_even_though_it_is_self_consistent():
    forged = sign(intent(), OTHER)
    with pytest.raises(KeyProblem) as e:
        verify_signature(forged, reg(AGENT))
    assert e.value.reason == "unknown_key"


def test_a_downgraded_or_missing_algorithm_signature_or_a_padded_signature_is_refused():
    base = sign(intent(), AGENT)
    cases = [({**base, "alg": "none"}, "algorithm"), ({k: v for k, v in base.items() if k != "sig"}, "malformed"), ({**base, "sig": base["sig"] + "=="}, "malformed"), ({**base, "sig": 5}, "malformed")]
    for forged, reason in cases:
        with pytest.raises(KeyProblem) as e:
            verify_signature(forged, reg(AGENT))
        assert e.value.reason == reason


def test_a_key_registered_for_another_role_cannot_vouch_for_an_agent_receipt():
    r = KeyRegistry()
    r.add(AGENT.public_bytes, "checkpoint")
    with pytest.raises(KeyProblem) as e:
        verify_signature(sign(intent(), AGENT), r)
    assert e.value.reason == "wrong_role"


def test_the_signing_time_decides_whether_the_key_was_valid_unless_a_time_is_given():
    r = KeyRegistry()
    r.add_signer(AGENT, valid_from=0, valid_to=150)
    rec = sign(intent(iat=100), AGENT)
    assert verify_signature(rec, r)
    with pytest.raises(KeyProblem) as e:
        verify_signature(rec, r, at=200)
    assert e.value.reason == "key_expired"


def test_signing_refuses_the_wrong_role_unknown_types_and_an_existing_signature():
    for payload, signer in ((intent(), Signer.generate("approver")), (intent(type="nonsense"), AGENT), ({**intent(), "sig": "x"}, AGENT), ({"type": APPROVAL}, AGENT), ({"type": CHECKPOINT}, AGENT)):
        with pytest.raises(ValueError):
            sign(payload, signer)
    assert ROLE_OF == {INTENT: "agent", RESULT: "agent", APPROVAL: "approver", CHECKPOINT: "checkpoint"}


def test_the_receipt_hash_covers_the_signature_and_is_stable():
    r = sign(intent(), AGENT)
    assert receipt_hash(r) == receipt_hash(copy.deepcopy(r)) and receipt_hash(r) != receipt_hash({**r, "sig": b64u(b"x" * 64)})


@pytest.mark.parametrize("mutate,fragment", [(lambda r: r.pop("tool"), "missing field 'tool'"), (lambda r: r.update(seq="0"), "'seq' has the wrong type"), (lambda r: r.update(seq=True), "'seq' has the wrong type"),
                                             (lambda r: r.update(prev=5), "'prev' has the wrong type"), (lambda r: r.pop("sig"), "missing signature"), (lambda r: r.update(type="x"), "unknown receipt type")])
def test_shape_problems_name_what_is_wrong(mutate, fragment):
    r = sign(intent(), AGENT)
    mutate(r)
    assert any(fragment in p for p in shape_problems(r))


def test_non_objects_and_a_well_formed_receipt():
    assert shape_problems("text") == ["not an object"] and shape_problems(None) and shape_problems(sign(intent(), AGENT)) == []
