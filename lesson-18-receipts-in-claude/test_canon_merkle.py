import hashlib
from decimal import Decimal

import pytest

from canon import CanonError, MAX_DEPTH, MAX_SAFE_INT, canonicalize, digest
from merkle import inclusion_proof, leaf_hash, root, verify_inclusion


# --- canonical JSON ------------------------------------------------------------------------------------------------------------------
def test_keys_are_sorted_and_there_is_no_whitespace():
    assert canonicalize({"b": 1, "a": 2}) == b'{"a":2,"b":1}'


def test_nested_objects_and_arrays_are_canonicalised_all_the_way_down():
    assert canonicalize({"z": [3, {"y": None, "x": True}], "a": {}}) == b'{"a":{},"z":[3,{"x":true,"y":null}]}'


def test_the_same_data_built_in_a_different_order_gives_the_same_bytes():
    a = {"one": 1, "two": {"b": 2, "a": 1}, "three": [1, 2]}
    b = {"three": [1, 2], "two": {"a": 1, "b": 2}, "one": 1}
    assert canonicalize(a) == canonicalize(b) and digest(a) == digest(b)


def test_true_and_one_are_different_documents():
    assert canonicalize(True) == b"true" and canonicalize(1) == b"1" and digest({"a": True}) != digest({"a": 1})


def test_empty_containers_and_null():
    assert canonicalize({}) == b"{}" and canonicalize([]) == b"[]" and canonicalize(None) == b"null" and canonicalize(0) == b"0" and canonicalize(-5) == b"-5"


def test_tuples_are_arrays():
    assert canonicalize((1, 2)) == b"[1,2]"


@pytest.mark.parametrize("text,expected", [
    ('a"b', b'"a\\"b"'), ("a\\b", b'"a\\\\b"'), ("a\nb", b'"a\\nb"'), ("a\tb", b'"a\\tb"'), ("a\bb", b'"a\\bb"'), ("a\fb", b'"a\\fb"'), ("a\rb", b'"a\\rb"'),
    ("\x01", b'"\\u0001"'), ("\x1f", b'"\\u001f"'), ("\x7f", b'"\x7f"'), ("é", '"é"'.encode()), ("日本", '"日本"'.encode()), (" ", '" "'.encode()), ("/", b'"/"')])
def test_strings_are_escaped_exactly_one_way(text, expected):
    assert canonicalize(text) == expected


def test_object_keys_are_sorted_by_utf16_code_units_not_by_code_point():
    """U+1F600 is the surrogate pair D83D DE00 in UTF-16, which sorts BEFORE U+E000. By code point it would sort after."""
    assert canonicalize({"": 1, "\U0001F600": 2}) == '{"\U0001F600":2,"":1}'.encode()


def test_plain_ordering_of_keys():
    assert canonicalize({"b": 1, "aa": 2, "a": 3, "B": 4}) == b'{"B":4,"a":3,"aa":2,"b":1}'


@pytest.mark.parametrize("value", [1.0, 0.5, float("nan"), float("inf"), {"price": 18.5}, [1, 2.5]])
def test_floats_are_refused_so_amounts_must_be_integers_or_strings(value):
    with pytest.raises(CanonError, match="floats"):
        canonicalize(value)


def test_integers_must_fit_exactly_in_a_json_number():
    assert canonicalize(MAX_SAFE_INT) == str(MAX_SAFE_INT).encode() and canonicalize(-MAX_SAFE_INT) == str(-MAX_SAFE_INT).encode()
    for big in (MAX_SAFE_INT + 1, -MAX_SAFE_INT - 1):
        with pytest.raises(CanonError):
            canonicalize(big)


def test_only_string_keys_and_known_types_are_accepted():
    for bad in ({1: "a"}, {None: 1}, {b"k": 1}, {"a": {1, 2}}, {"a": b"bytes"}, {"a": Decimal("1")}, object()):
        with pytest.raises(CanonError):
            canonicalize(bad)


def test_a_lone_surrogate_in_a_value_or_a_key_is_refused():
    for bad in ("\ud800", {"\ud800": 1}, {"k": "x\udfff"}):
        with pytest.raises(CanonError, match="surrogate"):
            canonicalize(bad)


def test_deep_nesting_is_refused():
    deep = 1
    for _ in range(MAX_DEPTH + 2):
        deep = [deep]
    with pytest.raises(CanonError, match="nesting"):
        canonicalize(deep)
    ok = 1
    for _ in range(10):
        ok = [ok]
    assert canonicalize(ok) == b"[" * 10 + b"1" + b"]" * 10


def test_an_enormous_document_is_refused():
    with pytest.raises(CanonError, match="larger"):
        canonicalize({"a": "x" * 1_100_000})


def test_the_digest_has_a_prefix_and_changes_when_anything_changes():
    d = digest({"a": 1})
    assert d.startswith("sha256:") and len(d) == 7 + 64 and d == "sha256:" + hashlib.sha256(b'{"a":1}').hexdigest()
    assert d != digest({"a": 2}) and d != digest({"b": 1})


# --- Merkle tree -----------------------------------------------------------------------------------------------------------------------
def leaves(n):
    return [f"receipt-{i}".encode() for i in range(n)]


def h1(*parts):
    return hashlib.sha256(b"\x01" + b"".join(parts)).digest()


def test_an_empty_tree_and_a_single_leaf_have_defined_roots():
    assert root([]) == hashlib.sha256(b"").digest() and root([b"a"]) == leaf_hash(b"a") == hashlib.sha256(b"\x00a").digest()


def test_two_and_three_leaf_trees_have_the_documented_shape():
    a, b, c = (leaf_hash(x) for x in (b"a", b"b", b"c"))
    assert root([b"a", b"b"]) == h1(a, b) and root([b"a", b"b", b"c"]) == h1(h1(a, b), c)


def test_leaf_order_matters():
    assert root([b"a", b"b"]) != root([b"b", b"a"])


def test_an_inner_node_cannot_be_passed_off_as_a_leaf():
    l, r = leaf_hash(b"a"), leaf_hash(b"b")
    assert root([l + r]) != h1(l, r) and root([b"a", b"b"]) != root([l + r])


@pytest.mark.parametrize("size", range(1, 18))
def test_every_leaf_of_every_small_tree_has_a_proof_that_verifies(size):
    ls = leaves(size)
    r = root(ls)
    for i in range(size):
        assert verify_inclusion(ls[i], i, size, inclusion_proof(ls, i), r), (size, i)


def test_a_proof_for_the_wrong_leaf_index_size_root_or_a_changed_proof_fails():
    ls = leaves(9)
    r = root(ls)
    proof = inclusion_proof(ls, 4)
    assert verify_inclusion(ls[4], 4, 9, proof, r)
    assert not verify_inclusion(b"other", 4, 9, proof, r)
    assert not verify_inclusion(ls[4], 5, 9, proof, r)
    assert not verify_inclusion(ls[4], 4, 17, proof, r)  # a size that changes the length of the proof
    assert not verify_inclusion(ls[4], 4, 9, proof, root(leaves(8)))
    assert not verify_inclusion(ls[4], 4, 9, proof[:-1], r) and not verify_inclusion(ls[4], 4, 9, proof + [b"x" * 32], r)
    assert not verify_inclusion(ls[4], 4, 9, [bytes(32)] + proof[1:], r)


def test_out_of_range_indexes():
    ls = leaves(4)
    with pytest.raises(IndexError):
        inclusion_proof(ls, 4)
    with pytest.raises(IndexError):
        inclusion_proof(ls, -1)
    assert not verify_inclusion(ls[0], 4, 4, [], root(ls)) and not verify_inclusion(ls[0], -1, 4, [], root(ls))


def test_a_single_leaf_proof_is_empty_and_a_non_empty_proof_for_it_fails():
    assert inclusion_proof([b"a"], 0) == [] and verify_inclusion(b"a", 0, 1, [], root([b"a"])) and not verify_inclusion(b"a", 0, 1, [bytes(32)], root([b"a"]))


def test_changing_any_leaf_changes_the_root():
    ls = leaves(11)
    base = root(ls)
    for i in range(11):
        changed = list(ls)
        changed[i] = b"changed"
        assert root(changed) != base


def test_the_tree_size_is_not_proved_by_the_proof_so_it_must_come_from_the_signed_checkpoint():
    """A leaf in the left half cannot tell whether its right sibling hash covers one leaf or two: the size-9 proof still verifies as 'size 10'. Documented, not fixed:
    the checkpoint signs `length` together with the root, and that is where the size is taken from."""
    ls = leaves(9)
    assert verify_inclusion(ls[4], 4, 10, inclusion_proof(ls, 4), root(ls))
    assert root(leaves(10)) != root(ls)


@pytest.mark.parametrize("size", [2, 4, 5, 8])
def test_a_real_proof_cannot_be_replayed_at_an_index_outside_the_tree(size):
    """Without a range check, a negative or too-large index walks the same path as a valid one and would verify a proof built for another position."""
    ls = leaves(size)
    r = root(ls)
    for i in range(size):
        proof = inclusion_proof(ls, i)
        assert verify_inclusion(ls[i], i, size, proof, r)
        assert not verify_inclusion(ls[i], i - size, size, proof, r)
        assert not verify_inclusion(ls[i], i + size, size, proof, r)
        assert not verify_inclusion(ls[i], -1, size, proof, r)
