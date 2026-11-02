"""A Merkle tree over receipts (the construction of RFC 6962, certificate transparency), for compact commitments and inclusion proofs.

A checkpoint signs ONE root that commits to every receipt so far. Anyone holding a receipt, its position and a short proof can show it was in the tree without seeing the others.
Leaves and inner nodes are hashed with different prefixes (0x00 and 0x01), so an inner node can never be passed off as a leaf. Built from the RFC's description, not cross-checked
against its published test vectors.
"""

import hashlib


def _h(prefix: bytes, *parts: bytes) -> bytes:
    return hashlib.sha256(prefix + b"".join(parts)).digest()


def leaf_hash(data: bytes) -> bytes:
    return _h(b"\x00", data)


def _split(n: int) -> int:
    """The largest power of two strictly less than n."""
    k = 1
    while k * 2 < n:
        k *= 2
    return k


def root(leaves: list[bytes]) -> bytes:
    if not leaves:
        return hashlib.sha256(b"").digest()
    if len(leaves) == 1:
        return leaf_hash(leaves[0])
    k = _split(len(leaves))
    return _h(b"\x01", root(leaves[:k]), root(leaves[k:]))


def inclusion_proof(leaves: list[bytes], index: int) -> list[bytes]:
    if not 0 <= index < len(leaves):
        raise IndexError("leaf index out of range")
    if len(leaves) == 1:
        return []
    k = _split(len(leaves))
    if index < k:
        return inclusion_proof(leaves[:k], index) + [root(leaves[k:])]
    return inclusion_proof(leaves[k:], index - k) + [root(leaves[:k])]


def verify_inclusion(leaf: bytes, index: int, size: int, proof: list[bytes], expected_root: bytes) -> bool:
    if not 0 <= index < size:
        return False

    def walk(idx: int, n: int, path: list[bytes]):
        if n == 1:
            return leaf_hash(leaf) if not path else None
        if not path:
            return None
        k = _split(n)
        if idx < k:
            left = walk(idx, k, path[:-1])
            return None if left is None else _h(b"\x01", left, path[-1])
        right = walk(idx - k, n - k, path[:-1])
        return None if right is None else _h(b"\x01", path[-1], right)
    return walk(index, size, list(proof)) == expected_root
