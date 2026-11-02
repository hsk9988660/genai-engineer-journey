"""Signing keys, and the registry a verifier PINS.

The lesson's key point, built into the types: a receipt names its signing key by `kid`, it does not carry the key. The verifier looks the `kid` up in a registry it obtained some other
way (a file you ship, a config you control, a published key set you pinned in advance). A receipt that arrives with its own key proves only that it agrees with itself.

Each key has a ROLE (agent, approver, checkpoint), so a key that may sign tool-call receipts cannot sign an approval, and a window of validity. `resolve` says exactly why a key is not
acceptable: unknown_key, wrong_role, not_yet_valid, expired, revoked.

LIMIT: validity and revocation are judged against the time the SIGNER wrote in the receipt, which a holder of a stolen old key can backdate. Checkpoints signed by a different key and
anchored elsewhere narrow that, they do not remove it.
"""

import hashlib
from dataclasses import dataclass

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

ROLES = ("agent", "approver", "checkpoint")


class KeyProblem(Exception):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def kid_for(public_raw: bytes) -> str:
    return "k_" + hashlib.sha256(public_raw).hexdigest()[:16]


class Signer:
    """Holds a private key. In production this would be a call to a key vault or an HSM, and the private bytes would never be in this process."""

    def __init__(self, private: Ed25519PrivateKey, role: str):
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        self._private, self.role = private, role
        self.public_bytes = private.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        self.kid = kid_for(self.public_bytes)

    @classmethod
    def generate(cls, role: str) -> "Signer":
        return cls(Ed25519PrivateKey.generate(), role)

    @classmethod
    def from_seed(cls, seed: bytes, role: str) -> "Signer":
        """Deterministic key from 32 bytes. For tests and fixtures only: a seed in source code is not a secret."""
        return cls(Ed25519PrivateKey.from_private_bytes(seed), role)

    def sign(self, data: bytes) -> bytes:
        return self._private.sign(data)


@dataclass
class KeyEntry:
    kid: str
    public: Ed25519PublicKey
    role: str
    valid_from: int
    valid_to: int | None
    revoked_at: int | None


class KeyRegistry:
    def __init__(self):
        self._keys: dict[str, KeyEntry] = {}

    def add(self, public_raw: bytes, role: str, valid_from: int = 0, valid_to: int | None = None) -> str:
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        if not isinstance(public_raw, (bytes, bytearray)) or len(public_raw) != 32:
            raise ValueError("an Ed25519 public key is exactly 32 bytes")
        if valid_to is not None and valid_to <= valid_from:
            raise ValueError("valid_to must be after valid_from")
        kid = kid_for(bytes(public_raw))
        if kid in self._keys:
            raise ValueError("that key is already registered")
        self._keys[kid] = KeyEntry(kid, Ed25519PublicKey.from_public_bytes(bytes(public_raw)), role, valid_from, valid_to, None)
        return kid

    def add_signer(self, signer: Signer, valid_from: int = 0, valid_to: int | None = None) -> str:
        return self.add(signer.public_bytes, signer.role, valid_from, valid_to)

    def revoke(self, kid: str, at: int) -> None:
        if kid not in self._keys:
            raise KeyError(kid)
        self._keys[kid].revoked_at = at

    def resolve(self, kid, role: str, at: int) -> KeyEntry:
        entry = self._keys.get(kid) if isinstance(kid, str) else None
        if entry is None:
            raise KeyProblem("unknown_key", "no registered key has that id")
        if entry.role != role:
            raise KeyProblem("wrong_role", f"this key is registered as {entry.role!r}, not {role!r}")
        if at < entry.valid_from:
            raise KeyProblem("key_not_yet_valid")
        if entry.valid_to is not None and at >= entry.valid_to:
            raise KeyProblem("key_expired")
        if entry.revoked_at is not None and at >= entry.revoked_at:
            raise KeyProblem("key_revoked")
        return entry

    def verify(self, kid, role: str, at: int, message: bytes, signature: bytes) -> KeyEntry:
        entry = self.resolve(kid, role, at)
        try:
            entry.public.verify(signature, message)
        except InvalidSignature:
            raise KeyProblem("bad_signature") from None
        return entry
