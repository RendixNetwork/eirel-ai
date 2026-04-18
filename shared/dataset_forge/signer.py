"""Bundle signing.

Bundles are canonicalized to a deterministic JSON byte string, sha256-hashed,
and signed with the owner hotkey. The owner-api loader then re-derives the
sha256 from the fetched bytes and verifies the signature against the public
key recorded in the ``OwnerDatasetBinding`` row.

The signing protocol is intentionally narrow so tests can pass a
``FakeKeypair`` without dragging in ``bittensor`` / substrate-interface.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol

from shared.core.evaluation_models import FamilyEvaluationBundle


class SignerError(RuntimeError):
    pass


class Keypair(Protocol):
    @property
    def ss58_address(self) -> str:
        ...

    def sign(self, data: bytes) -> bytes:
        ...

    def verify(self, data: bytes, signature: bytes) -> bool:
        ...


@dataclass(slots=True)
class SignedBundle:
    canonical_bytes: bytes
    sha256_hex: str
    signer_ss58: str
    signature_hex: str

    def to_metadata(self) -> dict[str, str]:
        return {
            "bundle_sha256": self.sha256_hex,
            "signer_ss58": self.signer_ss58,
            "signature_hex": self.signature_hex,
        }


def canonicalize_bundle(bundle: FamilyEvaluationBundle) -> bytes:
    """Produce a deterministic JSON byte string for hashing/signing.

    Uses ``sort_keys=True`` and ``separators=(",", ":")`` so two semantically
    identical bundles always produce the exact same bytes regardless of dict
    ordering at the source.
    """
    payload = bundle.model_dump(mode="json")
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sign_bundle(bundle: FamilyEvaluationBundle, keypair: Keypair) -> SignedBundle:
    canonical = canonicalize_bundle(bundle)
    digest = sha256_hex(canonical)
    try:
        signature = keypair.sign(canonical)
    except Exception as exc:
        raise SignerError(f"keypair.sign() failed: {exc}") from exc
    return SignedBundle(
        canonical_bytes=canonical,
        sha256_hex=digest,
        signer_ss58=keypair.ss58_address,
        signature_hex=signature.hex(),
    )


def verify_bundle(
    canonical_bytes: bytes,
    *,
    expected_sha256: str,
    signature_hex: str,
    keypair: Keypair,
) -> bool:
    actual = sha256_hex(canonical_bytes)
    if actual != expected_sha256:
        raise SignerError(
            f"sha256 mismatch: expected {expected_sha256}, got {actual}"
        )
    try:
        signature = bytes.fromhex(signature_hex)
    except ValueError as exc:
        raise SignerError(f"invalid hex signature: {exc}") from exc
    try:
        return bool(keypair.verify(canonical_bytes, signature))
    except Exception as exc:
        raise SignerError(f"keypair.verify() failed: {exc}") from exc


# -- testing helper -----------------------------------------------------


class FakeKeypair:
    """Deterministic in-process keypair for tests.

    Uses HMAC-SHA256 with a fixed secret as a stand-in for ed25519 — *not*
    cryptographically meaningful; tests only check the verify roundtrip and
    that the signature changes when the bundle changes.
    """

    def __init__(self, *, ss58_address: str = "5FakeOwnerHotkey", secret: bytes = b"forge-test") -> None:
        self._ss58 = ss58_address
        self._secret = secret

    @property
    def ss58_address(self) -> str:
        return self._ss58

    def sign(self, data: bytes) -> bytes:
        import hmac

        return hmac.new(self._secret, data, hashlib.sha256).digest()

    def verify(self, data: bytes, signature: bytes) -> bool:
        return self.sign(data) == signature
