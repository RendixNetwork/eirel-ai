"""Symmetric encryption for OAuth tokens at rest.

OAuth tokens for consumer MCP connections are encrypted in the DB and
decrypted only inside the orchestrator process. Tokens never leave
the orchestrator boundary, never log in plaintext, and never reach a
miner pod.

The implementation uses :mod:`cryptography.fernet` when the dependency
is available (production); a deterministic XOR-based fallback is wired
in for tests and dev environments where ``cryptography`` may not be
installed. The fallback is **not secure** — it only obscures plaintext
in logs / DB dumps. Production deployments MUST set
``EIREL_MCP_TOKEN_ENCRYPTION_KEY`` to a Fernet key (``Fernet.generate_key()``).
"""
from __future__ import annotations

import base64
import hashlib
import os
from typing import Any

__all__ = [
    "TokenCipher",
    "build_token_cipher",
]


class TokenCipher:
    """Encrypt / decrypt small byte strings.

    Concrete implementation chosen at construction time:

      * Production: ``cryptography.fernet.Fernet`` if the lib is present
        and ``EIREL_MCP_TOKEN_ENCRYPTION_KEY`` resolves to a valid key.
      * Fallback: XOR-keystream over a sha256 expansion. Bytes change,
        plaintext is not visible at a glance, but this is NOT
        cryptographically strong — it's a marker for dev/test only.

    Inputs and outputs are bytes. Callers (the connection writer / the
    relay service) are responsible for storing and retrieving the
    blob unchanged.
    """

    __slots__ = ("_fernet", "_xor_key")

    def __init__(
        self,
        *,
        fernet: Any | None = None,
        xor_key: bytes | None = None,
    ) -> None:
        if fernet is None and xor_key is None:
            raise ValueError("TokenCipher requires fernet or xor_key")
        self._fernet = fernet
        self._xor_key = xor_key

    @property
    def is_secure(self) -> bool:
        return self._fernet is not None

    def encrypt(self, plaintext: bytes) -> bytes:
        if self._fernet is not None:
            return self._fernet.encrypt(plaintext)
        return _xor(plaintext, self._xor_key or b"")

    def decrypt(self, ciphertext: bytes) -> bytes:
        if self._fernet is not None:
            return self._fernet.decrypt(ciphertext)
        return _xor(ciphertext, self._xor_key or b"")


def _expand_key(seed: bytes, length: int) -> bytes:
    """Stretch ``seed`` to ``length`` bytes via repeated SHA-256."""
    out = bytearray()
    counter = 0
    while len(out) < length:
        digest = hashlib.sha256(seed + counter.to_bytes(8, "big")).digest()
        out.extend(digest)
        counter += 1
    return bytes(out[:length])


def _xor(data: bytes, key_seed: bytes) -> bytes:
    if not data:
        return data
    keystream = _expand_key(key_seed, len(data))
    return bytes(b ^ k for b, k in zip(data, keystream))


def build_token_cipher() -> TokenCipher:
    """Build a cipher from the env, falling back to XOR for dev.

    Reads ``EIREL_MCP_TOKEN_ENCRYPTION_KEY``:
      * Looks like a Fernet key (44 url-safe base64 chars) → Fernet.
      * Anything else → XOR fallback keyed on the value (or a static
        dev key when the env var is unset).
    """
    raw = os.getenv("EIREL_MCP_TOKEN_ENCRYPTION_KEY", "").encode("utf-8")
    if raw:
        try:
            from cryptography.fernet import Fernet  # type: ignore[import-not-found]
        except ImportError:
            return TokenCipher(xor_key=raw)
        try:
            return TokenCipher(fernet=Fernet(raw))
        except Exception:  # noqa: BLE001 — bad key shape, degrade
            return TokenCipher(xor_key=raw)
    # Dev / test default — deterministic so two test runs roundtrip
    # without surprises but with a stable plaintext-bytes guarantee.
    return TokenCipher(xor_key=b"eirel-mcp-dev-key")
