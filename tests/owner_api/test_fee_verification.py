from __future__ import annotations

import time

import pytest

from control_plane.owner_api.fee_verifier import FeeVerifier, FeeVerificationResult


# -- Helpers ---------------------------------------------------------------


def _make_verifier(*, cache_ttl: float = 300.0) -> FeeVerifier:
    verifier = FeeVerifier(
        network="finney",
        treasury_address="5Treasury",
        fee_tao=0.1,
    )
    verifier._coldkey_cache_ttl = cache_ttl
    return verifier


# -- Fix 3: Coldkey cache tests -------------------------------------------


def test_coldkey_cache_hit_avoids_rpc():
    verifier = _make_verifier()
    call_count = 0
    original = verifier._get_hotkey_owner_uncached

    def _tracking_uncached(hotkey: str) -> str | None:
        nonlocal call_count
        call_count += 1
        return f"coldkey-for-{hotkey}"

    verifier._get_hotkey_owner_uncached = _tracking_uncached  # type: ignore[assignment]

    result1 = verifier._get_hotkey_owner_cached("hotkey-A")
    result2 = verifier._get_hotkey_owner_cached("hotkey-A")

    assert result1 == "coldkey-for-hotkey-A"
    assert result2 == "coldkey-for-hotkey-A"
    assert call_count == 1


def test_coldkey_cache_expires_after_ttl():
    verifier = _make_verifier(cache_ttl=0.01)
    call_count = 0

    def _tracking_uncached(hotkey: str) -> str | None:
        nonlocal call_count
        call_count += 1
        return f"coldkey-for-{hotkey}"

    verifier._get_hotkey_owner_uncached = _tracking_uncached  # type: ignore[assignment]

    verifier._get_hotkey_owner_cached("hotkey-B")
    time.sleep(0.05)  # wait for TTL to expire
    verifier._get_hotkey_owner_cached("hotkey-B")

    assert call_count == 2


# -- Fix 3: Retry tests ---------------------------------------------------


def test_verify_retries_on_connection_error():
    verifier = _make_verifier()

    # Stub the coldkey lookup to succeed immediately.
    verifier._get_hotkey_owner_uncached = lambda hk: "5ColdKey"  # type: ignore[assignment]

    attempt = 0
    _fake_receipt_attrs = {
        "from": "5ColdKey",
        "to": "5Treasury",
        "amount": 100_000_000,  # 0.1 TAO in RAO
    }

    class _FakeReceipt:
        is_success = True
        triggered_events = [
            {
                "module_id": "Balances",
                "event_id": "Transfer",
                "attributes": _fake_receipt_attrs,
            }
        ]

    class _FakeSubstrate:
        def retrieve_extrinsic_by_hash(self, block_hash, extrinsic_hash):
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                raise ConnectionError("transient network failure")
            return _FakeReceipt()

    class _FakeSubtensor:
        substrate = _FakeSubstrate()

    import bittensor as bt

    _original_subtensor = bt.Subtensor

    def _patched_subtensor(network: str = "finney"):
        return _FakeSubtensor()

    bt.Subtensor = _patched_subtensor  # type: ignore[assignment]
    try:
        result = verifier.verify_payment(
            "0xabc", "hotkey-C", block_hash="0xblock"
        )
    finally:
        bt.Subtensor = _original_subtensor

    assert result.valid
    assert attempt == 2


def test_verify_fails_after_retry_exhaustion():
    verifier = _make_verifier()
    verifier._get_hotkey_owner_uncached = lambda hk: "5ColdKey"  # type: ignore[assignment]

    class _FakeSubstrate:
        def retrieve_extrinsic_by_hash(self, block_hash, extrinsic_hash):
            raise ConnectionError("persistent failure")

    class _FakeSubtensor:
        substrate = _FakeSubstrate()

    import bittensor as bt

    _original_subtensor = bt.Subtensor

    def _patched_subtensor(network: str = "finney"):
        return _FakeSubtensor()

    bt.Subtensor = _patched_subtensor  # type: ignore[assignment]
    try:
        with pytest.raises(ValueError, match="retry"):
            verifier.verify_payment(
                "0xabc", "hotkey-D", block_hash="0xblock"
            )
    finally:
        bt.Subtensor = _original_subtensor
