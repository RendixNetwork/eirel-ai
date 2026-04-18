from __future__ import annotations

"""On-chain submission fee verification.

Miners must transfer the required fee to the treasury address before
submitting.  They include the extrinsic hash in the submission request
and this module verifies the transfer on-chain.
"""

import logging
import time
from dataclasses import dataclass

import bittensor as bt

logger = logging.getLogger(__name__)

TAO_TO_RAO = 1_000_000_000


@dataclass(slots=True)
class FeeVerificationResult:
    valid: bool
    reason: str
    sender: str | None = None
    destination: str | None = None
    amount_rao: int | None = None


class FeeVerifier:
    """Verifies that a miner has paid the submission fee to the treasury."""

    def __init__(
        self,
        *,
        network: str,
        treasury_address: str,
        fee_tao: float,
    ) -> None:
        self.network = network
        self.treasury_address = treasury_address
        self.fee_rao = int(fee_tao * TAO_TO_RAO)
        self._coldkey_cache: dict[str, tuple[str, float]] = {}
        self._coldkey_cache_ttl: float = 300.0  # 5 minutes

    def _get_hotkey_owner_uncached(self, hotkey: str) -> str | None:
        try:
            subtensor = bt.Subtensor(network=self.network)
            return subtensor.get_hotkey_owner(hotkey)
        except Exception:
            return None

    def _get_hotkey_owner_cached(self, hotkey: str) -> str | None:
        now = time.monotonic()
        cached = self._coldkey_cache.get(hotkey)
        if cached is not None and cached[1] > now:
            return cached[0]
        coldkey = self._get_hotkey_owner_uncached(hotkey)
        if coldkey is not None:
            self._coldkey_cache[hotkey] = (coldkey, now + self._coldkey_cache_ttl)
        return coldkey

    def verify_payment(
        self,
        extrinsic_hash: str,
        expected_hotkey: str,
        *,
        block_hash: str | None = None,
    ) -> FeeVerificationResult:
        """Verify that *extrinsic_hash* is a valid transfer to the treasury
        for at least the required fee, sent by the coldkey that owns
        *expected_hotkey*.

        Returns a ``FeeVerificationResult`` indicating success or failure with
        a human-readable reason.
        """
        # Resolve the coldkey that owns this hotkey (cached).
        expected_coldkey = self._get_hotkey_owner_cached(expected_hotkey)
        if not expected_coldkey:
            logger.warning("could not resolve coldkey for hotkey %s", expected_hotkey)
            return FeeVerificationResult(
                valid=False,
                reason=f"could not resolve coldkey for hotkey {expected_hotkey}",
            )
        if not block_hash:
            return FeeVerificationResult(
                valid=False,
                reason="block_hash is required for extrinsic verification",
            )
        # Fetch the extrinsic with a single retry on transient network errors.
        for attempt in range(2):
            try:
                subtensor = bt.Subtensor(network=self.network)
                substrate = subtensor.substrate
                receipt = substrate.retrieve_extrinsic_by_hash(
                    block_hash, extrinsic_hash,
                )
                break
            except (ConnectionError, OSError) as exc:
                if attempt == 0:
                    time.sleep(2.0)
                    continue
                raise ValueError(
                    f"fee verification failed after retry: {exc}"
                ) from exc
            except Exception as exc:
                logger.warning(
                    "extrinsic lookup failed for %s: %s", extrinsic_hash, exc,
                )
                return FeeVerificationResult(
                    valid=False,
                    reason=f"extrinsic lookup failed: {exc}",
                )
        if receipt is None:
            return FeeVerificationResult(
                valid=False,
                reason="extrinsic not found on chain",
            )
        if not receipt.is_success:
            return FeeVerificationResult(
                valid=False,
                reason="extrinsic did not succeed on chain",
            )
        # Extract transfer details from triggered events.
        try:
            transfer_event = None
            for event in receipt.triggered_events:
                if (
                    event.get("module_id") == "Balances"
                    and event.get("event_id") == "Transfer"
                ):
                    transfer_event = event
                    break
            if transfer_event is None:
                return FeeVerificationResult(
                    valid=False,
                    reason="no Balances.Transfer event found in extrinsic",
                )
            attrs = transfer_event.get("attributes", {})
            sender = attrs.get("from", "")
            destination = attrs.get("to", "")
            amount_rao = int(attrs.get("amount", 0))
        except Exception as exc:
            return FeeVerificationResult(
                valid=False,
                reason=f"failed to parse extrinsic events: {exc}",
            )
        if sender != expected_coldkey:
            return FeeVerificationResult(
                valid=False,
                reason="sender does not match hotkey owner",
                sender=sender,
                destination=destination,
                amount_rao=amount_rao,
            )
        if destination != self.treasury_address:
            return FeeVerificationResult(
                valid=False,
                reason="destination does not match treasury",
                sender=sender,
                destination=destination,
                amount_rao=amount_rao,
            )
        if amount_rao < self.fee_rao:
            return FeeVerificationResult(
                valid=False,
                reason="transferred amount below required fee",
                sender=sender,
                destination=destination,
                amount_rao=amount_rao,
            )
        return FeeVerificationResult(
            valid=True,
            reason="payment verified",
            sender=sender,
            destination=destination,
            amount_rao=amount_rao,
        )
