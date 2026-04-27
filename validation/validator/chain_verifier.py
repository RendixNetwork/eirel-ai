from __future__ import annotations

"""Chain verification for weight submissions.

After submitting weights via ``subtensor.set_weights()``, the caller should
poll the chain to confirm the weights actually landed.  This module provides
a lightweight verifier that reads committed weights and compares them to the
expected values within a configurable tolerance.
"""

import logging
import time

logger = logging.getLogger(__name__)

# Tolerance for floating-point comparison of on-chain vs submitted weights.
_DEFAULT_WEIGHT_TOLERANCE = 1e-4


def verify_weights_on_chain(
    subtensor,
    netuid: int,
    wallet_hotkey: str,
    expected_uids: list[int],
    expected_weights: list[float],
    *,
    tolerance: float = _DEFAULT_WEIGHT_TOLERANCE,
    max_poll_attempts: int = 3,
    poll_interval_seconds: float = 2.0,
) -> dict[str, object]:
    """Read committed weights from chain state and compare to expected values.

    Returns a dict with keys ``verified`` (bool), ``mismatches`` (list), and
    ``poll_attempts`` (int).
    """
    if not expected_uids:
        return {"verified": False, "reason": "no expected uids", "poll_attempts": 0}

    expected_map: dict[int, float] = dict(zip(expected_uids, expected_weights))
    last_error: str | None = None

    for attempt in range(1, max_poll_attempts + 1):
        try:
            # lite=False is required: the weight matrix ``W`` is not included
            # in the lite metagraph snapshot, so indexing ``W[validator_uid]``
            # would hit an empty array. Verification runs at most every
            # EIREL_WEIGHT_SET_INTERVAL_BLOCKS (~36 min), so the extra
            # bandwidth for the full metagraph is negligible.
            metagraph = subtensor.metagraph(netuid=netuid, lite=False)
            # Find the uid of the validator hotkey.
            uid_by_hotkey = {str(hk): uid for uid, hk in enumerate(metagraph.hotkeys)}
            validator_uid = uid_by_hotkey.get(wallet_hotkey)
            if validator_uid is None:
                last_error = f"validator hotkey {wallet_hotkey!r} not found in metagraph"
                if attempt < max_poll_attempts:
                    time.sleep(poll_interval_seconds)
                continue

            # Read the weight matrix row for this validator.
            chain_weights_row = metagraph.W[validator_uid] if hasattr(metagraph, "W") else None
            if chain_weights_row is None:
                last_error = "metagraph weight matrix not available"
                if attempt < max_poll_attempts:
                    time.sleep(poll_interval_seconds)
                continue

            mismatches: list[dict[str, object]] = []
            for uid, expected_w in expected_map.items():
                if uid >= len(chain_weights_row):
                    mismatches.append({"uid": uid, "expected": expected_w, "actual": None, "reason": "uid out of range"})
                    continue
                actual_w = float(chain_weights_row[uid])
                if abs(actual_w - expected_w) > tolerance:
                    mismatches.append({"uid": uid, "expected": expected_w, "actual": actual_w})

            return {
                "verified": len(mismatches) == 0,
                "mismatches": mismatches,
                "poll_attempts": attempt,
            }

        except Exception as exc:
            last_error = f"poll attempt {attempt} failed: {exc}"
            logger.warning("chain verification poll %d failed: %s", attempt, exc)
            if attempt < max_poll_attempts:
                time.sleep(poll_interval_seconds)

    return {
        "verified": False,
        "reason": last_error or "verification exhausted",
        "poll_attempts": max_poll_attempts,
    }
