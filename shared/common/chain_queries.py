from __future__ import annotations

"""Thin helpers over the bittensor package for subnet integration checks.

What these ARE for
------------------
Post-epoch integration queries that observe consensus outcomes:

* ``miner_incentive`` — the emission share the subnet is actually
  paying a hotkey, computed by the chain from *all* validators'
  revealed weights plus stake.  Non-zero incentive after a full epoch
  confirms "the subnet is rewarding this miner", independent of whose
  weight row produced it.

* ``weight_from_validator`` — a snapshot of a validator's weight row
  from the current (revealed) metagraph.

What these are NOT for
----------------------
Verifying that *our* validator just set weights.  Bittensor uses
commit-reveal: a weight row submitted at block N is committed, then
revealed at block N + commit_reveal_interval.  During that window the
metagraph shows the *previous* reveal (often zero or stale), so reading
``metagraph.W[our_uid]`` immediately after ``set_weights`` tells you
nothing about whether the extrinsic landed.

The authoritative signal for "did our validator publish?" is the
return value of ``subtensor.set_weights(wait_for_inclusion=True)``,
which yields ``(success, message)`` — the extrinsic either made it
into a block or it didn't.  The validator's weight-setting loop in
``validation/validator/engine.py`` already uses this signal and logs
``weight-setting: set_weights succeeded on attempt N``.  That log line
is the B4 verification path, not a metagraph read.

Everything here is synchronous — ``subtensor.metagraph`` is a blocking
RPC.  Callers inside async code should wrap via ``asyncio.to_thread``.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 15.0


class ChainQueryError(RuntimeError):
    """Raised when a metagraph query fails or times out."""


def _sync_metagraph(
    *,
    network: str,
    netuid: int,
    lite: bool,
    timeout_seconds: float,
) -> Any:
    """Fetch the metagraph synchronously with an explicit timeout.

    ``bittensor.Subtensor`` does not accept a timeout kwarg directly, so
    we rely on an underlying socket timeout via the websocket client and
    a hard ceiling via ``concurrent.futures``.  This prevents the calling
    loop from blocking forever on a dead endpoint — the failure mode the
    phase-5 E2E agent hit.
    """
    import bittensor as bt  # imported lazily so unit tests can stub
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutureTimeout

    def _load() -> Any:
        sub = bt.Subtensor(network=network)
        return sub.metagraph(netuid=netuid, lite=lite)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_load)
        try:
            return future.result(timeout=timeout_seconds)
        except _FutureTimeout as exc:
            raise ChainQueryError(
                f"metagraph query timed out after {timeout_seconds}s "
                f"(network={network!r}, netuid={netuid})"
            ) from exc
        except Exception as exc:
            raise ChainQueryError(f"metagraph query failed: {exc}") from exc


def miner_incentive(
    *,
    network: str,
    netuid: int,
    hotkey: str,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> float | None:
    """Return the incentive (0-1) the subnet currently pays a hotkey.

    Returns ``None`` if the hotkey isn't registered on this netuid.
    ``incentive == 1.0`` means the miner is receiving the full emission
    share (common for a single-miner testnet pool).
    """
    mg = _sync_metagraph(
        network=network, netuid=netuid, lite=False, timeout_seconds=timeout_seconds,
    )
    hotkeys = [str(h) for h in mg.hotkeys]
    if hotkey not in hotkeys:
        return None
    uid = hotkeys.index(hotkey)
    incentive_vec = getattr(mg, "I", None)
    if incentive_vec is None or uid >= len(incentive_vec):
        raise ChainQueryError("metagraph missing incentive vector")
    return float(incentive_vec[uid])


def weight_from_validator(
    *,
    network: str,
    netuid: int,
    validator_hotkey: str,
    miner_hotkey: str,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> float | None:
    """Return the last *revealed* weight a validator set for a miner.

    This reads ``metagraph.W`` which only reflects revealed weights.  A
    weight committed inside the current commit-reveal interval will not
    appear here until the reveal block, so this is not a real-time
    confirmation that a recent ``set_weights`` succeeded.  Use the tuple
    return of ``subtensor.set_weights(wait_for_inclusion=True)`` for
    that.  This helper is for post-reveal audits.

    Returns ``None`` if either hotkey isn't registered.  A return value
    of ``0.0`` is distinct from ``None`` — it means "registered, and the
    most recent reveal put zero weight here".
    """
    mg = _sync_metagraph(
        network=network, netuid=netuid, lite=False, timeout_seconds=timeout_seconds,
    )
    hotkeys = [str(h) for h in mg.hotkeys]
    try:
        v_uid = hotkeys.index(validator_hotkey)
        m_uid = hotkeys.index(miner_hotkey)
    except ValueError:
        return None
    weight_matrix = getattr(mg, "W", None)
    if weight_matrix is None or v_uid >= len(weight_matrix):
        raise ChainQueryError("metagraph missing weight matrix")
    row = weight_matrix[v_uid]
    if m_uid >= len(row):
        return None
    return float(row[m_uid])


__all__ = [
    "ChainQueryError",
    "miner_incentive",
    "weight_from_validator",
]
