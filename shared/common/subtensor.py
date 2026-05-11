from __future__ import annotations

"""Per-process singleton ``bittensor.Subtensor`` with lazy reconnect.

Each ``bt.Subtensor(network=...)`` constructor opens a fresh WebSocket
handshake to the chain endpoint. Repeated construction (one per call,
or one per loop cycle) hammers public endpoints and trips their
HTTP 429 rate limit on the WS upgrade. The substrate WebSocket is
designed to stay open and serve many RPCs, so the right shape is one
long-lived ``Subtensor`` per process.

This module exposes ``get_subtensor()`` returning that singleton, and
``reset_subtensor()`` for callers that observe a connection-level error
and want to force a reconnect on the next ``get_subtensor()``. Access
is guarded by a ``threading.Lock`` because ``bittensor`` is sync and
callers may be threads (e.g. ``chain_queries._sync_metagraph``).

The network arg is read from ``BITTENSOR_NETWORK`` on first build and
locked in for the process lifetime — flipping networks mid-process is
not a supported flow and would otherwise create an invisible second
connection.
"""

import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_instance: Any = None
_bound_network: str | None = None


def _build(network: str) -> Any:
    import bittensor as bt
    return bt.Subtensor(network=network)


def get_subtensor(network: str | None = None) -> Any:
    global _instance, _bound_network
    target = network or os.getenv("BITTENSOR_NETWORK", "finney")
    with _lock:
        if _instance is not None and _bound_network == target:
            return _instance
        if _instance is not None and _bound_network != target:
            logger.warning(
                "subtensor: rebinding network %r -> %r (process should not normally switch)",
                _bound_network, target,
            )
        _instance = _build(target)
        _bound_network = target
        logger.info("subtensor: opened singleton connection network=%s", target)
        return _instance


def reset_subtensor() -> None:
    """Drop the cached instance so the next ``get_subtensor`` reconnects.

    Call this from an ``except`` that observed a connection-level error
    (``websockets.exceptions.InvalidStatus``, ``ConnectionError``, etc.).
    """
    global _instance
    with _lock:
        if _instance is not None:
            logger.info("subtensor: resetting singleton (will reconnect on next call)")
        _instance = None
