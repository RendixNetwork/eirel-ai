"""Parallel oracle fanout with graceful per-vendor degradation.

Calls all configured oracle clients concurrently via ``asyncio.gather
(return_exceptions=True)``. Per-vendor exceptions are caught and
converted into ``OracleGrounding(status="error", ...)`` so a single
vendor outage doesn't take down the eval cycle.

Sequential mode (``parallel=False``) is supported for debugging
individual vendor failures.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable

from validation.validator.oracles.base import (
    OracleClient,
    OracleContext,
    OracleGrounding,
)

_logger = logging.getLogger(__name__)


class OracleFanout:
    """Run multiple ``OracleClient`` instances against the same context.

    Designed to be called once per claimed task at task-claim time;
    reuse the same fanout across many tasks within a batch (each
    underlying provider client holds its own httpx session).
    """

    def __init__(
        self,
        clients: Iterable[OracleClient],
        *,
        parallel: bool = True,
    ) -> None:
        self._clients: list[OracleClient] = list(clients)
        if not self._clients:
            raise ValueError("OracleFanout requires at least one client")
        self._parallel = bool(parallel)

    @property
    def vendors(self) -> list[str]:
        return [c.vendor for c in self._clients]

    async def run(self, context: OracleContext) -> list[OracleGrounding]:
        """Produce one ``OracleGrounding`` per configured client.

        Order matches the constructor order (so callers can index by
        vendor consistently across tasks). Exceptions raised by an
        oracle client (which shouldn't happen — clients are supposed
        to convert provider errors into ``status="error"`` themselves)
        get caught here as a last-resort safety net.
        """
        if self._parallel:
            results = await asyncio.gather(
                *(self._safe_call(c, context) for c in self._clients),
                return_exceptions=False,
            )
            return list(results)
        # Sequential: one vendor at a time, useful for debugging.
        out: list[OracleGrounding] = []
        for c in self._clients:
            out.append(await self._safe_call(c, context))
        return out

    @staticmethod
    async def _safe_call(
        client: OracleClient, context: OracleContext,
    ) -> OracleGrounding:
        try:
            return await client.produce_grounding(context)
        except Exception as exc:  # pragma: no cover — last-resort
            _logger.exception(
                "oracle client %s raised unexpectedly", client.vendor,
            )
            return OracleGrounding(
                vendor=client.vendor,
                status="error",
                error_msg=f"unexpected_exception: {exc!r}",
            )

    async def aclose(self) -> None:
        """Close every underlying provider client."""
        await asyncio.gather(
            *(c.aclose() for c in self._clients),
            return_exceptions=True,
        )


def successful_groundings(
    groundings: list[OracleGrounding],
) -> list[OracleGrounding]:
    """Filter to vendors that returned a usable answer (``status="ok"``)."""
    return [g for g in groundings if g.status == "ok"]


def vendor_status_map(
    groundings: list[OracleGrounding],
) -> dict[str, str]:
    """One-line per-vendor status map for telemetry / persistence."""
    return {g.vendor: g.status for g in groundings}


__all__ = [
    "OracleFanout",
    "successful_groundings",
    "vendor_status_map",
]
