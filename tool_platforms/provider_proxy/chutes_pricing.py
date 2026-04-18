from __future__ import annotations

"""Fetch Chutes model pricing from the canonical endpoint.

Chutes publishes current per-model rates at ``https://llm.chutes.ai/v1/models``.
Each entry looks like::

    {
        "id": "moonshotai/Kimi-K2.5-TEE",
        "price": {
            "input":  {"usd": 0.3827, "tao": ...},
            "output": {"usd": 1.72,   "tao": ...},
            "input_cache_read": {"usd": 0.19135, ...}
        },
        ...
    }

``price.input.usd`` / ``price.output.usd`` are $/1M tokens — the same unit
``LLMPrice`` uses — so refresh is a direct mapping with no rate conversion.

This module exposes ``fetch_chutes_pricing`` (one-shot HTTP fetch) and
``run_chutes_pricing_refresh_loop`` (background task used in the
provider-proxy lifespan).  Both fail-soft: network errors log a warning
and leave the current overlay in place.
"""

import asyncio
import logging
import os
from typing import Any

import httpx

from shared.common.tool_pricing import (
    LLMPrice,
    get_dynamic_pricing,
    update_dynamic_pricing,
)

_logger = logging.getLogger(__name__)

_DEFAULT_URL = "https://llm.chutes.ai/v1/models"
_DEFAULT_REFRESH_SECONDS = 3600.0  # 1 hour
_DEFAULT_TIMEOUT_SECONDS = 15.0


def _parse_models_response(payload: Any) -> dict[str, LLMPrice]:
    """Extract ``{chutes:<model_id>: LLMPrice}`` from the models endpoint body.

    Silently skips entries that lack a well-formed ``price.input.usd`` /
    ``price.output.usd`` pair — a malformed row shouldn't take down the
    whole refresh.
    """
    if not isinstance(payload, dict):
        raise ValueError("expected JSON object at top level")
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("expected ``data`` array")
    parsed: dict[str, LLMPrice] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("id")
        price = entry.get("price")
        if not isinstance(model_id, str) or not isinstance(price, dict):
            continue
        input_block = price.get("input")
        output_block = price.get("output")
        if not isinstance(input_block, dict) or not isinstance(output_block, dict):
            continue
        input_usd = input_block.get("usd")
        output_usd = output_block.get("usd")
        if not isinstance(input_usd, (int, float)) or not isinstance(output_usd, (int, float)):
            continue
        parsed[f"chutes:{model_id}"] = LLMPrice(
            input_per_mtok_usd=float(input_usd),
            output_per_mtok_usd=float(output_usd),
        )
    return parsed


async def fetch_chutes_pricing(
    *,
    url: str = _DEFAULT_URL,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    client: httpx.AsyncClient | None = None,
) -> dict[str, LLMPrice]:
    """Fetch current Chutes rates.  Raises on HTTP / parse errors."""
    owned_client = client is None
    c = client or httpx.AsyncClient(timeout=timeout_seconds)
    try:
        resp = await c.get(url)
        resp.raise_for_status()
        return _parse_models_response(resp.json())
    finally:
        if owned_client:
            await c.aclose()


async def refresh_chutes_pricing_once(
    *,
    url: str = _DEFAULT_URL,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> int:
    """Fetch + apply.  Returns the number of entries applied (0 on failure)."""
    try:
        pricing = await fetch_chutes_pricing(url=url, timeout_seconds=timeout_seconds)
    except (httpx.HTTPError, ValueError) as exc:
        _logger.warning(
            "chutes pricing refresh failed (keeping previous overlay with %d entries): %s",
            len(get_dynamic_pricing()), exc,
        )
        return 0
    if not pricing:
        _logger.warning(
            "chutes pricing refresh returned zero entries (keeping previous overlay)",
        )
        return 0
    update_dynamic_pricing(pricing)
    return len(pricing)


async def run_chutes_pricing_refresh_loop(
    *,
    url: str | None = None,
    refresh_seconds: float | None = None,
    timeout_seconds: float | None = None,
) -> None:
    """Background task: refresh on startup then every ``refresh_seconds``.

    Cancelled via the lifespan context when the proxy shuts down.
    Always re-raises ``CancelledError``.
    """
    url = url or os.getenv("EIREL_CHUTES_PRICING_URL", _DEFAULT_URL)
    refresh_seconds = refresh_seconds or float(
        os.getenv("EIREL_CHUTES_PRICING_REFRESH_SECONDS", str(_DEFAULT_REFRESH_SECONDS))
    )
    timeout_seconds = timeout_seconds or _DEFAULT_TIMEOUT_SECONDS

    # First fetch runs immediately so the overlay is populated before
    # the proxy handles its first request.
    await refresh_chutes_pricing_once(url=url, timeout_seconds=timeout_seconds)
    while True:
        try:
            await asyncio.sleep(refresh_seconds)
        except asyncio.CancelledError:
            raise
        await refresh_chutes_pricing_once(url=url, timeout_seconds=timeout_seconds)


__all__ = [
    "fetch_chutes_pricing",
    "refresh_chutes_pricing_once",
    "run_chutes_pricing_refresh_loop",
]
