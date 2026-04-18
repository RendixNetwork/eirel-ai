"""Miner HTTP client — invokes specialist family miners via their endpoints."""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

import httpx

_logger = logging.getLogger(__name__)
_DEFAULT_TIMEOUT = 30.0
_MAX_RETRIES = 2
_RETRYABLE_STATUS_CODES = {502, 503, 504}


async def invoke_miner(
    *,
    endpoint: str,
    payload: dict[str, Any],
    timeout_seconds: float = _DEFAULT_TIMEOUT,
    headers: dict[str, str] | None = None,
    max_retries: int = _MAX_RETRIES,
) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.post(
                    f"{endpoint.rstrip('/')}/v1/agent/infer",
                    json=payload,
                    headers=headers,
                )
                if response.status_code in _RETRYABLE_STATUS_CODES and attempt < max_retries:
                    _logger.warning(
                        "miner returned %s, retrying (%d/%d)",
                        response.status_code, attempt + 1, max_retries,
                    )
                    raise httpx.HTTPStatusError(
                        f"retryable {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                return response.json()
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPStatusError) as exc:
            last_exc = exc
            if attempt < max_retries:
                jitter = random.uniform(0, 0.5)
                await asyncio.sleep((2**attempt) * 0.5 + jitter)
                continue
            raise
    raise last_exc  # type: ignore[misc]
