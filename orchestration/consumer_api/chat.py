from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import httpx
from fastapi import HTTPException

from shared.common.circuit_breaker import CircuitBreaker, CircuitOpenError
from shared.common.config import get_settings

_logger = logging.getLogger(__name__)
_orchestrator_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)

# Opt-in traffic logging for evaluation sampling (Item 14).
# Set EIREL_TRAFFIC_LOGGING_ENABLED=1 to record anonymized request metadata
# that can later be sampled by the evaluation traffic sampler.
_TRAFFIC_LOGGING_ENABLED = bool(os.getenv("EIREL_TRAFFIC_LOGGING_ENABLED", ""))

_traffic_log: list[dict[str, Any]] = []
_traffic_log_lock = asyncio.Lock()
_TRAFFIC_LOG_MAX_SIZE = int(os.getenv("EIREL_TRAFFIC_LOG_MAX_SIZE", "10000"))


def get_traffic_log() -> list[dict[str, Any]]:
    """Return the in-memory traffic log for evaluation sampling."""
    return list(_traffic_log)


def clear_traffic_log() -> None:
    """Clear the in-memory traffic log."""
    _traffic_log.clear()


async def _record_traffic(
    *,
    prompt: str,
    user_id: str,
    session_id: str | None,
    status_code: int,
    latency_ms: float,
) -> None:
    """Record anonymized request metadata for evaluation sampling."""
    if not _TRAFFIC_LOGGING_ENABLED:
        return
    async with _traffic_log_lock:
        if len(_traffic_log) >= _TRAFFIC_LOG_MAX_SIZE:
            del _traffic_log[:_TRAFFIC_LOG_MAX_SIZE // 10]
        _traffic_log.append({
            "raw_input": prompt,
            "user_id": user_id,
            "session_id": session_id,
            "status": "completed" if 200 <= status_code < 300 else "failed",
            "mode": "sync",
            "latency_ms": round(latency_ms, 1),
            "logged_at": time.time(),
        })


async def route_chat_request(
    *,
    prompt: str,
    user_id: str = "anonymous",
    session_id: str | None = None,
    context_history: list[dict[str, Any]] | None = None,
):
    """Route a chat request to the orchestrator service.

    Previously routed to api-gateway. Now routes directly to the
    orchestrator which handles family selection, tool invocation,
    and specialist coordination.
    """
    payload = {
        "prompt": prompt,
        "user_id": user_id,
        "session_id": session_id,
        "context_history": context_history or [],
        "constraints": {},
        "metadata": {},
    }
    base_url = get_settings().orchestrator_url.rstrip("/")
    start = time.monotonic()

    async def _orchestrator_post() -> httpx.Response:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{base_url}/v1/orchestrate",
                json=payload,
            )
            resp.raise_for_status()
            return resp

    try:
        response = await _orchestrator_circuit_breaker.call("orchestrator", _orchestrator_post)
    except CircuitOpenError as exc:
        raise HTTPException(
            status_code=503,
            detail="service temporarily unavailable",
            headers={"Retry-After": str(int(exc.retry_after) + 1)},
        ) from exc
    elapsed_ms = (time.monotonic() - start) * 1000
    await _record_traffic(
        prompt=prompt,
        user_id=user_id,
        session_id=session_id,
        status_code=response.status_code,
        latency_ms=elapsed_ms,
    )
    return response.status_code, response.json()
