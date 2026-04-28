from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import HTTPException

from shared.common.circuit_breaker import CircuitBreaker, CircuitOpenError
from shared.common.config import get_settings

_logger = logging.getLogger(__name__)
_orchestrator_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)

# Total wall-clock budget for a streaming chat. The orchestrator's
# upstream budget is the same; we set an envelope here so the consumer
# connection times out around the same time the orchestrator gives up.
_CHAT_STREAM_TIMEOUT_SECONDS = float(
    os.getenv("EIREL_CONSUMER_CHAT_STREAM_TIMEOUT_SECONDS", "660")
)

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


# consumer-chat-api is now a thin SSE facade. Resolving the
# serving deployment, building the slim invocation body, and proxying
# NDJSON from the family pod all live in the orchestrator now. This
# module only translates between consumer-chat-api's SSE surface and
# the orchestrator's NDJSON streaming endpoint.
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8050")


def _sse_event(event: str, data: dict[str, Any]) -> bytes:
    """Format a Server-Sent Event frame.

    SSE spec: each event ends with a blank line. We use one `event:` line
    plus a `data:` line carrying compact JSON. The browser EventSource API
    consumes this directly.
    """
    return (
        f"event: {event}\n"
        f"data: {json.dumps(data, separators=(',', ':'))}\n\n"
    ).encode("utf-8")


async def stream_chat_request(
    *,
    prompt: str,
    user_id: str = "anonymous",
    session_id: str | None = None,
    context_history: list[dict[str, Any]] | None = None,
    mode: str = "instant",
    web_search: bool = False,
) -> AsyncIterator[bytes]:
    """Stream a chat response as SSE — thin facade in front of the orchestrator.

    Forwards to the orchestrator's ``/v1/orchestrate/chat/stream`` and
    re-emits each NDJSON ``StreamChunk`` as an SSE event with the
    chunk's ``event`` name (``started`` / ``delta`` / ``citation`` /
    ``tool_call`` / ``done``). The orchestrator owns:

      * session state (mode + web_search persistence),
      * routing to the right family (today: passthrough to
        ``general_chat``; later: DAG composition across families),
      * miner resolution + NDJSON proxy.

    A network failure to the orchestrator emits a terminal ``error``
    SSE event then closes the stream.
    """
    body = {
        "prompt": prompt,
        "user_id": user_id,
        "session_id": session_id,
        "context_history": context_history or [],
        "mode": mode,
        "web_search": bool(web_search),
    }

    start = time.monotonic()
    final_status = "completed"
    forward_url = f"{ORCHESTRATOR_URL.rstrip('/')}/v1/orchestrate/chat/stream"

    try:
        async with httpx.AsyncClient(timeout=_CHAT_STREAM_TIMEOUT_SECONDS) as client:
            async with client.stream("POST", forward_url, json=body) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        _logger.warning(
                            "malformed NDJSON from orchestrator: %r", line[:120],
                        )
                        continue
                    event = chunk.get("event") or "delta"
                    if event == "done":
                        final_status = chunk.get("status") or "completed"
                    yield _sse_event(event, chunk)
    except Exception as exc:  # noqa: BLE001
        _logger.exception("stream_chat_request failed: %s", exc)
        yield _sse_event("error", {"message": str(exc)})
        final_status = "failed"
        return

    elapsed_ms = (time.monotonic() - start) * 1000
    await _record_traffic(
        prompt=prompt,
        user_id=user_id,
        session_id=session_id,
        status_code=200 if final_status == "completed" else 500,
        latency_ms=elapsed_ms,
    )
