from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import HTTPException

from shared.common.circuit_breaker import CircuitBreaker, CircuitOpenError
from shared.common.config import get_settings

_logger = logging.getLogger(__name__)
_orchestrator_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)

OWNER_API_URL = os.getenv("OWNER_API_URL", "http://owner-api:8000")
INTERNAL_SERVICE_TOKEN = os.getenv("EIREL_INTERNAL_SERVICE_TOKEN", "")
# Default family for the consumer chat surface. The streaming endpoint
# resolves a serving miner for this family from owner-api and proxies its
# /v1/agent/infer/stream NDJSON back to the client as SSE.
_CHAT_FAMILY_ID = os.getenv("EIREL_CONSUMER_CHAT_FAMILY", "general_chat")
# Total wall-clock budget for a streaming chat. Should clear the slowest
# acceptable miner completion (thinking mode = 600s today).
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


async def _resolve_serving_miner(family_id: str) -> dict[str, Any] | None:
    """Fetch the current winner miner for a family from owner-api.

    Resolution order:
      1. `EIREL_CONSUMER_CHAT_MINER_OVERRIDE_ENDPOINT` env (test/debug
         override — bypasses owner-api entirely).
      2. `/v1/internal/serving/{family_id}` (production: a published
         serving release).
      3. `/v1/internal/managed-deployments/active/{family_id}` (fallback:
         any healthy active managed deployment, useful before the first
         serving release of a run is published).
    """
    override = os.getenv("EIREL_CONSUMER_CHAT_MINER_OVERRIDE_ENDPOINT", "").strip()
    if override:
        return {"endpoint": override, "hotkey": "override", "family_id": family_id}

    headers: dict[str, str] = {}
    if INTERNAL_SERVICE_TOKEN:
        headers["Authorization"] = f"Bearer {INTERNAL_SERVICE_TOKEN}"

    async with httpx.AsyncClient(timeout=10.0) as client:
        for path in (
            f"/v1/internal/serving/{family_id}",
            f"/v1/internal/managed-deployments/active/{family_id}",
        ):
            try:
                resp = await client.get(f"{OWNER_API_URL}{path}", headers=headers)
                if resp.status_code == 404:
                    continue
                resp.raise_for_status()
                data = resp.json()
                if data.get("endpoint"):
                    return data
            except Exception as exc:  # noqa: BLE001
                _logger.error(
                    "failed to resolve miner via %s for %s: %s",
                    path, family_id, exc,
                )
                continue
    return None


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
) -> AsyncIterator[bytes]:
    """Stream a chat response as SSE.

    1. Resolve the current serving miner for the chat family from owner-api.
    2. POST to that miner's `/v1/agent/infer/stream` (NDJSON, eirel SDK ≥ 0.2.3).
    3. Re-emit each NDJSON chunk as an SSE event with the chunk's `event`
       name (`delta`/`citation`/`tool_call`/`done`). Falls back to the
       unary endpoint on 404 and emits the whole answer as a single
       `delta` followed by `done` so the client UX is identical.

    Errors emit a final `error` SSE event then close the stream.
    """
    task_id = session_id or f"chat-{uuid.uuid4().hex[:12]}"
    yield _sse_event("started", {"task_id": task_id, "family_id": _CHAT_FAMILY_ID})

    miner = await _resolve_serving_miner(_CHAT_FAMILY_ID)
    if miner is None:
        yield _sse_event("error", {
            "message": f"no serving miner available for family {_CHAT_FAMILY_ID}",
        })
        return

    body = {
        "task_id": task_id,
        "family_id": _CHAT_FAMILY_ID,
        "primary_goal": prompt,
        "subtask": prompt,
        "inputs": {},
        "context_history": context_history or [],
        "metadata": {"user_id": user_id, "session_id": session_id},
    }
    headers = {"Content-Type": "application/json"}
    endpoint = miner["endpoint"].rstrip("/")
    stream_url = f"{endpoint}/v1/agent/infer/stream"
    unary_url = f"{endpoint}/v1/agent/infer"

    start = time.monotonic()
    used_stream = True
    final_status = "completed"

    try:
        async with httpx.AsyncClient(timeout=_CHAT_STREAM_TIMEOUT_SECONDS) as client:
            try:
                async with client.stream(
                    "POST", stream_url, json=body, headers=headers,
                ) as resp:
                    if resp.status_code == 404:
                        used_stream = False
                    else:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                chunk = json.loads(line)
                            except json.JSONDecodeError:
                                _logger.warning(
                                    "malformed NDJSON from miner: %r", line[:120],
                                )
                                continue
                            event = chunk.get("event") or "delta"
                            if event == "done":
                                final_status = chunk.get("status") or "completed"
                            yield _sse_event(event, chunk)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 404:
                    raise
                used_stream = False

            if not used_stream:
                # Older miners on eirel SDK < 0.2.3 — assemble a single
                # delta + done from the unary response so the client sees
                # the same SSE shape.
                resp = await client.post(unary_url, json=body, headers=headers)
                resp.raise_for_status()
                payload = resp.json() if resp.content else {}
                output = payload.get("output") or {}
                text = ""
                for key in ("answer", "response", "text", "content", "message"):
                    value = output.get(key)
                    if isinstance(value, str) and value:
                        text = value
                        break
                if text:
                    yield _sse_event("delta", {"event": "delta", "text": text})
                yield _sse_event("done", {
                    "event": "done",
                    "output": output,
                    "citations": payload.get("citations") or [],
                    "tool_calls": payload.get("tool_calls") or [],
                    "status": payload.get("status") or "completed",
                })
                final_status = payload.get("status") or "completed"
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
