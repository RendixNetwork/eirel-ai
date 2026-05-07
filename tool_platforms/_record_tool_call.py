"""Fire-and-forget helper for the server-attested tool-call ledger.

Tool services (web_search, url_fetch, sandbox, mcp_relay) call this after
a successful tool invocation so the eval pipeline can compute tool-use
KPIs from the orchestrator's authoritative log — never from
miner-emitted trace frames. Like ``charge_tool_cost``, the call is
best-effort: a transient owner-api outage leaves a single ledger row
unrecorded but never breaks the tool response. The downside of a missed
write is a dropped attestation signal for one item; the upside of
fail-open is that the tool platform stays available when owner-api hiccups.

The owner-api endpoint is ``POST /v1/internal/eval/tool_calls`` with the
``EIREL_INTERNAL_SERVICE_TOKEN`` bearer.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

import httpx

_logger = logging.getLogger(__name__)

_RECORD_TIMEOUT_SECONDS = 5.0
_DIGEST_MAX_CHARS = 512


def hash_args(args: dict[str, Any]) -> str:
    canonical = json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def digest_result(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, sort_keys=True, default=str)
        except (TypeError, ValueError):
            text = str(value)
    return text[:_DIGEST_MAX_CHARS]


async def record_tool_call(
    *,
    job_id: str | None,
    tool_name: str,
    args: dict[str, Any] | None,
    result: Any = None,
    latency_ms: int = 0,
    cost_usd: float = 0.0,
    status_str: str = "ok",
    error: str | None = None,
    owner_api_url: str | None = None,
    owner_api_token: str | None = None,
) -> None:
    """Best-effort POST of one tool-call row to the orchestrator ledger.

    A missing ``job_id`` (validator smoke test, direct curl) or missing
    owner-api config silently skips; production miner traffic always
    carries ``X-Eirel-Job-Id`` so this is a fast no-op when irrelevant.
    """
    if not job_id:
        return
    url = owner_api_url or os.getenv("EIREL_OWNER_API_URL", "")
    if not url:
        return
    token = owner_api_token or os.getenv("EIREL_INTERNAL_SERVICE_TOKEN", "")
    args_dict = args or {}
    payload = {
        "job_id": job_id,
        "tool_name": tool_name,
        "args_hash": hash_args(args_dict),
        "args_json": args_dict,
        "result_digest": digest_result(result),
        "latency_ms": int(max(0, latency_ms)),
        "cost_usd": float(max(0.0, cost_usd)),
        "status": status_str,
        "error": error,
    }
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    endpoint = f"{url.rstrip('/')}/v1/internal/eval/tool_calls"
    try:
        async with httpx.AsyncClient(timeout=_RECORD_TIMEOUT_SECONDS) as client:
            await client.post(endpoint, json=payload, headers=headers)
    except (httpx.HTTPError, OSError) as exc:
        _logger.warning(
            "record_tool_call failed: job=%s tool=%s err=%s",
            job_id, tool_name, exc,
        )


__all__ = ["record_tool_call", "hash_args", "digest_result"]
