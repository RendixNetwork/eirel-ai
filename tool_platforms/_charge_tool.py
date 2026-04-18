from __future__ import annotations

"""Shared best-effort helper for reporting per-request tool cost.

Tool services (web_search, x, semantic_scholar, sandbox) call into this
after handling a successful miner request so the provider-proxy can
attribute the USD cost to the job. The attribution is fire-and-forget:
failure to reach the proxy must never break the tool handler — the
penalty is at most an undercharge in one ``DeploymentScoreRecord``.

The provider-proxy endpoint accepts either the master
``EIREL_PROVIDER_PROXY_TOKEN`` or a per-job HMAC-derived token. Tool
services run on the control-plane side so they hold the master token
via ``EIREL_PROVIDER_PROXY_TOKEN`` / ``EIREL_INTERNAL_SERVICE_TOKEN``.
"""

import logging
import os

import httpx


_logger = logging.getLogger(__name__)

_CHARGE_TOOL_TIMEOUT_SECONDS = 5.0


async def charge_tool_cost(
    *,
    job_id: str | None,
    tool_name: str,
    amount_usd: float,
    proxy_url: str | None = None,
    proxy_token: str | None = None,
) -> None:
    """Fire-and-forget charge to the provider-proxy.

    A missing ``job_id`` (validator smoke test, direct curl) or missing
    proxy config silently skips — the design assumption is that
    production miner traffic always carries ``X-Eirel-Job-Id``.
    """
    if amount_usd <= 0.0:
        return
    if not job_id:
        return
    url = proxy_url or os.getenv("EIREL_PROVIDER_PROXY_URL", "")
    if not url:
        return
    token = proxy_token or os.getenv("EIREL_PROVIDER_PROXY_TOKEN", "")
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    endpoint = f"{url.rstrip('/')}/v1/jobs/{job_id}/charge_tool"
    payload = {"tool_name": tool_name, "amount_usd": float(amount_usd)}
    try:
        async with httpx.AsyncClient(timeout=_CHARGE_TOOL_TIMEOUT_SECONDS) as client:
            await client.post(endpoint, json=payload, headers=headers)
    except (httpx.HTTPError, OSError) as exc:
        # Best-effort: missing cost attribution is preferable to failing
        # the user-facing tool response.
        _logger.warning(
            "charge_tool failed: job=%s tool=%s amount=%.6f err=%s",
            job_id, tool_name, amount_usd, exc,
        )


__all__ = ["charge_tool_cost"]
