from __future__ import annotations

"""Minimal miner-invocation helper for the validator loop.

The legacy benchmark package composed this helper out of many family-aware
utilities (builder long-running, research ledger fetch, multimodal
artifact evaluation). For the general_chat-only world we only need a
thin HTTP client call that pushes a prompt to the miner and captures
the response as a :class:`BenchmarkTaskRun`.
"""

import asyncio
import logging
import time
from typing import Any

import httpx

from shared.core.evaluation_models import BenchmarkTaskRun, MinerBenchmarkTarget


_logger = logging.getLogger(__name__)

# Upstream-transient status codes we retry once on. 502/503/504 are the
# canonical "your downstream is unavailable" signals; the owner-api
# internal proxy returns 502 when the miner pod or provider-proxy times
# out. A single short retry recovers from most chutes hiccups without
# stretching the validator's 90s per-task budget.
_RETRYABLE_STATUS_CODES = frozenset({502, 503, 504})
_RETRY_BACKOFF_SECONDS = 1.0


async def _invoke_task(
    *,
    miner: MinerBenchmarkTarget,
    task: Any,
    timeout_seconds: float = 90.0,
) -> BenchmarkTaskRun:
    """POST a single task to a miner's endpoint and wrap the response."""
    endpoint = (miner.endpoint or "").rstrip("/")
    prompt = getattr(task, "prompt", "") or ""
    family_id = getattr(task, "family_id", "general_chat")
    task_id = getattr(task, "task_id", "")
    expected_output = getattr(task, "expected_output", {}) or {}
    inputs = getattr(task, "inputs", {}) or {}
    metadata = dict(getattr(task, "metadata", {}) or {})

    headers: dict[str, str] = {"Content-Type": "application/json"}
    auth_headers = (miner.metadata or {}).get("auth_headers") or {}
    if isinstance(auth_headers, dict):
        headers.update({str(k): str(v) for k, v in auth_headers.items()})

    body = {
        "task_id": task_id,
        "family_id": family_id,
        "primary_goal": prompt,
        "subtask": prompt,
        "inputs": {**inputs, "expected_output": expected_output} if expected_output else inputs,
        "metadata": metadata,
    }

    url = f"{endpoint}/v1/agent/infer" if endpoint else ""
    if not url:
        return BenchmarkTaskRun(
            task_id=task_id,
            family_id=family_id,
            prompt=prompt,
            expected_output=expected_output,
            response={},
            status="failed",
            error="missing_miner_endpoint",
            metadata=metadata,
        )

    t0 = time.perf_counter()
    attempts = 2  # one initial + one retry on transient upstream errors
    last_exc: Exception | None = None
    payload: dict[str, Any] = {}
    for attempt in range(1, attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                resp = await client.post(url, json=body, headers=headers)
                resp.raise_for_status()
                payload = resp.json() if resp.content else {}
            last_exc = None
            break
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            status_code = exc.response.status_code
            if status_code in _RETRYABLE_STATUS_CODES and attempt < attempts:
                _logger.warning(
                    "miner invocation hit %d on attempt %d/%d, retrying: task=%s miner=%s",
                    status_code, attempt, attempts, task_id, miner.hotkey[:16],
                )
                await asyncio.sleep(_RETRY_BACKOFF_SECONDS)
                continue
            break
        except httpx.HTTPError as exc:
            last_exc = exc
            break
    if last_exc is not None:
        _logger.warning("miner invocation failed: %s", last_exc)
        return BenchmarkTaskRun(
            task_id=task_id,
            family_id=family_id,
            prompt=prompt,
            expected_output=expected_output,
            response={},
            status="failed",
            error=str(last_exc),
            metadata={**metadata, "latency_seconds": time.perf_counter() - t0},
        )

    return BenchmarkTaskRun(
        task_id=task_id,
        family_id=family_id,
        prompt=prompt,
        expected_output=expected_output,
        response=payload if isinstance(payload, dict) else {"raw": payload},
        status="completed",
        error=None,
        metadata={**metadata, "latency_seconds": time.perf_counter() - t0},
    )
