from __future__ import annotations

"""Miner-invocation helper for the validator loop.

Prefers the streaming endpoint (``POST /v1/agent/infer/stream``, NDJSON)
because that's the wire format the consumer chat UI uses; the validator
exercises the same path so a streaming-only regression is caught here
rather than in production. Falls back to the legacy non-streaming
endpoint (``POST /v1/agent/infer``) on a 404 so miners on eirel SDK
< 0.2.3 keep working. Only completion latency is recorded — TTFB is not
measured or scored.
"""

import asyncio
import json
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
# stretching the validator's per-task budget.
_RETRYABLE_STATUS_CODES = frozenset({502, 503, 504})
_RETRY_BACKOFF_SECONDS = 1.0


def _build_body(*, task: Any, prompt: str, family_id: str, task_id: str) -> dict[str, Any]:
    expected_output = getattr(task, "expected_output", {}) or {}
    inputs = getattr(task, "inputs", {}) or {}
    metadata = dict(getattr(task, "metadata", {}) or {})
    return {
        "task_id": task_id,
        "family_id": family_id,
        "primary_goal": prompt,
        "subtask": prompt,
        "inputs": (
            {**inputs, "expected_output": expected_output} if expected_output else inputs
        ),
        "metadata": metadata,
    }


def _auth_headers(miner: MinerBenchmarkTarget) -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    auth_headers = (miner.metadata or {}).get("auth_headers") or {}
    if isinstance(auth_headers, dict):
        headers.update({str(k): str(v) for k, v in auth_headers.items()})
    return headers


async def _invoke_stream(
    *,
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: float,
) -> dict[str, Any]:
    """Stream NDJSON chunks from the miner's `/v1/agent/infer/stream`.

    Returns a payload mirroring the non-streaming response body shape —
    assembled from `delta` chunks (concatenated into output.answer) and
    the final `done` chunk's fields. The rest of the pipeline doesn't
    care whether streaming or unary was used.

    Raises HTTPStatusError on non-2xx (caller decides whether to fall back
    or retry). Raises httpx.HTTPError for network failures.
    """
    accumulated_text: list[str] = []
    citations: list[Any] = []
    tool_calls: list[dict[str, Any]] = []
    final_output: dict[str, Any] = {}
    final_metadata: dict[str, Any] = {}
    final_status: str | None = None
    final_error: str | None = None

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        async with client.stream("POST", url, json=body, headers=headers) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    _logger.warning("malformed NDJSON chunk from miner: %r", line[:120])
                    continue
                event = chunk.get("event")
                if event == "delta":
                    text = chunk.get("text") or ""
                    if text:
                        accumulated_text.append(text)
                elif event == "citation":
                    cit = chunk.get("citation")
                    if cit is not None:
                        citations.append(cit)
                elif event == "tool_call":
                    tc = chunk.get("tool_call")
                    if isinstance(tc, dict):
                        tool_calls.append(tc)
                elif event == "done":
                    if isinstance(chunk.get("output"), dict):
                        final_output = chunk["output"]
                    if isinstance(chunk.get("citations"), list):
                        # `done.citations` is the source of truth — the SDK
                        # mirrors the non-streaming response. Per-chunk
                        # `citation` events are bonus diagnostics for now.
                        citations = list(chunk["citations"])
                    if isinstance(chunk.get("tool_calls"), list):
                        tool_calls = list(chunk["tool_calls"])
                    if isinstance(chunk.get("metadata"), dict):
                        final_metadata = chunk["metadata"]
                    final_status = chunk.get("status") or "completed"
                    final_error = chunk.get("error")

    # If the agent never sent a final output but emitted deltas, build one
    # from the accumulated text so the judge has something to compare.
    if not final_output and accumulated_text:
        final_output = {"answer": "".join(accumulated_text)}
    elif accumulated_text and "answer" not in final_output:
        final_output = {**final_output, "answer": "".join(accumulated_text)}

    payload: dict[str, Any] = {
        "task_id": body.get("task_id"),
        "family_id": body.get("family_id"),
        "status": final_status or "completed",
        "output": final_output,
        "citations": citations,
        "tool_calls": tool_calls,
        "metadata": final_metadata,
    }
    if final_error:
        payload["error"] = final_error
    return payload


async def _invoke_unary(
    *,
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: float,
) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        resp = await client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        return resp.json() if resp.content else {}


async def _invoke_task(
    *,
    miner: MinerBenchmarkTarget,
    task: Any,
    timeout_seconds: float = 90.0,
) -> BenchmarkTaskRun:
    """POST a single task to a miner's endpoint and wrap the response.

    Tries the streaming endpoint first. Falls back to the legacy unary
    endpoint on 404. Records total completion latency in `metadata`.
    """
    endpoint = (miner.endpoint or "").rstrip("/")
    prompt = getattr(task, "prompt", "") or ""
    family_id = getattr(task, "family_id", "general_chat")
    task_id = getattr(task, "task_id", "")
    expected_output = getattr(task, "expected_output", {}) or {}
    metadata = dict(getattr(task, "metadata", {}) or {})

    if not endpoint:
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

    headers = _auth_headers(miner)
    body = _build_body(
        task=task, prompt=prompt, family_id=family_id, task_id=task_id,
    )
    stream_url = f"{endpoint}/v1/agent/infer/stream"
    unary_url = f"{endpoint}/v1/agent/infer"

    t0 = time.perf_counter()
    attempts = 2  # one initial + one retry on transient upstream errors
    last_exc: Exception | None = None
    payload: dict[str, Any] = {}
    used_stream = True

    for attempt in range(1, attempts + 1):
        try:
            if used_stream:
                payload = await _invoke_stream(
                    url=stream_url, body=body, headers=headers,
                    timeout_seconds=timeout_seconds,
                )
            else:
                payload = await _invoke_unary(
                    url=unary_url, body=body, headers=headers,
                    timeout_seconds=timeout_seconds,
                )
            last_exc = None
            break
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            status_code = exc.response.status_code
            # 404 on the stream URL → miner is on an older SDK without
            # the streaming route. Fall back permanently for this call.
            if used_stream and status_code == 404:
                _logger.info(
                    "miner %s lacks streaming endpoint (404); falling back to unary: task=%s",
                    miner.hotkey[:16], task_id,
                )
                used_stream = False
                t0 = time.perf_counter()  # reset clock for the unary attempt
                continue
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

    elapsed = time.perf_counter() - t0
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
            metadata={**metadata, "latency_seconds": elapsed},
        )

    out_metadata: dict[str, Any] = {**metadata, "latency_seconds": elapsed}
    out_metadata["streamed"] = used_stream

    # If the miner's `done` chunk explicitly reported failed, surface it on
    # the run object so the validator's _judge_miner gates this as an error
    # (otherwise we'd quietly score a failed agent call as a completed
    # response with no answer text).
    payload_status = (
        payload.get("status") if isinstance(payload, dict) else None
    ) or "completed"
    payload_error = payload.get("error") if isinstance(payload, dict) else None

    return BenchmarkTaskRun(
        task_id=task_id,
        family_id=family_id,
        prompt=prompt,
        expected_output=expected_output,
        response=payload if isinstance(payload, dict) else {"raw": payload},
        status="completed" if payload_status == "completed" else "failed",
        error=payload_error if payload_status != "completed" else None,
        metadata=out_metadata,
    )
