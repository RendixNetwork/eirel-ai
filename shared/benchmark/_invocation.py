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
import os
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


def _build_body(
    *,
    task: Any,
    prompt: str,
    family_id: str,
    task_id: str,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the slim 0.3.0 invocation body.

    Family agents are stateless specialists; they receive only the
    prompt, the per-turn knobs, and any prior conversation. Anything
    eval-internal (``expected_output``, ``category``, ``difficulty``,
    grading hints) stays server-side and never crosses the wire — old
    builds leaked the answer key here via ``inputs.expected_output``.

    By default we also populate the legacy fields
    (``primary_goal``/``subtask``/``inputs.{mode,web_search}``) so
    miners on 0.2.x keep working through the migration window. Set
    ``EIREL_VALIDATOR_SLIM_ONLY=1`` to suppress the legacy mirror —
    used to airtight-test that miners are reading the 0.3.0 contract
    (any miner that secretly relied on the legacy fields will
    immediately fail the run with empty prompts).
    """
    inputs = getattr(task, "inputs", {}) or {}
    mode = inputs.get("mode") or "instant"
    web_search = bool(inputs.get("web_search", False))

    # Multi-turn replay: caller passes ``history`` accumulated from
    # prior turns of the same fixture. Single-turn tasks pass None /
    # empty. We don't read ``inputs.history`` anymore — that was only
    # relevant in the pre-Phase-B shape.
    cleaned_history = [
        {"role": h.get("role"), "content": h.get("content")}
        for h in (history or [])
        if isinstance(h, dict) and h.get("role") in ("user", "assistant")
    ]

    body: dict[str, Any] = {
        # Slim 0.3.0 contract — what new miners read.
        "turn_id": task_id,
        "prompt": prompt,
        "mode": mode,
        "web_search": web_search,
        "history": cleaned_history,
    }
    if os.getenv("EIREL_VALIDATOR_SLIM_ONLY", "0") not in {"1", "true", "yes"}:
        body.update({
            "task_id": task_id,
            "family_id": family_id,
            "primary_goal": prompt,
            "subtask": prompt,
            "inputs": {"mode": mode, "web_search": web_search},
        })
    return body


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
                    if isinstance(chunk.get("metadata"), dict):
                        final_metadata = chunk["metadata"]
                    # Tool calls: 0.3.0 emits them under
                    # ``metadata.executed_tool_calls``; 0.2.x emitted a
                    # top-level ``tool_calls``. Read both during the
                    # migration window — slim wins when both present.
                    meta_tcs = final_metadata.get("executed_tool_calls")
                    if isinstance(meta_tcs, list):
                        tool_calls = list(meta_tcs)
                    elif isinstance(chunk.get("tool_calls"), list):
                        tool_calls = list(chunk["tool_calls"])
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


async def _invoke_one_turn(
    *,
    miner: MinerBenchmarkTarget,
    task: Any,
    prompt: str,
    history: list[dict[str, Any]],
    turn_id: str,
    timeout_seconds: float,
) -> tuple[dict[str, Any], float, bool, Exception | None]:
    """Invoke the miner for one turn (streaming-first, unary fallback).

    Returns ``(payload, elapsed_seconds, used_stream, last_exc)``.
    Caller decides whether to surface ``last_exc`` as a failure or
    treat the partial payload as recoverable. Used by both single-turn
    and multi-turn replay paths.
    """
    endpoint = (miner.endpoint or "").rstrip("/")
    family_id = getattr(task, "family_id", "general_chat")
    headers = _auth_headers(miner)
    body = _build_body(
        task=task,
        prompt=prompt,
        family_id=family_id,
        task_id=turn_id,
        history=history,
    )
    stream_url = f"{endpoint}/v1/agent/infer/stream"
    unary_url = f"{endpoint}/v1/agent/infer"

    t0 = time.perf_counter()
    attempts = 2
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
            if used_stream and status_code == 404:
                _logger.info(
                    "miner %s lacks streaming endpoint (404); falling back to unary: turn=%s",
                    miner.hotkey[:16], turn_id,
                )
                used_stream = False
                t0 = time.perf_counter()
                continue
            if status_code in _RETRYABLE_STATUS_CODES and attempt < attempts:
                _logger.warning(
                    "miner invocation hit %d on attempt %d/%d, retrying: turn=%s miner=%s",
                    status_code, attempt, attempts, turn_id, miner.hotkey[:16],
                )
                await asyncio.sleep(_RETRY_BACKOFF_SECONDS)
                continue
            break
        except httpx.HTTPError as exc:
            last_exc = exc
            break

    return payload, time.perf_counter() - t0, used_stream, last_exc


def _extract_answer_text(payload: dict[str, Any] | None) -> str:
    """Pull the assistant text out of a normalised invocation payload.

    Used by the multi-turn replay loop to feed each turn's reply back
    into the next turn's history.
    """
    if not isinstance(payload, dict):
        return ""
    out = payload.get("output") or {}
    if isinstance(out, dict):
        for key in ("answer", "response", "text", "content", "message"):
            v = out.get(key)
            if isinstance(v, str) and v:
                return v
    return ""


async def _invoke_task(
    *,
    miner: MinerBenchmarkTarget,
    task: Any,
    timeout_seconds: float = 90.0,
) -> BenchmarkTaskRun:
    """POST a single task (single-turn or multi-turn) to a miner.

    Single-turn (``task.turns`` empty/None): one HTTP call, history is
    empty. Multi-turn (``task.turns`` populated): replay each turn in
    sequence, accumulating the miner's own replies as ``assistant``
    history entries between turns. Scripted turns (``assistant`` set on
    the fixture) are inserted into history without calling the miner;
    live turns (``assistant=None``) call the miner and record its
    reply. The miner is always called for the final turn, and its
    answer is what the judge scores.

    Recorded latency in ``metadata.latency_seconds`` is the **sum** of
    per-turn wall clocks (matches "total time the user waited"). The
    per-turn max is recorded as ``metadata.max_turn_latency_seconds``
    so the validator's mode-budget gate can fire on any turn that
    overruns. ``metadata.turns`` carries the per-turn breakdown for the
    dashboard.
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

    # Build the turn script.
    raw_turns = list(getattr(task, "turns", None) or [])
    if not raw_turns:
        # Single-turn: synthesize a one-turn script from the legacy
        # ``prompt`` field so the loop below is uniform.
        raw_turns = [{"user": prompt, "assistant": None}]

    history: list[dict[str, Any]] = []
    turn_breakdown: list[dict[str, Any]] = []
    final_payload: dict[str, Any] = {}
    final_used_stream = True
    final_status = "completed"
    final_error: str | None = None
    total_elapsed = 0.0
    max_turn_elapsed = 0.0

    for idx, raw in enumerate(raw_turns):
        is_last = (idx == len(raw_turns) - 1)
        if hasattr(raw, "user"):
            user_text = raw.user
            scripted = raw.assistant
        elif isinstance(raw, dict):
            user_text = str(raw.get("user") or "")
            scripted = raw.get("assistant")
        else:
            continue
        if not isinstance(user_text, str) or not user_text:
            continue

        # Scripted intermediate turn — no miner call, just inject the
        # canned exchange into history.
        if scripted is not None and not is_last:
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": str(scripted)})
            turn_breakdown.append({
                "turn_index": idx,
                "scripted": True,
                "latency_seconds": 0.0,
            })
            continue

        # Live turn — call the miner.
        turn_id = f"{task_id}-t{idx}" if len(raw_turns) > 1 else task_id
        payload, elapsed, used_stream, exc = await _invoke_one_turn(
            miner=miner,
            task=task,
            prompt=user_text,
            history=list(history),
            turn_id=turn_id,
            timeout_seconds=timeout_seconds,
        )
        total_elapsed += elapsed
        if elapsed > max_turn_elapsed:
            max_turn_elapsed = elapsed
        if exc is not None:
            _logger.warning(
                "miner invocation failed at turn %d/%d: %s",
                idx + 1, len(raw_turns), exc,
            )
            return BenchmarkTaskRun(
                task_id=task_id,
                family_id=family_id,
                prompt=prompt,
                expected_output=expected_output,
                response={},
                status="failed",
                error=str(exc),
                metadata={
                    **metadata,
                    "latency_seconds": total_elapsed,
                    "max_turn_latency_seconds": max_turn_elapsed,
                    "failed_at_turn": idx,
                    "turns": turn_breakdown + [{
                        "turn_index": idx,
                        "scripted": False,
                        "latency_seconds": elapsed,
                        "error": str(exc),
                    }],
                },
            )

        miner_reply = _extract_answer_text(payload)
        # Append this turn to history before moving on (last-turn append
        # is harmless — judge reads the payload directly).
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": miner_reply})
        turn_breakdown.append({
            "turn_index": idx,
            "scripted": False,
            "latency_seconds": elapsed,
            "streamed": used_stream,
        })
        final_payload = payload
        final_used_stream = used_stream
        # Surface a per-turn `done.status: failed` from the miner.
        ps = (payload.get("status") if isinstance(payload, dict) else None) or "completed"
        if ps != "completed":
            final_status = "failed"
            final_error = payload.get("error") if isinstance(payload, dict) else None

    out_metadata: dict[str, Any] = {
        **metadata,
        "latency_seconds": total_elapsed,
        "max_turn_latency_seconds": max_turn_elapsed,
        "streamed": final_used_stream,
        "turns": turn_breakdown,
        "turn_count": len(raw_turns),
    }

    return BenchmarkTaskRun(
        task_id=task_id,
        family_id=family_id,
        prompt=prompt,
        expected_output=expected_output,
        response=final_payload if isinstance(final_payload, dict) else {"raw": final_payload},
        status="completed" if final_status == "completed" else "failed",
        error=final_error if final_status != "completed" else None,
        metadata=out_metadata,
    )
