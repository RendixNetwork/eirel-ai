"""Streaming chat passthrough for the orchestrator.

The orchestrator owns conversation state (per the multi-family
architecture); for now we only route to ``general_chat``, but the
session bookkeeping and the wire shape are already family-agnostic so
adding a DAG composition path later is local to this module.

Two responsibilities:

  * **Session state** — load/upsert the per-session ``mode`` and
    ``web_search`` toggles on ``ConsumerSessionState``. The toggles
    persist across turns so the consumer-chat-api (and the user's
    browser tab) don't have to re-assert them on every prompt.

  * **Miner proxy** — resolve the current serving deployment for the
    target family from owner-api, build the slim 0.3.0 invocation body,
    and stream the miner's NDJSON back to the caller line-by-line.
    Falls back to the unary endpoint with a synthetic ``delta`` +
    ``done`` if the miner pod doesn't expose ``/v1/agent/infer/stream``
    (eirel SDK < 0.2.3).
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx

from shared.common.database import Database
from shared.common.models import ConsumerSessionState, utcnow

_logger = logging.getLogger(__name__)

OWNER_API_URL = os.getenv("OWNER_API_URL", "http://owner-api:8000")
INTERNAL_SERVICE_TOKEN = os.getenv("EIREL_INTERNAL_SERVICE_TOKEN", "")
# Total wall-clock budget for a streaming chat. Should clear the slowest
# acceptable miner completion (thinking mode = 600s today).
_CHAT_STREAM_TIMEOUT_SECONDS = float(
    os.getenv("EIREL_ORCHESTRATOR_CHAT_STREAM_TIMEOUT_SECONDS", "660")
)


class ChatSessionStore:
    """Thin facade over ``ConsumerSessionState`` for the chat path.

    All writes go through this so the orchestrator owns the single
    write-path to the session row. Consumer-chat-api will eventually be
    a stateless facade — no DB writes from there.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    def resolve_toggles(
        self,
        *,
        session_id: str | None,
        user_id: str,
        mode_override: str | None,
        web_search_override: bool | None,
    ) -> tuple[str, str, bool]:
        """Load (and upsert) mode/web_search for the session.

        Resolution rules:
          * If ``session_id`` is None, generate a new one and persist
            with the override values (or defaults).
          * If the session exists, override values win when present;
            otherwise reuse the stored row values.
          * If the session is missing, create it with override-or-default.

        Returns ``(session_id, mode, web_search)``.
        """
        sid = session_id or str(uuid.uuid4())
        with self._db.sessionmaker() as s:
            row = s.get(ConsumerSessionState, sid)
            if row is None:
                row = ConsumerSessionState(
                    session_id=sid,
                    user_id=user_id,
                    status="active",
                    messages_json=[],
                    mode=mode_override or "instant",
                    web_search=bool(web_search_override) if web_search_override is not None else False,
                )
                s.add(row)
            else:
                if mode_override is not None:
                    row.mode = mode_override
                if web_search_override is not None:
                    row.web_search = bool(web_search_override)
                row.updated_at = utcnow()
            mode = row.mode
            web_search = bool(row.web_search)
            s.commit()
        return sid, mode, web_search


async def _resolve_serving_endpoint(family_id: str) -> dict[str, Any] | None:
    """Look up the currently serving deployment for *family_id* in owner-api.

    Resolution order:
      1. ``EIREL_ORCHESTRATOR_MINER_OVERRIDE_ENDPOINT`` env override —
         test-only, bypasses owner-api entirely.
      2. ``/v1/internal/serving/{family_id}`` — production winner pod
         that won the most recent epoch for this family.
      3. ``/v1/internal/managed-deployments/active/{family_id}`` —
         fallback when no serving release has been published yet
         (e.g. the first run after a fresh subnet bring-up). Useful so
         the chat surface still works in early-cluster states.
    """
    override = os.getenv("EIREL_ORCHESTRATOR_MINER_OVERRIDE_ENDPOINT", "").strip()
    if override:
        return {"endpoint": override, "hotkey": "override", "family_id": family_id}

    headers: dict[str, str] = {}
    if INTERNAL_SERVICE_TOKEN:
        headers["Authorization"] = f"Bearer {INTERNAL_SERVICE_TOKEN}"

    paths = [
        f"/v1/internal/serving/{family_id}",
        f"/v1/internal/managed-deployments/active/{family_id}",
    ]
    async with httpx.AsyncClient(timeout=10.0) as client:
        for path in paths:
            try:
                resp = await client.get(f"{OWNER_API_URL}{path}", headers=headers)
                if resp.status_code == 404:
                    continue
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, dict) and data.get("endpoint"):
                    return data
            except Exception as exc:
                _logger.warning(
                    "orchestrator: serving lookup %s for %s failed: %s",
                    path, family_id, exc,
                )
                continue
    return None


def _ndjson(chunk: dict[str, Any]) -> bytes:
    return (json.dumps(chunk, separators=(",", ":")) + "\n").encode("utf-8")


def _build_invocation_body(
    *,
    prompt: str,
    mode: str,
    web_search: bool,
    history: list[dict[str, Any]],
    turn_id: str,
    family_id: str,
) -> dict[str, Any]:
    """Construct the slim 0.3.0 family-agent body, with legacy fallbacks.

    Mirrors validator's ``_build_body`` so the orchestrator and the
    eval pipeline send identical wire shapes — the family agent only
    learns one contract.
    """
    body: dict[str, Any] = {
        # Slim 0.3.0 contract.
        "turn_id": turn_id,
        "prompt": prompt,
        "mode": mode,
        "web_search": web_search,
        "history": history,
    }
    # Legacy mirror suppressed when ``EIREL_VALIDATOR_SLIM_ONLY=1`` —
    # same kill switch the validator uses, so a slim-only test exercises
    # both paths consistently. Drops in 0.4.0 either way.
    if os.getenv("EIREL_VALIDATOR_SLIM_ONLY", "0") not in {"1", "true", "yes"}:
        body.update({
            "task_id": turn_id,
            "family_id": family_id,
            "primary_goal": prompt,
            "subtask": prompt,
            "inputs": {"mode": mode, "web_search": web_search},
            "context_history": history,
        })
    return body


async def stream_family_chat(
    *,
    store: ChatSessionStore,
    family_id: str,
    prompt: str,
    user_id: str,
    session_id: str | None,
    context_history: list[dict[str, Any]],
    mode_override: str | None,
    web_search_override: bool | None,
) -> AsyncIterator[bytes]:
    """Yield NDJSON StreamChunks for one chat turn.

    Sequence:
      1. Resolve session toggles (and create the session row if new).
      2. Resolve serving family deployment endpoint via owner-api.
      3. Stream the miner's ``/v1/agent/infer/stream``, line-by-line.
      4. On 404 streaming, fall back to the unary endpoint and emit a
         synthetic single ``delta`` + ``done`` so the wire contract is
         identical regardless of miner SDK version.
      5. Always end with a ``done`` chunk — error or success.
    """
    sid, mode, web_search = store.resolve_toggles(
        session_id=session_id,
        user_id=user_id,
        mode_override=mode_override,
        web_search_override=web_search_override,
    )
    turn_id = f"chat-{uuid.uuid4().hex[:12]}"

    # The very first chunk is informational so the consumer-chat-api can
    # echo session_id + family back to the browser before content arrives.
    yield _ndjson({
        "event": "started",
        "metadata": {
            "session_id": sid,
            "family_id": family_id,
            "mode": mode,
            "web_search": web_search,
            "turn_id": turn_id,
        },
    })

    miner = await _resolve_serving_endpoint(family_id)
    if miner is None:
        yield _ndjson({
            "event": "done",
            "status": "failed",
            "error": f"no serving deployment available for family {family_id}",
        })
        return

    history = [
        {"role": h.get("role"), "content": h.get("content")}
        for h in (context_history or [])
        if isinstance(h, dict) and h.get("role") in ("user", "assistant")
    ]
    body = _build_invocation_body(
        prompt=prompt,
        mode=mode,
        web_search=web_search,
        history=history,
        turn_id=turn_id,
        family_id=family_id,
    )

    endpoint = miner["endpoint"].rstrip("/")
    stream_url = f"{endpoint}/v1/agent/infer/stream"
    unary_url = f"{endpoint}/v1/agent/infer"

    started_at = time.monotonic()
    used_stream = True
    try:
        async with httpx.AsyncClient(timeout=_CHAT_STREAM_TIMEOUT_SECONDS) as client:
            async with client.stream("POST", stream_url, json=body) as resp:
                if resp.status_code == 404:
                    used_stream = False
                else:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        yield (line + "\n").encode("utf-8")
                    return
    except httpx.HTTPError as exc:
        _logger.warning(
            "orchestrator: stream call to %s failed: %s",
            stream_url, exc,
        )
        used_stream = False

    if not used_stream:
        try:
            async with httpx.AsyncClient(timeout=_CHAT_STREAM_TIMEOUT_SECONDS) as client:
                resp = await client.post(unary_url, json=body)
                resp.raise_for_status()
                payload = resp.json() if resp.content else {}
        except Exception as exc:
            yield _ndjson({"event": "done", "status": "failed", "error": str(exc)})
            return

        text = ""
        if isinstance(payload, dict):
            out = payload.get("output") or {}
            if isinstance(out, dict):
                for key in ("answer", "response", "text", "content", "message"):
                    val = out.get(key)
                    if isinstance(val, str) and val:
                        text = val
                        break
        if text:
            yield _ndjson({"event": "delta", "text": text})
        yield _ndjson({
            "event": "done",
            "status": payload.get("status") if isinstance(payload, dict) else "completed",
            "output": payload.get("output") if isinstance(payload, dict) else {},
            "citations": payload.get("citations") if isinstance(payload, dict) else [],
            "metadata": {
                **(payload.get("metadata", {}) if isinstance(payload, dict) else {}),
                "fallback": "unary",
                "elapsed_seconds": round(time.monotonic() - started_at, 3),
            },
        })


__all__ = ["ChatSessionStore", "stream_family_chat"]
