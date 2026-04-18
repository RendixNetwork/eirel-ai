from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import defaultdict
from typing import Any, Protocol, runtime_checkable

import httpx

from shared.common.tool_pricing import cost_for_call
from shared.core.evaluation_models import ConversationTrace, TraceEntry
from shared.core.honeytokens import inject_honeytokens_into_search_payload


_logger = logging.getLogger(__name__)


@runtime_checkable
class TraceStore(Protocol):
    async def append(self, conversation_id: str, entry: TraceEntry) -> None: ...
    async def get_trace(self, conversation_id: str) -> ConversationTrace: ...
    async def clear(self, conversation_id: str) -> None: ...
    async def clear_many(self, conversation_ids: list[str]) -> None: ...


class InMemoryTraceStore:
    """Per-conversation trace store.

    Threadsafe via a single asyncio lock because control-plane workers run
    on a single event loop. Conversations are keyed by ``conversation_id``
    which the miner runtime is expected to propagate via the
    ``X-Eirel-Conversation-Id`` header on every tool-proxy request.
    """

    def __init__(self) -> None:
        self._traces: dict[str, list[TraceEntry]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def append(self, conversation_id: str, entry: TraceEntry) -> None:
        async with self._lock:
            self._traces[conversation_id].append(entry)

    async def get_trace(self, conversation_id: str) -> ConversationTrace:
        async with self._lock:
            entries = list(self._traces.get(conversation_id, []))
        return ConversationTrace(conversation_id=conversation_id, entries=entries)

    async def clear(self, conversation_id: str) -> None:
        async with self._lock:
            self._traces.pop(conversation_id, None)

    async def clear_many(self, conversation_ids: list[str]) -> None:
        async with self._lock:
            for cid in conversation_ids:
                self._traces.pop(cid, None)

    def snapshot_for_tests(self) -> dict[str, list[TraceEntry]]:
        """Synchronous snapshot used exclusively by tests."""
        return {
            conversation_id: list(entries)
            for conversation_id, entries in self._traces.items()
        }


_REDIS_CLIENT_REF: Any = None

_BODY_EXCERPT_MAX_CHARS = 2048


def _digest_result(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, (bytes, bytearray)):
        return hashlib.sha256(bytes(payload)).hexdigest()
    try:
        import json

        serialized = json.dumps(payload, sort_keys=True, default=str)
    except (TypeError, ValueError):
        serialized = str(payload)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _body_excerpt(payload: Any) -> str:
    """Return a lowercased, length-capped excerpt of a tool response body.

    Used by ``verify_trace_integrity`` to do substring / token-overlap
    checks against cited content. Structure-preserving JSON serialization
    keeps field names available for matching.
    """
    if payload is None:
        return ""
    if isinstance(payload, (bytes, bytearray)):
        try:
            text = bytes(payload).decode("utf-8", errors="replace")
        except Exception:  # pragma: no cover - defensive fallback
            text = ""
    else:
        try:
            import json

            text = json.dumps(payload, default=str)
        except (TypeError, ValueError):
            text = str(payload)
    return text[:_BODY_EXCERPT_MAX_CHARS].lower()


class ToolBudgetExhaustedError(RuntimeError):
    """Raised when the provider proxy rejects a tool charge due to budget exhaustion."""


class ToolProxy:
    """Thin forwarder from owner-api to a downstream tool service.

    Records a :class:`TraceEntry` per call so the scoring pipeline can
    later verify the miner's attribution claims match what actually
    happened. The proxy does not own retry logic — downstream services
    implement their own backoff; we only record the final outcome.

    When ``provider_proxy_url``, ``provider_proxy_token``, and ``job_id``
    are supplied, each tool call is charged against the run's USD budget
    via ``POST /v1/jobs/{job_id}/charge_tool`` before the downstream
    request is issued.
    """

    def __init__(
        self,
        *,
        store: TraceStore,
        http_client: httpx.AsyncClient | None = None,
        provider_proxy_url: str | None = None,
        provider_proxy_token: str | None = None,
        job_id: str | None = None,
        active_honeytokens: list[str] | None = None,
        honeytoken_injection_rate: float = 0.02,
    ) -> None:
        self._store = store
        self._client = http_client or httpx.AsyncClient(timeout=30.0)
        self._owns_client = http_client is None
        self._provider_proxy_url = provider_proxy_url
        self._provider_proxy_token = provider_proxy_token
        self._job_id = job_id
        self._active_honeytokens = list(active_honeytokens or [])
        self._honeytoken_rate = honeytoken_injection_rate
        # Per-conversation counter for deterministic injection keying.
        self._call_counters: dict[str, int] = {}

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _charge_tool_budget(self, tool_name: str, amount_usd: float) -> None:
        if not (self._provider_proxy_url and self._job_id):
            return
        charge_url = f"{self._provider_proxy_url}/v1/jobs/{self._job_id}/charge_tool"
        charge_headers: dict[str, str] = {}
        if self._provider_proxy_token:
            charge_headers["Authorization"] = f"Bearer {self._provider_proxy_token}"
        resp = await self._client.post(
            charge_url,
            json={"tool_name": tool_name, "amount_usd": amount_usd},
            headers=charge_headers,
        )
        if resp.status_code == 429:
            raise ToolBudgetExhaustedError(
                f"run budget exhausted for tool {tool_name}"
            )
        resp.raise_for_status()

    async def proxy_call(
        self,
        *,
        conversation_id: str,
        tool_name: str,
        target_url: str,
        args: dict[str, Any],
        headers: dict[str, str] | None = None,
        method: str = "POST",
    ) -> dict[str, Any]:
        cost_usd = cost_for_call(tool_name)

        try:
            await self._charge_tool_budget(tool_name, cost_usd)
        except ToolBudgetExhaustedError:
            entry = TraceEntry(
                tool_name=tool_name,
                args=dict(args),
                result_digest="",
                latency_ms=0,
                cost_usd=0.0,
                metadata={
                    "status_code": 429,
                    "error": "run_budget_exhausted",
                    "target_url": target_url,
                },
            )
            await self._store.append(conversation_id, entry)
            raise

        t0 = time.perf_counter()
        error_text: str | None = None
        payload: Any = None
        status_code = 0
        try:
            response = await self._client.request(
                method=method,
                url=target_url,
                json=args if method.upper() != "GET" else None,
                params=args if method.upper() == "GET" else None,
                headers=headers or {},
            )
            status_code = response.status_code
            try:
                payload = response.json()
            except ValueError:
                payload = response.text
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            error_text = f"http_status_{exc.response.status_code}"
        except httpx.HTTPError as exc:
            error_text = f"http_error:{type(exc).__name__}"
        latency_ms = int((time.perf_counter() - t0) * 1000)

        # Honeytoken injection — only applies to search-style tools that
        # return a list payload. Runs after the real fetch so the injected
        # entry is visible to the miner alongside legitimate results. Rate
        # is intentionally low (~2%) so honest miners rarely encounter one.
        if self._active_honeytokens and error_text is None:
            call_index = self._call_counters.get(conversation_id, 0)
            self._call_counters[conversation_id] = call_index + 1
            payload = inject_honeytokens_into_search_payload(
                payload,
                active_set=self._active_honeytokens,
                conversation_id=conversation_id,
                call_index=call_index,
                rate=self._honeytoken_rate,
            )
        entry = TraceEntry(
            tool_name=tool_name,
            args=dict(args),
            result_digest=_digest_result(payload),
            result_body_excerpt=_body_excerpt(payload),
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            metadata={
                "status_code": status_code,
                "error": error_text or "",
                "target_url": target_url,
            },
        )
        await self._store.append(conversation_id, entry)
        return {
            "tool_name": tool_name,
            "status_code": status_code,
            "latency_ms": latency_ms,
            "cost_usd": cost_usd,
            "payload": payload,
            "error": error_text,
        }


# Convenience singleton so simple call sites can share a store without
# passing it explicitly. Tests should construct their own store via
# :class:`InMemoryTraceStore` rather than relying on the singleton.
_default_store: TraceStore | None = None


def default_trace_store() -> TraceStore:
    global _default_store, _REDIS_CLIENT_REF
    if _default_store is None:
        from shared.common.config import get_settings

        settings = get_settings()
        if settings.trace_store_backend == "redis":
            from redis import asyncio as redis_asyncio

            from control_plane.owner_api.middleware.redis_trace_store import (
                RedisTraceStore,
            )

            url = settings.trace_store_redis_url or settings.redis_url
            client = redis_asyncio.from_url(url, decode_responses=True)
            _REDIS_CLIENT_REF = client
            _default_store = RedisTraceStore(client)
        else:
            _default_store = InMemoryTraceStore()
    return _default_store


async def _close_redis_client() -> None:
    global _REDIS_CLIENT_REF
    if _REDIS_CLIENT_REF is not None:
        try:
            await _REDIS_CLIENT_REF.aclose()
        except Exception:
            pass
        _REDIS_CLIENT_REF = None


def reset_default_trace_store() -> None:
    global _default_store, _REDIS_CLIENT_REF
    _REDIS_CLIENT_REF = None
    _default_store = None
