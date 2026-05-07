from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import PlainTextResponse

from pydantic import BaseModel as _BaseModel
from pydantic import Field as _Field

from redis import asyncio as redis_asyncio

from shared.common.tool_pricing import LLM_PRICING, llm_cost_for
from tool_platforms.provider_proxy.chutes_pricing import (
    run_chutes_pricing_refresh_loop,
)
from tool_platforms.provider_proxy.models import (
    PROVIDER_ENV_MAP,
    ProviderProxyRequest,
    ProviderProxyResponse,
    ProviderProxyUsage,
    SUPPORTED_PROVIDER_IDS,
)
from tool_platforms.provider_proxy.redis_store import (
    JobUsage,
    ProviderJobStore,
)

logger = logging.getLogger(__name__)


class ChargeToolRequest(_BaseModel):
    tool_name: str
    amount_usd: float = _Field(ge=0.0)
    span_id: str | None = None
    parent_span_id: str | None = None


class ReserveBatchEstimate(_BaseModel):
    estimated_cost: float = _Field(ge=0.0)
    estimated_tokens: int = _Field(ge=0)
    provider: str
    model: str
    span_id: str | None = None


class ReserveBatchRequest(_BaseModel):
    """Atomic batch reservation for graph parallel-tool dispatch.

    All ``estimates`` must fit under the run budget *together* — if the
    sum would exceed it, none are reserved and the call returns a 429.
    This eliminates the race where N concurrent reserve_estimate calls
    each pass an individually-affordable check but collectively blow
    the cap.
    """

    max_usd_budget: float = _Field(ge=0.0)
    estimates: list[ReserveBatchEstimate]


@dataclass(slots=True)
class AppState:
    auth_token: str
    store: ProviderJobStore
    metrics: dict[str, float]


def _make_redis_client(url: str) -> redis_asyncio.Redis:
    """Factory isolated for tests to monkeypatch with fakeredis."""
    return redis_asyncio.from_url(url, decode_responses=True)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        redis_client = _make_redis_client(redis_url)
        store = ProviderJobStore(redis_client)
        app.state.services = AppState(
            auth_token=os.getenv("EIREL_PROVIDER_PROXY_MASTER_TOKEN", "")
            or os.getenv("PROVIDER_PROXY_AUTH_TOKEN", ""),
            store=store,
            metrics={
                "requests_total": 0.0,
                "quota_rejections_total": 0.0,
                "provider_openai_requests_total": 0.0,
                "provider_anthropic_requests_total": 0.0,
                "provider_openrouter_requests_total": 0.0,
                "provider_chutes_requests_total": 0.0,
            },
        )
        # Keep the dynamic Chutes pricing overlay fresh so we never bill
        # miners at a stale rate after Chutes publishes a price change.
        # Refresh runs once at startup and then every ~1 h; disabled
        # when EIREL_CHUTES_PRICING_REFRESH_ENABLED=false (tests).
        pricing_task: asyncio.Task | None = None
        if os.getenv("EIREL_CHUTES_PRICING_REFRESH_ENABLED", "true").lower() not in (
            "0", "false", "no",
        ):
            pricing_task = asyncio.create_task(run_chutes_pricing_refresh_loop())
        try:
            yield
        finally:
            if pricing_task is not None:
                pricing_task.cancel()
                try:
                    await pricing_task
                except (asyncio.CancelledError, Exception):
                    pass
            await redis_client.aclose()

    app = FastAPI(title="Eirel Provider Proxy", version="0.1.0", lifespan=lifespan)

    async def require_auth(authorization: str | None = Header(default=None)) -> str:
        auth_token: str = app.state.services.auth_token
        if not auth_token:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="provider proxy is not configured",
            )
        if authorization != f"Bearer {auth_token}":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid provider proxy token",
            )
        return auth_token

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(
            content=_provider_metrics_text(app.state.services.metrics),
            media_type="text/plain; version=0.0.4",
        )

    @app.get("/v1/operators/summary")
    async def operator_summary() -> dict[str, Any]:
        jobs = await app.state.services.store.list_all()
        active_jobs = {
            job_id: {
                "request_count": usage.request_count,
                "estimated_total_tokens": usage.estimated_total_tokens,
                "actual_total_tokens": usage.actual_total_tokens,
                "provider_request_counts": dict(usage.provider_request_counts),
                "model_request_counts": dict(usage.model_request_counts),
                "started_at_monotonic": usage.started_at,
                "elapsed_seconds": max(time.monotonic() - usage.started_at, 0.0),
            }
            for job_id, usage in sorted(jobs.items())
        }
        return {
            "active_job_count": len(active_jobs),
            "metrics": dict(app.state.services.metrics),
            "active_jobs": active_jobs,
        }

    @app.get("/v1/jobs/{job_id}/usage")
    async def job_usage(job_id: str, _: str = Depends(require_auth)) -> dict[str, Any]:
        usage = await app.state.services.store.get(job_id)
        if usage is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="provider proxy job usage not found",
            )
        return {
            "job_id": job_id,
            "request_count": usage.request_count,
            "estimated_total_tokens": usage.estimated_total_tokens,
            "actual_total_tokens": usage.actual_total_tokens,
            "provider_request_counts": dict(usage.provider_request_counts),
            "model_request_counts": dict(usage.model_request_counts),
            "elapsed_seconds": max(time.monotonic() - usage.started_at, 0.0),
        }

    @app.get("/v1/jobs/{job_id}/cost")
    async def job_cost(job_id: str, _: str = Depends(require_auth)) -> dict[str, Any]:
        usage = await app.state.services.store.get(job_id)
        if usage is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="job not found",
            )
        # ``cost_by_provider`` keys come in two shapes:
        #   * bare LLM provider names ("chutes", "openai", ...) —
        #     written by /chat/completions reconciliation
        #   * ``tool:<name>`` — written by ``charge_tool``
        # Split on those prefixes so ScoringManager.populate_cost_columns
        # gets truthful ``DeploymentScoreRecord.{llm_cost_usd,tool_cost_usd}``
        # instead of lumping everything together.
        llm_cost_usd = 0.0
        tool_cost_usd = 0.0
        for key, value in usage.cost_by_provider.items():
            if key.startswith("tool:"):
                tool_cost_usd += float(value)
            else:
                llm_cost_usd += float(value)
        # Aggregate per-span totals across buckets (tool/llm).
        cost_by_span_totals: dict[str, float] = {}
        for key, value in (usage.cost_by_span or {}).items():
            span_id, _, _bucket = key.partition("::")
            cost_by_span_totals[span_id] = round(
                cost_by_span_totals.get(span_id, 0.0) + float(value), 8
            )
        return {
            "cost_usd_used": usage.cost_usd_used,
            "llm_cost_usd": round(llm_cost_usd, 8),
            "tool_cost_usd": round(tool_cost_usd, 8),
            "max_usd_budget": usage.max_usd_budget,
            "cost_rejections": usage.cost_rejections,
            "per_provider": dict(usage.cost_by_provider),
            "per_span": cost_by_span_totals,
            "per_span_buckets": dict(usage.cost_by_span or {}),
            "span_parents": dict(usage.span_parents or {}),
        }

    @app.post("/v1/jobs/{job_id}/charge_tool")
    async def charge_tool(
        job_id: str,
        body: ChargeToolRequest,
        _: str = Depends(require_auth),
        x_eirel_span_id: str | None = Header(default=None),
        x_eirel_parent_span_id: str | None = Header(default=None),
    ) -> dict[str, Any]:
        store = app.state.services.store
        if not await store.exists(job_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="job not found",
            )
        # Header takes precedence over body.span_id so callers that wire
        # tracing once at the HTTP layer don't have to plumb span ids
        # through every payload. Body fields are kept as a fallback for
        # codepaths that can't set headers (e.g. retries on a queued tool
        # call replay).
        span_id = x_eirel_span_id or body.span_id
        parent_span_id = x_eirel_parent_span_id or body.parent_span_id
        accepted, cost_used = await store.charge_tool(
            job_id=job_id,
            tool_name=body.tool_name,
            amount_usd=body.amount_usd,
            span_id=span_id,
            parent_span_id=parent_span_id,
        )
        if not accepted:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="run budget exhausted",
            )
        return {"cost_usd_used": cost_used, "span_id": span_id}

    @app.post("/v1/jobs/{job_id}/reserve_batch_estimate")
    async def reserve_batch_estimate(
        job_id: str,
        body: ReserveBatchRequest,
        _: str = Depends(require_auth),
    ) -> dict[str, Any]:
        """Atomic batch reservation for parallel-tool dispatch.

        On success returns the per-estimate ``span_id`` of every
        reservation that landed alongside the new ``cost_usd_used``.
        On budget exhaustion, NONE of the estimates are recorded and
        the call 429s — that's the whole point of batching.
        """
        store = app.state.services.store
        if not body.estimates:
            return {"cost_usd_used": 0.0, "reserved": []}
        accepted, cost_used = await store.reserve_batch_estimate(
            job_id=job_id,
            max_usd_budget=body.max_usd_budget,
            estimates=[
                {
                    "estimated_cost": e.estimated_cost,
                    "estimated_tokens": e.estimated_tokens,
                    "provider": e.provider,
                    "model": e.model,
                    "span_id": e.span_id,
                }
                for e in body.estimates
            ],
        )
        if not accepted:
            app.state.services.metrics["quota_rejections_total"] += 1
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="run budget exhausted",
            )
        return {
            "cost_usd_used": cost_used,
            "reserved": [e.span_id for e in body.estimates],
        }

    @app.post("/v1/chat/completions", response_model=ProviderProxyResponse)
    async def chat_completions(
        payload: ProviderProxyRequest,
        _: str = Depends(require_auth),
        x_eirel_job_id: str | None = Header(default=None),
        x_eirel_run_budget_usd: str | None = Header(default=None),
    ) -> ProviderProxyResponse:
        if payload.provider not in SUPPORTED_PROVIDER_IDS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="unsupported provider",
            )
        if not x_eirel_job_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="missing job id header",
            )
        if not x_eirel_run_budget_usd:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="missing run budget header",
            )
        max_usd_budget = float(x_eirel_run_budget_usd)
        estimated_tokens = _estimate_tokens(payload.payload)
        max_output_tokens = int(payload.payload.get("max_tokens") or 4096)
        estimated_cost = _estimate_usd_cost(
            payload.provider, payload.model, estimated_tokens, max_output_tokens,
        )
        store = app.state.services.store
        accepted, cost_used_after_reserve = await store.reserve_estimate(
            job_id=x_eirel_job_id,
            estimated_cost=estimated_cost,
            max_usd_budget=max_usd_budget,
            estimated_tokens=estimated_tokens,
            provider=payload.provider,
            model=payload.model,
        )
        if not accepted:
            app.state.services.metrics["quota_rejections_total"] += 1
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="run budget exhausted",
            )
        app.state.services.metrics["requests_total"] += 1
        app.state.services.metrics[f"provider_{payload.provider}_requests_total"] += 1

        started = time.perf_counter()
        try:
            upstream_response = await asyncio.to_thread(
                _dispatch_provider_request,
                provider=payload.provider,
                model=payload.model,
                payload=payload.payload,
                timeout_seconds=payload.per_request_timeout_seconds,
            )
        except BaseException:
            # Refund the pre-reserved estimate if the upstream call
            # never completed — otherwise the estimate leaks into
            # ``cost_usd_used`` without a matching ``cost_by_provider``
            # entry and shows up as ghost cost in downstream reporting.
            try:
                await store.refund_estimate(
                    job_id=x_eirel_job_id,
                    estimated_cost=estimated_cost,
                    estimated_tokens=estimated_tokens,
                )
            except Exception as refund_exc:
                logger.warning(
                    "refund_estimate failed for job=%s: %s",
                    x_eirel_job_id, refund_exc,
                )
            raise
        latency = max(0.0, time.perf_counter() - started)

        actual_usage = _extract_upstream_usage(payload.provider, upstream_response)

        actual_cost = llm_cost_for(
            provider=payload.provider,
            model=payload.model,
            prompt_tokens=actual_usage["prompt_tokens"],
            completion_tokens=actual_usage["completion_tokens"],
            reasoning_tokens=actual_usage.get("reasoning_tokens", 0),
        )
        await store.reconcile_actual_cost(
            job_id=x_eirel_job_id,
            provider=payload.provider,
            delta_cost=actual_cost - estimated_cost,
            actual_cost=actual_cost,
            actual_total_tokens=actual_usage["total_tokens"],
        )

        snap = await store.get(x_eirel_job_id)
        snap_cost_used = snap.cost_usd_used if snap else 0.0
        snap_max_budget = snap.max_usd_budget if snap else max_usd_budget
        snap_request_count = snap.request_count if snap else 1
        snap_estimated_tokens = snap.estimated_total_tokens if snap else estimated_tokens

        logger.info(
            "provider_call: job=%s provider=%s model=%s latency=%.2fs "
            "prompt_tokens=%d completion_tokens=%d total_tokens=%d cost_usd=%.6f",
            x_eirel_job_id, payload.provider, payload.model, latency,
            actual_usage["prompt_tokens"],
            actual_usage["completion_tokens"],
            actual_usage["total_tokens"],
            actual_cost,
        )

        return ProviderProxyResponse(
            upstream_response=upstream_response,
            usage=ProviderProxyUsage(
                provider=payload.provider,
                model=payload.model,
                request_count=snap_request_count,
                estimated_total_tokens=snap_estimated_tokens,
                max_requests=payload.max_requests,
                max_total_tokens=payload.max_total_tokens,
                actual_prompt_tokens=actual_usage["prompt_tokens"],
                actual_completion_tokens=actual_usage["completion_tokens"],
                actual_total_tokens=actual_usage["total_tokens"],
                latency_seconds=round(latency, 4),
                cost_usd=round(actual_cost, 8),
                cost_usd_used=round(snap_cost_used, 8),
                cost_remaining_usd=round(max(0.0, snap_max_budget - snap_cost_used), 8),
            ),
        )

    return app


def _estimate_usd_cost(
    provider: str, model: str, prompt_tokens: int, max_output_tokens: int,
) -> float:
    key = f"{provider}:{model}"
    price = LLM_PRICING.get(key) or LLM_PRICING.get(f"{provider}:*")
    if price is None:
        return 0.0
    return (
        prompt_tokens * price.input_per_mtok_usd
        + max_output_tokens * price.output_per_mtok_usd
    ) / 1_000_000


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------


_RETRYABLE_STATUS_CODES = {429, 502, 503, 504}
_MAX_RETRIES = int(os.getenv("EIREL_PROVIDER_PROXY_MAX_RETRIES", "3"))
_RETRY_BASE_DELAY = float(os.getenv("EIREL_PROVIDER_PROXY_RETRY_BASE_DELAY", "1.0"))


def _dispatch_provider_request(
    *,
    provider: str,
    model: str,
    payload: dict[str, Any],
    timeout_seconds: int,
) -> dict[str, Any]:
    base_url_env, api_key_env, default_url = PROVIDER_ENV_MAP[provider]
    url = os.getenv(base_url_env, default_url)
    api_key = os.getenv(api_key_env, "")
    if not api_key:
        if _mock_provider_enabled():
            return _mock_provider_response(provider=provider, model=model, payload=payload)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"provider {provider} is not configured",
        )

    headers, request_body = _build_provider_request(
        provider=provider, model=model, api_key=api_key, payload=payload,
    )
    raw = json.dumps(request_body).encode("utf-8")

    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        req = urllib.request.Request(url, data=raw, headers=dict(headers), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
                upstream = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            if exc.code in _RETRYABLE_STATUS_CODES and attempt < _MAX_RETRIES:
                retry_after = _parse_retry_after(exc.headers.get("Retry-After"))
                delay = retry_after if retry_after is not None else _RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "provider_retry: provider=%s model=%s status=%d attempt=%d/%d delay=%.1fs",
                    provider, model, exc.code, attempt + 1, _MAX_RETRIES, delay,
                )
                time.sleep(delay)
                last_exc = exc
                continue
            logger.warning(
                "provider_error: provider=%s model=%s status=%d detail=%s",
                provider, model, exc.code, detail[:500],
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"provider upstream error ({provider}): {detail or exc.reason}",
            ) from exc
        except urllib.error.URLError as exc:
            # Don't retry timeouts — they're unlikely to succeed and waste time
            logger.warning("provider_unavailable: provider=%s error=%s", provider, exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"provider upstream unavailable ({provider}): {exc}",
            ) from exc
    else:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"provider upstream error ({provider}): retries exhausted",
        ) from last_exc

    # Normalize Anthropic response back to OpenAI format
    if provider == "anthropic":
        upstream = _anthropic_response_to_openai(upstream)

    return upstream


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a Retry-After header value (seconds) if present."""
    if not value:
        return None
    try:
        return max(0.5, min(float(value), 30.0))
    except (ValueError, TypeError):
        return None


def _build_provider_request(
    *,
    provider: str,
    model: str,
    api_key: str,
    payload: dict[str, Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Build provider-specific headers and request body."""
    if provider == "anthropic":
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        body = _openai_payload_to_anthropic(model, payload)
        return headers, body

    # OpenAI, OpenRouter, Chutes — all OpenAI-compatible
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = dict(payload)
    body["model"] = model
    return headers, body


# ---------------------------------------------------------------------------
# Anthropic format converters
# ---------------------------------------------------------------------------


def _openai_payload_to_anthropic(model: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI-format chat completion request to Anthropic Messages format.

    - Extracts system messages into top-level ``system`` field.
    - Strips system role from ``messages``.
    - Ensures ``max_tokens`` is set (required by Anthropic).
    """
    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []
    for message in payload.get("messages", []):
        role = message.get("role")
        content = message.get("content")
        if role == "system" and isinstance(content, str):
            system_parts.append(content)
            continue
        if role in {"user", "assistant"}:
            messages.append({"role": role, "content": content or ""})

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": payload.get("max_tokens") or 4096,
    }
    if system_parts:
        body["system"] = "\n".join(system_parts)
    if "temperature" in payload:
        body["temperature"] = payload["temperature"]
    if "top_p" in payload:
        body["top_p"] = payload["top_p"]
    if "stop" in payload:
        body["stop_sequences"] = (
            payload["stop"] if isinstance(payload["stop"], list)
            else [payload["stop"]]
        )
    return body


def _anthropic_response_to_openai(response: dict[str, Any]) -> dict[str, Any]:
    """Convert an Anthropic Messages response to OpenAI chat completion format.

    Normalizes content blocks, stop_reason, and usage fields so the miner
    always receives a consistent OpenAI-shaped response regardless of provider.
    """
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in response.get("content", []):
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(str(block.get("text", "")))
        elif block_type == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                },
            })

    message: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts)}
    if tool_calls:
        message["tool_calls"] = tool_calls

    # Normalize usage: Anthropic → OpenAI field names
    anthropic_usage = response.get("usage", {})
    prompt_tokens = int(anthropic_usage.get("input_tokens", 0))
    completion_tokens = int(anthropic_usage.get("output_tokens", 0))

    return {
        "id": response.get("id", ""),
        "object": "chat.completion",
        "model": response.get("model", ""),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": response.get("stop_reason") or "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# ---------------------------------------------------------------------------
# Token usage extraction
# ---------------------------------------------------------------------------


def _extract_upstream_usage(provider: str, response: dict[str, Any]) -> dict[str, int]:
    """Extract actual token counts from the (already-normalized) upstream response."""
    usage = response.get("usage", {})
    if not isinstance(usage, dict):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0}
    prompt = int(usage.get("prompt_tokens", 0))
    completion = int(usage.get("completion_tokens", 0))
    total = int(usage.get("total_tokens", 0)) or (prompt + completion)
    reasoning = 0
    details = usage.get("completion_tokens_details")
    if isinstance(details, dict):
        reasoning = int(details.get("reasoning_tokens", 0))
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "reasoning_tokens": reasoning,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_provider_enabled() -> bool:
    return os.getenv("EIREL_PROVIDER_PROXY_ALLOW_MOCK", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _mock_provider_response(
    *,
    provider: str,
    model: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    latest_user_message = ""
    for message in reversed(payload.get("messages", [])):
        if isinstance(message, dict) and message.get("role") == "user":
            latest_user_message = str(message.get("content", "")).strip()
            if latest_user_message:
                break
    content = (
        "Staging mock provider response."
        if not latest_user_message
        else f"Staging mock provider response for: {latest_user_message}"
    )
    return {
        "id": "mock-provider-response",
        "object": "chat.completion",
        "provider": provider,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": _estimate_tokens(payload),
            "completion_tokens": max(1, len(content) // 4),
            "total_tokens": _estimate_tokens(payload) + max(1, len(content) // 4),
        },
    }


def _estimate_tokens(payload: dict[str, Any]) -> int:
    try:
        messages = payload.get("messages", [])
        serialized = json.dumps(messages, sort_keys=True)
        return max(1, len(serialized) // 4)
    except Exception:
        return 1


def _provider_metrics_text(metrics: dict[str, float]) -> str:
    lines = [
        "# HELP eirel_provider_proxy_up Provider-proxy service is serving metrics.",
        "# TYPE eirel_provider_proxy_up gauge",
        "eirel_provider_proxy_up 1",
        "# HELP eirel_provider_proxy_requests_total Total upstream provider requests accepted.",
        "# TYPE eirel_provider_proxy_requests_total counter",
        f"eirel_provider_proxy_requests_total {int(metrics['requests_total'])}",
        "# HELP eirel_provider_proxy_quota_rejections_total Total provider requests rejected by quota.",
        "# TYPE eirel_provider_proxy_quota_rejections_total counter",
        f"eirel_provider_proxy_quota_rejections_total {int(metrics['quota_rejections_total'])}",
        "# HELP eirel_provider_proxy_provider_requests_total Total provider requests by provider.",
        "# TYPE eirel_provider_proxy_provider_requests_total counter",
    ]
    for provider in SUPPORTED_PROVIDER_IDS:
        lines.append(
            f'eirel_provider_proxy_provider_requests_total{{provider="{provider}"}} '
            f"{int(metrics[f'provider_{provider}_requests_total'])}"
        )
    return "\n".join(lines) + "\n"


app = create_app()
