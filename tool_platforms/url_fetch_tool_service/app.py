"""FastAPI app for the URL-fetch tool service.

Mirrors the auth + ledger pattern of ``sandbox_tool_service``: bearer
token (master) + per-job HMAC token, per-job request-count cap via the
shared :class:`JobLedger`, per-host rate-limit + body-size cap on the
HTTP fetch itself.
"""
from __future__ import annotations

import asyncio
import hmac
import os
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import httpx
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from shared.common.redis_job_ledger import (
    JobLedger,
    JobUsageRecord,
    create_job_ledger,
)
from tool_platforms._charge_tool import charge_tool_cost
from tool_platforms._record_tool_call import record_tool_call
from tool_platforms.url_fetch_tool_service.extractor import (
    ExtractedDocument,
    extract_text,
)
from tool_platforms.url_fetch_tool_service.ssrf import (
    UrlFetchSSRFError,
    validate_url,
)


# Hard cap on response body bytes — protects against memory blowups
# from a hostile or unbounded server. Configurable per-deploy but
# enforced per-request inside the streaming read.
_DEFAULT_MAX_RESPONSE_BYTES = 1 * 1024 * 1024  # 1 MB
_DEFAULT_FETCH_TIMEOUT_SECONDS = 15.0
_DEFAULT_MAX_REDIRECTS = 5
_DEFAULT_PER_HOST_RATE = 4  # max requests per window per host
_DEFAULT_PER_HOST_WINDOW_SECONDS = 10.0


def generate_job_token(master_token: str, job_id: str) -> str:
    return hmac.new(master_token.encode(), job_id.encode(), "sha256").hexdigest()


def verify_job_token(master_token: str, job_id: str, token: str) -> bool:
    expected = generate_job_token(master_token, job_id)
    return hmac.compare_digest(expected, token)


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


# -- Request / response models ----------------------------------------------


class UrlFetchRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)
    max_chars: int = Field(default=50_000, ge=100, le=500_000)
    include_links: bool = True


class UrlFetchResponse(BaseModel):
    url: str
    final_url: str
    title: str
    content: str
    links: list[dict[str, str]] = Field(default_factory=list)
    status_code: int
    content_type: str
    bytes_read: int
    truncated: bool


# -- Per-host rate limiter --------------------------------------------------


class _PerHostRateLimiter:
    """Token-bucket-style sliding-window rate limit, per-host.

    In-process state — fine for the single-replica deployment we ship
    with. Multi-replica scale-out moves to Redis with the same window
    semantics.
    """

    def __init__(self, *, max_per_window: int, window_seconds: float) -> None:
        self._max = max_per_window
        self._window = window_seconds
        self._buckets: dict[str, list[float]] = {}
        self._lock = asyncio.Lock()

    async def check(self, host: str) -> bool:
        now = time.monotonic()
        async with self._lock:
            bucket = self._buckets.setdefault(host, [])
            cutoff = now - self._window
            while bucket and bucket[0] < cutoff:
                bucket.pop(0)
            if len(bucket) >= self._max:
                return False
            bucket.append(now)
            return True


# -- App factory ------------------------------------------------------------


def create_app(
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    ledger: JobLedger | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.auth_token = os.getenv("EIREL_URL_FETCH_API_TOKEN", "")
        app.state.default_max_requests = int(
            os.getenv("EIREL_URL_FETCH_DEFAULT_MAX_REQUESTS", "12")
        )
        app.state.max_response_bytes = int(
            os.getenv("EIREL_URL_FETCH_MAX_RESPONSE_BYTES", str(_DEFAULT_MAX_RESPONSE_BYTES))
        )
        app.state.fetch_timeout_seconds = float(
            os.getenv("EIREL_URL_FETCH_TIMEOUT_SECONDS", str(_DEFAULT_FETCH_TIMEOUT_SECONDS))
        )
        app.state.max_redirects = int(
            os.getenv("EIREL_URL_FETCH_MAX_REDIRECTS", str(_DEFAULT_MAX_REDIRECTS))
        )
        per_host_rate = int(
            os.getenv("EIREL_URL_FETCH_PER_HOST_RATE", str(_DEFAULT_PER_HOST_RATE))
        )
        per_host_window = float(
            os.getenv(
                "EIREL_URL_FETCH_PER_HOST_WINDOW_SECONDS",
                str(_DEFAULT_PER_HOST_WINDOW_SECONDS),
            )
        )
        app.state.rate_limiter = _PerHostRateLimiter(
            max_per_window=per_host_rate, window_seconds=per_host_window,
        )
        app.state.transport = transport
        app.state.user_agent = os.getenv(
            "EIREL_URL_FETCH_USER_AGENT", "EirelUrlFetch/0.1 (+https://eirel.ai)",
        )
        if ledger is not None:
            app.state.job_ledger = ledger
        else:
            app.state.job_ledger = create_job_ledger(os.getenv("REDIS_URL", ""))
        app.state.metrics = {
            "requests_total": 0,
            "quota_rejections_total": 0,
            "ssrf_blocks_total": 0,
            "rate_limit_blocks_total": 0,
            "fetch_failures_total": 0,
            "size_cap_truncations_total": 0,
        }
        try:
            yield
        finally:
            await app.state.job_ledger.close()

    app = FastAPI(
        title="url-fetch-tool-service",
        version="0.1.0",
        lifespan=lifespan,
    )

    async def require_auth(
        authorization: str | None = Header(default=None),
        x_eirel_job_id: str | None = Header(default=None),
    ) -> None:
        auth_token: str = app.state.auth_token
        if not auth_token:
            return
        if authorization == f"Bearer {auth_token}":
            return
        if x_eirel_job_id and authorization:
            bearer = authorization.removeprefix("Bearer ").strip()
            if verify_job_token(auth_token, x_eirel_job_id.strip(), bearer):
                return
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid url-fetch tool auth token",
        )

    async def require_job(
        x_eirel_job_id: str | None = Header(default=None),
        x_eirel_max_requests: str | None = Header(default=None),
    ) -> tuple[str, int]:
        job_id = (x_eirel_job_id or "").strip()
        if not job_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="missing X-Eirel-Job-Id header",
            )
        max_requests = app.state.default_max_requests
        if x_eirel_max_requests:
            try:
                max_requests = max(1, int(x_eirel_max_requests))
            except ValueError:
                pass
        return job_id, max_requests

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "now": _utcnow_iso()}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> str:
        m: dict[str, Any] = app.state.metrics
        lines = [
            "# HELP eirel_url_fetch_requests_total Successful URL fetch requests.",
            "# TYPE eirel_url_fetch_requests_total counter",
            f"eirel_url_fetch_requests_total {m['requests_total']}",
            "# HELP eirel_url_fetch_quota_rejections_total Job-level quota rejections.",
            "# TYPE eirel_url_fetch_quota_rejections_total counter",
            f"eirel_url_fetch_quota_rejections_total {m['quota_rejections_total']}",
            "# HELP eirel_url_fetch_ssrf_blocks_total SSRF policy denials.",
            "# TYPE eirel_url_fetch_ssrf_blocks_total counter",
            f"eirel_url_fetch_ssrf_blocks_total {m['ssrf_blocks_total']}",
            "# HELP eirel_url_fetch_rate_limit_blocks_total Per-host rate-limit denials.",
            "# TYPE eirel_url_fetch_rate_limit_blocks_total counter",
            f"eirel_url_fetch_rate_limit_blocks_total {m['rate_limit_blocks_total']}",
            "# HELP eirel_url_fetch_failures_total Upstream fetch failures.",
            "# TYPE eirel_url_fetch_failures_total counter",
            f"eirel_url_fetch_failures_total {m['fetch_failures_total']}",
            "# HELP eirel_url_fetch_truncations_total Responses truncated by size cap.",
            "# TYPE eirel_url_fetch_truncations_total counter",
            f"eirel_url_fetch_truncations_total {m['size_cap_truncations_total']}",
        ]
        return "\n".join(lines) + "\n"

    async def check_budget(
        *, job_id: str, max_requests: int, tool_name: str
    ) -> JobUsageRecord:
        ledger: JobLedger = app.state.job_ledger
        usage = await ledger.get_or_create(job_id)
        if usage.request_count >= max_requests:
            app.state.metrics["quota_rejections_total"] += 1
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={"error": "url_fetch_quota_exhausted", "max_requests": max_requests},
            )
        usage.request_count += 1
        usage.tool_counts[tool_name] = usage.tool_counts.get(tool_name, 0) + 1
        await ledger.save(job_id, usage)
        return usage

    @app.post("/v1/fetch", response_model=UrlFetchResponse)
    async def fetch(
        body: UrlFetchRequest,
        _: None = Depends(require_auth),
        job: tuple[str, int] = Depends(require_job),
    ) -> UrlFetchResponse:
        job_id, max_requests = job

        # SSRF guard: validate scheme + DNS + IP ranges before any
        # network IO. Fails loud so attempts are visible in metrics.
        try:
            scheme, host = validate_url(body.url)
        except UrlFetchSSRFError as exc:
            app.state.metrics["ssrf_blocks_total"] += 1
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "ssrf_blocked", "reason": str(exc)},
            )
        del scheme  # unused — validation enforces http/https

        # Per-host rate limit applied AFTER SSRF (so SSRF metrics are
        # accurate even under sustained probes).
        if not await app.state.rate_limiter.check(host):
            app.state.metrics["rate_limit_blocks_total"] += 1
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={"error": "per_host_rate_limit_exceeded", "host": host},
            )

        # Per-job quota check + ledger increment, AFTER SSRF + rate-limit
        # so a blocked URL never burns budget.
        await check_budget(
            job_id=job_id, max_requests=max_requests, tool_name="url_fetch",
        )

        client_kwargs: dict[str, Any] = {
            "timeout": app.state.fetch_timeout_seconds,
            "follow_redirects": True,
            "max_redirects": app.state.max_redirects,
            "headers": {"User-Agent": app.state.user_agent},
        }
        if app.state.transport is not None:
            client_kwargs["transport"] = app.state.transport

        try:
            async with httpx.AsyncClient(**client_kwargs) as client:
                async with client.stream("GET", body.url) as resp:
                    chunks: list[bytes] = []
                    truncated = False
                    cap = app.state.max_response_bytes
                    kept = 0
                    async for chunk in resp.aiter_bytes():
                        if not chunk:
                            continue
                        if kept + len(chunk) > cap:
                            # Take only the head up to the cap.
                            remaining = max(0, cap - kept)
                            if remaining:
                                chunks.append(chunk[:remaining])
                                kept += remaining
                            truncated = True
                            break
                        chunks.append(chunk)
                        kept += len(chunk)
                    raw = b"".join(chunks)
                    bytes_read = kept
                    final_url = str(resp.url)
                    status_code = resp.status_code
                    content_type = resp.headers.get("content-type", "")
        except httpx.HTTPError as exc:
            app.state.metrics["fetch_failures_total"] += 1
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={"error": "fetch_failed", "reason": str(exc)},
            ) from None

        # Best-effort provider-proxy charge so per-task tool cost
        # surfaces in DeploymentScoreRecord.tool_cost_usd. Failures
        # are logged but never break the fetch.
        per_call_cost = float(os.getenv("EIREL_URL_FETCH_PER_CALL_COST_USD", "0.0005"))
        await charge_tool_cost(
            job_id=job_id, tool_name="url_fetch", amount_usd=per_call_cost,
        )
        await record_tool_call(
            job_id=job_id, tool_name="url_fetch",
            args={
                "url": body.url, "max_chars": body.max_chars,
                "include_links": body.include_links,
            },
            result={
                "final_url": final_url, "status_code": status_code,
                "bytes_read": bytes_read, "content_type": content_type,
            },
            cost_usd=per_call_cost,
            status_str="ok" if status_code < 400 else "error",
        )

        if truncated:
            app.state.metrics["size_cap_truncations_total"] += 1
        app.state.metrics["requests_total"] += 1

        # Heuristic: only run the HTML extractor if content-type smells
        # like HTML. Otherwise return the raw text best-effort decoded.
        is_html = "html" in content_type.lower()
        if is_html:
            try:
                text = raw.decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                text = raw.decode("latin-1", errors="replace")
            extracted = extract_text(
                text,
                base_url=final_url,
                max_chars=body.max_chars,
                include_links=body.include_links,
            )
            return UrlFetchResponse(
                url=body.url,
                final_url=final_url,
                title=extracted.title,
                content=extracted.content,
                links=extracted.links,
                status_code=status_code,
                content_type=content_type,
                bytes_read=bytes_read,
                truncated=truncated,
            )

        # Non-HTML: surface raw text up to the cap. Useful for
        # plain-text/markdown/json — the LLM consumer can interpret.
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            text = raw.decode("latin-1", errors="replace")
        if len(text) > body.max_chars:
            text = text[: body.max_chars] + "..."
        return UrlFetchResponse(
            url=body.url,
            final_url=final_url,
            title="",
            content=text,
            links=[],
            status_code=status_code,
            content_type=content_type,
            bytes_read=bytes_read,
            truncated=truncated,
        )

    return app


def main() -> None:
    port = int(os.getenv("EIREL_URL_FETCH_PORT", "8087"))
    uvicorn.run(
        "tool_platforms.url_fetch_tool_service.app:create_app",
        host="0.0.0.0",
        port=port,
        factory=True,
    )
