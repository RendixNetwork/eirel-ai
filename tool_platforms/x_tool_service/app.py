from __future__ import annotations

import hashlib
import hmac
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import httpx
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import PlainTextResponse

from shared.common.redis_job_ledger import (
    JobLedger,
    JobUsageRecord,
    create_job_ledger,
)
from tool_platforms._charge_tool import charge_tool_cost
from tool_platforms.x_tool_service.models import (
    XSearchRequest,
    XSearchResponse,
    XTweet,
)

DEFAULT_X_API_BASE_URL = "https://api.x.com/2"

# X API is an expensive upstream. The product-level hard cap is
# **1 call per task** — enforced at owner-api before requests reach
# this service. This service additionally honours the X-Eirel-Max-Requests
# header (default 1) as a defense-in-depth guard.


def generate_job_token(master_token: str, job_id: str) -> str:
    return hmac.new(
        master_token.encode(), job_id.encode(), "sha256"
    ).hexdigest()


def verify_job_token(master_token: str, job_id: str, token: str) -> bool:
    expected = generate_job_token(master_token, job_id)
    return hmac.compare_digest(expected, token)


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def create_app(
    *,
    x_transport: httpx.AsyncBaseTransport | None = None,
    ledger: JobLedger | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.auth_token = os.getenv("EIREL_X_TOOL_API_TOKEN", "")
        app.state.x_api_base_url = os.getenv("EIREL_X_API_BASE_URL", DEFAULT_X_API_BASE_URL).rstrip("/")
        app.state.x_bearer_token = os.getenv("EIREL_X_BEARER_TOKEN", "")
        # X API caps at 1 call per task by default (enforced at owner-api too).
        app.state.default_max_requests = int(os.getenv("EIREL_X_TOOL_DEFAULT_MAX_REQUESTS", "1"))
        app.state.fetch_timeout_seconds = float(os.getenv("EIREL_X_TOOL_FETCH_TIMEOUT_SECONDS", "15"))
        if ledger is not None:
            app.state.job_ledger = ledger
        else:
            app.state.job_ledger = create_job_ledger(os.getenv("REDIS_URL", ""))
        app.state.x_transport = x_transport
        app.state.metrics = {
            "requests_total": 0,
            "quota_rejections_total": 0,
            "search_requests_total": 0,
        }
        try:
            yield
        finally:
            await app.state.job_ledger.close()

    app = FastAPI(title="x-tool-service", version="0.1.0", lifespan=lifespan)

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
            detail="invalid x tool auth token",
        )

    async def require_job(
        x_eirel_job_id: str | None = Header(default=None),
        x_eirel_max_requests: str | None = Header(default=None),
    ) -> tuple[str, int]:
        job_id = (x_eirel_job_id or "").strip()
        if not job_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="missing X-Eirel-Job-Id")
        try:
            max_requests = int(x_eirel_max_requests or app.state.default_max_requests)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid X-Eirel-Max-Requests") from exc
        return job_id, max(1, max_requests)

    async def check_budget(*, job_id: str, max_requests: int, tool_name: str) -> JobUsageRecord:
        ledger: JobLedger = app.state.job_ledger
        usage = await ledger.get_or_create(job_id)
        if usage.request_count >= max_requests:
            app.state.metrics["quota_rejections_total"] += 1
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="x tool request budget exceeded")
        usage.request_count += 1
        usage.tool_counts[tool_name] = usage.tool_counts.get(tool_name, 0) + 1
        app.state.metrics["requests_total"] += 1
        await ledger.save(job_id, usage)
        return usage

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> PlainTextResponse:
        m = app.state.metrics
        lines = [
            "# HELP eirel_x_tool_requests_total Total successful X tool requests.",
            "# TYPE eirel_x_tool_requests_total counter",
            f"eirel_x_tool_requests_total {m['requests_total']}",
            "# HELP eirel_x_tool_quota_rejections_total Total X tool quota rejections.",
            "# TYPE eirel_x_tool_quota_rejections_total counter",
            f"eirel_x_tool_quota_rejections_total {m['quota_rejections_total']}",
        ]
        return PlainTextResponse("\n".join(lines) + "\n")

    @app.get("/v1/jobs/{job_id}/usage")
    async def job_usage(job_id: str, _: None = Depends(require_auth)) -> dict[str, Any]:
        ledger: JobLedger = app.state.job_ledger
        usage = await ledger.get_usage(job_id)
        if usage is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job usage not found")
        return {
            "job_id": job_id,
            "retrieval_ledger_id": usage.ledger_id,
            "request_count": usage.request_count,
            "tool_counts": dict(usage.tool_counts),
        }

    @app.post("/v1/search", response_model=XSearchResponse)
    async def search(
        payload: XSearchRequest,
        _: None = Depends(require_auth),
        job: tuple[str, int] = Depends(require_job),
    ) -> XSearchResponse:
        job_id, max_requests = job
        usage = await check_budget(job_id=job_id, max_requests=max_requests, tool_name="x_search")
        app.state.metrics["search_requests_total"] += 1
        retrieved_at = _utcnow()

        tweets = await _x_api_search(
            base_url=app.state.x_api_base_url,
            bearer_token=app.state.x_bearer_token,
            query=payload.query,
            since=payload.since,
            until=payload.until,
            max_results=payload.max_results,
            transport=app.state.x_transport,
            timeout_seconds=app.state.fetch_timeout_seconds,
        )

        usage.searches.append({
            "query": payload.query,
            "since": payload.since,
            "until": payload.until,
            "verified_only": payload.verified_only,
            "retrieved_at": retrieved_at,
            "result_count": len(tweets),
        })
        ledger: JobLedger = app.state.job_ledger
        await ledger.save(job_id, usage)

        if payload.verified_only:
            tweets = [t for t in tweets if t.verified]

        per_query_cost = float(os.getenv("EIREL_X_API_PER_QUERY_COST_USD", "0.0015"))
        await charge_tool_cost(
            job_id=job_id, tool_name="x_api", amount_usd=per_query_cost,
        )

        return XSearchResponse(
            query=payload.query,
            tweets=tweets,
            result_count=len(tweets),
            retrieved_at=retrieved_at,
            retrieval_ledger_id=usage.ledger_id,
            metadata={"backend": "x_api_v2"},
        )

    return app


async def _x_api_search(
    *,
    base_url: str,
    bearer_token: str,
    query: str,
    since: str | None,
    until: str | None,
    max_results: int,
    transport: httpx.AsyncBaseTransport | None,
    timeout_seconds: float,
) -> list[XTweet]:
    if not bearer_token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="X API bearer token is not configured",
        )
    params: dict[str, Any] = {
        "query": query,
        "max_results": max(10, min(max_results, 100)),
        "tweet.fields": "created_at,public_metrics,author_id",
        "user.fields": "username,verified",
        "expansions": "author_id",
    }
    if since:
        params["start_time"] = since
    if until:
        params["end_time"] = until

    async with httpx.AsyncClient(
        timeout=timeout_seconds,
        transport=transport,
    ) as client:
        response = await client.get(
            f"{base_url}/tweets/search/recent",
            params=params,
            headers={
                "Authorization": f"Bearer {bearer_token}",
                "User-Agent": "EIREL-X-Tool/0.1",
            },
        )
        response.raise_for_status()
        payload = response.json()

    users_by_id: dict[str, dict[str, Any]] = {}
    for user in payload.get("includes", {}).get("users", []):
        users_by_id[user.get("id", "")] = user

    tweets: list[XTweet] = []
    for item in payload.get("data", []):
        if not isinstance(item, dict):
            continue
        tweet_id = str(item.get("id", ""))
        author_id = str(item.get("author_id", ""))
        user = users_by_id.get(author_id, {})
        text = str(item.get("text", ""))
        metrics = item.get("public_metrics", {})
        tweets.append(XTweet(
            tweet_id=tweet_id,
            author_id=author_id,
            author_username=str(user.get("username", "")),
            text=text,
            created_at=str(item.get("created_at", "")),
            verified=bool(user.get("verified", False)),
            retweet_count=int(metrics.get("retweet_count", 0)),
            like_count=int(metrics.get("like_count", 0)),
            reply_count=int(metrics.get("reply_count", 0)),
            url=f"https://x.com/{user.get('username', '_')}/status/{tweet_id}",
            content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        ))
    return tweets[:max_results]


app = create_app()


def main() -> None:
    port = int(os.getenv("EIREL_X_TOOL_PORT", "8086"))
    uvicorn.run("tool_platforms.x_tool_service.app:app", host="0.0.0.0", port=port, reload=False)
