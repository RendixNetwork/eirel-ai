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
from tool_platforms.semantic_scholar_tool_service.models import (
    SemanticScholarBatchRequest,
    SemanticScholarBatchResponse,
    SemanticScholarPaper,
    SemanticScholarSearchRequest,
    SemanticScholarSearchResponse,
)

DEFAULT_SEMANTIC_SCHOLAR_API_BASE_URL = "https://api.semanticscholar.org/graph/v1"

# Fields we pull from Semantic Scholar per paper. Comma-separated.
_PAPER_FIELDS = (
    "paperId,title,abstract,authors,year,venue,"
    "citationCount,influentialCitationCount,externalIds,openAccessPdf,url"
)


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
    semantic_scholar_transport: httpx.AsyncBaseTransport | None = None,
    ledger: JobLedger | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.auth_token = os.getenv("EIREL_SEMANTIC_SCHOLAR_TOOL_API_TOKEN", "")
        app.state.api_base_url = os.getenv(
            "EIREL_SEMANTIC_SCHOLAR_API_BASE_URL",
            DEFAULT_SEMANTIC_SCHOLAR_API_BASE_URL,
        ).rstrip("/")
        app.state.api_key = os.getenv("EIREL_SEMANTIC_SCHOLAR_API_KEY", "")
        app.state.default_max_requests = int(
            os.getenv("EIREL_SEMANTIC_SCHOLAR_TOOL_DEFAULT_MAX_REQUESTS", "12")
        )
        app.state.fetch_timeout_seconds = float(
            os.getenv("EIREL_SEMANTIC_SCHOLAR_TOOL_FETCH_TIMEOUT_SECONDS", "15")
        )
        if ledger is not None:
            app.state.job_ledger = ledger
        else:
            app.state.job_ledger = create_job_ledger(os.getenv("REDIS_URL", ""))
        app.state.semantic_scholar_transport = semantic_scholar_transport
        app.state.metrics = {
            "requests_total": 0,
            "quota_rejections_total": 0,
            "search_requests_total": 0,
            "batch_requests_total": 0,
            "upstream_429_total": 0,
        }
        try:
            yield
        finally:
            await app.state.job_ledger.close()

    app = FastAPI(
        title="semantic-scholar-tool-service",
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
            detail="invalid semantic scholar tool auth token",
        )

    async def require_job(
        x_eirel_job_id: str | None = Header(default=None),
        x_eirel_max_requests: str | None = Header(default=None),
    ) -> tuple[str, int]:
        job_id = (x_eirel_job_id or "").strip()
        if not job_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="missing X-Eirel-Job-Id",
            )
        try:
            max_requests = int(x_eirel_max_requests or app.state.default_max_requests)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="invalid X-Eirel-Max-Requests",
            ) from exc
        return job_id, max(1, max_requests)

    async def check_budget(
        *, job_id: str, max_requests: int, tool_name: str
    ) -> JobUsageRecord:
        ledger: JobLedger = app.state.job_ledger
        usage = await ledger.get_or_create(job_id)
        if usage.request_count >= max_requests:
            app.state.metrics["quota_rejections_total"] += 1
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="semantic scholar tool request budget exceeded",
            )
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
            "# HELP eirel_semantic_scholar_tool_requests_total Total successful Semantic Scholar tool requests.",
            "# TYPE eirel_semantic_scholar_tool_requests_total counter",
            f"eirel_semantic_scholar_tool_requests_total {m['requests_total']}",
            "# HELP eirel_semantic_scholar_tool_quota_rejections_total Total Semantic Scholar tool quota rejections.",
            "# TYPE eirel_semantic_scholar_tool_quota_rejections_total counter",
            f"eirel_semantic_scholar_tool_quota_rejections_total {m['quota_rejections_total']}",
            "# HELP eirel_semantic_scholar_upstream_429_total Total upstream rate-limit responses from Semantic Scholar.",
            "# TYPE eirel_semantic_scholar_upstream_429_total counter",
            f"eirel_semantic_scholar_upstream_429_total {m['upstream_429_total']}",
        ]
        return PlainTextResponse("\n".join(lines) + "\n")

    @app.get("/v1/jobs/{job_id}/usage")
    async def job_usage(
        job_id: str, _: None = Depends(require_auth)
    ) -> dict[str, Any]:
        ledger: JobLedger = app.state.job_ledger
        usage = await ledger.get_usage(job_id)
        if usage is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="job usage not found",
            )
        return {
            "job_id": job_id,
            "retrieval_ledger_id": usage.ledger_id,
            "request_count": usage.request_count,
            "tool_counts": dict(usage.tool_counts),
        }

    @app.post("/v1/search", response_model=SemanticScholarSearchResponse)
    async def search(
        payload: SemanticScholarSearchRequest,
        _: None = Depends(require_auth),
        job: tuple[str, int] = Depends(require_job),
    ) -> SemanticScholarSearchResponse:
        job_id, max_requests = job
        usage = await check_budget(
            job_id=job_id, max_requests=max_requests, tool_name="semantic_scholar_search"
        )
        app.state.metrics["search_requests_total"] += 1
        retrieved_at = _utcnow()

        papers, total_matching = await _semantic_scholar_search(
            base_url=app.state.api_base_url,
            api_key=app.state.api_key,
            query=payload.query,
            year=payload.year,
            fields_of_study=payload.fields_of_study,
            venue=payload.venue,
            open_access_only=payload.open_access_only,
            max_results=payload.max_results,
            transport=app.state.semantic_scholar_transport,
            timeout_seconds=app.state.fetch_timeout_seconds,
            metrics=app.state.metrics,
        )

        usage.searches.append({
            "query": payload.query,
            "year": payload.year,
            "fields_of_study": list(payload.fields_of_study),
            "retrieved_at": retrieved_at,
            "result_count": len(papers),
        })
        ledger: JobLedger = app.state.job_ledger
        await ledger.save(job_id, usage)

        per_query_cost = float(os.getenv("EIREL_SEMANTIC_SCHOLAR_PER_QUERY_COST_USD", "0.0"))
        await charge_tool_cost(
            job_id=job_id, tool_name="semantic_scholar", amount_usd=per_query_cost,
        )

        return SemanticScholarSearchResponse(
            query=payload.query,
            papers=papers,
            result_count=len(papers),
            total_matching=total_matching,
            retrieved_at=retrieved_at,
            retrieval_ledger_id=usage.ledger_id,
            metadata={"backend": "semantic_scholar_api"},
        )

    @app.post("/v1/batch", response_model=SemanticScholarBatchResponse)
    async def batch(
        payload: SemanticScholarBatchRequest,
        _: None = Depends(require_auth),
        job: tuple[str, int] = Depends(require_job),
    ) -> SemanticScholarBatchResponse:
        job_id, max_requests = job
        usage = await check_budget(
            job_id=job_id, max_requests=max_requests, tool_name="semantic_scholar_batch"
        )
        app.state.metrics["batch_requests_total"] += 1
        retrieved_at = _utcnow()

        papers = await _semantic_scholar_batch(
            base_url=app.state.api_base_url,
            api_key=app.state.api_key,
            paper_ids=payload.paper_ids,
            transport=app.state.semantic_scholar_transport,
            timeout_seconds=app.state.fetch_timeout_seconds,
            metrics=app.state.metrics,
        )

        ledger: JobLedger = app.state.job_ledger
        await ledger.save(job_id, usage)

        return SemanticScholarBatchResponse(
            papers=papers,
            result_count=len(papers),
            retrieved_at=retrieved_at,
            retrieval_ledger_id=usage.ledger_id,
            metadata={"backend": "semantic_scholar_api"},
        )

    return app


def _build_search_params(
    *,
    query: str,
    year: str | None,
    fields_of_study: list[str],
    venue: str | None,
    open_access_only: bool,
    max_results: int,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "query": query,
        "limit": max(1, min(max_results, 100)),
        "offset": 0,
        "fields": _PAPER_FIELDS,
    }
    if year:
        params["year"] = year
    if fields_of_study:
        params["fieldsOfStudy"] = ",".join(fields_of_study)
    if venue:
        params["venue"] = venue
    if open_access_only:
        params["openAccessPdf"] = "true"
    return params


async def _semantic_scholar_search(
    *,
    base_url: str,
    api_key: str,
    query: str,
    year: str | None,
    fields_of_study: list[str],
    venue: str | None,
    open_access_only: bool,
    max_results: int,
    transport: httpx.AsyncBaseTransport | None,
    timeout_seconds: float,
    metrics: dict[str, int],
) -> tuple[list[SemanticScholarPaper], int]:
    params = _build_search_params(
        query=query,
        year=year,
        fields_of_study=fields_of_study,
        venue=venue,
        open_access_only=open_access_only,
        max_results=max_results,
    )
    headers = {"User-Agent": "EIREL-Semantic-Scholar-Tool/0.1"}
    if api_key:
        headers["x-api-key"] = api_key

    async with httpx.AsyncClient(
        timeout=timeout_seconds,
        transport=transport,
    ) as client:
        response = await client.get(
            f"{base_url}/paper/search",
            params=params,
            headers=headers,
        )
        if response.status_code == 429:
            metrics["upstream_429_total"] += 1
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="semantic scholar upstream rate limited",
                headers={"Retry-After": response.headers.get("Retry-After", "1")},
            )
        response.raise_for_status()
        body = response.json()

    total_matching = int(body.get("total", 0) or 0)
    data = body.get("data") or []
    papers: list[SemanticScholarPaper] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        papers.append(_paper_from_raw(item))
    return papers[:max_results], total_matching


async def _semantic_scholar_batch(
    *,
    base_url: str,
    api_key: str,
    paper_ids: list[str],
    transport: httpx.AsyncBaseTransport | None,
    timeout_seconds: float,
    metrics: dict[str, int],
) -> list[SemanticScholarPaper]:
    headers = {
        "User-Agent": "EIREL-Semantic-Scholar-Tool/0.1",
        "Content-Type": "application/json",
    }
    if api_key:
        headers["x-api-key"] = api_key

    async with httpx.AsyncClient(
        timeout=timeout_seconds,
        transport=transport,
    ) as client:
        response = await client.post(
            f"{base_url}/paper/batch",
            params={"fields": _PAPER_FIELDS},
            headers=headers,
            json={"ids": list(paper_ids)},
        )
        if response.status_code == 429:
            metrics["upstream_429_total"] += 1
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="semantic scholar upstream rate limited",
                headers={"Retry-After": response.headers.get("Retry-After", "1")},
            )
        response.raise_for_status()
        body = response.json()

    papers: list[SemanticScholarPaper] = []
    items = body if isinstance(body, list) else []
    for item in items:
        if not isinstance(item, dict):
            continue
        papers.append(_paper_from_raw(item))
    return papers


def _paper_from_raw(raw: dict[str, Any]) -> SemanticScholarPaper:
    paper_id = str(raw.get("paperId") or "")
    title = str(raw.get("title") or "").strip()
    abstract = str(raw.get("abstract") or "").strip()
    authors_raw = raw.get("authors") or []
    authors: list[str] = []
    if isinstance(authors_raw, list):
        for author in authors_raw:
            if isinstance(author, dict):
                name = str(author.get("name") or "").strip()
                if name:
                    authors.append(name)
    year_raw = raw.get("year")
    year = int(year_raw) if isinstance(year_raw, int) else None
    venue = str(raw.get("venue") or "").strip()
    citation_count = int(raw.get("citationCount") or 0)
    influential = int(raw.get("influentialCitationCount") or 0)
    external_ids_raw = raw.get("externalIds") or {}
    external_ids: dict[str, str] = {}
    if isinstance(external_ids_raw, dict):
        for key, value in external_ids_raw.items():
            if value is not None:
                external_ids[str(key)] = str(value)
    arxiv_id = external_ids.get("ArXiv", "")
    doi = external_ids.get("DOI", "")
    open_access = raw.get("openAccessPdf") or {}
    open_access_pdf_url = ""
    if isinstance(open_access, dict):
        open_access_pdf_url = str(open_access.get("url") or "")
    url = str(raw.get("url") or "")
    if not url and paper_id:
        url = f"https://www.semanticscholar.org/paper/{paper_id}"
    content_sha256 = hashlib.sha256(
        (abstract or title).encode("utf-8")
    ).hexdigest()
    return SemanticScholarPaper(
        paper_id=paper_id,
        title=title,
        abstract=abstract[:4000],
        authors=authors,
        year=year,
        venue=venue,
        citation_count=citation_count,
        influential_citation_count=influential,
        arxiv_id=arxiv_id,
        doi=doi,
        open_access_pdf_url=open_access_pdf_url,
        url=url,
        external_ids=external_ids,
        content_sha256=content_sha256,
    )


app = create_app()


def main() -> None:
    port = int(os.getenv("EIREL_SEMANTIC_SCHOLAR_TOOL_PORT", "8087"))
    uvicorn.run(
        "tool_platforms.semantic_scholar_tool_service.app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
