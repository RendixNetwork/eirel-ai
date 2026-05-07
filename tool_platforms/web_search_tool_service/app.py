from __future__ import annotations

import hashlib
import hmac
import logging
import json
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html import unescape
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import PlainTextResponse

from tool_platforms._charge_tool import charge_tool_cost
from tool_platforms._record_tool_call import record_tool_call
from tool_platforms.web_search_tool_service.backends import (
    DEFAULT_BRAVE_SEARCH_BASE_URL,
    DEFAULT_FETCH_USER_AGENT,
    AllBackendsFailedError,
    BraveBackend,
    CatalogBackend,
    FallbackSearchBackend,
    HardBackendError,
    ResearchCatalogStore,
    ResearchDocumentRecord,
    RetryableBackendError,
    SerperBackend,
    TavilyBackend,
    _canonical_url,
    _extract_published_at,
    _published_at_from_text,
)
from tool_platforms.web_search_tool_service.models import (
    FindOnPageRequest,
    FindOnPageResponse,
    PageLink,
    PageOpenRequest,
    PageOpenResponse,
    RetrievedSourceSnapshot,
    RetrievalLedgerResponse,
    SearchDocument,
    SearchRequest,
    SearchResponse,
)


_logger = logging.getLogger(__name__)

VISIBLE_BREAK_TAG_PATTERN = re.compile(r"</(?:p|div|section|article|li|ul|ol|h[1-6]|br|tr|table)>", re.IGNORECASE)
STRIP_BLOCK_PATTERN = re.compile(r"<(script|style|noscript|svg|nav|footer|header|form)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"[ \t\r\f\v]+")


def generate_job_token(master_token: str, job_id: str) -> str:
    """Derive a per-job scoped token from the master token via HMAC-SHA256."""
    return hmac.new(
        master_token.encode(), job_id.encode(), "sha256"
    ).hexdigest()


def verify_job_token(master_token: str, job_id: str, token: str) -> bool:
    """Verify a per-job scoped token against the master token."""
    expected = generate_job_token(master_token, job_id)
    return hmac.compare_digest(expected, token)


def default_catalog_path() -> Path | None:
    override = os.getenv("EIREL_WEB_SEARCH_TOOL_CATALOG_PATH", "").strip()
    if override:
        return Path(override)
    return None


@dataclass(slots=True)
class JobUsageRecord:
    request_count: int = 0
    tool_counts: dict[str, int] = field(default_factory=dict)
    execution_mode: str = "live_web"
    ledger_id: str = ""
    searches: list[dict[str, Any]] = field(default_factory=list)
    opened_pages: dict[str, dict[str, Any]] = field(default_factory=dict)
    opened_page_text: dict[str, str] = field(default_factory=dict)
    search_index: dict[str, dict[str, Any]] = field(default_factory=dict)
    find_on_page_events: list[dict[str, Any]] = field(default_factory=list)


def load_catalog_store(catalog_path: Path | None) -> ResearchCatalogStore:
    if catalog_path is None or not catalog_path.exists():
        return ResearchCatalogStore(documents={})
    payload = json.loads(catalog_path.read_text())
    documents: dict[str, ResearchDocumentRecord] = {}
    for item in payload.get("documents", []):
        record = ResearchDocumentRecord(
            document_id=str(item["document_id"]),
            title=str(item["title"]),
            url=str(item["url"]),
            snippet=str(item["snippet"]),
            content=str(item.get("content", "")),
            links=[link for link in item.get("links", []) if isinstance(link, dict)],
            metadata=dict(item.get("metadata", {}) or {}),
        )
        documents[record.document_id] = record
    return ResearchCatalogStore(documents=documents)


def create_app(
    catalog_store: ResearchCatalogStore | None = None,
    *,
    backend: str | None = None,
    search_transport: httpx.AsyncBaseTransport | None = None,
    fetch_transport: httpx.AsyncBaseTransport | None = None,
    per_backend_timeout: float | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.catalog_store = catalog_store or load_catalog_store(default_catalog_path())
        app.state.auth_token = os.getenv("EIREL_WEB_SEARCH_TOOL_API_TOKEN", "")
        backends_str = backend
        if not backends_str:
            backends_str = os.getenv("EIREL_WEB_SEARCH_TOOL_BACKENDS")
        if not backends_str:
            backends_str = os.getenv("EIREL_WEB_SEARCH_TOOL_BACKEND")
        if not backends_str:
            has_brave = bool(os.getenv("EIREL_BRAVE_SEARCH_API_KEY", "").strip())
            has_serper = bool(os.getenv("EIREL_SERPER_API_KEY", "").strip())
            has_tavily = bool(os.getenv("EIREL_TAVILY_API_KEY", "").strip())
            if has_brave and has_serper and has_tavily:
                backends_str = "brave,serper,tavily"
            elif has_brave and has_serper:
                backends_str = "brave,serper"
            elif has_brave and has_tavily:
                backends_str = "brave,tavily"
            elif has_brave:
                backends_str = "brave"
            elif has_serper:
                backends_str = "serper"
            elif has_tavily:
                backends_str = "tavily"
            else:
                backends_str = "catalog"
        app.state.backend = (backends_str or "catalog").strip() or "catalog"
        app.state.fetch_timeout_seconds = float(
            os.getenv(
                "EIREL_WEB_SEARCH_FETCH_TIMEOUT_SECONDS",
                os.getenv("EIREL_RESEARCH_FETCH_TIMEOUT_SECONDS", "15"),
            )
        )
        backend_names = [n.strip() for n in app.state.backend.split(",") if n.strip()]
        search_backends: list[CatalogBackend | BraveBackend | SerperBackend | TavilyBackend] = []
        for bname in backend_names:
            if bname in ("brave", "brave_live_web"):
                search_backends.append(BraveBackend(
                    api_key=os.getenv("EIREL_BRAVE_SEARCH_API_KEY", ""),
                    base_url=os.getenv("EIREL_BRAVE_SEARCH_BASE_URL", DEFAULT_BRAVE_SEARCH_BASE_URL),
                    timeout_seconds=app.state.fetch_timeout_seconds,
                    transport=search_transport,
                ))
            elif bname == "serper":
                serper_key = os.getenv("EIREL_SERPER_API_KEY", "").strip()
                if serper_key:
                    search_backends.append(SerperBackend(api_key=serper_key, transport=search_transport))
                else:
                    _logger.warning("skipping serper backend: EIREL_SERPER_API_KEY not set")
            elif bname == "tavily":
                tavily_key = os.getenv("EIREL_TAVILY_API_KEY", "").strip()
                if tavily_key:
                    search_backends.append(TavilyBackend(api_key=tavily_key, transport=search_transport))
                else:
                    _logger.warning("skipping tavily backend: EIREL_TAVILY_API_KEY not set")
            elif bname == "catalog":
                search_backends.append(CatalogBackend(app.state.catalog_store))
            else:
                _logger.warning("unknown search backend %s, skipping", bname)
        if not search_backends:
            _logger.warning("no search backends resolved; falling back to catalog")
            search_backends.append(CatalogBackend(app.state.catalog_store))
        resolved_timeout = per_backend_timeout if per_backend_timeout is not None else float(
            os.getenv("EIREL_WEB_SEARCH_PER_BACKEND_TIMEOUT_SECONDS", "10.0")
        )
        app.state.search_backend = FallbackSearchBackend(
            search_backends, per_backend_timeout=resolved_timeout,
        )
        app.state.fetch_max_bytes = int(
            os.getenv(
                "EIREL_WEB_SEARCH_FETCH_MAX_BYTES",
                os.getenv("EIREL_RESEARCH_FETCH_MAX_BYTES", str(1024 * 1024)),
            )
        )
        app.state.fetch_user_agent = os.getenv(
            "EIREL_WEB_SEARCH_FETCH_USER_AGENT",
            os.getenv("EIREL_RESEARCH_FETCH_USER_AGENT", DEFAULT_FETCH_USER_AGENT),
        )
        app.state.fetch_redirect_limit = int(
            os.getenv("EIREL_WEB_SEARCH_FETCH_REDIRECT_LIMIT", "5")
        )
        app.state.default_max_requests = int(
            os.getenv("EIREL_WEB_SEARCH_TOOL_DEFAULT_MAX_REQUESTS", "12")
        )
        app.state.job_usage: dict[str, JobUsageRecord] = {}
        app.state.fetch_transport = fetch_transport
        app.state.metrics = {
            "requests_total": 0,
            "quota_rejections_total": 0,
            "search_requests_total": 0,
            "open_page_requests_total": 0,
            "find_on_page_requests_total": 0,
        }
        yield

    app = FastAPI(title="web-search-tool-service", version="0.1.0", lifespan=lifespan)

    async def require_auth(
        authorization: str | None = Header(default=None),
        x_eirel_job_id: str | None = Header(default=None),
    ) -> None:
        auth_token: str = app.state.auth_token
        if not auth_token:
            return
        if authorization == f"Bearer {auth_token}":
            return  # Master token — always valid
        # Per-job scoped token: requires matching job ID header
        if x_eirel_job_id and authorization:
            bearer = authorization.removeprefix("Bearer ").strip()
            if verify_job_token(auth_token, x_eirel_job_id.strip(), bearer):
                return
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid web search tool auth token",
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

    def _usage_record(job_id: str) -> JobUsageRecord:
        usage: JobUsageRecord = app.state.job_usage.setdefault(job_id, JobUsageRecord())
        if not usage.ledger_id:
            usage.ledger_id = f"ledger:{job_id}"
        return usage

    def check_budget(*, job_id: str, max_requests: int, tool_name: str) -> None:
        usage = _usage_record(job_id)
        if usage.request_count >= max_requests:
            app.state.metrics["quota_rejections_total"] += 1
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="web search tool request budget exceeded")
        usage.request_count += 1
        usage.tool_counts[tool_name] = usage.tool_counts.get(tool_name, 0) + 1
        app.state.metrics["requests_total"] += 1

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> PlainTextResponse:
        metrics = app.state.metrics
        lines = [
            "# HELP eirel_web_search_tool_requests_total Total successful web search tool requests.",
            "# TYPE eirel_web_search_tool_requests_total counter",
            f"eirel_web_search_tool_requests_total {metrics['requests_total']}",
            "# HELP eirel_web_search_tool_quota_rejections_total Total web search tool quota rejections.",
            "# TYPE eirel_web_search_tool_quota_rejections_total counter",
            f"eirel_web_search_tool_quota_rejections_total {metrics['quota_rejections_total']}",
        ]
        return PlainTextResponse("\n".join(lines) + "\n")

    @app.get("/v1/operators/summary")
    async def operator_summary(_: None = Depends(require_auth)) -> dict[str, Any]:
        return {
            "backend": app.state.backend,
            "document_count": len(app.state.catalog_store.documents),
            "active_job_count": len(app.state.job_usage),
            "active_jobs": {
                job_id: {
                    "request_count": usage.request_count,
                    "tool_counts": dict(usage.tool_counts),
                }
                for job_id, usage in sorted(app.state.job_usage.items())
            },
            "metrics": dict(app.state.metrics),
        }

    @app.get("/v1/jobs/{job_id}/usage")
    async def job_usage(job_id: str, _: None = Depends(require_auth)) -> dict[str, Any]:
        usage: JobUsageRecord | None = app.state.job_usage.get(job_id)
        if usage is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job usage not found")
        return {
            "job_id": job_id,
            "retrieval_ledger_id": usage.ledger_id,
            "execution_mode": usage.execution_mode,
            "request_count": usage.request_count,
            "tool_counts": dict(usage.tool_counts),
        }

    @app.post("/v1/usage/reset")
    async def reset_usage(_: None = Depends(require_auth)) -> dict[str, Any]:
        """Clear all per-job usage counters.

        Called by owner-api on run advance so each deployment's request
        budget refills for the new run. Miners deliberately use a sticky
        job_id (miner-<deployment_id>) for cost attribution, so without
        this periodic reset they'd accumulate requests monotonically and
        eventually hit the per-job cap. Returns the number of jobs cleared.
        """
        cleared = len(app.state.job_usage)
        app.state.job_usage.clear()
        return {"cleared_job_count": cleared}

    @app.get("/v1/jobs/{job_id}/ledger", response_model=RetrievalLedgerResponse)
    async def job_ledger(job_id: str, _: None = Depends(require_auth)) -> RetrievalLedgerResponse:
        usage: JobUsageRecord | None = app.state.job_usage.get(job_id)
        if usage is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job ledger not found")
        return RetrievalLedgerResponse(
            job_id=job_id,
            retrieval_ledger_id=usage.ledger_id or f"ledger:{job_id}",
            execution_mode=usage.execution_mode or "live_web",
            searches=list(usage.searches),
            opened_pages=[RetrievedSourceSnapshot.model_validate(item) for item in usage.opened_pages.values()],
            find_on_page_events=list(usage.find_on_page_events),
        )

    @app.post("/v1/search", response_model=SearchResponse)
    async def search(
        payload: SearchRequest,
        _: None = Depends(require_auth),
        job: tuple[str, int] = Depends(require_job),
    ) -> SearchResponse:
        job_id, max_requests = job
        check_budget(job_id=job_id, max_requests=max_requests, tool_name="search")
        usage = _usage_record(job_id)
        app.state.metrics["search_requests_total"] += 1
        retrieved_at = _utcnow()
        try:
            result = await app.state.search_backend.search(
                query=payload.query, count=payload.top_k,
            )
        except AllBackendsFailedError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="all search backends failed",
            )
        documents = result.documents
        if payload.preferred_domain_families:
            for doc in documents:
                bonus = _preferred_domain_bonus(
                    doc.metadata.get("domain", ""),
                    payload.preferred_domain_families,
                )
                if bonus > 0:
                    doc.score = round(doc.score + bonus, 4)
            documents.sort(key=lambda item: (-item.score, item.document_id))
            documents = documents[:payload.top_k]
        for item in documents:
            usage.search_index[item.document_id] = {
                "document_id": item.document_id,
                "title": item.title,
                "url": item.url,
                "snippet": item.snippet,
                "metadata": dict(item.metadata),
            }
        usage.searches.append(
            {
                "query": payload.query,
                "top_k": payload.top_k,
                "preferred_domain_families": list(payload.preferred_domain_families),
                "retrieved_at": retrieved_at,
                "snapshot_id": payload.snapshot_id,
                "results": [
                    {
                        "document_id": item.document_id,
                        "title": item.title,
                        "url": item.url,
                        "score": item.score,
                        "canonical_url": item.metadata.get("canonical_url"),
                        "published_at": item.metadata.get("published_at"),
                        "domain": item.metadata.get("domain"),
                    }
                    for item in documents
                ],
            }
        )
        # Report per-query cost so DeploymentScoreRecord.tool_cost_usd is
        # populated during aggregation.  Fire-and-forget — a proxy outage
        # leaves cost unattributed but never breaks the tool response.
        per_query_cost = float(os.getenv("EIREL_WEB_SEARCH_PER_QUERY_COST_USD", "0.003"))
        await charge_tool_cost(
            job_id=job_id, tool_name="web_search", amount_usd=per_query_cost,
        )
        # Server-attested ledger row for the eval pipeline. Fire-and-forget;
        # owner-api outage drops one row but never breaks the tool.
        await record_tool_call(
            job_id=job_id, tool_name="web_search",
            args={"query": payload.query, "top_k": payload.top_k},
            result={"backend": result.backend_name, "n_results": len(documents)},
            cost_usd=per_query_cost,
            status_str="ok",
        )
        return SearchResponse(
            query=payload.query,
            snapshot_id=payload.snapshot_id,
            documents=documents,
            retrieval_ledger_id=usage.ledger_id,
            retrieved_at=retrieved_at,
            metadata={
                "execution_mode": usage.execution_mode,
                "search_backend": result.backend_name,
                "search_backends_attempted": result.attempted,
            },
        )

    @app.post("/v1/open-page", response_model=PageOpenResponse)
    async def open_page(
        payload: PageOpenRequest,
        _: None = Depends(require_auth),
        job: tuple[str, int] = Depends(require_job),
    ) -> PageOpenResponse:
        job_id, max_requests = job
        check_budget(job_id=job_id, max_requests=max_requests, tool_name="open_page")
        usage = _usage_record(job_id)
        app.state.metrics["open_page_requests_total"] += 1
        if app.state.backend == "catalog":
            document = _document_or_404(app.state.catalog_store, payload.document_id)
            snapshot_row = _catalog_snapshot_row(document)
            usage.opened_pages[document.document_id] = snapshot_row
            usage.opened_page_text[document.document_id] = document.content
            return PageOpenResponse(
                snapshot_id=payload.snapshot_id,
                document_id=document.document_id,
                title=document.title,
                url=document.url,
                content=document.content,
                snippet=document.snippet,
                links=[
                    PageLink(
                        title=str(item.get("title", "")),
                        url=str(item.get("url", "")),
                        target_document_id=(
                            str(item["target_document_id"])
                            if item.get("target_document_id") is not None
                            else None
                        ),
                    )
                    for item in document.links
                ],
                metadata={
                    **document.metadata,
                    "canonical_url": snapshot_row["canonical_url"],
                    "published_at": snapshot_row["published_at"],
                    "backend": "catalog",
                },
                retrieval_ledger_id=usage.ledger_id,
                canonical_url=snapshot_row["canonical_url"],
                final_url=snapshot_row["final_url"],
                published_at=snapshot_row["published_at"],
                retrieved_at=snapshot_row["retrieved_at"],
                content_hash=snapshot_row["content_hash"],
                extraction_confidence=snapshot_row["extraction_confidence"],
            )

        search_row = usage.search_index.get(payload.document_id)
        if not isinstance(search_row, dict):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="document not found in retrieval ledger")
        fetched = await _fetch_live_page(
            url=str(search_row.get("url") or ""),
            transport=app.state.fetch_transport,
            timeout_seconds=app.state.fetch_timeout_seconds,
            max_bytes=app.state.fetch_max_bytes,
            user_agent=app.state.fetch_user_agent,
            redirect_limit=app.state.fetch_redirect_limit,
        )
        snapshot_row = {
            "document_id": payload.document_id,
            "canonical_url": fetched["canonical_url"],
            "final_url": fetched["final_url"],
            "title": str(search_row.get("title") or fetched.get("title") or ""),
            "published_at": fetched["published_at"],
            "retrieved_at": fetched["retrieved_at"],
            "content_hash": fetched["content_hash"],
            "date_confidence": fetched["date_confidence"],
            "content_type": fetched["content_type"],
            "http_status": fetched["http_status"],
            "extraction_confidence": fetched["extraction_confidence"],
            "support_spans": [],
            "extracted_support_spans": [],
            "metadata": {
                **dict(search_row.get("metadata") or {}),
                "backend": app.state.backend,
                "title": fetched.get("title") or search_row.get("title"),
                **{key: value for key, value in fetched["metadata"].items() if value not in (None, "", [], {})},
            },
        }
        usage.opened_pages[payload.document_id] = snapshot_row
        usage.opened_page_text[payload.document_id] = fetched["content"]
        await record_tool_call(
            job_id=job_id, tool_name="web_search.open_page",
            args={"document_id": payload.document_id},
            result={"final_url": fetched["final_url"], "n_chars": len(fetched["content"])},
            status_str="ok",
        )
        return PageOpenResponse(
            snapshot_id=payload.snapshot_id,
            document_id=payload.document_id,
            title=str(search_row.get("title") or fetched.get("title") or ""),
            url=str(search_row.get("url") or fetched["final_url"]),
            content=fetched["content"],
            snippet=str(search_row.get("snippet") or ""),
            links=[],
            metadata=snapshot_row["metadata"],
            retrieval_ledger_id=usage.ledger_id,
            canonical_url=fetched["canonical_url"],
            final_url=fetched["final_url"],
            published_at=fetched["published_at"],
            retrieved_at=fetched["retrieved_at"],
            content_hash=fetched["content_hash"],
            date_confidence=fetched["date_confidence"],
            content_type=fetched["content_type"],
            http_status=fetched["http_status"],
            extraction_confidence=fetched["extraction_confidence"],
        )

    @app.post("/v1/find-on-page", response_model=FindOnPageResponse)
    async def find_on_page(
        payload: FindOnPageRequest,
        _: None = Depends(require_auth),
        job: tuple[str, int] = Depends(require_job),
    ) -> FindOnPageResponse:
        job_id, max_requests = job
        check_budget(job_id=job_id, max_requests=max_requests, tool_name="find_on_page")
        usage = _usage_record(job_id)
        app.state.metrics["find_on_page_requests_total"] += 1
        if app.state.backend == "catalog":
            document = _document_or_404(app.state.catalog_store, payload.document_id)
            text = document.content
            title = document.title
            url = document.url
        else:
            text = usage.opened_page_text.get(payload.document_id, "")
            snapshot = usage.opened_pages.get(payload.document_id)
            if not snapshot or not text.strip():
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="find-on-page requires an opened page with extracted text",
                )
            title = str(snapshot.get("title") or "")
            url = str(snapshot.get("final_url") or snapshot.get("canonical_url") or "")
        matches = _find_text_matches(text, payload.pattern)
        retrieved_at = _utcnow()
        canonical_url = _canonical_url(url) if url else None
        usage.find_on_page_events.append(
            {
                "document_id": payload.document_id,
                "pattern": payload.pattern,
                "matches": list(matches),
                "retrieved_at": retrieved_at,
                "canonical_url": canonical_url,
            }
        )
        existing_snapshot = usage.opened_pages.get(payload.document_id)
        if existing_snapshot is not None and matches:
            existing_snapshot["support_spans"] = list(
                dict.fromkeys([*existing_snapshot.get("support_spans", []), *matches])
            )
            existing_snapshot["extracted_support_spans"] = list(
                dict.fromkeys([*existing_snapshot.get("extracted_support_spans", []), *matches])
            )
        return FindOnPageResponse(
            snapshot_id=payload.snapshot_id,
            document_id=payload.document_id,
            pattern=payload.pattern,
            matches=matches,
            url=url or None,
            title=title or None,
            retrieval_ledger_id=usage.ledger_id,
            canonical_url=canonical_url,
            retrieved_at=retrieved_at,
            support_spans=matches,
        )

    return app


async def _fetch_live_page(
    *,
    url: str,
    transport: httpx.AsyncBaseTransport | None,
    timeout_seconds: float,
    max_bytes: int,
    user_agent: str,
    redirect_limit: int,
) -> dict[str, Any]:
    if not url:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="missing page url")
    async with httpx.AsyncClient(
        timeout=timeout_seconds,
        transport=transport,
        follow_redirects=True,
        max_redirects=max(1, redirect_limit),
        headers={"User-Agent": user_agent, "Accept": "text/html,application/xhtml+xml"},
    ) as client:
        response = await client.get(url)
    final_url = str(response.url)
    content_type = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
    raw_bytes = response.content[: max(1, max_bytes)]
    decoded = raw_bytes.decode(response.encoding or "utf-8", errors="ignore")
    published_at, date_confidence = _extract_published_at(decoded)
    content = ""
    extraction_confidence = 0.0
    metadata: dict[str, Any] = {}
    if response.status_code >= 400:
        metadata["open_error_kind"] = "http_error"
    elif content_type and "html" not in content_type:
        metadata["open_error_kind"] = "unsupported_content_type"
    else:
        content = _extract_visible_text(decoded)
        extraction_confidence = _extraction_confidence(content)
        if not content.strip():
            metadata["open_error_kind"] = "empty_or_unreadable"
        elif _looks_like_paywall(content):
            metadata["open_error_kind"] = "paywall_or_access_limited"
            extraction_confidence = min(extraction_confidence, 0.25)
    content_hash = hashlib.sha256(raw_bytes).hexdigest()
    title = _extract_title(decoded)
    return {
        "canonical_url": _canonical_url(final_url or url),
        "final_url": final_url or url,
        "title": title,
        "content": content,
        "retrieved_at": _utcnow(),
        "content_hash": content_hash,
        "published_at": published_at,
        "date_confidence": date_confidence,
        "content_type": content_type or None,
        "http_status": response.status_code,
        "extraction_confidence": extraction_confidence,
        "metadata": metadata,
    }


def _catalog_snapshot_row(document: ResearchDocumentRecord) -> dict[str, Any]:
    retrieved_at = _utcnow()
    published_at = _published_at_from_text(
        document.metadata.get("published_at"),
        "\n".join([document.title, document.snippet, document.content]),
    )
    return {
        "document_id": document.document_id,
        "canonical_url": _canonical_url(document.url),
        "final_url": document.url,
        "title": document.title,
        "published_at": published_at,
        "retrieved_at": retrieved_at,
        "content_hash": hashlib.sha256(document.content.encode("utf-8")).hexdigest(),
        "date_confidence": 1.0 if published_at else 0.0,
        "content_type": "text/plain",
        "http_status": 200,
        "extraction_confidence": 1.0 if document.content.strip() else 0.0,
        "support_spans": [],
        "extracted_support_spans": [],
        "metadata": dict(document.metadata),
    }


def _document_or_404(catalog_store: ResearchCatalogStore, document_id: str) -> ResearchDocumentRecord:
    document = catalog_store.documents.get(document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="document not found")
    return document


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _preferred_domain_bonus(domain: str, preferred_domain_families: list[str]) -> float:
    normalized = [str(item).strip().lower() for item in preferred_domain_families if str(item).strip()]
    for family in normalized:
        if family.startswith(".") and domain.endswith(family):
            return 0.15
        if domain == family or domain.endswith(f".{family}"):
            return 0.25
    return 0.0


def _extract_title(html: str) -> str | None:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return WHITESPACE_PATTERN.sub(" ", unescape(match.group(1))).strip() or None


def _extract_visible_text(html: str) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return _fallback_extract_visible_text(html)
    soup = BeautifulSoup(html, "html.parser")
    for selector in ("script", "style", "noscript", "svg", "nav", "footer", "header", "form"):
        for tag in soup.select(selector):
            tag.decompose()
    for tag in soup.select("[hidden], [aria-hidden='true']"):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [WHITESPACE_PATTERN.sub(" ", unescape(line)).strip() for line in text.splitlines()]
    visible = [line for line in lines if line]
    return "\n".join(visible[:4000])


def _fallback_extract_visible_text(html: str) -> str:
    without_blocks = STRIP_BLOCK_PATTERN.sub(" ", html)
    with_breaks = VISIBLE_BREAK_TAG_PATTERN.sub("\n", without_blocks)
    text = TAG_PATTERN.sub(" ", with_breaks)
    text = unescape(text)
    lines = [WHITESPACE_PATTERN.sub(" ", line).strip() for line in text.splitlines()]
    visible = [line for line in lines if line]
    return "\n".join(visible[:4000])


def _extraction_confidence(text: str) -> float:
    words = len(text.split())
    if words >= 250:
        return 0.95
    if words >= 120:
        return 0.82
    if words >= 40:
        return 0.58
    if words >= 10:
        return 0.30
    return 0.0


def _looks_like_paywall(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "subscribe to continue",
            "sign in to continue",
            "subscribe now",
            "remaining free article",
            "membership required",
        )
    )


def _find_text_matches(text: str, pattern: str) -> list[str]:
    lowered_pattern = pattern.strip().lower()
    if not lowered_pattern:
        return []
    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not raw_lines:
        raw_lines = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]
    matches = [line for line in raw_lines if lowered_pattern in line.lower()]
    return list(dict.fromkeys(matches))[:8]


app = create_app()


def main() -> None:
    uvicorn.run("tool_platforms.web_search_tool_service.app:app", host="0.0.0.0", port=8085, reload=False)
