"""FastAPI app for the RAG tool service.

Two surfaces:
  * ``POST /v1/rag/corpora`` (master-token-only) — operator indexes a
    document set into a corpus. Replaces an existing corpus on
    re-post; deletion is implicit by re-posting an empty document
    list. ``ProductOrchestrator`` later calls this to index per-
    project user uploads.
  * ``POST /v1/rag/retrieve`` (per-job-token-gated) — miner agents
    call this with ``corpus_id + query`` and get top-k chunks back.
    Every successful call writes to ``OrchestratorToolCallLog`` so
    the validator's ``tool_attestation_factor`` and the eval's
    retrieval-quality scoring can attribute calls to ground truth.

Mirrors the auth + ledger pattern of ``sandbox_tool_service`` and
``url_fetch_tool_service``: master bearer token (``Authorization``)
or per-job HMAC token (``Authorization`` + ``X-Eirel-Job-Id``); per-
job request-count cap via the shared ``JobLedger``; per-call ledger
write via ``record_tool_call``.
"""
from __future__ import annotations

import hmac
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from shared.common.redis_job_ledger import (
    JobLedger,
    JobUsageRecord,
    create_job_ledger,
)
from tool_platforms._record_tool_call import record_tool_call
from tool_platforms.rag_tool_service.chunker import chunk_document
from tool_platforms.rag_tool_service.corpus_store import (
    CorpusStore,
    RetrievedChunk,
)
from tool_platforms.rag_tool_service.embedder import (
    EmbeddingClient,
    EmbeddingError,
)

_logger = logging.getLogger(__name__)

# --- Defaults --------------------------------------------------------------
_DEFAULT_MAX_REQUESTS_PER_JOB = 30  # how many retrieve calls per job_id
_DEFAULT_MAX_K = 10
_DEFAULT_MAX_DOCUMENTS_PER_INDEX = 200
_DEFAULT_MAX_CHARS_PER_DOC = 200_000


# -- Job-token signing (mirrors url_fetch_tool_service) -------------------


def generate_job_token(master_token: str, job_id: str) -> str:
    return hmac.new(
        master_token.encode("utf-8"), job_id.encode("utf-8"),
        digestmod="sha256",
    ).hexdigest()


def verify_job_token(master_token: str, job_id: str, token: str) -> bool:
    expected = generate_job_token(master_token, job_id)
    return hmac.compare_digest(expected, token)


def _utcnow_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# -- Wire models -----------------------------------------------------------


class IndexDocument(BaseModel):
    """One document to index. ``doc_id`` is operator-defined and stable
    across re-indexes (used as the chunk-id prefix)."""

    doc_id: str = Field(..., min_length=1, max_length=128)
    content: str = Field(..., min_length=1)
    metadata: dict[str, Any] | None = None


class IndexCorpusRequest(BaseModel):
    corpus_id: str = Field(..., min_length=1, max_length=128)
    documents: list[IndexDocument] = Field(default_factory=list)
    chunk_chars: int | None = Field(default=None, ge=200, le=4000)
    overlap_chars: int | None = Field(default=None, ge=0, le=1000)


class IndexCorpusResponse(BaseModel):
    corpus_id: str
    n_documents: int
    n_chunks: int


class RetrieveRequest(BaseModel):
    corpus_id: str = Field(..., min_length=1, max_length=128)
    query: str = Field(..., min_length=1, max_length=4000)
    k: int = Field(default=5, ge=1, le=_DEFAULT_MAX_K)


class RetrievedChunkPayload(BaseModel):
    doc_id: str
    chunk_id: str
    text: str
    score: float
    char_start: int
    char_end: int


class RetrieveResponse(BaseModel):
    corpus_id: str
    query: str
    chunks: list[RetrievedChunkPayload]


# -- App factory -----------------------------------------------------------


def create_app(
    *,
    embedding_client: EmbeddingClient | None = None,
    corpus_store: CorpusStore | None = None,
    ledger: JobLedger | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.auth_token = os.getenv("EIREL_RAG_TOOL_API_TOKEN", "")
        app.state.default_max_requests = int(
            os.getenv(
                "EIREL_RAG_DEFAULT_MAX_REQUESTS",
                str(_DEFAULT_MAX_REQUESTS_PER_JOB),
            ),
        )
        app.state.max_documents_per_index = int(
            os.getenv(
                "EIREL_RAG_MAX_DOCUMENTS_PER_INDEX",
                str(_DEFAULT_MAX_DOCUMENTS_PER_INDEX),
            ),
        )
        app.state.max_chars_per_doc = int(
            os.getenv(
                "EIREL_RAG_MAX_CHARS_PER_DOC",
                str(_DEFAULT_MAX_CHARS_PER_DOC),
            ),
        )
        app.state.embedding_client = (
            embedding_client if embedding_client is not None
            else EmbeddingClient()
        )
        app.state.corpus_store = (
            corpus_store if corpus_store is not None
            else CorpusStore()
        )
        if ledger is not None:
            app.state.job_ledger = ledger
        else:
            app.state.job_ledger = create_job_ledger(os.getenv("REDIS_URL", ""))
        app.state.metrics = {
            "index_requests_total": 0,
            "retrieve_requests_total": 0,
            "quota_rejections_total": 0,
            "embedding_failures_total": 0,
            "unknown_corpus_total": 0,
        }
        # Owner-api ledger write target — same as the other tool
        # services. ``record_tool_call`` reads these from kwargs;
        # passing them explicitly per-call keeps the helper stateless.
        app.state.owner_api_url = os.getenv("OWNER_API_URL", "").rstrip("/")
        app.state.internal_token = os.getenv(
            "EIREL_INTERNAL_SERVICE_TOKEN", "",
        )
        try:
            yield
        finally:
            await app.state.embedding_client.aclose()
            await app.state.job_ledger.close()

    app = FastAPI(
        title="rag-tool-service",
        version="0.1.0",
        lifespan=lifespan,
    )

    # -- Auth helpers (mirror url_fetch_tool_service) ---------------------

    async def require_master_auth(
        authorization: str | None = Header(default=None),
    ) -> None:
        """Operator-only: raw master token, NEVER a per-job token.

        Indexing is privileged — we do NOT want miners to be able to
        upload corpora. Per-job tokens fail this check; only the
        operator's master token in ``Authorization: Bearer ...``
        passes.
        """
        auth_token: str = app.state.auth_token
        if not auth_token:
            # No token configured = open service (dev only). Same
            # convention as the other tool services.
            return
        if authorization == f"Bearer {auth_token}":
            return
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="rag tool: master token required for /v1/rag/corpora",
        )

    async def require_job_or_master(
        authorization: str | None = Header(default=None),
        x_eirel_job_id: str | None = Header(default=None),
    ) -> None:
        """Miner-callable: master token OR per-job HMAC."""
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
            detail="rag tool: invalid auth",
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

    async def check_budget(
        *, job_id: str, max_requests: int, tool_name: str,
    ) -> JobUsageRecord:
        ledger: JobLedger = app.state.job_ledger
        usage = await ledger.get_or_create(job_id)
        if usage.request_count >= max_requests:
            app.state.metrics["quota_rejections_total"] += 1
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "rag_quota_exhausted",
                    "max_requests": max_requests,
                },
            )
        usage.request_count += 1
        usage.tool_counts[tool_name] = usage.tool_counts.get(tool_name, 0) + 1
        await ledger.save(job_id, usage)
        return usage

    # -- Routes -----------------------------------------------------------

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "now": _utcnow_iso()}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> str:
        m: dict[str, Any] = app.state.metrics
        n_corpora = len(app.state.corpus_store.list_corpora())
        lines = [
            "# HELP eirel_rag_index_requests_total Successful corpus indexes.",
            "# TYPE eirel_rag_index_requests_total counter",
            f"eirel_rag_index_requests_total {m['index_requests_total']}",
            "# HELP eirel_rag_retrieve_requests_total Successful retrieves.",
            "# TYPE eirel_rag_retrieve_requests_total counter",
            f"eirel_rag_retrieve_requests_total {m['retrieve_requests_total']}",
            "# HELP eirel_rag_quota_rejections_total Job-quota rejections.",
            "# TYPE eirel_rag_quota_rejections_total counter",
            f"eirel_rag_quota_rejections_total {m['quota_rejections_total']}",
            "# HELP eirel_rag_embedding_failures_total Embedding upstream failures.",
            "# TYPE eirel_rag_embedding_failures_total counter",
            f"eirel_rag_embedding_failures_total {m['embedding_failures_total']}",
            "# HELP eirel_rag_unknown_corpus_total Retrieve calls for unknown corpus_id.",
            "# TYPE eirel_rag_unknown_corpus_total counter",
            f"eirel_rag_unknown_corpus_total {m['unknown_corpus_total']}",
            "# HELP eirel_rag_corpora Active corpora count (gauge).",
            "# TYPE eirel_rag_corpora gauge",
            f"eirel_rag_corpora {n_corpora}",
        ]
        return "\n".join(lines) + "\n"

    @app.get("/v1/rag/corpora")
    async def list_corpora(
        _: None = Depends(require_master_auth),
    ) -> dict[str, Any]:
        return {"corpora": app.state.corpus_store.list_corpora()}

    @app.post("/v1/rag/corpora", response_model=IndexCorpusResponse)
    async def index_corpus(
        body: IndexCorpusRequest,
        _: None = Depends(require_master_auth),
    ) -> IndexCorpusResponse:
        # Caps: bound the operator's blast radius if a misconfigured
        # script tries to upload a giant corpus. Documented limits;
        # raise via env knob when needed.
        if len(body.documents) > app.state.max_documents_per_index:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail={
                    "error": "too_many_documents",
                    "limit": app.state.max_documents_per_index,
                },
            )
        for d in body.documents:
            if len(d.content) > app.state.max_chars_per_doc:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail={
                        "error": "document_too_long",
                        "doc_id": d.doc_id,
                        "limit_chars": app.state.max_chars_per_doc,
                    },
                )

        # Chunk every document. Empty-document corpora are valid (they
        # clear the corpus); chunking handles them gracefully.
        chunk_kwargs: dict[str, Any] = {}
        if body.chunk_chars is not None:
            chunk_kwargs["chunk_chars"] = body.chunk_chars
        if body.overlap_chars is not None:
            chunk_kwargs["overlap_chars"] = body.overlap_chars
        chunks = []
        for d in body.documents:
            chunks.extend(
                chunk_document(doc_id=d.doc_id, content=d.content, **chunk_kwargs),
            )

        # Embed all chunks in batches.
        if chunks:
            try:
                embeddings = await app.state.embedding_client.embed(
                    [c.text for c in chunks],
                )
            except EmbeddingError as exc:
                app.state.metrics["embedding_failures_total"] += 1
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail={"error": "embedding_failed", "reason": str(exc)},
                )
        else:
            embeddings = []

        n_chunks = await app.state.corpus_store.upsert(
            corpus_id=body.corpus_id, chunks=chunks, embeddings=embeddings,
        )
        app.state.metrics["index_requests_total"] += 1
        return IndexCorpusResponse(
            corpus_id=body.corpus_id,
            n_documents=len(body.documents),
            n_chunks=n_chunks,
        )

    @app.post("/v1/rag/retrieve", response_model=RetrieveResponse)
    async def retrieve(
        body: RetrieveRequest,
        _: None = Depends(require_job_or_master),
        job: tuple[str, int] = Depends(require_job),
    ) -> RetrieveResponse:
        job_id, max_requests = job

        # 404 fast on unknown corpus — no budget burned, clear error.
        if not app.state.corpus_store.has(body.corpus_id):
            app.state.metrics["unknown_corpus_total"] += 1
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "unknown_corpus",
                    "corpus_id": body.corpus_id,
                },
            )

        # Per-job request-count cap.
        await check_budget(
            job_id=job_id, max_requests=max_requests, tool_name="rag.retrieve",
        )

        t0 = time.perf_counter()
        # Embed the query.
        try:
            q_vec = (await app.state.embedding_client.embed([body.query]))[0]
        except EmbeddingError as exc:
            app.state.metrics["embedding_failures_total"] += 1
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={"error": "embedding_failed", "reason": str(exc)},
            )

        results: list[RetrievedChunk] = await app.state.corpus_store.retrieve(
            corpus_id=body.corpus_id, query_embedding=q_vec, k=body.k,
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)

        payload_chunks = [
            RetrievedChunkPayload(
                doc_id=r.doc_id,
                chunk_id=r.chunk_id,
                text=r.text,
                score=r.score,
                char_start=r.char_start,
                char_end=r.char_end,
            )
            for r in results
        ]

        # Server-attested ledger write — best-effort. The result is
        # the ordered list of returned chunk_ids so a downstream judge
        # can ask "did the right chunks come back?" via the
        # ``OrchestratorToolCallLog.result_digest`` column.
        await record_tool_call(
            job_id=job_id,
            tool_name="rag.retrieve",
            args={"corpus_id": body.corpus_id, "query": body.query, "k": body.k},
            result={"chunk_ids": [r.chunk_id for r in results]},
            latency_ms=latency_ms,
            cost_usd=0.0,  # embedding cost negligible per call
            status_str="ok",
            owner_api_url=app.state.owner_api_url or None,
            owner_api_token=app.state.internal_token or None,
        )
        app.state.metrics["retrieve_requests_total"] += 1
        return RetrieveResponse(
            corpus_id=body.corpus_id,
            query=body.query,
            chunks=payload_chunks,
        )

    return app


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    host = os.getenv("EIREL_RAG_TOOL_HOST", "0.0.0.0")
    port = int(os.getenv("EIREL_RAG_TOOL_PORT", "8088"))
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
