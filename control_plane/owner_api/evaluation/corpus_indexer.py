"""Bundle → rag-tool-service corpus indexer.

Owner-api invokes this at run-open time after the bundle is built.
For every ``RagBundleCorpus`` carried by the bundle, POST the corpus
to the local rag-tool-service so subsequent ``rag.retrieve`` calls
from miner pods have an indexed source.

Sync because run-open is sync. Each POST is a single HTTP call with
a generous timeout — embeddings are batched server-side, so even
medium corpora (~50 documents) finish well inside the timeout.

Fail-soft: a missing rag-tool-service URL or a transient indexing
failure logs a warning and lets the rest of the run proceed. The
``rag_required`` tasks then fail individually (404 on retrieve →
miner answer = wrong → composite = 0 with knockout reason
``unknown_corpus``), so the operator notices on the dashboard
without breaking the whole run.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Iterable

import httpx

_logger = logging.getLogger(__name__)

__all__ = ["index_bundle_corpora"]


_DEFAULT_TIMEOUT_SECONDS = 60.0


def _rag_tool_url() -> str:
    return (
        os.getenv("EIREL_RAG_TOOL_URL", "http://rag-tool-service:8088")
        .rstrip("/")
    )


def _api_token() -> str:
    return os.getenv("EIREL_RAG_TOOL_API_TOKEN", "")


def index_bundle_corpora(
    corpora: Iterable[Any],
    *,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, int]:
    """POST every corpus to the rag-tool-service.

    Accepts either ``RagBundleCorpus`` Pydantic models OR raw dicts
    (so callers can pass un-validated bundle JSON without a Pydantic
    round-trip). Returns ``{corpus_id: n_chunks}`` for successful
    uploads; corpora that failed are logged + skipped.

    Idempotent on ``corpus_id``: re-indexing replaces the prior
    upload. Safe to call repeatedly across run reboots.
    """
    indexed: dict[str, int] = {}
    items = list(corpora)
    if not items:
        return indexed
    base_url = _rag_tool_url()
    headers: dict[str, str] = {"Content-Type": "application/json"}
    token = _api_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with httpx.Client(timeout=timeout_seconds) as client:
        for raw in items:
            corpus = raw.model_dump() if hasattr(raw, "model_dump") else dict(raw)
            corpus_id = str(corpus.get("corpus_id") or "").strip()
            if not corpus_id:
                _logger.warning("rag corpus index: skipping entry with empty corpus_id")
                continue
            documents_in: list[dict[str, Any]] = []
            for d in corpus.get("documents") or []:
                if not isinstance(d, dict):
                    continue
                doc_payload = {"doc_id": d["doc_id"], "content": d["content"]}
                if d.get("title"):
                    doc_payload["metadata"] = {"title": d["title"]}
                documents_in.append(doc_payload)
            payload = {"corpus_id": corpus_id, "documents": documents_in}
            try:
                resp = client.post(
                    f"{base_url}/v1/rag/corpora",
                    json=payload, headers=headers,
                )
            except httpx.HTTPError as exc:
                _logger.warning(
                    "rag corpus index failed: corpus_id=%s url=%s err=%s",
                    corpus_id, base_url, exc,
                )
                continue
            if resp.status_code != 200:
                _logger.warning(
                    "rag corpus index non-200: corpus_id=%s status=%d body=%s",
                    corpus_id, resp.status_code, (resp.text or "")[:256],
                )
                continue
            try:
                data = resp.json()
            except ValueError:
                data = {}
            indexed[corpus_id] = int(data.get("n_chunks", 0))
            _logger.info(
                "rag corpus indexed: corpus_id=%s n_documents=%d n_chunks=%d",
                corpus_id, len(documents_in), indexed[corpus_id],
            )
    return indexed
