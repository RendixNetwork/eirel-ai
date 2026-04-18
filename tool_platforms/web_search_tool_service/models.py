from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RetrievedSourceSnapshot(BaseModel):
    document_id: str
    canonical_url: str | None = None
    final_url: str | None = None
    title: str | None = None
    published_at: str | None = None
    retrieved_at: str | None = None
    content_hash: str | None = None
    date_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    content_type: str | None = None
    http_status: int | None = None
    extraction_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    support_spans: list[str] = Field(default_factory=list)
    extracted_support_spans: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=10)
    snapshot_id: str | None = None
    preferred_domain_families: list[str] = Field(default_factory=list)


class SearchDocument(BaseModel):
    document_id: str
    title: str
    url: str
    snippet: str
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    documents: list[SearchDocument] = Field(default_factory=list)
    snapshot_id: str | None = None
    query: str
    retrieval_ledger_id: str | None = None
    retrieved_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PageOpenRequest(BaseModel):
    document_id: str
    snapshot_id: str | None = None


class PageLink(BaseModel):
    title: str
    url: str
    target_document_id: str | None = None


class PageOpenResponse(BaseModel):
    document_id: str
    title: str
    url: str
    content: str
    snippet: str
    links: list[PageLink] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    snapshot_id: str | None = None
    retrieval_ledger_id: str | None = None
    canonical_url: str | None = None
    final_url: str | None = None
    published_at: str | None = None
    retrieved_at: str | None = None
    content_hash: str | None = None
    date_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    content_type: str | None = None
    http_status: int | None = None
    extraction_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class FindOnPageRequest(BaseModel):
    document_id: str
    pattern: str
    snapshot_id: str | None = None


class FindOnPageResponse(BaseModel):
    document_id: str
    pattern: str
    matches: list[str] = Field(default_factory=list)
    url: str | None = None
    title: str | None = None
    snapshot_id: str | None = None
    retrieval_ledger_id: str | None = None
    canonical_url: str | None = None
    retrieved_at: str | None = None
    support_spans: list[str] = Field(default_factory=list)


class RetrievalLedgerResponse(BaseModel):
    job_id: str
    retrieval_ledger_id: str
    execution_mode: str
    searches: list[dict[str, Any]] = Field(default_factory=list)
    opened_pages: list[RetrievedSourceSnapshot] = Field(default_factory=list)
    find_on_page_events: list[dict[str, Any]] = Field(default_factory=list)
