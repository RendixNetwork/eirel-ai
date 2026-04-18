from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SemanticScholarSearchRequest(BaseModel):
    query: str
    year: str | None = None  # "2020-" or "2020-2024" per Semantic Scholar syntax
    fields_of_study: list[str] = Field(default_factory=list)
    venue: str | None = None
    open_access_only: bool = False
    max_results: int = Field(default=10, ge=1, le=100)


class SemanticScholarBatchRequest(BaseModel):
    paper_ids: list[str] = Field(default_factory=list, min_length=1, max_length=500)


class SemanticScholarPaper(BaseModel):
    paper_id: str
    title: str
    abstract: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str = ""
    citation_count: int = 0
    influential_citation_count: int = 0
    arxiv_id: str = ""
    doi: str = ""
    open_access_pdf_url: str = ""
    url: str = ""
    external_ids: dict[str, str] = Field(default_factory=dict)
    content_sha256: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class SemanticScholarSearchResponse(BaseModel):
    query: str
    papers: list[SemanticScholarPaper] = Field(default_factory=list)
    result_count: int = 0
    total_matching: int = 0
    retrieved_at: str | None = None
    retrieval_ledger_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SemanticScholarBatchResponse(BaseModel):
    papers: list[SemanticScholarPaper] = Field(default_factory=list)
    result_count: int = 0
    retrieved_at: str | None = None
    retrieval_ledger_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
