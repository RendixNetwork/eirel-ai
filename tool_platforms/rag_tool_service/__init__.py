"""RAG tool service.

Subnet-owned vector search service that miner agents call to retrieve
top-k chunks from a curated document corpus. Mirrors the Claude
Projects / ChatGPT Projects pattern: operator (or product layer)
indexes documents into a corpus, miners query that corpus by id +
natural-language query, and answers cite the retrieved chunks.

Two surfaces:
  * ``POST /v1/rag/corpora`` (master-token-gated) — index a batch of
    documents into a corpus. Idempotent on ``corpus_id``: re-posting
    the same corpus replaces it. Operators populate fixture corpora
    for eval; ``ProductOrchestrator`` populates per-project corpora
    when a user uploads docs.
  * ``POST /v1/rag/retrieve`` (per-job-token-gated, miner-callable) —
    retrieve top-k chunks for a query. Server-attested via the
    ``OrchestratorToolCallLog`` ledger so the validator can score
    retrieval-quality from authoritative records.
"""
from __future__ import annotations

from tool_platforms.rag_tool_service.app import create_app, main

__all__ = ["create_app", "main"]
