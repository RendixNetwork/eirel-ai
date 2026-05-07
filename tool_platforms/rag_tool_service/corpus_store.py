"""In-process corpus store + cosine-similarity search.

Why in-memory: eval corpora are bounded (operator-curated, ~hundreds
of chunks per corpus) and recreated per epoch. Persistence adds infra
(pgvector / Qdrant) without changing the eval surface today. When
product-mode RAG over user-uploaded docs lands at scale, swap this
backend for pgvector behind the same ``CorpusStore`` interface.

Backend:
  * Embeddings stacked as a numpy array per corpus → matrix-vector
    cosine similarity for retrieval. ~tens of microseconds per query
    at thousand-chunk scale.
  * Indexes are dict[corpus_id] → ``IndexedCorpus``. Replacement on
    re-index is atomic at the dict level.

Thread-safety: the FastAPI app uses ``asyncio.Lock`` per corpus_id
when mutating, so concurrent retrievers don't see a half-written index.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from tool_platforms.rag_tool_service.chunker import Chunk

_logger = logging.getLogger(__name__)

__all__ = [
    "CorpusStore",
    "IndexedCorpus",
    "RetrievedChunk",
]


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk in a retrieve response — text + score + provenance."""

    doc_id: str
    chunk_id: str
    text: str
    score: float
    char_start: int
    char_end: int


@dataclass
class IndexedCorpus:
    """One corpus's stacked embeddings + chunk metadata.

    ``embeddings`` shape is ``(n_chunks, embedding_dim)``, L2-normalized
    so cosine similarity reduces to a dot product.
    """

    corpus_id: str
    chunks: list[Chunk] = field(default_factory=list)
    embeddings: np.ndarray | None = None  # (n_chunks, dim) float32

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)


class CorpusStore:
    """In-memory dict[corpus_id] → IndexedCorpus."""

    def __init__(self) -> None:
        self._corpora: dict[str, IndexedCorpus] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, corpus_id: str) -> asyncio.Lock:
        if corpus_id not in self._locks:
            self._locks[corpus_id] = asyncio.Lock()
        return self._locks[corpus_id]

    async def upsert(
        self,
        *,
        corpus_id: str,
        chunks: Sequence[Chunk],
        embeddings: list[list[float]],
    ) -> int:
        """Replace or create a corpus. Returns the new chunk count.

        ``embeddings`` must align 1:1 with ``chunks`` and share a
        consistent dim across all entries.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"upsert mismatch: {len(chunks)} chunks vs "
                f"{len(embeddings)} embeddings"
            )
        async with self._get_lock(corpus_id):
            if not chunks:
                # Empty upsert clears the corpus rather than leaving
                # the prior one in place — matches REST PUT semantics.
                self._corpora[corpus_id] = IndexedCorpus(corpus_id=corpus_id)
                return 0
            mat = np.asarray(embeddings, dtype=np.float32)
            # L2-normalize so cosine similarity = dot product. Avoid
            # divide-by-zero on degenerate (all-zero) vectors.
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            mat = mat / norms
            self._corpora[corpus_id] = IndexedCorpus(
                corpus_id=corpus_id,
                chunks=list(chunks),
                embeddings=mat,
            )
            return len(chunks)

    def list_corpora(self) -> list[dict[str, int]]:
        return [
            {"corpus_id": cid, "n_chunks": c.n_chunks}
            for cid, c in self._corpora.items()
        ]

    def has(self, corpus_id: str) -> bool:
        return corpus_id in self._corpora

    async def retrieve(
        self,
        *,
        corpus_id: str,
        query_embedding: list[float],
        k: int,
    ) -> list[RetrievedChunk]:
        """Top-k cosine search. Returns empty list when corpus is
        unknown or empty.
        """
        corpus = self._corpora.get(corpus_id)
        if corpus is None or corpus.embeddings is None or corpus.n_chunks == 0:
            return []
        if k <= 0:
            return []
        q = np.asarray(query_embedding, dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0.0:
            return []
        q = q / q_norm
        scores = corpus.embeddings @ q  # (n_chunks,)
        # argpartition for top-k then sort the slice
        n = scores.shape[0]
        k_eff = min(k, n)
        top_idx = np.argpartition(-scores, k_eff - 1)[:k_eff]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        out: list[RetrievedChunk] = []
        for i in top_idx:
            ch = corpus.chunks[int(i)]
            out.append(
                RetrievedChunk(
                    doc_id=ch.doc_id,
                    chunk_id=ch.chunk_id,
                    text=ch.text,
                    score=float(scores[int(i)]),
                    char_start=ch.char_start,
                    char_end=ch.char_end,
                ),
            )
        return out
