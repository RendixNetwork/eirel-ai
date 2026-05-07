"""Project memory: chunk → embed → store; query → embed → top-K recall.

Backs the ``ConsumerProjectMemory`` table. The ``ProductOrchestrator``
schedules :meth:`ProjectMemoryWriter.write_message` after persisting
each assistant turn, so over time the project's memory accumulates.
:meth:`ProjectMemoryReader.recall` runs at the start of each turn and
injects top-K snippets into ``metadata.recalled_memory``.

Cosine similarity is computed in Python — fine for the small N this
table holds per project. Production scale-out moves to a pgvector
index (or a dedicated vector store via the SDK's ``VectorStore``
adapters) without changing the read/write contract.
"""
from __future__ import annotations

import logging
import re
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from shared.common.database import Database
from shared.common.models import ConsumerProjectMemory

from orchestration.orchestrator.embedding_client import EmbeddingClient

_logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "MAX_CHUNKS_PER_MESSAGE",
    "MemoryHit",
    "ProjectMemoryWriter",
    "ProjectMemoryReader",
    "chunk_text",
    "encode_embedding",
    "decode_embedding",
]


DEFAULT_CHUNK_SIZE: int = 800
DEFAULT_CHUNK_OVERLAP: int = 100
MAX_CHUNKS_PER_MESSAGE: int = 32


# -- Embedding (de)serialization -------------------------------------------
#
# Storage shape: little-endian float32 array. ``LargeBinary`` column. No
# numpy dependency in the orchestrator path; struct.pack/unpack is
# fast enough for the per-turn write rate we expect.


def encode_embedding(vec: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def decode_embedding(blob: bytes) -> list[float]:
    n = len(blob) // 4
    if n * 4 != len(blob):
        raise ValueError(f"embedding blob length {len(blob)} is not a multiple of 4")
    return list(struct.unpack(f"<{n}f", blob))


# -- Chunker ----------------------------------------------------------------


_PARAGRAPH_BOUNDARY = re.compile(r"\n{2,}")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def chunk_text(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    max_chunks: int = MAX_CHUNKS_PER_MESSAGE,
) -> list[str]:
    """Split ``text`` into LLM-recall-friendly chunks.

    Strategy: paragraph-first, then sentence-first within a paragraph,
    falling back to a hard byte cap. ``overlap`` is the trailing slice
    of the previous chunk that gets prepended to the next — keeps a
    sentence that crosses a chunk boundary recoverable.

    Returns at most ``max_chunks`` chunks. A message longer than
    ``chunk_size * max_chunks`` is silently truncated; the caller is
    expected to enforce a higher-level cap on persisted text length.
    """
    text = (text or "").strip()
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    # First split on paragraph boundaries; long paragraphs get further
    # sentence-split below.
    paragraphs = [p.strip() for p in _PARAGRAPH_BOUNDARY.split(text) if p.strip()]
    units: list[str] = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            units.append(para)
            continue
        # Sentence split inside the paragraph.
        sentences = _SENTENCE_BOUNDARY.split(para)
        buf = ""
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            candidate = (buf + " " + sent).strip() if buf else sent
            if len(candidate) <= chunk_size:
                buf = candidate
                continue
            if buf:
                units.append(buf)
            if len(sent) <= chunk_size:
                buf = sent
                continue
            # A single sentence too long for the chunk — slice it hard.
            for i in range(0, len(sent), chunk_size - overlap):
                units.append(sent[i : i + chunk_size])
            buf = ""
        if buf:
            units.append(buf)

    # Apply overlap by re-stitching.
    chunks: list[str] = []
    for u in units:
        if chunks and overlap:
            tail = chunks[-1][-overlap:]
            merged = (tail + " " + u).strip()
            if len(merged) <= chunk_size:
                # Replace last chunk with the merged version when it
                # fits — avoids near-duplicates.
                chunks[-1] = merged
                continue
        chunks.append(u)
        if len(chunks) >= max_chunks:
            break
    return chunks


# -- Reader ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MemoryHit:
    """One result returned by :meth:`ProjectMemoryReader.recall`."""

    vector_id: str
    text: str
    score: float
    metadata: dict[str, Any]
    source_message_id: str | None


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


class ProjectMemoryReader:
    """Top-K cosine-similarity retrieval over ``ConsumerProjectMemory``."""

    def __init__(
        self,
        *,
        database: Database,
        embedding_client: EmbeddingClient,
    ) -> None:
        self._db = database
        self._embed = embedding_client

    async def recall(
        self,
        *,
        project_id: str,
        query: str,
        k: int = 5,
    ) -> list[MemoryHit]:
        """Embed ``query`` and return the K closest snippets for ``project_id``.

        Ranking is in-process cosine similarity. Returns an empty list
        when the project has no memory or the embedding call fails —
        recall must never break the chat turn.
        """
        if not project_id or k <= 0:
            return []
        try:
            query_vecs = await self._embed.aembed([query])
        except Exception as exc:  # noqa: BLE001 — recall is best-effort
            _logger.warning("memory recall embed failed: %s", exc)
            return []
        if not query_vecs:
            return []
        query_vec = query_vecs[0]

        with self._db.sessionmaker() as session:
            stmt = select(ConsumerProjectMemory).where(
                ConsumerProjectMemory.project_id == project_id,
            )
            rows = list(session.scalars(stmt))
            scored: list[tuple[float, ConsumerProjectMemory]] = []
            for row in rows:
                try:
                    vec = decode_embedding(row.embedding or b"")
                except ValueError:
                    continue
                score = _cosine(query_vec, vec)
                scored.append((score, row))
            scored.sort(key=lambda pair: pair[0], reverse=True)
            top = scored[:k]
            return [
                MemoryHit(
                    vector_id=row.vector_id,
                    text=row.text,
                    score=score,
                    metadata=dict(row.metadata_json or {}),
                    source_message_id=row.source_message_id,
                )
                for score, row in top
            ]


# -- Writer ----------------------------------------------------------------


class ProjectMemoryWriter:
    """Chunk, embed, and persist a message's contribution to project memory."""

    def __init__(
        self,
        *,
        database: Database,
        embedding_client: EmbeddingClient,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
        max_chunks: int = MAX_CHUNKS_PER_MESSAGE,
    ) -> None:
        self._db = database
        self._embed = embedding_client
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._max_chunks = max_chunks

    async def write_message(
        self,
        *,
        project_id: str,
        text: str,
        source_message_id: str | None = None,
        role: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> int:
        """Embed + persist the chunks of ``text`` under ``project_id``.

        Returns the number of rows written. Idempotent on
        ``(project_id, source_message_id)``: if rows already exist for
        that pair, they're replaced with the fresh chunks rather than
        duplicated. Failures are swallowed and logged — the orchestrator
        calls this in a fire-and-forget task.
        """
        if not project_id:
            return 0
        chunks = chunk_text(
            text,
            chunk_size=self._chunk_size,
            overlap=self._overlap,
            max_chunks=self._max_chunks,
        )
        if not chunks:
            return 0
        try:
            vectors = await self._embed.aembed(chunks)
        except Exception as exc:  # noqa: BLE001 — best-effort
            _logger.warning("project memory embed failed: %s", exc)
            return 0
        if len(vectors) != len(chunks):
            _logger.warning(
                "embedding count mismatch: %d vectors for %d chunks",
                len(vectors), len(chunks),
            )
            return 0

        with self._db.sessionmaker() as session:
            # Idempotency: clear any previous rows for this
            # (project_id, source_message_id) pair before writing.
            if source_message_id:
                stale = list(session.scalars(
                    select(ConsumerProjectMemory).where(
                        ConsumerProjectMemory.project_id == project_id,
                        ConsumerProjectMemory.source_message_id == source_message_id,
                    ),
                ))
                for row in stale:
                    session.delete(row)

            n_written = 0
            for idx, (chunk, vec) in enumerate(zip(chunks, vectors)):
                vector_id = (
                    f"{source_message_id}:{idx}"
                    if source_message_id
                    else f"adhoc:{idx}:{abs(hash(chunk)) % 10**12}"
                )
                metadata: dict[str, Any] = {
                    "chunk_index": idx,
                    "chunk_count": len(chunks),
                }
                if role:
                    metadata["role"] = role
                if extra_metadata:
                    metadata.update(extra_metadata)
                row = ConsumerProjectMemory(
                    project_id=project_id,
                    vector_id=vector_id,
                    embedding=encode_embedding(vec),
                    text=chunk,
                    source_message_id=source_message_id,
                    metadata_json=metadata,
                )
                session.add(row)
                n_written += 1
            session.commit()
            return n_written
