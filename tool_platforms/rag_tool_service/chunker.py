"""Text chunking for the RAG indexer.

Simple character-window chunker with a soft sentence-aware boundary.
Tokenizer-aware chunking would be more accurate (~4x cost, ~2x
correctness for chunk boundaries), but we keep it character-based
because the embedding model handles short inputs gracefully and
operator-curated corpora rarely hit pathological cases.

Defaults are tuned for ``text-embedding-3-small`` (8192-token context,
~32K-character input safe-zone). Each chunk is ~1500 chars with
~200-char overlap — small enough that a single retrieved chunk fits
comfortably in a typical agent's context budget, large enough that
~3-5 chunks cover most question-answer spans.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["Chunk", "chunk_document"]


_DEFAULT_CHUNK_CHARS = 1500
_DEFAULT_OVERLAP_CHARS = 200
# Soft boundary search window — when looking for a sentence break to
# end a chunk on, scan back at most this many chars from the hard
# limit before falling through to a hard slice.
_BOUNDARY_LOOKBACK = 200

_SENTENCE_END_RE = re.compile(r"[.!?]\s+")


@dataclass(frozen=True)
class Chunk:
    """One chunk of a document, ready for embedding + indexing."""

    doc_id: str
    chunk_id: str
    text: str
    char_start: int
    char_end: int


def chunk_document(
    *,
    doc_id: str,
    content: str,
    chunk_chars: int = _DEFAULT_CHUNK_CHARS,
    overlap_chars: int = _DEFAULT_OVERLAP_CHARS,
) -> list[Chunk]:
    """Split a document into overlapping chunks.

    Returns at least one chunk per non-empty document. Chunk ids are
    ``{doc_id}#{ordinal}``, zero-padded to 4 digits so lexical sort
    matches ordinal order.
    """
    text = (content or "").strip()
    if not text:
        return []
    if chunk_chars <= 0:
        raise ValueError(f"chunk_chars must be positive; got {chunk_chars}")
    if overlap_chars < 0 or overlap_chars >= chunk_chars:
        raise ValueError(
            f"overlap_chars must be in [0, {chunk_chars}); got {overlap_chars}"
        )

    chunks: list[Chunk] = []
    pos = 0
    ordinal = 0
    text_len = len(text)
    while pos < text_len:
        end = min(pos + chunk_chars, text_len)
        # Try to end on a sentence boundary within the lookback window.
        if end < text_len:
            window = text[max(pos, end - _BOUNDARY_LOOKBACK):end]
            match = None
            for m in _SENTENCE_END_RE.finditer(window):
                match = m  # last match in the window
            if match is not None:
                end = max(pos, end - _BOUNDARY_LOOKBACK) + match.end()
        chunk_text = text[pos:end].strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}#{ordinal:04d}",
                    text=chunk_text,
                    char_start=pos,
                    char_end=end,
                ),
            )
            ordinal += 1
        if end >= text_len:
            break
        pos = max(end - overlap_chars, pos + 1)
    return chunks
