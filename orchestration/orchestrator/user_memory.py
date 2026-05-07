"""User-level memory: stable facts that persist across projects.

Distinct from :mod:`orchestration.orchestrator.project_memory`:

  * **Project memory** chunks every assistant turn under a project and
    recalls top-K snippets for the same project.
  * **User memory** holds *stable facts* about the user themselves —
    "works in Python", "prefers concise answers", "lives in Tokyo" —
    that should surface across every project they touch.

Pipeline:

  1. **Pre-filter** the user's message with a cheap regex. If no
     "I am / I work / I prefer / My ..." pattern matches, do nothing —
     no LLM call. This keeps the cost ≈ zero on most turns.
  2. **Extract** when the regex hits: ask a small LLM to return a JSON
     array of ``{kind, text, confidence}`` items. Empty array → no-op.
  3. **Embed + persist** each extracted fact into
     :class:`ConsumerUserMemory`. Idempotency dedupes near-duplicates by
     (user_id, normalized text).
  4. **Recall** at the start of every turn: cosine top-K over the
     user's rows; injected into ``request.metadata.user_facts``.

All write paths are best-effort — failures swallow with a log line so
the chat turn never blocks on extraction.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from sqlalchemy import select

from shared.common.database import Database
from shared.common.models import ConsumerUserMemory

from orchestration.orchestrator.embedding_client import EmbeddingClient
from orchestration.orchestrator.project_memory import (
    decode_embedding,
    encode_embedding,
)

_logger = logging.getLogger(__name__)

__all__ = [
    "UserFactCandidate",
    "UserMemoryHit",
    "UserMemoryReader",
    "UserMemoryWriter",
    "PreFilter",
    "default_pre_filter",
]


# Cheap regex pre-filter: matches messages that talk about the user
# themselves. Hit rate matters more than precision — a false positive
# costs one cheap extractor call; a false negative costs a missed fact
# that we'll catch the next time the user volunteers it.
_PRE_FILTER_RE = re.compile(
    r"("
    r"\bI['’]m\b"
    r"|\bI\s+(?:am|work|prefer|like|don['’]?t\s+like|use|hate|love|study|live|speak)\b"
    r"|\bMy\s+(?:name|job|role|company|team|preferred|favorite)\b"
    r"|\bCall\s+me\b"
    r")",
    re.IGNORECASE,
)


PreFilter = "PreFilter"  # forward type alias for docs


def default_pre_filter(text: str) -> bool:
    """Return ``True`` when ``text`` looks like it talks about the user.

    Conservative: matches "I am / I work / I prefer / My job / Call me"
    and a few related verbs. Misses are tolerated; false positives are
    cheap (one extractor call that returns ``[]``).
    """
    return bool(_PRE_FILTER_RE.search(text or ""))


# Reserved kinds for the ``kind`` column. New kinds can be added by
# extending this set; the retrieval path doesn't validate against it,
# so unknown kinds round-trip without breaking.
_VALID_KINDS = frozenset({"fact", "preference", "skill"})


_DEFAULT_INSTRUCTION = (
    "Extract durable facts about the USER from their message. Return a "
    "JSON array. Each item is "
    '{"kind": "fact"|"preference"|"skill", "text": str, "confidence": float}. '
    "Confidence is 0.0–1.0. Only include facts the user is plainly "
    "stating about themselves; do not invent or paraphrase. If the "
    "message has no such facts, return an empty array []. Reply with "
    "ONLY the JSON array — no prose, no fences."
)


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text or "").strip()


@dataclass(frozen=True, slots=True)
class UserFactCandidate:
    """One extracted fact about the user, before persistence."""

    text: str
    kind: str = "fact"
    confidence: float = 1.0

    def normalized(self) -> str:
        return " ".join((self.text or "").strip().casefold().split())


@dataclass(frozen=True, slots=True)
class UserMemoryHit:
    """One result returned by :meth:`UserMemoryReader.recall`."""

    vector_id: str
    text: str
    kind: str
    confidence: float
    score: float
    metadata: dict[str, Any]


# -- Cosine ----------------------------------------------------------------


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


# -- Reader ----------------------------------------------------------------


class UserMemoryReader:
    """Top-K cosine-similarity retrieval over :class:`ConsumerUserMemory`."""

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
        user_id: str,
        query: str,
        k: int = 3,
    ) -> list[UserMemoryHit]:
        """Embed ``query`` and return the K closest facts for ``user_id``.

        Returns an empty list when the user has no memory or the
        embedding call fails — recall must never break the chat turn.
        """
        if not user_id or k <= 0:
            return []
        try:
            query_vecs = await self._embed.aembed([query])
        except Exception as exc:  # noqa: BLE001 — best-effort
            _logger.warning("user memory recall embed failed: %s", exc)
            return []
        if not query_vecs:
            return []
        query_vec = query_vecs[0]

        with self._db.sessionmaker() as session:
            stmt = select(ConsumerUserMemory).where(
                ConsumerUserMemory.user_id == user_id,
            )
            rows = list(session.scalars(stmt))
            scored: list[tuple[float, ConsumerUserMemory]] = []
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
                UserMemoryHit(
                    vector_id=row.vector_id,
                    text=row.text,
                    kind=row.kind,
                    confidence=float(row.confidence),
                    score=score,
                    metadata=dict(row.metadata_json or {}),
                )
                for score, row in top
            ]


# -- Extractor protocol ----------------------------------------------------


class _ExtractorLLM(Protocol):
    async def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]: ...


# -- Writer ----------------------------------------------------------------


class UserMemoryWriter:
    """Pre-filter + LLM-extract + embed + persist user facts.

    Parameters
    ----------
    database
        Backing store.
    embedding_client
        Embeds extracted fact text for retrieval.
    extractor_llm
        Object satisfying :meth:`AgentProviderClient.chat_completions`.
    pre_filter
        Cheap test: when it returns ``False``, the extractor LLM is
        not called. Default: :func:`default_pre_filter` (regex).
    instruction
        System message used when calling the extractor.
    max_tokens
        Output-token allowance for the extractor call.
    """

    def __init__(
        self,
        *,
        database: Database,
        embedding_client: EmbeddingClient,
        extractor_llm: _ExtractorLLM,
        pre_filter: "callable[[str], bool] | None" = None,
        instruction: str = _DEFAULT_INSTRUCTION,
        max_tokens: int = 512,
    ) -> None:
        self._db = database
        self._embed = embedding_client
        self._llm = extractor_llm
        self._pre_filter = pre_filter or default_pre_filter
        self._instruction = instruction
        self._max_tokens = max_tokens

    # --- extraction ------------------------------------------------------

    def _parse_candidates(self, text: str) -> list[UserFactCandidate]:
        cleaned = _strip_fences(text)
        if not cleaned:
            return []
        # Tolerate prose preamble: find first '[' and last ']'.
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start < 0 or end < start:
            return []
        try:
            raw = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return []
        if not isinstance(raw, list):
            return []
        out: list[UserFactCandidate] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            text_val = item.get("text")
            if not isinstance(text_val, str) or not text_val.strip():
                continue
            kind = item.get("kind") or "fact"
            if not isinstance(kind, str) or kind not in _VALID_KINDS:
                kind = "fact"
            try:
                confidence = float(item.get("confidence", 1.0))
            except (TypeError, ValueError):
                confidence = 1.0
            confidence = max(0.0, min(1.0, confidence))
            out.append(UserFactCandidate(
                text=text_val.strip(), kind=kind, confidence=confidence,
            ))
        return out

    async def _extract(self, user_message: str) -> list[UserFactCandidate]:
        try:
            response = await self._llm.chat_completions(
                {
                    "messages": [
                        {"role": "system", "content": self._instruction},
                        {"role": "user", "content": user_message},
                    ],
                    "max_tokens": self._max_tokens,
                }
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            _logger.warning("user memory extract failed: %s", exc)
            return []
        try:
            content = response["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            return []
        return self._parse_candidates(content)

    # --- persistence -----------------------------------------------------

    async def extract_and_write(
        self,
        *,
        user_id: str,
        user_message: str,
        source_conversation_id: str | None = None,
        source_message_id: str | None = None,
    ) -> int:
        """Pre-filter → extract → embed → persist. Returns rows written.

        Cost shape:
          - regex miss: 0 LLM calls, 0 rows written.
          - regex hit, extractor returns []: 1 LLM call, 0 rows written.
          - regex hit, extractor returns N facts: 1 LLM call, 1 embed
            call (batched over N candidates), 0–N rows written
            (existing duplicates are skipped).
        """
        if not user_id or not user_message or not user_message.strip():
            return 0
        if not self._pre_filter(user_message):
            return 0

        candidates = await self._extract(user_message)
        if not candidates:
            return 0

        # Batch-embed all candidate texts in one call.
        try:
            vectors = await self._embed.aembed([c.text for c in candidates])
        except Exception as exc:  # noqa: BLE001 — best-effort
            _logger.warning("user memory embed failed: %s", exc)
            return 0
        if len(vectors) != len(candidates):
            _logger.warning(
                "user memory embedding count mismatch: %d vectors for %d candidates",
                len(vectors), len(candidates),
            )
            return 0

        # Persist with dedup by normalized text.
        n_written = 0
        with self._db.sessionmaker() as session:
            existing = list(session.scalars(
                select(ConsumerUserMemory).where(
                    ConsumerUserMemory.user_id == user_id,
                )
            ))
            existing_norms = {
                " ".join((row.text or "").strip().casefold().split())
                for row in existing
            }
            for cand, vec in zip(candidates, vectors):
                norm = cand.normalized()
                if not norm or norm in existing_norms:
                    continue
                vector_id = (
                    f"{source_message_id}:{n_written}"
                    if source_message_id
                    else f"user:{user_id}:{abs(hash(norm)) % 10**12}"
                )
                row = ConsumerUserMemory(
                    user_id=user_id,
                    vector_id=vector_id,
                    embedding=encode_embedding(vec),
                    text=cand.text,
                    kind=cand.kind,
                    confidence=cand.confidence,
                    source_conversation_id=source_conversation_id,
                    source_message_id=source_message_id,
                    metadata_json={},
                )
                session.add(row)
                existing_norms.add(norm)
                n_written += 1
            session.commit()
        return n_written
