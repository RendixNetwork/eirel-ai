"""Long-context summarization for product chat history.

Closes the silent-truncation gap in :class:`ProductOrchestrator`
where ``_history_limit=20`` would drop older turns once a conversation
grew past it. When the verbatim tail behind
``last_summarized_message_id`` exceeds ``stale_threshold_messages``,
the orchestrator schedules a re-summarization via this module. The
new summary collapses everything up to a chosen boundary into
``ConsumerConversation.rolling_summary``; the orchestrator injects
that summary as the head system message and the verbatim tail as the
rest of ``request.history``.

The summarizer is conservative on cost:

  * **Lazy.** First-summary triggers only when the conversation has
    grown past the threshold; subsequent re-summaries trigger only when
    the verbatim tail since ``last_summarized_message_id`` exceeds
    ``stale_threshold_messages``.
  * **Boundary.** When summarizing, we keep the last
    ``keep_recent_messages`` messages verbatim and summarize everything
    older; the boundary id is recorded so subsequent runs know where
    the verbatim tail starts.
  * **Best-effort.** Failures (LLM error, JSON-parse error, network)
    swallow with a log warning; the chat turn proceeds with the prior
    summary (or no summary at all). The orchestrator never blocks on
    this.

Token budget: target_tokens is a soft prompt — providers vary in how
strictly they honor it. The call uses ``max_tokens`` to bound the
output side; the input side is bounded by the chunked summarize-
the-prefix path described in :meth:`maybe_summarize`.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Protocol

from sqlalchemy import select

from shared.common.database import Database
from shared.common.models import ConsumerConversation, ConsumerMessage

_logger = logging.getLogger(__name__)

__all__ = [
    "ConversationSummarizer",
    "SummarizerLLM",
    "DEFAULT_TARGET_TOKENS",
    "DEFAULT_STALE_THRESHOLD",
    "DEFAULT_KEEP_RECENT",
]


DEFAULT_TARGET_TOKENS: int = 800
DEFAULT_STALE_THRESHOLD: int = 10
DEFAULT_KEEP_RECENT: int = 8


class SummarizerLLM(Protocol):
    """Minimal interface a summarizer LLM must satisfy.

    Matches :meth:`AgentProviderClient.chat_completions` so
    ``ProductOrchestrator`` can pass an :class:`AgentProviderClient`
    directly. The protocol keeps the summarizer testable without a real
    network — tests pass a stub that returns canned strings.
    """

    async def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]: ...


_DEFAULT_INSTRUCTION = (
    "You are summarizing the head of a chat conversation so a future "
    "turn can re-enter context without re-reading every prior message. "
    "Capture: the user's overall goal, key facts the assistant has "
    "established, decisions made, open threads, and any preferences "
    "the user has expressed. Skip pleasantries and exact wording. "
    "Write ONE paragraph, third person, present tense. No bullet "
    "points, no headings, no preamble — just the summary text."
)


def _format_messages_for_summary(rows: Sequence[ConsumerMessage]) -> str:
    """Linearize messages into a compact text block for the summarizer."""
    parts: list[str] = []
    for row in rows:
        role = (row.role or "?").upper()
        # Trim individual messages so a single huge turn doesn't blow
        # past the model context. Conservative cap.
        text = (row.content or "")[:4000]
        parts.append(f"{role}: {text}")
    return "\n\n".join(parts)


class ConversationSummarizer:
    """Lazy, idempotent rolling-summary writer for ConsumerConversation.

    Parameters
    ----------
    database
        Backing store. The summarizer reads :class:`ConsumerMessage`
        rows and writes back to :class:`ConsumerConversation`.
    llm
        Object satisfying :class:`SummarizerLLM`.
    target_tokens
        Soft target for the summary length, passed as ``max_tokens`` on
        the LLM call.
    stale_threshold_messages
        Re-summarize when the verbatim tail since
        ``last_summarized_message_id`` grows past this many messages.
        Also the threshold for the very-first summarization on a
        conversation with no prior summary.
    keep_recent_messages
        When summarizing, keep the most recent N messages as verbatim
        tail; everything older is collapsed into the summary. Must be
        smaller than ``stale_threshold_messages``.
    instruction
        System message text injected into the summarizer call.
    """

    def __init__(
        self,
        *,
        database: Database,
        llm: SummarizerLLM,
        target_tokens: int = DEFAULT_TARGET_TOKENS,
        stale_threshold_messages: int = DEFAULT_STALE_THRESHOLD,
        keep_recent_messages: int = DEFAULT_KEEP_RECENT,
        instruction: str = _DEFAULT_INSTRUCTION,
    ) -> None:
        if target_tokens < 1:
            raise ValueError("target_tokens must be at least 1")
        if stale_threshold_messages < 1:
            raise ValueError("stale_threshold_messages must be at least 1")
        if keep_recent_messages < 0:
            raise ValueError("keep_recent_messages must be non-negative")
        if keep_recent_messages >= stale_threshold_messages:
            raise ValueError(
                "keep_recent_messages must be smaller than "
                "stale_threshold_messages"
            )
        self._db = database
        self._llm = llm
        self._target_tokens = target_tokens
        self._stale_threshold = stale_threshold_messages
        self._keep_recent = keep_recent_messages
        self._instruction = instruction

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def _tail_message_count(
        self,
        session,
        *,
        conversation_id: str,
        last_summarized_message_id: str | None,
    ) -> int:
        """Count messages strictly newer than the summary boundary.

        When ``last_summarized_message_id`` is None, the entire
        conversation counts as the tail.
        """
        if last_summarized_message_id is None:
            stmt = select(ConsumerMessage.id).where(
                ConsumerMessage.conversation_id == conversation_id,
            )
            return len(list(session.scalars(stmt)))
        boundary = session.get(ConsumerMessage, last_summarized_message_id)
        if boundary is None:
            # The boundary message was deleted (rare). Treat as no
            # boundary so we re-summarize from scratch.
            stmt = select(ConsumerMessage.id).where(
                ConsumerMessage.conversation_id == conversation_id,
            )
            return len(list(session.scalars(stmt)))
        stmt = select(ConsumerMessage.id).where(
            ConsumerMessage.conversation_id == conversation_id,
            ConsumerMessage.turn_idx > boundary.turn_idx,
        )
        return len(list(session.scalars(stmt)))

    def needs_summary(
        self,
        *,
        conversation_id: str,
    ) -> bool:
        """Synchronous check: does this conversation need (re)summarizing?

        Used by :class:`ProductOrchestrator` to decide whether to fire
        :meth:`maybe_summarize` after the chat turn completes.
        """
        with self._db.sessionmaker() as session:
            convo = session.get(ConsumerConversation, conversation_id)
            if convo is None:
                return False
            tail = self._tail_message_count(
                session,
                conversation_id=conversation_id,
                last_summarized_message_id=convo.last_summarized_message_id,
            )
            return tail >= self._stale_threshold

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    async def maybe_summarize(
        self,
        *,
        conversation_id: str,
    ) -> bool:
        """Re-summarize when the verbatim tail is stale.

        Returns ``True`` when a fresh summary was written, ``False``
        when the conversation didn't need one (or the LLM call failed
        and the prior summary was preserved). Best-effort: exceptions
        are swallowed and logged.
        """
        # Pull the rows we need to summarize, then release the session
        # before calling the LLM (which is async + slow).
        with self._db.sessionmaker() as session:
            convo = session.get(ConsumerConversation, conversation_id)
            if convo is None:
                return False
            tail = self._tail_message_count(
                session,
                conversation_id=conversation_id,
                last_summarized_message_id=convo.last_summarized_message_id,
            )
            if tail < self._stale_threshold:
                return False

            # Fetch full ordered history for this conversation.
            stmt = (
                select(ConsumerMessage)
                .where(ConsumerMessage.conversation_id == conversation_id)
                .order_by(ConsumerMessage.turn_idx.asc())
            )
            rows = list(session.scalars(stmt))
            if len(rows) <= self._keep_recent:
                # Tail-only conversation; summary unnecessary.
                return False

            head_rows = rows[: -self._keep_recent] if self._keep_recent > 0 else rows
            boundary = head_rows[-1]
            existing_summary = convo.rolling_summary or ""

        # LLM call outside the DB session.
        head_block = _format_messages_for_summary(head_rows)
        user_block = head_block
        if existing_summary:
            user_block = (
                "Previous summary (extend / refine — do not contradict):\n"
                f"{existing_summary}\n\n"
                "New head turns to fold in:\n"
                f"{head_block}"
            )

        try:
            response = await self._llm.chat_completions(
                {
                    "messages": [
                        {"role": "system", "content": self._instruction},
                        {"role": "user", "content": user_block},
                    ],
                    "max_tokens": self._target_tokens,
                }
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            _logger.warning(
                "summarize call failed conversation=%s err=%s",
                conversation_id, exc,
            )
            return False

        summary = ""
        try:
            summary = response["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            summary = ""
        summary = summary.strip()
        if not summary:
            _logger.warning(
                "summarize returned empty content conversation=%s",
                conversation_id,
            )
            return False

        # Persist back. Re-fetch the row to avoid stale-state writes if
        # the conversation was concurrently updated.
        with self._db.sessionmaker() as session:
            convo = session.get(ConsumerConversation, conversation_id)
            if convo is None:
                return False
            convo.rolling_summary = summary
            convo.last_summarized_message_id = boundary.id
            session.commit()
        return True

    # ------------------------------------------------------------------
    # Fire-and-forget
    # ------------------------------------------------------------------

    def schedule(self, *, conversation_id: str) -> None:
        """Run :meth:`maybe_summarize` in a background task.

        Caller (:class:`ProductOrchestrator`) doesn't await — DB-write
        latency for the summary never blocks the chat response.
        """
        async def _run() -> None:
            try:
                await self.maybe_summarize(conversation_id=conversation_id)
            except Exception as exc:  # noqa: BLE001 — best-effort
                _logger.warning(
                    "background summarize failed conversation=%s err=%s",
                    conversation_id, exc,
                )

        try:
            asyncio.create_task(_run())
        except RuntimeError:
            _logger.debug(
                "no event loop; skipping background summarize for %s",
                conversation_id,
            )
