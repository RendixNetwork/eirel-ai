"""Product-mode orchestrator: real users → owner-api serving runtime.

Mirrors :class:`~orchestration.orchestrator.graph_orchestrator.GraphOrchestrator`
but routes against ``ServingDeployment`` rows and owns user-facing state
(history, preferences, project memory). The agent on the other end of
the wire receives the same ``AgentInvocationRequest`` envelope as in
eval mode — only the source of ``history`` / ``metadata`` differs.

State partition (locked by design):

  * **User-facing** (history, preferences, projects) lives here, in the
    eirel-ai product DB. Persists across promotions.
  * **Agent-internal** (intra-turn scratchpad, mid-turn Interrupt resume)
    lives in the agent's graph state via the SDK Checkpointer.

The orchestrator generates a fresh graph ``thread_id`` per turn so the
agent never sees cross-turn state on its own pod.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any
from uuid import uuid4

import httpx
from sqlalchemy import select

from shared.common.database import Database
from shared.common.models import (
    ConsumerAttachment,
    ConsumerConversation,
    ConsumerMessage,
    ConsumerPreference,
    ConsumerProject,
    ConsumerProjectMemory,
    ConsumerUser,
)

from orchestration.orchestrator.conversation_summarizer import (
    ConversationSummarizer,
    SummarizerLLM,
)
from orchestration.orchestrator.mcp_dispatcher import MCPToolDispatcher
from orchestration.orchestrator.safety_pipeline import (
    SafetyOutcome,
    SafetyPipeline,
)
from orchestration.orchestrator.embedding_client import (
    EmbeddingClient,
    build_default_embedding_client,
)
from orchestration.orchestrator.project_memory import (
    ProjectMemoryReader,
    ProjectMemoryWriter,
)
from orchestration.orchestrator.serving_picker import (
    NoEligibleServingDeploymentError,
    ServingCandidate,
    ServingPicker,
)
from orchestration.orchestrator.user_memory import (
    UserMemoryReader,
    UserMemoryWriter,
)

_logger = logging.getLogger(__name__)

__all__ = ["ProductOrchestrator", "ProductOrchestratorError"]


# Default knobs — overridable per request.
_DEFAULT_HISTORY_LIMIT = 20
_DEFAULT_RECALL_K = 5
_DEFAULT_USER_FACT_K = 3

# Event taxonomy from the graph runtime — drop ``trace`` before yielding
# to consumers (the runtime proxy already tees it to eiretes).
_PASSTHROUGH_EVENTS = frozenset({
    "delta", "tool_call", "tool_result", "citation", "checkpoint", "done",
})
_DROP_EVENTS = frozenset({"trace"})


class ProductOrchestratorError(RuntimeError):
    """Raised when the product orchestrator can't serve the request."""


class ProductOrchestrator:
    """User-prompt-in, NDJSON-out orchestrator for the product runtime."""

    def __init__(
        self,
        *,
        database: Database,
        serving_picker: ServingPicker,
        owner_api_url: str | None = None,
        internal_service_token: str | None = None,
        timeout_seconds: float = 1800.0,
        transport: httpx.AsyncBaseTransport | None = None,
        history_limit: int = _DEFAULT_HISTORY_LIMIT,
        recall_k: int = _DEFAULT_RECALL_K,
        user_fact_k: int = _DEFAULT_USER_FACT_K,
        embedding_client: EmbeddingClient | None = None,
        summarizer: ConversationSummarizer | None = None,
        user_memory_writer: UserMemoryWriter | None = None,
        safety_pipeline: SafetyPipeline | None = None,
        mcp_dispatcher: MCPToolDispatcher | None = None,
    ):
        self._db = database
        self._picker = serving_picker
        self._owner_api_url = (
            owner_api_url or os.getenv("OWNER_API_URL", "http://owner-api:8000")
        ).rstrip("/")
        self._internal_token = (
            internal_service_token
            or os.getenv("EIREL_INTERNAL_SERVICE_TOKEN", "")
        )
        self._timeout = timeout_seconds
        self._transport = transport
        self._history_limit = max(0, int(history_limit))
        self._recall_k = max(0, int(recall_k))
        self._user_fact_k = max(0, int(user_fact_k))
        # Embedding client is optional — when not supplied we lazily
        # build one from env config. Tests pass a stub directly so the
        # recall + write paths exercise without real network.
        self._embed = embedding_client or build_default_embedding_client()
        self._memory_reader = ProjectMemoryReader(
            database=database, embedding_client=self._embed,
        )
        self._memory_writer = ProjectMemoryWriter(
            database=database, embedding_client=self._embed,
        )
        # Long-context summarization + user-level memory are optional.
        # When injected, the orchestrator runs them; when absent,
        # history loading falls back to last-N slicing and
        # metadata.user_facts stays empty. Keeps the eval/test path
        # lightweight (no LLM extraction unless wired explicitly).
        self._summarizer = summarizer
        self._user_memory_writer = user_memory_writer
        self._user_memory_reader = UserMemoryReader(
            database=database, embedding_client=self._embed,
        )
        # Orchestrator-boundary safety pipeline. None ≡ off (eval/dev).
        # Product runs ship PIIRedactionGuard + PromptInjectionGuard.
        self._safety = safety_pipeline
        # Orchestrator-only MCP dispatcher. Miners never see MCP — the
        # dispatcher decides + executes on the user's behalf and the
        # orchestrator injects results into the envelope's
        # ``metadata.mcp_tool_results`` block.
        self._mcp = mcp_dispatcher

    # ------------------------------------------------------------------
    # Headers / client
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._internal_token:
            headers["Authorization"] = f"Bearer {self._internal_token}"
        return headers

    def _client_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"timeout": self._timeout}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return kwargs

    # ------------------------------------------------------------------
    # Context loading
    # ------------------------------------------------------------------

    def _load_user(self, session, user_id: str) -> ConsumerUser:
        user = session.get(ConsumerUser, user_id)
        if user is None:
            raise ProductOrchestratorError(f"unknown user_id {user_id!r}")
        return user

    def _load_project(
        self, session, project_id: str | None
    ) -> ConsumerProject | None:
        if not project_id:
            return None
        project = session.get(ConsumerProject, project_id)
        if project is None:
            raise ProductOrchestratorError(f"unknown project_id {project_id!r}")
        return project

    def _load_or_create_conversation(
        self,
        session,
        *,
        user_id: str,
        conversation_id: str | None,
        project_id: str | None,
        family_id: str,
    ) -> ConsumerConversation:
        if conversation_id is not None:
            convo = session.get(ConsumerConversation, conversation_id)
            if convo is None:
                raise ProductOrchestratorError(
                    f"unknown conversation_id {conversation_id!r}"
                )
            if convo.user_id != user_id:
                raise ProductOrchestratorError(
                    "conversation does not belong to this user"
                )
            return convo
        # Fresh conversation.
        convo = ConsumerConversation(
            conversation_id=str(uuid4()),
            user_id=user_id,
            project_id=project_id,
            family_id=family_id,
            title=None,
            last_message_at=None,
        )
        session.add(convo)
        session.flush()
        return convo

    def _load_history(
        self, session, conversation_id: str
    ) -> list[dict[str, Any]]:
        """Load history with optional rolling-summary head.

        When the conversation has a populated ``rolling_summary`` (set
        by :class:`ConversationSummarizer` on prior turns), the
        envelope's ``history`` becomes ``[summary system message, ...
        verbatim tail]``. Verbatim tail is the messages strictly newer
        than ``last_summarized_message_id``, capped at
        ``_history_limit`` newest entries. Without a summary, this
        falls back to the last-N slice.
        """
        if self._history_limit == 0:
            return []
        convo = session.get(ConsumerConversation, conversation_id)
        summary_text: str | None = None
        boundary_turn_idx: int | None = None
        if convo is not None:
            summary_text = (convo.rolling_summary or "").strip() or None
            if summary_text and convo.last_summarized_message_id:
                boundary = session.get(
                    ConsumerMessage, convo.last_summarized_message_id
                )
                if boundary is not None:
                    boundary_turn_idx = boundary.turn_idx
        stmt = (
            select(ConsumerMessage)
            .where(ConsumerMessage.conversation_id == conversation_id)
        )
        if boundary_turn_idx is not None:
            stmt = stmt.where(ConsumerMessage.turn_idx > boundary_turn_idx)
        stmt = stmt.order_by(ConsumerMessage.turn_idx.desc()).limit(
            self._history_limit
        )
        rows = list(session.scalars(stmt))
        rows.reverse()  # oldest first for the agent
        out: list[dict[str, Any]] = []
        if summary_text:
            out.append({
                "role": "system",
                "content": (
                    "Earlier conversation summary (compressed by the "
                    f"orchestrator):\n{summary_text}"
                ),
            })
        out.extend({"role": r.role, "content": r.content} for r in rows)
        return out

    def _load_preferences(
        self,
        session,
        *,
        user_id: str,
        project_id: str | None,
    ) -> dict[str, Any]:
        stmt = select(ConsumerPreference).where(
            ConsumerPreference.user_id == user_id,
        )
        result: dict[str, Any] = {"global": {}, "project": {}}
        for row in session.scalars(stmt):
            if row.scope == "global":
                result["global"][row.key] = row.value_json
            elif row.scope == "project" and row.project_id == project_id:
                result["project"][row.key] = row.value_json
        return result

    def _load_attachments(
        self,
        session,
        *,
        user_id: str,
        attachment_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Hydrate uploaded files into the metadata.attached_files block.

        Validates ownership (the user that uploaded must match the
        user invoking the chat). Skips silently for unknown ids
        rather than raising — matches ChatGPT/Claude behavior where a
        stale attachment id from an old session doesn't kill the turn.
        """
        if not attachment_ids:
            return []
        stmt = select(ConsumerAttachment).where(
            ConsumerAttachment.id.in_(attachment_ids),
            ConsumerAttachment.user_id == user_id,
        )
        out: list[dict[str, Any]] = []
        rows_by_id = {row.id: row for row in session.scalars(stmt)}
        # Preserve caller-supplied ordering so the agent sees files in
        # upload order.
        for attachment_id in attachment_ids:
            row = rows_by_id.get(attachment_id)
            if row is None:
                continue
            out.append({
                "attachment_id": row.id,
                "filename": row.filename,
                "content_type": row.content_type,
                "extracted_text": row.extracted_text,
                "extraction_status": row.extraction_status,
                "metadata": dict(row.extraction_metadata_json or {}),
            })
        return out

    async def _recall_user_facts(
        self,
        *,
        user_id: str,
        query: str,
    ) -> list[dict[str, Any]]:
        """Top-K stable user facts via embedding-based recall.

        Returns ``[]`` when the user has no memory, when ``user_fact_k``
        is zero, or when the embedding call fails. Best-effort —
        recall must never break the chat turn.
        """
        if not user_id or self._user_fact_k == 0 or not query.strip():
            return []
        try:
            hits = await self._user_memory_reader.recall(
                user_id=user_id, query=query, k=self._user_fact_k,
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            _logger.warning("user fact recall failed: %s", exc)
            return []
        return [
            {
                "vector_id": h.vector_id,
                "text": h.text,
                "kind": h.kind,
                "confidence": round(h.confidence, 3),
                "score": round(h.score, 6),
            }
            for h in hits
        ]

    def _schedule_user_memory_write(
        self,
        *,
        user_id: str,
        prompt: str,
        conversation_id: str,
        source_message_id: str | None,
    ) -> None:
        """Fire-and-forget extract-and-write for the user's prompt.

        No-op when the orchestrator was constructed without a
        ``user_memory_writer``. Same swallow-and-log discipline as
        :meth:`_schedule_memory_write`.
        """
        if self._user_memory_writer is None or not prompt.strip():
            return
        import asyncio as _asyncio

        async def _run() -> None:
            try:
                await self._user_memory_writer.extract_and_write(
                    user_id=user_id,
                    user_message=prompt,
                    source_conversation_id=conversation_id,
                    source_message_id=source_message_id,
                )
            except Exception as exc:  # noqa: BLE001 — best-effort
                _logger.warning(
                    "background user memory write failed: user=%s err=%s",
                    user_id, exc,
                )

        try:
            _asyncio.create_task(_run())
        except RuntimeError:
            _logger.debug("no event loop; skipping background user memory write")

    async def _safety_pre_input(
        self,
        *,
        prompt: str,
        user_id: str,
        conversation_id: str | None,
        project_id: str | None,
    ) -> SafetyOutcome:
        if self._safety is None or self._safety.empty:
            return SafetyOutcome(allow=True, text=prompt)
        ctx = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "project_id": project_id,
            "stage": "pre_input",
        }
        return await self._safety.pre_input(prompt, ctx)

    async def _safety_post_output(
        self,
        *,
        content: str,
        user_id: str,
        conversation_id: str,
        project_id: str | None,
    ) -> SafetyOutcome:
        if self._safety is None or self._safety.empty:
            return SafetyOutcome(allow=True, text=content)
        ctx = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "project_id": project_id,
            "stage": "post_output",
        }
        return await self._safety.post_output(content, ctx)

    async def _run_mcp_dispatch(
        self,
        *,
        user_id: str,
        prompt: str,
        history: list[dict[str, Any]],
        conversation_id: str | None,
    ) -> list[dict[str, Any]]:
        """Decide + execute MCP tool calls; return result block for metadata.

        Best-effort: any failure (no dispatcher, no active connections,
        planner LLM down, relay timeouts) returns ``[]`` so the chat
        turn proceeds with no MCP context. The audit log captures
        per-call outcomes regardless.
        """
        if self._mcp is None or not user_id:
            return []
        available = self._mcp.available_tools(user_id=user_id)
        if not available:
            return []
        calls = await self._mcp.decide_calls(
            prompt=prompt, history=history, available=available,
        )
        if not calls:
            return []
        results = await self._mcp.execute_calls(
            user_id=user_id, calls=calls,
            conversation_id=conversation_id,
        )
        return [
            {
                "integration_slug": r.integration_slug,
                "tool_name": r.tool_name,
                "args": r.args,
                "ok": r.ok,
                "result_summary": r.result_summary,
                "error": r.error,
                "latency_ms": r.latency_ms,
                "cost_usd": round(r.cost_usd, 6),
            }
            for r in results
        ]

    def _schedule_summarize(self, *, conversation_id: str) -> None:
        """Fire-and-forget rolling-summary refresh.

        No-op when no summarizer was injected. The summarizer itself
        decides whether the tail is stale — this just kicks off the
        check.
        """
        if self._summarizer is None:
            return
        self._summarizer.schedule(conversation_id=conversation_id)

    async def _recall_memory(
        self,
        *,
        project_id: str | None,
        query: str,
    ) -> list[dict[str, Any]]:
        """Top-K project memory snippets via embedding-based recall.

        Async because the embedding call is async; the writer is fired
        in a background task so this read is the only blocking memory
        op on the request path.
        """
        if not project_id or self._recall_k == 0 or not query.strip():
            return []
        hits = await self._memory_reader.recall(
            project_id=project_id, query=query, k=self._recall_k,
        )
        return [
            {
                "vector_id": h.vector_id,
                "text": h.text,
                "score": round(h.score, 6),
                "metadata": dict(h.metadata),
                "source_message_id": h.source_message_id,
            }
            for h in hits
        ]

    def _schedule_memory_write(
        self,
        *,
        project_id: str | None,
        message_id: str,
        text: str,
        role: str,
    ) -> None:
        """Fire-and-forget chunk-and-embed for an assistant turn.

        Wrapped in a closure that swallows exceptions so a failed
        write never surfaces as a chat-turn error. Returns immediately
        — the caller doesn't await.
        """
        if not project_id or not text.strip():
            return
        import asyncio as _asyncio

        async def _run() -> None:
            try:
                await self._memory_writer.write_message(
                    project_id=project_id,
                    text=text,
                    source_message_id=message_id,
                    role=role,
                )
            except Exception as exc:  # noqa: BLE001 — best-effort
                _logger.warning(
                    "background memory write failed: project=%s message=%s err=%s",
                    project_id, message_id, exc,
                )

        try:
            _asyncio.create_task(_run())
        except RuntimeError:
            # No running loop (e.g. invoked from sync context); skip.
            _logger.debug("no event loop; skipping background memory write")

    # ------------------------------------------------------------------
    # Envelope construction
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        *,
        prompt: str,
        history: list[dict[str, Any]],
        preferences: dict[str, Any],
        project: ConsumerProject | None,
        recalled_memory: list[dict[str, Any]],
        user_facts: list[dict[str, Any]],
        attached_files: list[dict[str, Any]],
        mcp_tool_results: list[dict[str, Any]],
        thread_id: str,
        mode: str,
        web_search: bool,
        run_budget_usd: float | None,
        extra_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Construct the AgentInvocationRequest envelope.

        Same shape as the eval-path envelope; the only difference is the
        ``metadata`` block carries product-mode context the agent can
        opt into reading. Agents that don't read it fall back to their
        own defaults.
        """
        metadata: dict[str, Any] = {
            "user_preferences": preferences,
            "recalled_memory": recalled_memory,
            "user_facts": user_facts,
            "attached_files": attached_files,
            "mcp_tool_results": mcp_tool_results,
        }
        if project is not None:
            metadata["project_context"] = {
                "project_id": project.project_id,
                "name": project.name,
                "custom_instructions": project.custom_instructions or "",
            }
        if run_budget_usd is not None:
            metadata["run_budget_usd"] = float(run_budget_usd)
        if extra_metadata:
            metadata.update(extra_metadata)
        return {
            "prompt": prompt,
            "history": list(history),
            "mode": mode,
            "web_search": web_search,
            "turn_id": thread_id,
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _next_turn_idx(self, session, conversation_id: str) -> int:
        from sqlalchemy import func

        result = session.execute(
            select(func.coalesce(func.max(ConsumerMessage.turn_idx), -1))
            .where(ConsumerMessage.conversation_id == conversation_id)
        ).scalar_one()
        return int(result) + 1

    def _persist_user_turn(
        self,
        session,
        *,
        conversation: ConsumerConversation,
        prompt: str,
    ) -> ConsumerMessage:
        idx = self._next_turn_idx(session, conversation.conversation_id)
        msg = ConsumerMessage(
            conversation_id=conversation.conversation_id,
            turn_idx=idx,
            role="user",
            content=prompt,
        )
        session.add(msg)
        session.flush()
        return msg

    def _persist_assistant_turn(
        self,
        session,
        *,
        conversation: ConsumerConversation,
        content: str,
        citations: list[dict[str, Any]],
        tool_calls: list[dict[str, Any]],
        served_by: ServingCandidate,
        latency_ms: int,
        cost_usd: float,
        extra_metadata: dict[str, Any] | None = None,
    ) -> ConsumerMessage:
        idx = self._next_turn_idx(session, conversation.conversation_id)
        msg = ConsumerMessage(
            conversation_id=conversation.conversation_id,
            turn_idx=idx,
            role="assistant",
            content=content,
            citations_json=list(citations),
            tool_calls_json=list(tool_calls),
            served_by_deployment_id=served_by.deployment_id,
            served_by_release_id=served_by.serving_release_id,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            metadata_json=dict(extra_metadata or {}),
        )
        session.add(msg)
        conversation.last_message_at = datetime.utcnow()
        session.flush()
        return msg

    async def _persist_safety_refusal(
        self,
        *,
        user_id: str,
        user_prompt: str,
        conversation_id: str | None,
        project_id: str | None,
        stage: str,
        refusal_text: str,
        safety_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Short-circuit path when a guard denies. Returns invoke()-shape dict."""
        with self._db.sessionmaker() as session:
            user = self._load_user(session, user_id)
            project = self._load_project(session, project_id)
            family_id = (project.default_family_id if project else "general_chat")
            convo = self._load_or_create_conversation(
                session,
                user_id=user.user_id,
                conversation_id=conversation_id,
                project_id=project.project_id if project else None,
                family_id=family_id,
            )
            self._persist_user_turn(session, conversation=convo, prompt=user_prompt)
            idx = self._next_turn_idx(session, convo.conversation_id)
            msg = ConsumerMessage(
                conversation_id=convo.conversation_id,
                turn_idx=idx,
                role="assistant",
                content=refusal_text,
                metadata_json={"safety_verdict": dict(safety_metadata),
                                "refused_at": stage},
            )
            session.add(msg)
            convo.last_message_at = datetime.utcnow()
            session.flush()
            session.commit()
            convo_id = convo.conversation_id
            message_id = msg.id
        return {
            "conversation_id": convo_id,
            "message_id": message_id,
            "response": {
                "task_id": None,
                "family_id": "general_chat",
                "status": "refused",
                "output": {"answer": refusal_text},
                "citations": [],
                "metadata": {
                    "orchestrator": {
                        "kind": "product",
                        "refused_at": stage,
                        "conversation_id": convo_id,
                    },
                    "safety_verdict": dict(safety_metadata),
                },
            },
        }

    # ------------------------------------------------------------------
    # Public surface — invoke + astream
    # ------------------------------------------------------------------

    async def invoke(
        self,
        *,
        user_id: str,
        prompt: str,
        conversation_id: str | None = None,
        project_id: str | None = None,
        attachment_ids: list[str] | None = None,
        mode: str = "instant",
        web_search: bool = False,
        run_budget_usd: float | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Unary product invocation.

        Returns ``{conversation_id, message_id, response}`` where
        ``response`` is the agent's :class:`AgentInvocationResponse`
        body verbatim, with an ``orchestrator`` block stamped into
        ``metadata``.
        """
        # pre_input safety. A deny short-circuits the turn before we
        # call the miner. Persist the user turn (with redacted text if
        # the guard suggested it) so the audit trail is complete, then
        # persist a canned refusal as the assistant reply.
        pre_safety = await self._safety_pre_input(
            prompt=prompt, user_id=user_id,
            conversation_id=conversation_id, project_id=project_id,
        )
        if not pre_safety.allow:
            return await self._persist_safety_refusal(
                user_id=user_id,
                # The user's row records what they sent (may be redacted
                # if a PII guard fired before the deny). The refusal we
                # send back is `pre_safety.text`.
                user_prompt=prompt,
                conversation_id=conversation_id,
                project_id=project_id,
                stage="pre_input",
                refusal_text=pre_safety.text,
                safety_metadata=pre_safety.metadata,
            )
        prompt = pre_safety.text  # may have been redacted

        # Load context + persist the user turn.
        attached_files: list[dict[str, Any]] = []
        with self._db.sessionmaker() as session:
            user = self._load_user(session, user_id)
            project = self._load_project(session, project_id)
            family_id = (project.default_family_id if project else "general_chat")
            convo = self._load_or_create_conversation(
                session,
                user_id=user.user_id,
                conversation_id=conversation_id,
                project_id=project.project_id if project else None,
                family_id=family_id,
            )
            history = self._load_history(session, convo.conversation_id)
            preferences = self._load_preferences(
                session, user_id=user.user_id, project_id=project.project_id if project else None,
            )
            attached_files = self._load_attachments(
                session, user_id=user.user_id, attachment_ids=list(attachment_ids or []),
            )
            self._persist_user_turn(session, conversation=convo, prompt=prompt)
            session.commit()
            convo_snapshot = (convo.conversation_id, convo.family_id)
            project_id_resolved = project.project_id if project else None

        # Recall happens outside the DB session because it issues an
        # async embedding call. Failure is swallowed — recall must not
        # block the chat turn.
        recalled = await self._recall_memory(
            project_id=project_id_resolved, query=prompt,
        )
        user_facts = await self._recall_user_facts(
            user_id=user_id, query=prompt,
        )
        # Orchestrator-side MCP dispatch.
        mcp_results = await self._run_mcp_dispatch(
            user_id=user_id, prompt=prompt, history=history,
            conversation_id=convo_snapshot[0],
        )

        # Pick a serving deployment + call it.
        try:
            served_by = self._picker.pick(family_id=convo_snapshot[1])
        except NoEligibleServingDeploymentError as exc:
            raise ProductOrchestratorError(str(exc)) from exc

        thread_id = uuid4().hex
        payload = self._build_payload(
            prompt=prompt,
            history=history,
            preferences=preferences,
            project=project,
            recalled_memory=recalled,
            user_facts=user_facts,
            attached_files=attached_files,
            mcp_tool_results=mcp_results,
            thread_id=thread_id,
            mode=mode,
            web_search=web_search,
            run_budget_usd=run_budget_usd,
            extra_metadata=extra_metadata,
        )
        url = (
            f"{self._owner_api_url}/runtime/serving/{served_by.deployment_id}"
            "/v1/agent/infer"
        )
        started_ms = _now_ms()
        async with httpx.AsyncClient(**self._client_kwargs()) as client:
            resp = await client.post(url, json=payload, headers=self._headers())
            resp.raise_for_status()
            agent_response = resp.json()
        latency_ms = _now_ms() - started_ms

        # Persist the assistant turn.
        output = agent_response.get("output") or {}
        text = ""
        if isinstance(output, dict):
            for key in ("answer", "response", "content", "text"):
                value = output.get(key)
                if isinstance(value, str) and value:
                    text = value
                    break
        citations = list(agent_response.get("citations") or [])
        meta = agent_response.get("metadata") or {}

        # post_output safety. Deny replaces the persisted + returned
        # content with the canned refusal; redactions quietly rewrite
        # the assistant text.
        post_safety = await self._safety_post_output(
            content=text, user_id=user_id,
            conversation_id=convo_snapshot[0], project_id=project_id_resolved,
        )
        text = post_safety.text
        # Update the agent_response output to reflect any redaction so
        # the API caller sees the same text that was persisted.
        if isinstance(output, dict):
            for key in ("answer", "response", "content", "text"):
                if key in output:
                    output[key] = text
                    break
            else:
                output["answer"] = text
            agent_response["output"] = output
        executed = meta.get("executed_tool_calls") if isinstance(meta, dict) else None
        tool_calls = list(executed) if isinstance(executed, list) else []
        cost = 0.0
        if isinstance(meta, dict):
            cost = float(meta.get("proxy_cost_usd") or 0.0) + float(meta.get("proxy_tool_cost_usd") or 0.0)

        with self._db.sessionmaker() as session:
            convo = session.get(ConsumerConversation, convo_snapshot[0])
            msg = self._persist_assistant_turn(
                session,
                conversation=convo,
                content=text,
                citations=[{"url": c} if isinstance(c, str) else c for c in citations],
                tool_calls=tool_calls,
                served_by=served_by,
                latency_ms=latency_ms,
                cost_usd=cost,
                extra_metadata={"safety_verdict": post_safety.metadata}
                if not post_safety.allow or post_safety.verdicts
                else None,
            )
            session.commit()
            message_id = msg.id

        # Background: chunk + embed + persist the assistant turn into
        # project memory. Fire-and-forget so DB latency for embedding
        # writes never blocks the response. Caller's prompt is
        # recorded too so questions and answers are both recallable.
        self._schedule_memory_write(
            project_id=project_id_resolved,
            message_id=message_id,
            text=text,
            role="assistant",
        )
        # Extract stable user facts from the user's prompt (if a
        # UserMemoryWriter was injected) and refresh the rolling
        # summary when the verbatim tail has grown stale.
        self._schedule_user_memory_write(
            user_id=user_id,
            prompt=prompt,
            conversation_id=convo_snapshot[0],
            source_message_id=message_id,
        )
        self._schedule_summarize(conversation_id=convo_snapshot[0])

        # Stamp orchestrator audit block onto the response.
        response_meta = agent_response.get("metadata")
        if not isinstance(response_meta, dict):
            response_meta = {}
        response_meta.setdefault("orchestrator", {
            "kind": "product",
            "deployment_id": served_by.deployment_id,
            "serving_release_id": served_by.serving_release_id,
            "miner_hotkey": served_by.miner_hotkey,
            "runtime_kind": served_by.runtime_kind,
            "thread_id": thread_id,
            "conversation_id": convo_snapshot[0],
        })
        agent_response["metadata"] = response_meta

        return {
            "conversation_id": convo_snapshot[0],
            "message_id": message_id,
            "response": agent_response,
        }

    async def astream(
        self,
        *,
        user_id: str,
        prompt: str,
        conversation_id: str | None = None,
        project_id: str | None = None,
        attachment_ids: list[str] | None = None,
        mode: str = "instant",
        web_search: bool = False,
        run_budget_usd: float | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """NDJSON streaming product invocation.

        Yields events the consumer can render directly:
          - one ``conversation`` event up front carrying the
            ``conversation_id`` so the client can echo it on follow-up
          - ``delta`` / ``tool_call`` / ``tool_result`` / ``citation`` /
            ``checkpoint`` passthrough from the serving deployment
          - ``done`` terminal event with the orchestrator audit block

        ``trace`` events are dropped — they've already been teed to
        eiretes by the owner-api runtime proxy.
        """
        # pre_input safety. On deny we persist the refusal synchronously
        # then yield a single ``done`` frame so the consumer surfaces
        # the refusal cleanly.
        pre_safety = await self._safety_pre_input(
            prompt=prompt, user_id=user_id,
            conversation_id=conversation_id, project_id=project_id,
        )
        if not pre_safety.allow:
            refused = await self._persist_safety_refusal(
                user_id=user_id,
                user_prompt=prompt,
                conversation_id=conversation_id,
                project_id=project_id,
                stage="pre_input",
                refusal_text=pre_safety.text,
                safety_metadata=pre_safety.metadata,
            )
            yield {"event": "conversation",
                   "conversation_id": refused["conversation_id"]}
            yield {
                "event": "done",
                "status": "refused",
                "output": {"answer": pre_safety.text},
                "citations": [],
                "metadata": {
                    "conversation_id": refused["conversation_id"],
                    "safety_verdict": dict(pre_safety.metadata),
                    "orchestrator": {
                        "kind": "product",
                        "refused_at": "pre_input",
                        "conversation_id": refused["conversation_id"],
                    },
                },
            }
            return
        prompt = pre_safety.text

        # Load context + persist user turn (same as invoke).
        attached_files: list[dict[str, Any]] = []
        with self._db.sessionmaker() as session:
            user = self._load_user(session, user_id)
            project = self._load_project(session, project_id)
            family_id = (project.default_family_id if project else "general_chat")
            convo = self._load_or_create_conversation(
                session,
                user_id=user.user_id,
                conversation_id=conversation_id,
                project_id=project.project_id if project else None,
                family_id=family_id,
            )
            history = self._load_history(session, convo.conversation_id)
            preferences = self._load_preferences(
                session, user_id=user.user_id, project_id=project.project_id if project else None,
            )
            attached_files = self._load_attachments(
                session, user_id=user.user_id, attachment_ids=list(attachment_ids or []),
            )
            self._persist_user_turn(session, conversation=convo, prompt=prompt)
            session.commit()
            convo_id = convo.conversation_id
            convo_family = convo.family_id
            project_id_resolved = project.project_id if project else None
            project_snapshot = (
                {
                    "project_id": project.project_id,
                    "name": project.name,
                    "custom_instructions": project.custom_instructions or "",
                }
                if project is not None
                else None
            )

        # Embedding-based recall outside the DB session.
        recalled = await self._recall_memory(
            project_id=project_id_resolved, query=prompt,
        )
        user_facts = await self._recall_user_facts(
            user_id=user_id, query=prompt,
        )
        mcp_results = await self._run_mcp_dispatch(
            user_id=user_id, prompt=prompt, history=history,
            conversation_id=convo_id,
        )

        yield {"event": "conversation", "conversation_id": convo_id}

        try:
            served_by = self._picker.pick(family_id=convo_family)
        except NoEligibleServingDeploymentError as exc:
            yield {
                "event": "done",
                "status": "failed",
                "error": f"orchestrator: {exc}",
                "metadata": {"conversation_id": convo_id},
            }
            return

        thread_id = uuid4().hex
        payload = self._build_payload(
            prompt=prompt,
            history=history,
            preferences=preferences,
            project=project if project_snapshot else None,
            recalled_memory=recalled,
            user_facts=user_facts,
            attached_files=attached_files,
            mcp_tool_results=mcp_results,
            thread_id=thread_id,
            mode=mode,
            web_search=web_search,
            run_budget_usd=run_budget_usd,
            extra_metadata=extra_metadata,
        )
        url = (
            f"{self._owner_api_url}/runtime/serving/{served_by.deployment_id}"
            "/v1/agent/infer/stream"
        )
        started_ms = _now_ms()
        accumulated_text: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        citations: list[dict[str, Any]] = []
        cost = 0.0

        async with httpx.AsyncClient(**self._client_kwargs()) as client:
            async with client.stream(
                "POST", url, json=payload, headers=self._headers()
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        frame = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(frame, dict):
                        continue
                    event = frame.get("event")
                    if event in _DROP_EVENTS:
                        continue
                    if event == "delta":
                        text = frame.get("text")
                        if isinstance(text, str):
                            accumulated_text.append(text)
                    elif event == "tool_call":
                        tc = frame.get("tool_call")
                        if isinstance(tc, dict):
                            tool_calls.append(tc)
                    elif event == "citation":
                        cit = frame.get("citation")
                        if isinstance(cit, dict):
                            citations.append(cit)
                    elif event == "done":
                        meta = frame.get("metadata")
                        if isinstance(meta, dict):
                            cost = (
                                float(meta.get("proxy_cost_usd") or 0.0)
                                + float(meta.get("proxy_tool_cost_usd") or 0.0)
                            )
                            executed = meta.get("executed_tool_calls")
                            if isinstance(executed, list):
                                tool_calls = list(executed)
                            output = frame.get("output")
                            if not accumulated_text and isinstance(output, dict):
                                # Non-streaming agent: full answer in
                                # the done frame's output.
                                for key in ("answer", "response", "content", "text"):
                                    value = output.get(key)
                                    if isinstance(value, str) and value:
                                        accumulated_text.append(value)
                                        break
                            meta.setdefault("orchestrator", {
                                "kind": "product",
                                "deployment_id": served_by.deployment_id,
                                "serving_release_id": served_by.serving_release_id,
                                "miner_hotkey": served_by.miner_hotkey,
                                "runtime_kind": served_by.runtime_kind,
                                "thread_id": thread_id,
                                "conversation_id": convo_id,
                            })
                            frame["metadata"] = meta
                    yield frame

        latency_ms = _now_ms() - started_ms
        # Persist after the stream closes — the consumer has already
        # received the full response, so DB latency doesn't slow them.
        assistant_text = "".join(accumulated_text)

        # post_output safety. The streamed text already reached the
        # consumer verbatim, so deny doesn't undo what they saw — but
        # we DO refuse to persist sensitive content. Redaction is
        # applied to the persisted record only.
        post_safety = await self._safety_post_output(
            content=assistant_text, user_id=user_id,
            conversation_id=convo_id, project_id=project_id_resolved,
        )
        persisted_text = post_safety.text

        message_id: str | None = None
        with self._db.sessionmaker() as session:
            convo = session.get(ConsumerConversation, convo_id)
            msg = self._persist_assistant_turn(
                session,
                conversation=convo,
                content=persisted_text,
                citations=citations,
                tool_calls=tool_calls,
                served_by=served_by,
                latency_ms=latency_ms,
                cost_usd=cost,
                extra_metadata={"safety_verdict": post_safety.metadata}
                if not post_safety.allow or post_safety.verdicts
                else None,
            )
            session.commit()
            message_id = msg.id

        # Background memory write — same fire-and-forget pattern as
        # ``invoke``. Recall on the next turn will see this content.
        if message_id is not None:
            self._schedule_memory_write(
                project_id=project_id_resolved,
                message_id=message_id,
                text=assistant_text,
                role="assistant",
            )
        # Same fire-and-forget pattern as invoke().
        self._schedule_user_memory_write(
            user_id=user_id,
            prompt=prompt,
            conversation_id=convo_id,
            source_message_id=message_id,
        )
        self._schedule_summarize(conversation_id=convo_id)


def _now_ms() -> int:
    import time
    return int(time.monotonic() * 1000)
