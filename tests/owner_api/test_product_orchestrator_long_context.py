"""End-to-end: long-context summary head + user_facts metadata.

Verifies:
  * Pre-existing rolling summary lands as a system-role message at the
    head of ``request.history`` and the verbatim tail follows.
  * ``metadata.user_facts`` is populated from ``ConsumerUserMemory``.
  * After the turn, the orchestrator schedules the user-memory writer
    + summarizer (best-effort fire-and-forget).
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any
from uuid import uuid4

import httpx

from shared.common.database import Database
from shared.common.models import (
    ConsumerConversation,
    ConsumerMessage,
    ConsumerProject,
    ConsumerUser,
    ConsumerUserMemory,
    ManagedDeployment,
    ManagedMinerSubmission,
    ServingDeployment,
    ServingRelease,
)
from orchestration.orchestrator.conversation_summarizer import (
    ConversationSummarizer,
)
from orchestration.orchestrator.embedding_client import StubEmbeddingClient
from orchestration.orchestrator.product_orchestrator import ProductOrchestrator
from orchestration.orchestrator.project_memory import encode_embedding
from orchestration.orchestrator.serving_picker import ServingPicker
from orchestration.orchestrator.user_memory import UserMemoryWriter


# -- fixtures ---------------------------------------------------------------


def _make_db(tmp_path) -> Database:
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'lc.db'}")
    db.create_all()
    return db


def _seed_serving(session) -> str:
    deployment_id = "serving-lc"
    submission_id = str(uuid4())
    session.add(ManagedMinerSubmission(
        id=submission_id, miner_hotkey="hk", submission_seq=1,
        family_id="general_chat", status="deployed", artifact_id=str(uuid4()),
        manifest_json={"runtime": {"kind": "graph"}},
        archive_sha256="0" * 64, submission_block=0,
    ))
    source_id = str(uuid4())
    session.add(ManagedDeployment(
        id=source_id, submission_id=submission_id, miner_hotkey="hk",
        family_id="general_chat", deployment_revision=str(uuid4()),
        image_ref="img:x", endpoint="http://eval.test:8080",
        status="active", health_status="healthy", placement_status="placed",
    ))
    release_id = str(uuid4())
    session.add(ServingRelease(
        id=release_id, trigger_type="t",
        status="published", published_at=datetime.utcnow(),
    ))
    session.add(ServingDeployment(
        id=deployment_id, release_id=release_id, family_id="general_chat",
        source_deployment_id=source_id, source_submission_id=submission_id,
        miner_hotkey="hk", source_deployment_revision=str(uuid4()),
        endpoint=f"http://serving-{deployment_id}.test:8080",
        status="healthy", health_status="healthy",
        published_at=datetime.utcnow(),
    ))
    session.commit()
    return deployment_id


def _build_orch(
    db: Database, *, captured: list[dict[str, Any]],
    summarizer: ConversationSummarizer | None = None,
    user_memory_writer: UserMemoryWriter | None = None,
) -> ProductOrchestrator:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode("utf-8")) if request.content else {})
        return httpx.Response(200, json={
            "task_id": "x",
            "family_id": "general_chat",
            "status": "completed",
            "output": {"answer": "ack"},
            "citations": [],
            "metadata": {"runtime_kind": "graph", "executed_tool_calls": []},
        })

    return ProductOrchestrator(
        database=db,
        serving_picker=ServingPicker(database=db),
        owner_api_url="http://owner-api.test",
        internal_service_token="t",
        transport=httpx.MockTransport(handler),
        embedding_client=StubEmbeddingClient(),
        summarizer=summarizer,
        user_memory_writer=user_memory_writer,
        # Smaller history limit so a 30-message convo exercises tail
        # truncation predictably.
        history_limit=8,
    )


# -- tests ------------------------------------------------------------------


async def test_envelope_carries_summary_head_and_verbatim_tail(tmp_path):
    """A pre-existing rolling_summary is injected as system msg + tail."""
    db = _make_db(tmp_path)
    user_id = str(uuid4())
    project_id = str(uuid4())
    convo_id = str(uuid4())
    with db.sessionmaker() as session:
        session.add(ConsumerUser(
            user_id=user_id, auth_subject=f"k:{user_id}", display_name="X",
        ))
        session.add(ConsumerProject(
            project_id=project_id, user_id=user_id, name="P",
        ))
        session.add(ConsumerConversation(
            conversation_id=convo_id, user_id=user_id,
            project_id=project_id, family_id="general_chat",
            rolling_summary="Earlier: user discussed Python projects.",
            last_summarized_message_id=None,  # set below
        ))
        # 30 prior messages, alternating roles.
        ids: list[str] = []
        for i in range(30):
            role = "user" if i % 2 == 0 else "assistant"
            msg = ConsumerMessage(
                conversation_id=convo_id, turn_idx=i, role=role,
                content=f"prior-{role}-{i}",
            )
            session.add(msg)
            session.flush()
            ids.append(msg.id)
        # Mark the 21st message (index 20) as the summary boundary →
        # tail starts at message 22 and is 9 messages long.
        convo = session.get(ConsumerConversation, convo_id)
        convo.last_summarized_message_id = ids[20]
        _seed_serving(session)
        session.commit()

    captured: list[dict[str, Any]] = []
    orch = _build_orch(db, captured=captured)
    await orch.invoke(
        user_id=user_id, prompt="continuing the chat",
        conversation_id=convo_id, project_id=project_id,
    )

    history = captured[-1]["history"]
    # First entry is the orchestrator-injected summary.
    assert history[0]["role"] == "system"
    assert "Earlier conversation summary" in history[0]["content"]
    assert "Python projects" in history[0]["content"]
    # The rest are the verbatim tail. `history_limit=8` caps the count.
    tail = history[1:]
    assert len(tail) <= 8
    # All tail messages have turn_idx > 20 (i.e., from prior-*-21 onwards).
    for entry in tail:
        # contents are like "prior-user-25" / "prior-assistant-26"
        assert entry["content"].startswith("prior-")
        idx = int(entry["content"].rsplit("-", 1)[1])
        assert idx > 20


async def test_envelope_falls_back_to_last_n_when_no_summary(tmp_path):
    """Without a rolling_summary, the envelope falls back to last-N slicing."""
    db = _make_db(tmp_path)
    user_id = str(uuid4())
    convo_id = str(uuid4())
    with db.sessionmaker() as session:
        session.add(ConsumerUser(
            user_id=user_id, auth_subject=f"k:{user_id}", display_name="X",
        ))
        session.add(ConsumerConversation(
            conversation_id=convo_id, user_id=user_id,
            family_id="general_chat",
        ))
        for i in range(30):
            session.add(ConsumerMessage(
                conversation_id=convo_id, turn_idx=i, role="user",
                content=f"m{i}",
            ))
        _seed_serving(session)
        session.commit()

    captured: list[dict[str, Any]] = []
    orch = _build_orch(db, captured=captured)
    await orch.invoke(
        user_id=user_id, prompt="hi",
        conversation_id=convo_id,
    )
    history = captured[-1]["history"]
    # No summary → no system head; just the last 8 verbatim.
    roles = {h["role"] for h in history}
    assert "system" not in roles
    assert len(history) == 8


async def test_envelope_carries_user_facts_metadata(tmp_path):
    """A pre-seeded ConsumerUserMemory row surfaces in metadata.user_facts."""
    db = _make_db(tmp_path)
    user_id = str(uuid4())
    embed = StubEmbeddingClient()
    with db.sessionmaker() as session:
        session.add(ConsumerUser(
            user_id=user_id, auth_subject=f"k:{user_id}", display_name="X",
        ))
        # Seed a user-memory row directly (skip the writer path).
        vec = (await embed.aembed(["I work in Python"]))[0]
        session.add(ConsumerUserMemory(
            id=str(uuid4()), user_id=user_id, vector_id="seed-1",
            embedding=encode_embedding(vec), text="I work in Python",
            kind="skill", confidence=0.95,
        ))
        _seed_serving(session)
        session.commit()

    captured: list[dict[str, Any]] = []
    orch = _build_orch(db, captured=captured)
    await orch.invoke(
        user_id=user_id, prompt="what language do I use",
    )
    metadata = captured[-1]["metadata"]
    assert "user_facts" in metadata
    facts = metadata["user_facts"]
    assert len(facts) == 1
    assert facts[0]["text"] == "I work in Python"
    assert facts[0]["kind"] == "skill"


async def test_post_turn_user_memory_writer_fires(tmp_path):
    """When a UserMemoryWriter is injected, the orchestrator schedules it."""
    db = _make_db(tmp_path)
    user_id = str(uuid4())
    with db.sessionmaker() as session:
        session.add(ConsumerUser(
            user_id=user_id, auth_subject=f"k:{user_id}", display_name="X",
        ))
        _seed_serving(session)
        session.commit()

    class _FactExtractor:
        def __init__(self) -> None:
            self.calls = 0

        async def chat_completions(self, payload):
            self.calls += 1
            return {"choices": [{"message": {"content": '[{"text": "I work in Rust"}]'}}]}

    extractor = _FactExtractor()
    writer = UserMemoryWriter(
        database=db,
        embedding_client=StubEmbeddingClient(),
        extractor_llm=extractor,
    )
    captured: list[dict[str, Any]] = []
    orch = _build_orch(db, captured=captured, user_memory_writer=writer)
    await orch.invoke(
        user_id=user_id, prompt="I work in Rust on a side project.",
    )
    # Drain background tasks.
    for _ in range(20):
        await asyncio.sleep(0)
    # Pre-filter hit + extractor called once + row persisted.
    assert extractor.calls == 1
    with db.sessionmaker() as session:
        rows = session.query(ConsumerUserMemory).filter_by(user_id=user_id).all()
        assert any(r.text == "I work in Rust" for r in rows)


async def test_post_turn_summarizer_fires_when_stale(tmp_path):
    """When the verbatim tail exceeds the threshold, summarizer runs."""
    db = _make_db(tmp_path)
    user_id = str(uuid4())
    convo_id = str(uuid4())
    with db.sessionmaker() as session:
        session.add(ConsumerUser(
            user_id=user_id, auth_subject=f"k:{user_id}", display_name="X",
        ))
        session.add(ConsumerConversation(
            conversation_id=convo_id, user_id=user_id,
            family_id="general_chat",
        ))
        for i in range(30):
            session.add(ConsumerMessage(
                conversation_id=convo_id, turn_idx=i, role="user",
                content=f"m{i}",
            ))
        _seed_serving(session)
        session.commit()

    class _SummaryLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def chat_completions(self, payload):
            self.calls += 1
            return {"choices": [{"message": {"content": "Summary text"}}]}

    summary_llm = _SummaryLLM()
    summarizer = ConversationSummarizer(
        database=db, llm=summary_llm,
        stale_threshold_messages=10, keep_recent_messages=4,
    )
    captured: list[dict[str, Any]] = []
    orch = _build_orch(db, captured=captured, summarizer=summarizer)
    await orch.invoke(
        user_id=user_id, prompt="hi",
        conversation_id=convo_id,
    )
    # Drain background tasks.
    for _ in range(20):
        await asyncio.sleep(0)
    assert summary_llm.calls == 1
    with db.sessionmaker() as session:
        convo = session.get(ConsumerConversation, convo_id)
        assert convo.rolling_summary == "Summary text"
        assert convo.last_summarized_message_id is not None
