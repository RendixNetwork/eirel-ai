"""Tests for project memory: chunker, writer, reader, orchestrator wiring."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any
from uuid import uuid4

import httpx
import pytest

from shared.common.database import Database
from shared.common.models import (
    ConsumerProject,
    ConsumerProjectMemory,
    ConsumerUser,
    ManagedDeployment,
    ManagedMinerSubmission,
    ServingDeployment,
    ServingRelease,
)
from orchestration.orchestrator.embedding_client import (
    DEFAULT_DIMENSION,
    EmbeddingClient,
    ProxyEmbeddingClient,
    StubEmbeddingClient,
)
from orchestration.orchestrator.product_orchestrator import ProductOrchestrator
from orchestration.orchestrator.project_memory import (
    DEFAULT_CHUNK_SIZE,
    MAX_CHUNKS_PER_MESSAGE,
    MemoryHit,
    ProjectMemoryReader,
    ProjectMemoryWriter,
    chunk_text,
    decode_embedding,
    encode_embedding,
)
from orchestration.orchestrator.serving_picker import ServingPicker


# -- chunker ---------------------------------------------------------------


def test_chunker_returns_short_text_unchanged():
    out = chunk_text("Hello world.", chunk_size=200)
    assert out == ["Hello world."]


def test_chunker_empty_returns_empty():
    assert chunk_text("") == []
    assert chunk_text("   \n\n   ") == []


def test_chunker_paragraph_split():
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    out = chunk_text(text, chunk_size=20, overlap=0)
    assert "Paragraph one." in out
    assert any("two" in c for c in out)
    assert any("three" in c for c in out)


def test_chunker_long_paragraph_sentence_split():
    sentences = [f"Sentence {i}." for i in range(20)]
    text = " ".join(sentences)
    out = chunk_text(text, chunk_size=50, overlap=10)
    assert len(out) > 1
    for chunk in out:
        assert len(chunk) <= 100  # rough upper bound including overlap


def test_chunker_respects_max_chunks():
    long_text = ("para. " * 5000)
    out = chunk_text(long_text, chunk_size=80, overlap=0, max_chunks=4)
    assert len(out) <= 4


def test_chunker_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("x", chunk_size=10, overlap=10)
    with pytest.raises(ValueError):
        chunk_text("x", chunk_size=0)


# -- embedding (de)serialization -------------------------------------------


def test_embedding_encode_decode_roundtrip():
    vec = [0.1, -0.2, 0.3, 0.4, -0.5]
    blob = encode_embedding(vec)
    decoded = decode_embedding(blob)
    for orig, got in zip(vec, decoded):
        assert abs(orig - got) < 1e-6


def test_decode_embedding_rejects_bad_blob_length():
    with pytest.raises(ValueError):
        decode_embedding(b"\x00\x00\x00")  # not multiple of 4


# -- StubEmbeddingClient ---------------------------------------------------


async def test_stub_embedding_is_deterministic_and_unit_length():
    stub = StubEmbeddingClient()
    a = (await stub.aembed(["hello world"]))[0]
    b = (await stub.aembed(["hello world"]))[0]
    assert a == b
    norm = sum(x * x for x in a) ** 0.5
    assert abs(norm - 1.0) < 1e-6


async def test_stub_embedding_overlapping_tokens_have_higher_cosine():
    """Two strings sharing tokens should be closer than disjoint strings."""
    from orchestration.orchestrator.project_memory import _cosine

    stub = StubEmbeddingClient()
    base, similar, distinct = await stub.aembed([
        "the user works in python and golang",
        "what language does the user write",
        "completely unrelated weather forecast for tomorrow",
    ])
    assert _cosine(base, similar) > _cosine(base, distinct)


async def test_stub_embedding_empty_text_returns_zeroish_vector():
    stub = StubEmbeddingClient()
    [vec] = await stub.aembed([""])
    assert all(v == 0 for v in vec)


# -- ProxyEmbeddingClient (transport-mocked) -------------------------------


async def test_proxy_embedding_calls_openai_compatible_endpoint():
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        captured.append({"url": str(request.url), "body": body, "auth": request.headers.get("Authorization")})
        # OpenAI shape: {data: [{embedding: [...], index: i}, ...]}
        return httpx.Response(200, json={
            "data": [
                {"embedding": [0.1] * 8, "index": i}
                for i in range(len(body.get("input", [])))
            ],
        })

    client = ProxyEmbeddingClient(
        base_url="http://embed.test/v1", api_key="tok",
        model="text-embedding-3-small", dimension=8,
        transport=httpx.MockTransport(handler),
    )
    try:
        result = await client.aembed(["a", "b"])
    finally:
        await client.aclose()
    assert len(result) == 2
    assert all(len(v) == 8 for v in result)
    assert captured[0]["body"]["model"] == "text-embedding-3-small"
    assert captured[0]["auth"] == "Bearer tok"


async def test_proxy_embedding_raises_without_base_url():
    client = ProxyEmbeddingClient(base_url="", api_key="tok")
    with pytest.raises(RuntimeError, match="EIREL_EMBEDDING_BASE_URL"):
        await client.aembed(["x"])


# -- Writer / Reader -------------------------------------------------------


def _make_db(tmp_path) -> Database:
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'pm.db'}")
    db.create_all()
    return db


def _seed_user_and_project(session, *, user_id: str = "user-1", project_id: str | None = None) -> tuple[str, str]:
    project_id = project_id or str(uuid4())
    session.add(ConsumerUser(
        user_id=user_id, auth_subject=f"api-key:{user_id}", display_name="X",
    ))
    session.add(ConsumerProject(
        project_id=project_id, user_id=user_id, name="Test",
    ))
    session.commit()
    return user_id, project_id


async def test_writer_persists_chunks_with_embeddings(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _user_id, project_id = _seed_user_and_project(session)

    writer = ProjectMemoryWriter(database=db, embedding_client=StubEmbeddingClient())
    n = await writer.write_message(
        project_id=project_id,
        text="The user prefers Python over JavaScript for backend work.",
        source_message_id="msg-1",
        role="assistant",
    )
    assert n >= 1
    with db.sessionmaker() as session:
        rows = list(session.scalars(
            ConsumerProjectMemory.__table__.select().where(
                ConsumerProjectMemory.project_id == project_id,
            )
        ))
        # Workaround: use ORM .query for cleanliness.
        rows = session.query(ConsumerProjectMemory).filter_by(project_id=project_id).all()
        assert len(rows) == n
        for row in rows:
            assert row.source_message_id == "msg-1"
            assert row.metadata_json["role"] == "assistant"
            decoded = decode_embedding(row.embedding)
            assert len(decoded) > 0


async def test_writer_idempotent_on_rewrite_same_message(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _, project_id = _seed_user_and_project(session)

    writer = ProjectMemoryWriter(database=db, embedding_client=StubEmbeddingClient())
    await writer.write_message(
        project_id=project_id, text="initial content.", source_message_id="msg-x",
    )
    await writer.write_message(
        project_id=project_id, text="replacement content.", source_message_id="msg-x",
    )
    with db.sessionmaker() as session:
        rows = session.query(ConsumerProjectMemory).filter_by(project_id=project_id).all()
        # Old rows replaced, not duplicated.
        assert all("replacement" in r.text for r in rows)


async def test_writer_empty_text_no_op(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _, project_id = _seed_user_and_project(session)
    writer = ProjectMemoryWriter(database=db, embedding_client=StubEmbeddingClient())
    n = await writer.write_message(project_id=project_id, text="   ", source_message_id="empty")
    assert n == 0


async def test_writer_failed_embedding_swallows_and_returns_zero(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _, project_id = _seed_user_and_project(session)

    class _BoomClient(EmbeddingClient):
        @property
        def dimension(self) -> int:
            return 8
        async def aembed(self, texts):
            raise RuntimeError("upstream down")

    writer = ProjectMemoryWriter(database=db, embedding_client=_BoomClient())
    n = await writer.write_message(
        project_id=project_id, text="some content", source_message_id="m",
    )
    assert n == 0
    with db.sessionmaker() as session:
        assert session.query(ConsumerProjectMemory).count() == 0


async def test_reader_top_k_ranking(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _, project_id = _seed_user_and_project(session)

    stub = StubEmbeddingClient()
    writer = ProjectMemoryWriter(database=db, embedding_client=stub)
    await writer.write_message(
        project_id=project_id,
        text="The user prefers python and django for backend.",
        source_message_id="msg-py",
    )
    await writer.write_message(
        project_id=project_id,
        text="Their favorite vacation spot is the mountains in Iceland.",
        source_message_id="msg-trip",
    )

    reader = ProjectMemoryReader(database=db, embedding_client=stub)
    hits = await reader.recall(
        project_id=project_id, query="what programming language does the user use", k=2,
    )
    assert len(hits) >= 1
    # Top hit should be the python row, not the vacation one.
    assert "python" in hits[0].text.lower()


async def test_reader_cross_project_isolation(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_user_and_project(session, user_id="u1", project_id="proj-a")
        # Second project under a different user.
        session.add(ConsumerUser(user_id="u2", auth_subject="x"))
        session.add(ConsumerProject(project_id="proj-b", user_id="u2", name="B"))
        session.commit()

    writer = ProjectMemoryWriter(database=db, embedding_client=StubEmbeddingClient())
    await writer.write_message(
        project_id="proj-a", text="secret a content", source_message_id="a",
    )
    await writer.write_message(
        project_id="proj-b", text="secret b content", source_message_id="b",
    )
    reader = ProjectMemoryReader(database=db, embedding_client=StubEmbeddingClient())
    hits_a = await reader.recall(project_id="proj-a", query="secret", k=10)
    hits_b = await reader.recall(project_id="proj-b", query="secret", k=10)
    assert all("a content" in h.text for h in hits_a)
    assert all("b content" in h.text for h in hits_b)


async def test_reader_empty_project_returns_empty(tmp_path):
    db = _make_db(tmp_path)
    reader = ProjectMemoryReader(database=db, embedding_client=StubEmbeddingClient())
    hits = await reader.recall(project_id="ghost", query="anything", k=5)
    assert hits == []


async def test_reader_swallows_failed_embedding(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _, project_id = _seed_user_and_project(session)

    class _BoomClient(EmbeddingClient):
        @property
        def dimension(self) -> int:
            return 8
        async def aembed(self, texts):
            raise RuntimeError("down")

    reader = ProjectMemoryReader(database=db, embedding_client=_BoomClient())
    hits = await reader.recall(project_id=project_id, query="x", k=5)
    assert hits == []


# -- Orchestrator end-to-end ----------------------------------------------


def _seed_serving_deployment(session, *, deployment_id: str = "serving-mem") -> str:
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


def _build_orchestrator(
    db: Database, *, captured: list[dict[str, Any]],
    embedding_client: EmbeddingClient,
    canned_answer: str = "noted",
) -> ProductOrchestrator:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        captured.append(body)
        return httpx.Response(200, json={
            "task_id": body.get("turn_id"),
            "family_id": "general_chat",
            "status": "completed",
            "output": {"answer": canned_answer},
            "citations": [],
            "metadata": {"runtime_kind": "graph", "executed_tool_calls": []},
        })

    return ProductOrchestrator(
        database=db,
        serving_picker=ServingPicker(database=db),
        owner_api_url="http://owner-api.test",
        internal_service_token="t",
        transport=httpx.MockTransport(handler),
        embedding_client=embedding_client,
    )


async def test_orchestrator_writes_and_recalls_assistant_turns(tmp_path):
    """End-to-end: turn 1 establishes a fact; turn 2 recall surfaces it."""
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        user_id, project_id = _seed_user_and_project(session)
        _seed_serving_deployment(session)

    captured: list[dict[str, Any]] = []
    stub = StubEmbeddingClient()
    orch = _build_orchestrator(
        db, captured=captured, embedding_client=stub,
        canned_answer="The user writes Python code daily.",
    )

    # Turn 1 — assistant says "The user writes Python code daily." which
    # is the content the writer will embed.
    first = await orch.invoke(
        user_id=user_id, prompt="what do you remember about me",
        project_id=project_id,
    )
    convo_id = first["conversation_id"]

    # Background memory write is fire-and-forget — give the loop a tick
    # so create_task() runs before turn 2.
    for _ in range(5):
        await asyncio.sleep(0)

    # Confirm the row landed.
    with db.sessionmaker() as session:
        rows = session.query(ConsumerProjectMemory).filter_by(project_id=project_id).all()
        assert rows, "expected at least one memory row after turn 1"

    # Turn 2 — query about programming language; recall should surface
    # the Python answer.
    captured.clear()
    await orch.invoke(
        user_id=user_id, prompt="what programming language does the user use",
        conversation_id=convo_id, project_id=project_id,
    )
    metadata = captured[-1]["metadata"]
    recalled_texts = " ".join(s["text"] for s in metadata["recalled_memory"])
    assert "Python" in recalled_texts


async def test_orchestrator_recalled_memory_empty_without_project(tmp_path):
    """Conversations without a project_id never get recall snippets."""
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        session.add(ConsumerUser(user_id="u", auth_subject="x"))
        session.commit()
        _seed_serving_deployment(session)

    captured: list[dict[str, Any]] = []
    orch = _build_orchestrator(
        db, captured=captured, embedding_client=StubEmbeddingClient(),
    )
    await orch.invoke(user_id="u", prompt="hi")
    assert captured[-1]["metadata"]["recalled_memory"] == []


async def test_orchestrator_recall_is_resilient_to_embedding_failure(tmp_path):
    """A broken embedding client must not break the chat turn."""
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        user_id, project_id = _seed_user_and_project(session)
        _seed_serving_deployment(session)

    class _BoomClient(EmbeddingClient):
        @property
        def dimension(self) -> int:
            return 8
        async def aembed(self, texts):
            raise RuntimeError("embedding down")

    captured: list[dict[str, Any]] = []
    orch = _build_orchestrator(
        db, captured=captured, embedding_client=_BoomClient(),
    )
    # Should not raise.
    result = await orch.invoke(
        user_id=user_id, prompt="hi", project_id=project_id,
    )
    assert result["response"]["status"] == "completed"
    assert captured[-1]["metadata"]["recalled_memory"] == []
