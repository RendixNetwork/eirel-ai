"""End-to-end tests for ProductOrchestrator (eval-mode → product-mode split).

Covers:
  * Context loading: history (last N) + preferences (global + project)
    + project memory (top-K) all flow into AgentInvocationRequest.metadata
  * Persistence: user turn + assistant turn land in ConsumerMessage
    with served_by_* audit fields populated
  * Streaming: trace frames dropped, conversation event up front, done
    frame carries orchestrator audit block
  * Idempotent conversation create: passing conversation_id reuses, omit creates
  * Eval/product byte-compat: same envelope shape into both paths
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import uuid4

import httpx
import pytest

from shared.common.database import Database
from shared.common.models import (
    ConsumerConversation,
    ConsumerMessage,
    ConsumerPreference,
    ConsumerProject,
    ConsumerProjectMemory,
    ConsumerUser,
    ManagedDeployment,
    ManagedMinerSubmission,
    ServingDeployment,
    ServingRelease,
)
from orchestration.orchestrator.product_orchestrator import (
    ProductOrchestrator,
    ProductOrchestratorError,
)
from orchestration.orchestrator.serving_picker import ServingPicker


# -- Fixtures ---------------------------------------------------------------


def _make_db(tmp_path) -> Database:
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'po.db'}")
    db.create_all()
    return db


def _seed_serving_deployment(session, *, deployment_id: str = "serving-1", family_id: str = "general_chat") -> str:
    submission_id = str(uuid4())
    session.add(ManagedMinerSubmission(
        id=submission_id, miner_hotkey="hk-winner", submission_seq=1,
        family_id=family_id, status="deployed", artifact_id=str(uuid4()),
        manifest_json={"runtime": {"kind": "graph"}},
        archive_sha256="0" * 64, submission_block=0,
    ))
    source_id = str(uuid4())
    session.add(ManagedDeployment(
        id=source_id, submission_id=submission_id, miner_hotkey="hk-winner",
        family_id=family_id, deployment_revision=str(uuid4()),
        image_ref="img:test", endpoint="http://eval.test:8080",
        status="active", health_status="healthy", placement_status="placed",
    ))
    release_id = str(uuid4())
    session.add(ServingRelease(
        id=release_id, trigger_type="test",
        status="published", published_at=datetime.utcnow(),
    ))
    session.add(ServingDeployment(
        id=deployment_id, release_id=release_id, family_id=family_id,
        source_deployment_id=source_id, source_submission_id=submission_id,
        miner_hotkey="hk-winner", source_deployment_revision=str(uuid4()),
        endpoint=f"http://serving-{deployment_id}.test:8080",
        status="healthy", health_status="healthy",
        published_at=datetime.utcnow(),
    ))
    session.commit()
    return deployment_id


def _seed_user(session, *, user_id: str = "user-1", auth_subject: str = "api-key:test") -> str:
    session.add(ConsumerUser(
        user_id=user_id, auth_subject=auth_subject, display_name="Test",
    ))
    session.commit()
    return user_id


def _build_orchestrator(
    db: Database,
    *,
    captured_payloads: list[dict[str, Any]] | None = None,
    ndjson_lines: list[str] | None = None,
) -> ProductOrchestrator:
    """Build an orchestrator with a mock owner-api transport.

    Captures every outbound request body in ``captured_payloads`` so
    tests can assert the AgentInvocationRequest envelope is built
    correctly. ``ndjson_lines`` drives the streaming response.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        if captured_payloads is not None:
            captured_payloads.append({"url": str(request.url), "body": body})
        if "/v1/agent/infer/stream" in str(request.url):
            data = "\n".join(ndjson_lines or []) + "\n"
            return httpx.Response(
                200, content=data.encode("utf-8"),
                headers={"content-type": "application/x-ndjson"},
            )
        if str(request.url).endswith("/v1/agent/infer"):
            return httpx.Response(200, json={
                "task_id": body.get("turn_id"),
                "family_id": "general_chat",
                "status": "completed",
                "output": {"answer": f"echo:{body.get('prompt')}"},
                "citations": [],
                "metadata": {
                    "runtime_kind": "graph",
                    "proxy_cost_usd": 0.001,
                    "executed_tool_calls": [],
                },
            })
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    return ProductOrchestrator(
        database=db,
        serving_picker=ServingPicker(database=db),
        owner_api_url="http://owner-api.test",
        internal_service_token="t",
        transport=transport,
    )


# -- Unary invoke -----------------------------------------------------------


async def test_invoke_creates_conversation_and_persists_both_turns(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_user(session)
        _seed_serving_deployment(session)

    captured: list[dict[str, Any]] = []
    orch = _build_orchestrator(db, captured_payloads=captured)

    result = await orch.invoke(user_id="user-1", prompt="hello")
    assert result["conversation_id"]
    assert result["message_id"]
    assert result["response"]["status"] == "completed"
    assert result["response"]["output"]["answer"] == "echo:hello"

    # Both user + assistant turns persisted.
    with db.sessionmaker() as session:
        msgs = session.query(ConsumerMessage).order_by(ConsumerMessage.turn_idx).all()
        assert [m.role for m in msgs] == ["user", "assistant"]
        assert msgs[0].content == "hello"
        assert msgs[1].content == "echo:hello"
        assert msgs[1].served_by_deployment_id == "serving-1"
        assert msgs[1].served_by_release_id is not None
        assert msgs[1].cost_usd == pytest.approx(0.001)

    # Orchestrator audit block on the response.
    audit = result["response"]["metadata"]["orchestrator"]
    assert audit["kind"] == "product"
    assert audit["deployment_id"] == "serving-1"
    assert audit["miner_hotkey"] == "hk-winner"


async def test_invoke_loads_history_into_request_envelope(tmp_path):
    """Multi-turn: prior messages must appear in request.history on the
    next turn — the agent never persists user-visible state itself."""
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_user(session)
        _seed_serving_deployment(session)

    captured: list[dict[str, Any]] = []
    orch = _build_orchestrator(db, captured_payloads=captured)

    first = await orch.invoke(user_id="user-1", prompt="hi")
    convo_id = first["conversation_id"]
    second = await orch.invoke(
        user_id="user-1", prompt="follow-up", conversation_id=convo_id,
    )
    # Second turn's outbound payload should carry the first user+assistant
    # turn in history.
    second_body = captured[-1]["body"]
    assert second_body["prompt"] == "follow-up"
    history_roles = [m["role"] for m in second_body["history"]]
    assert history_roles == ["user", "assistant"]
    assert second_body["history"][0]["content"] == "hi"


async def test_invoke_loads_preferences_into_metadata(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_user(session)
        _seed_serving_deployment(session)
        session.add(ConsumerPreference(
            user_id="user-1", scope="global", project_id=None,
            key="tone", value_json="friendly",
        ))
        session.commit()

    captured: list[dict[str, Any]] = []
    orch = _build_orchestrator(db, captured_payloads=captured)
    await orch.invoke(user_id="user-1", prompt="hi")

    metadata = captured[-1]["body"]["metadata"]
    assert metadata["user_preferences"]["global"] == {"tone": "friendly"}
    assert metadata["user_preferences"]["project"] == {}


async def test_invoke_loads_project_context_and_memory(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_user(session)
        _seed_serving_deployment(session)
        proj_id = str(uuid4())
        session.add(ConsumerProject(
            project_id=proj_id, user_id="user-1",
            name="My Project",
            custom_instructions="Always answer in haiku.",
        ))
        session.add(ConsumerPreference(
            user_id="user-1", scope="project", project_id=proj_id,
            key="style", value_json="terse",
        ))
        # Seed a memory row with a real encoded embedding so the
        # cosine-similarity reader finds it. We use the
        # StubEmbeddingClient to embed the same way the orchestrator
        # will when the user's prompt arrives.
        from orchestration.orchestrator.embedding_client import StubEmbeddingClient
        from orchestration.orchestrator.project_memory import encode_embedding
        stub = StubEmbeddingClient()
        memory_text = "The user works in Python."
        embedded = (await stub.aembed([memory_text]))[0]
        session.add(ConsumerProjectMemory(
            project_id=proj_id, vector_id="v1",
            embedding=encode_embedding(embedded), text=memory_text,
        ))
        session.commit()

    captured: list[dict[str, Any]] = []
    orch = _build_orchestrator(db, captured_payloads=captured)
    # Use a prompt that shares tokens with the seeded memory so the
    # stub embedder yields a high cosine similarity.
    await orch.invoke(user_id="user-1", prompt="what language does the user work in", project_id=proj_id)

    metadata = captured[-1]["body"]["metadata"]
    assert metadata["project_context"]["name"] == "My Project"
    assert metadata["project_context"]["custom_instructions"] == "Always answer in haiku."
    assert metadata["user_preferences"]["project"] == {"style": "terse"}
    assert any(
        snippet["text"] == "The user works in Python."
        for snippet in metadata["recalled_memory"]
    )


async def test_invoke_unknown_user_raises(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_serving_deployment(session)
    orch = _build_orchestrator(db)
    with pytest.raises(ProductOrchestratorError):
        await orch.invoke(user_id="ghost", prompt="hi")


async def test_invoke_unknown_conversation_raises(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_user(session)
        _seed_serving_deployment(session)
    orch = _build_orchestrator(db)
    with pytest.raises(ProductOrchestratorError):
        await orch.invoke(user_id="user-1", prompt="hi", conversation_id="ghost")


async def test_invoke_no_serving_deployment_raises(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_user(session)
    orch = _build_orchestrator(db)
    with pytest.raises(ProductOrchestratorError):
        await orch.invoke(user_id="user-1", prompt="hi")


# -- Streaming astream -----------------------------------------------------


async def test_astream_emits_conversation_event_then_drops_trace_frames(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_user(session)
        _seed_serving_deployment(session)

    ndjson = [
        json.dumps({"event": "delta", "text": "hel"}),
        json.dumps({"event": "trace", "node": "planner", "kind": "node_enter"}),
        json.dumps({"event": "delta", "text": "lo"}),
        json.dumps({"event": "citation", "citation": {"url": "https://x"}}),
        json.dumps({"event": "trace", "node": "planner", "kind": "node_exit"}),
        json.dumps({
            "event": "done",
            "status": "completed",
            "output": {"answer": "hello"},
            "metadata": {"proxy_cost_usd": 0.0005, "executed_tool_calls": []},
        }),
    ]
    orch = _build_orchestrator(db, ndjson_lines=ndjson)
    events = []
    async for ev in orch.astream(user_id="user-1", prompt="hi"):
        events.append(ev)
    types = [e.get("event") for e in events]
    # First event is the conversation announcement; trace frames are
    # dropped; passthrough order preserved otherwise.
    assert types[0] == "conversation"
    assert "trace" not in types
    assert types[1:] == ["delta", "delta", "citation", "done"]
    final = events[-1]
    assert final["status"] == "completed"
    assert final["metadata"]["orchestrator"]["kind"] == "product"


async def test_astream_persists_assistant_after_stream_closes(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_user(session)
        _seed_serving_deployment(session)
    ndjson = [
        json.dumps({"event": "delta", "text": "hi "}),
        json.dumps({"event": "delta", "text": "there"}),
        json.dumps({"event": "done", "status": "completed", "metadata": {}}),
    ]
    orch = _build_orchestrator(db, ndjson_lines=ndjson)
    convo_id = None
    async for ev in orch.astream(user_id="user-1", prompt="ping"):
        if ev.get("event") == "conversation":
            convo_id = ev["conversation_id"]

    assert convo_id is not None
    with db.sessionmaker() as session:
        msgs = (
            session.query(ConsumerMessage)
            .filter_by(conversation_id=convo_id)
            .order_by(ConsumerMessage.turn_idx)
            .all()
        )
        roles = [m.role for m in msgs]
        contents = [m.content for m in msgs]
        assert roles == ["user", "assistant"]
        assert contents == ["ping", "hi there"]


async def test_astream_emits_failed_done_when_no_serving(tmp_path):
    """Empty fleet — astream must still yield conversation + terminal done."""
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_user(session)
    orch = _build_orchestrator(db, ndjson_lines=[])
    events = []
    async for ev in orch.astream(user_id="user-1", prompt="hi"):
        events.append(ev)
    types = [e.get("event") for e in events]
    assert types == ["conversation", "done"]
    assert events[-1]["status"] == "failed"


# -- Eval / product byte-compat -------------------------------------------


async def test_envelope_shape_matches_eval_path_keys(tmp_path):
    """The miner-side AgentInvocationRequest contract is the same in
    eval and product. Assert the keys produced by ProductOrchestrator
    are a superset of the eval-path keys (eval has no metadata.* user
    fields; product adds them under metadata)."""
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_user(session)
        _seed_serving_deployment(session)
    captured: list[dict[str, Any]] = []
    orch = _build_orchestrator(db, captured_payloads=captured)
    await orch.invoke(user_id="user-1", prompt="hi")

    body = captured[-1]["body"]
    # Contract keys that must be present in both eval and product:
    for key in ("prompt", "history", "mode", "web_search", "turn_id"):
        assert key in body, f"missing required envelope key: {key}"
    # Product-only metadata block:
    assert "metadata" in body
    md = body["metadata"]
    for key in ("user_preferences", "recalled_memory"):
        assert key in md
