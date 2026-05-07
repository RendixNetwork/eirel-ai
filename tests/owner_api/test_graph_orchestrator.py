"""End-to-end tests for the graph-runtime orchestrator.

Covers:
  * family_selection: single-family fast path
  * graph_plan: single-step plan
  * miner_picker: round-robin among healthy deployments, thread-id pinning,
    fall-forward when pinned deployment is gone
  * graph_executor: unary + streaming flows; trace frames dropped from
    consumer stream; orchestrator metadata block stamped on done frame
  * graph_orchestrator: end-to-end stream with mocked owner-api transport
  * NoEligibleMinerError surfaces as a final failed-done event in astream
"""
from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import httpx
import pytest

from shared.common.database import Database
from shared.common.models import (
    ConversationThread,
    ManagedDeployment,
    ManagedMinerSubmission,
)

from orchestration.orchestrator.family_selection import (
    FamilySelection,
    select_family_for_prompt,
)
from orchestration.orchestrator.graph_executor import GraphExecutor
from orchestration.orchestrator.graph_orchestrator import (
    GraphOrchestrator,
    OrchestratorError,
)
from orchestration.orchestrator.graph_plan import build_graph_plan
from orchestration.orchestrator.miner_picker import (
    MinerPicker,
    NoEligibleMinerError,
)


# -- Fixtures ----------------------------------------------------------------


def _make_db(tmp_path) -> Database:
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'orch.db'}")
    db.create_all()
    return db


def _seed_deployment(
    session,
    *,
    deployment_id: str,
    miner_hotkey: str,
    family_id: str = "general_chat",
    status: str = "active",
    health: str = "healthy",
    latency_p50: int = 100,
    runtime_kind: str = "graph",
) -> str:
    submission_id = str(uuid4())
    session.add(
        ManagedMinerSubmission(
            id=submission_id,
            miner_hotkey=miner_hotkey,
            submission_seq=1,
            family_id=family_id,
            status="deployed",
            artifact_id=str(uuid4()),
            manifest_json={"runtime": {"kind": runtime_kind}},
            archive_sha256="0" * 64,
            submission_block=0,
        )
    )
    session.add(
        ManagedDeployment(
            id=deployment_id,
            submission_id=submission_id,
            miner_hotkey=miner_hotkey,
            family_id=family_id,
            deployment_revision=str(uuid4()),
            image_ref="img:test",
            endpoint=f"http://miner-{deployment_id}.test:8080",
            status=status,
            health_status=health,
            placement_status="placed",
            latency_ms_p50=latency_p50,
        )
    )
    session.commit()
    return deployment_id


# -- family_selection -------------------------------------------------------


def test_family_selection_picks_general_chat_fast_path():
    sel = select_family_for_prompt(prompt="hi")
    assert sel.family_id == "general_chat"
    assert sel.confidence == 1.0
    assert sel.rationale == "single_family_fast_path"


def test_family_selection_falls_back_when_general_chat_unavailable():
    sel = select_family_for_prompt(
        prompt="hi", available_families=("general_chat",)
    )
    assert sel.family_id == "general_chat"


def test_family_selection_rejects_empty_available():
    with pytest.raises(ValueError):
        select_family_for_prompt(prompt="hi", available_families=())


# -- graph_plan -------------------------------------------------------------


def test_graph_plan_single_step():
    sel = select_family_for_prompt(prompt="hi")
    plan = build_graph_plan(selection=sel)
    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert step.family_id == "general_chat"
    assert step.step_id == "step-1"
    assert plan.metadata["single_family_fast_path"] is True


# -- miner_picker -----------------------------------------------------------


def test_miner_picker_round_robins_top_k(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        # Three healthy deployments with ascending latencies — top_k=2
        # restricts the round-robin to the two fastest.
        _seed_deployment(session, deployment_id="d-fast", miner_hotkey="h1", latency_p50=50)
        _seed_deployment(session, deployment_id="d-mid", miner_hotkey="h2", latency_p50=100)
        _seed_deployment(session, deployment_id="d-slow", miner_hotkey="h3", latency_p50=500)
    picker = MinerPicker(database=db, top_k=2)
    chosen = [picker.pick(family_id="general_chat").deployment_id for _ in range(4)]
    # Round-robin between d-fast and d-mid; d-slow excluded by top_k.
    assert "d-slow" not in chosen
    assert {chosen[0], chosen[1]} == {"d-fast", "d-mid"}
    # Stable cycling — same two ids reappear.
    assert {chosen[2], chosen[3]} == {"d-fast", "d-mid"}


def test_miner_picker_skips_unhealthy(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_deployment(session, deployment_id="d-fast", miner_hotkey="h1", health="degraded")
        _seed_deployment(session, deployment_id="d-ok", miner_hotkey="h2", health="healthy")
    picker = MinerPicker(database=db)
    cand = picker.pick(family_id="general_chat")
    assert cand.deployment_id == "d-ok"


def test_miner_picker_raises_when_no_eligible(tmp_path):
    db = _make_db(tmp_path)
    picker = MinerPicker(database=db)
    with pytest.raises(NoEligibleMinerError):
        picker.pick(family_id="general_chat")


def test_miner_picker_pins_thread_id_to_known_deployment(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_deployment(session, deployment_id="d-a", miner_hotkey="h1", latency_p50=50)
        _seed_deployment(session, deployment_id="d-b", miner_hotkey="h2", latency_p50=100)
        # Bind the thread to the slower deployment.
        session.add(
            ConversationThread(
                thread_id="t-pin",
                user_id="u",
                deployment_id="d-b",
                family_id="general_chat",
            )
        )
        session.commit()
    picker = MinerPicker(database=db)
    cand = picker.pick(family_id="general_chat", thread_id="t-pin")
    # Pinned deployment wins regardless of latency rank.
    assert cand.deployment_id == "d-b"


def test_miner_picker_falls_forward_when_pinned_deployment_unhealthy(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_deployment(
            session, deployment_id="d-pin", miner_hotkey="h1",
            health="degraded",  # not eligible
        )
        _seed_deployment(session, deployment_id="d-ok", miner_hotkey="h2", health="healthy")
        session.add(
            ConversationThread(
                thread_id="t-orphan",
                user_id="u",
                deployment_id="d-pin",
                family_id="general_chat",
            )
        )
        session.commit()
    picker = MinerPicker(database=db)
    cand = picker.pick(family_id="general_chat", thread_id="t-orphan")
    assert cand.deployment_id == "d-ok"


def test_miner_picker_reports_runtime_kind(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_deployment(session, deployment_id="d-graph", miner_hotkey="h1", runtime_kind="graph")
    picker = MinerPicker(database=db)
    cand = picker.pick(family_id="general_chat")
    assert cand.runtime_kind == "graph"


# -- graph_executor + graph_orchestrator ------------------------------------


def _stub_handler_factory(*, ndjson_lines: list[str]):
    """Build an httpx.MockTransport handler for owner-api streaming routes."""

    def handler(request: httpx.Request) -> httpx.Response:
        # Streaming routes
        if "/v1/agent/infer/stream" in str(request.url):
            body = ("\n".join(ndjson_lines) + "\n").encode("utf-8")
            return httpx.Response(
                200,
                content=body,
                headers={"content-type": "application/x-ndjson"},
            )
        # Unary route — echo the prompt
        if request.url.path.endswith("/v1/agent/infer"):
            payload = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                json={
                    "task_id": "t-1",
                    "family_id": "general_chat",
                    "status": "completed",
                    "output": {"answer": f"echo:{payload.get('prompt')}"},
                    "citations": [],
                    "metadata": {"runtime_kind": "graph"},
                },
            )
        return httpx.Response(404)

    return handler


def _build_orchestrator(db: Database, ndjson_lines: list[str]) -> GraphOrchestrator:
    transport = httpx.MockTransport(_stub_handler_factory(ndjson_lines=ndjson_lines))
    picker = MinerPicker(database=db)
    executor = GraphExecutor(
        miner_picker=picker,
        owner_api_url="http://owner-api.test",
        internal_service_token="t",
        transport=transport,
    )
    return GraphOrchestrator(miner_picker=picker, executor=executor)


async def test_orchestrator_unary_invoke_stamps_metadata(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_deployment(session, deployment_id="d-1", miner_hotkey="hk-1", runtime_kind="graph")
    orch = _build_orchestrator(db, ndjson_lines=[])
    resp = await orch.invoke(prompt="ping", thread_id="t-x")
    assert resp["status"] == "completed"
    assert resp["output"]["answer"] == "echo:ping"
    meta = resp["metadata"]["orchestrator"]
    assert meta["deployment_id"] == "d-1"
    assert meta["miner_hotkey"] == "hk-1"
    assert meta["thread_id"] == "t-x"
    assert meta["selection"]["family_id"] == "general_chat"


async def test_orchestrator_stream_drops_trace_frames_and_passes_others(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_deployment(session, deployment_id="d-1", miner_hotkey="hk-1")
    ndjson = [
        json.dumps({"event": "delta", "text": "hel"}),
        json.dumps({"event": "trace", "node": "planner", "kind": "node_enter"}),
        json.dumps({"event": "delta", "text": "lo"}),
        json.dumps({"event": "tool_call", "tool_call": {"name": "ws"}}),
        json.dumps({"event": "trace", "node": "planner", "kind": "node_exit"}),
        json.dumps({"event": "citation", "citation": {"url": "https://x"}}),
        json.dumps({"event": "done", "status": "completed", "output": {"answer": "hello"}}),
    ]
    orch = _build_orchestrator(db, ndjson_lines=ndjson)
    events = []
    async for ev in orch.astream(prompt="hi", thread_id="t-1"):
        events.append(ev)
    event_types = [e.get("event") for e in events]
    assert "trace" not in event_types
    assert event_types == ["delta", "delta", "tool_call", "citation", "done"]
    final = events[-1]
    assert final["status"] == "completed"
    assert final["metadata"]["orchestrator"]["deployment_id"] == "d-1"


async def test_orchestrator_stream_emits_failed_done_when_no_miner(tmp_path):
    """Empty fleet — astream must still yield a terminal done event."""
    db = _make_db(tmp_path)  # no deployments
    orch = _build_orchestrator(db, ndjson_lines=[])
    events = []
    async for ev in orch.astream(prompt="hi"):
        events.append(ev)
    assert len(events) == 1
    assert events[0]["event"] == "done"
    assert events[0]["status"] == "failed"
    assert "no healthy deployment" in events[0]["error"]


async def test_orchestrator_invoke_raises_orchestrator_error_on_no_miner(tmp_path):
    db = _make_db(tmp_path)
    orch = _build_orchestrator(db, ndjson_lines=[])
    with pytest.raises(OrchestratorError):
        await orch.invoke(prompt="hi")


async def test_orchestrator_stream_passthrough_unknown_event_types(tmp_path):
    """Future event types we haven't taught the orchestrator about must
    pass through verbatim — the wire is forward-compat."""
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_deployment(session, deployment_id="d-1", miner_hotkey="hk-1")
    ndjson = [
        json.dumps({"event": "delta", "text": "hi"}),
        json.dumps({"event": "future_event", "data": "experimental"}),
        json.dumps({"event": "done", "status": "completed"}),
    ]
    orch = _build_orchestrator(db, ndjson_lines=ndjson)
    events = [e async for e in orch.astream(prompt="x")]
    types = [e.get("event") for e in events]
    assert "future_event" in types  # forwarded
