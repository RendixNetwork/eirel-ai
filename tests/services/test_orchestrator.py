from __future__ import annotations

"""Tests for orchestrator — family selector, composition planner,
execution coordinator, and the main Orchestrator class + FastAPI app."""

import asyncio
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestration.orchestrator.family_selector import (
    RoutingDecision,
    select_route,
)
from orchestration.orchestrator.composition_planner import (
    CompositionPlan,
    ExecutionStep,
    build_plan,
)
from orchestration.orchestrator.execution_coordinator import ExecutionCoordinator
from orchestration.orchestrator.platform_tools.base import PlatformTool, ToolResult
from orchestration.orchestrator.platform_tools.tools_registry import ToolsRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeTool(PlatformTool):
    @property
    def name(self) -> str:
        return "fake_tool"

    @property
    def description(self) -> str:
        return "A fake tool for testing."

    async def execute(self, *, params: dict[str, Any]) -> ToolResult:
        return ToolResult(
            tool_name=self.name,
            success=True,
            output={"echo": params.get("query", "")},
        )


def _make_registry_with_fake() -> ToolsRegistry:
    registry = ToolsRegistry()
    registry.register(_FakeTool())
    return registry


# ===================================================================
# Family selector tests
# ===================================================================


def test_default_route_is_general_chat():
    d = select_route(prompt="hello")
    assert d.route_type == "specialist"
    assert d.families == ["general_chat"]


def test_long_prompt_routes_to_general_chat():
    d = select_route(prompt="tell me about the implications of this new policy on global trade and economics")
    assert d.route_type == "specialist"
    assert d.families == ["general_chat"]


def test_general_chat_exclusion_returns_direct():
    d = select_route(
        prompt="write python code to implement a binary search algorithm",
        families_excluded=["general_chat"],
    )
    assert d.route_type == "direct"
    assert d.families == []


# ===================================================================
# Composition planner tests
# ===================================================================


def test_direct_plan_single_step():
    decision = RoutingDecision(route_type="direct")
    plan = build_plan(decision=decision, prompt="hello")
    assert len(plan.steps) == 1
    assert plan.steps[0].step_type == "direct"


def test_specialist_plan_ordered_steps_verifier_last():
    decision = RoutingDecision(
        route_type="specialist",
        families=["analyst", "verifier"],
        workflow_template="direct_analysis",
    )
    plan = build_plan(decision=decision, prompt="explain this")
    family_order = [s.family_id for s in plan.steps]
    assert family_order[-1] == "verifier"
    assert "analyst" in family_order


def test_tool_plan_synthesis_step_appended():
    decision = RoutingDecision(
        route_type="platform_tool",
        platform_tools=["code_exec", "web_search"],
        workflow_template="tool_execution",
    )
    plan = build_plan(decision=decision, prompt="run and search")
    types = [s.step_type for s in plan.steps]
    assert types[-1] == "synthesis"
    assert types.count("platform_tool") == 2


def test_composite_plan_tools_before_specialists():
    decision = RoutingDecision(
        route_type="composite",
        families=["analyst", "verifier"],
        platform_tools=["web_search"],
        workflow_template="direct_analysis",
    )
    plan = build_plan(decision=decision, prompt="search and analyze")
    types = [s.step_type for s in plan.steps]
    first_specialist = next(i for i, t in enumerate(types) if t == "specialist")
    last_tool = max(i for i, t in enumerate(types) if t == "platform_tool")
    assert last_tool < first_specialist


# ===================================================================
# Execution coordinator tests
# ===================================================================


async def test_direct_plan_execution():
    registry = _make_registry_with_fake()
    coordinator = ExecutionCoordinator(tools_registry=registry)
    plan = CompositionPlan(
        steps=[ExecutionStep(step_id="direct", step_type="direct", subtask="hello world")],
        route_type="direct",
    )
    result = await coordinator.execute_plan(plan=plan, prompt="hello world")
    assert result["status"] == "completed"
    assert result["response"]["response"] == "hello world"


async def test_tool_step_invokes_registry():
    registry = _make_registry_with_fake()
    coordinator = ExecutionCoordinator(tools_registry=registry)
    plan = CompositionPlan(
        steps=[
            ExecutionStep(
                step_id="t1",
                step_type="platform_tool",
                tool_name="fake_tool",
                params={"query": "test query"},
            )
        ],
        route_type="platform_tool",
    )
    result = await coordinator.execute_plan(plan=plan, prompt="test query")
    assert result["status"] == "completed"
    step = result["steps"][0]
    assert step["status"] == "completed"
    assert step["tool_name"] == "fake_tool"


async def test_specialist_failure_returns_partial(monkeypatch):
    registry = _make_registry_with_fake()
    coordinator = ExecutionCoordinator(tools_registry=registry)

    async def _fail_miner(family_id: str) -> None:
        return None

    monkeypatch.setattr(coordinator, "_get_serving_miner", _fail_miner)

    plan = CompositionPlan(
        steps=[
            ExecutionStep(
                step_id="s1",
                step_type="specialist",
                family_id="analyst",
                subtask="analyze this",
            )
        ],
        route_type="specialist",
    )
    result = await coordinator.execute_plan(plan=plan, prompt="analyze this")
    assert result["status"] == "partial"
    assert result["steps"][0]["status"] == "failed"


async def test_synthesis_step_combines_results():
    registry = _make_registry_with_fake()
    coordinator = ExecutionCoordinator(tools_registry=registry)
    plan = CompositionPlan(
        steps=[
            ExecutionStep(step_id="t1", step_type="platform_tool", tool_name="fake_tool", params={"query": "a"}),
            ExecutionStep(step_id="t2", step_type="platform_tool", tool_name="fake_tool", params={"query": "b"}),
            ExecutionStep(step_id="syn", step_type="synthesis", subtask="combine", depends_on=["t1", "t2"]),
        ],
        route_type="platform_tool",
    )
    result = await coordinator.execute_plan(plan=plan, prompt="a and b")
    assert result["status"] == "completed"
    syn_step = [s for s in result["steps"] if s["step_id"] == "syn"][0]
    assert len(syn_step["output"]["combined_results"]) == 2


# ===================================================================
# End-to-end orchestrator + FastAPI tests
# ===================================================================


async def test_orchestrator_handle_request_timeout(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_REQUEST_TIMEOUT_SECONDS", "0.01")
    # Re-import to pick up new env value
    import importlib
    import orchestration.orchestrator.orchestrator as oo

    importlib.reload(oo)
    try:
        orch = oo.Orchestrator()

        original_inner = orch._handle_request_inner

        async def _slow(**kwargs):
            await asyncio.sleep(5)
            return await original_inner(**kwargs)

        monkeypatch.setattr(orch, "_handle_request_inner", _slow)

        result = await orch.handle_request(prompt="hello")
        assert result["status"] == "failed"
        assert "timed out" in result["error"]
    finally:
        monkeypatch.setenv("ORCHESTRATOR_REQUEST_TIMEOUT_SECONDS", "120")
        importlib.reload(oo)


async def test_fastapi_healthz():
    from orchestration.orchestrator.main import app

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/healthz")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert "tools_registry" in data["checks"]


async def test_fastapi_list_tools():
    from orchestration.orchestrator.main import app

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/tools")
            assert resp.status_code == 200
            data = resp.json()
            assert data["count"] >= 5
            assert len(data["tools"]) == data["count"]


async def test_fastapi_orchestrate_endpoint():
    from orchestration.orchestrator.main import app

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/orchestrate",
                json={"prompt": "hi there", "user_id": "tester"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "request_id" in data
            assert data["status"] in ("completed", "partial")


# ---------------------------------------------------------------------------
# /v1/orchestrate/chat/stream
# ---------------------------------------------------------------------------


async def test_chat_stream_persists_session_toggles_and_proxies_ndjson(
    monkeypatch, tmp_path,
):
    """End-to-end: orchestrator persists mode/web_search on the session
    row and proxies the miner pod's NDJSON back to the caller."""
    import json as _json
    from orchestration.orchestrator import chat_stream as cs
    from orchestration.orchestrator.main import app

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/cs.db")

    # Stub the serving-deployment lookup so the test doesn't need a live
    # owner-api. Force the new env override path so we don't need to
    # mock owner-api at all.
    monkeypatch.setenv(
        "EIREL_ORCHESTRATOR_MINER_OVERRIDE_ENDPOINT",
        "http://miner.local",
    )

    body = (
        _json.dumps({"event": "delta", "text": "hello"}) + "\n"
        + _json.dumps({
            "event": "done",
            "status": "completed",
            "output": {"answer": "hello"},
            "citations": [],
        }) + "\n"
    ).encode("utf-8")

    class _Transport(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.bodies: list[dict] = []

        async def handle_async_request(self, request):
            self.calls.append(request.url.path)
            try:
                self.bodies.append(_json.loads(request.content or b"{}"))
            except Exception:
                self.bodies.append({})
            if request.url.path.endswith("/v1/agent/infer/stream"):
                return httpx.Response(
                    200, content=body,
                    headers={"content-type": "application/x-ndjson"},
                )
            return httpx.Response(404)

    transport = _Transport()
    original = cs.httpx.AsyncClient

    def _patched(*args, **kwargs):
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    monkeypatch.setattr(cs.httpx, "AsyncClient", _patched)

    async with app.router.lifespan_context(app):
        asgi = ASGITransport(app=app)
        async with AsyncClient(transport=asgi, base_url="http://test") as client:
            # First turn: set mode=thinking + web_search=True; orchestrator
            # should persist these on the session row.
            resp = await client.post(
                "/v1/orchestrate/chat/stream",
                json={
                    "prompt": "hi",
                    "user_id": "u1",
                    "session_id": "sess-A",
                    "mode": "thinking",
                    "web_search": True,
                },
            )
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("application/x-ndjson")
            lines = [ln for ln in resp.content.decode("utf-8").split("\n") if ln.strip()]
            chunks = [_json.loads(ln) for ln in lines]
            assert chunks[0]["event"] == "started"
            assert chunks[0]["metadata"]["session_id"] == "sess-A"
            assert chunks[0]["metadata"]["mode"] == "thinking"
            assert chunks[0]["metadata"]["web_search"] is True
            assert chunks[-1]["event"] == "done"
            assert chunks[-1]["status"] == "completed"

            # The body sent to the miner carries the slim contract.
            miner_body = transport.bodies[0]
            assert miner_body["prompt"] == "hi"
            assert miner_body["mode"] == "thinking"
            assert miner_body["web_search"] is True

            # Second turn for the same session without explicit toggles —
            # orchestrator must reuse the persisted values.
            resp2 = await client.post(
                "/v1/orchestrate/chat/stream",
                json={"prompt": "follow up", "user_id": "u1", "session_id": "sess-A"},
            )
            assert resp2.status_code == 200
            chunks2 = [
                _json.loads(ln)
                for ln in resp2.content.decode("utf-8").split("\n")
                if ln.strip()
            ]
            started2 = chunks2[0]
            assert started2["metadata"]["mode"] == "thinking"
            assert started2["metadata"]["web_search"] is True


async def test_chat_stream_emits_done_failed_when_no_serving_deployment(
    monkeypatch, tmp_path,
):
    """If owner-api has no serving deployment for the family and no
    override is set, the orchestrator must still close the stream with
    a terminal ``done/failed`` chunk so the consumer-api facade can
    render an error UI."""
    import json as _json
    from orchestration.orchestrator import chat_stream as cs
    from orchestration.orchestrator.main import app

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/cs.db")
    monkeypatch.delenv("EIREL_ORCHESTRATOR_MINER_OVERRIDE_ENDPOINT", raising=False)

    async def _no_serving(family_id):
        return None

    monkeypatch.setattr(cs, "_resolve_serving_endpoint", _no_serving)

    async with app.router.lifespan_context(app):
        asgi = ASGITransport(app=app)
        async with AsyncClient(transport=asgi, base_url="http://test") as client:
            resp = await client.post(
                "/v1/orchestrate/chat/stream",
                json={"prompt": "hi"},
            )
            assert resp.status_code == 200
            lines = [ln for ln in resp.content.decode("utf-8").split("\n") if ln.strip()]
            chunks = [_json.loads(ln) for ln in lines]
            assert chunks[-1]["event"] == "done"
            assert chunks[-1]["status"] == "failed"
            assert "no serving deployment" in chunks[-1]["error"]
