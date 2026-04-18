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
