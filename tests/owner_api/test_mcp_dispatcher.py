"""Tests for orchestration.orchestrator.mcp_dispatcher."""
from __future__ import annotations

import asyncio
import json
from typing import Any
from uuid import uuid4

from shared.common.database import Database
from shared.common.models import (
    ConsumerMcpConnection,
    ConsumerMcpToolCall,
    ConsumerUser,
    McpIntegration,
)
from orchestration.orchestrator.mcp_dispatcher import (
    DispatcherLLM,
    MCPCallResult,
    MCPRelayClient,
    MCPToolDescriptor,
    MCPToolDispatcher,
    PendingMCPCall,
)


# -- fixtures --------------------------------------------------------------


def _make_db(tmp_path) -> Database:
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'mcpd.db'}")
    db.create_all()
    return db


def _seed_active_integration(
    session,
    *,
    user_id: str,
    slug: str,
    capabilities: list[dict[str, Any]] | None = None,
    capabilities_hash: str = "h1",
    integration_status: str = "active",
    connection_status: str = "active",
) -> tuple[str, str]:
    integration_id = str(uuid4())
    session.add(McpIntegration(
        id=integration_id, slug=slug, display_name=slug.title(),
        vendor=slug.title(), base_url=f"https://{slug}.test/mcp",
        capabilities_json=capabilities or [
            {"name": "search", "description": "search docs"},
        ],
        capabilities_hash=capabilities_hash,
        status=integration_status,
    ))
    connection_id = str(uuid4())
    session.add(ConsumerMcpConnection(
        id=connection_id, user_id=user_id, integration_id=integration_id,
        status=connection_status,
    ))
    session.commit()
    return integration_id, connection_id


def _seed_user(session) -> str:
    user_id = str(uuid4())
    session.add(ConsumerUser(
        user_id=user_id, auth_subject=f"k:{user_id}", display_name="X",
    ))
    session.commit()
    return user_id


class _StubLLM:
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    async def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(payload)
        if not self._replies:
            raise AssertionError("StubLLM exhausted")
        return {"choices": [{"message": {"content": self._replies.pop(0)}}]}


class _FakeRelay:
    """Records dispatched calls; returns canned outcomes."""

    def __init__(
        self,
        *,
        outcomes: dict[tuple[str, str], dict[str, Any]] | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self._outcomes = outcomes or {}

    async def call(
        self, *, connection_id, tool_name, args,
        capabilities_hash, timeout_seconds,
    ):
        self.calls.append({
            "connection_id": connection_id,
            "tool_name": tool_name,
            "args": args,
            "capabilities_hash": capabilities_hash,
            "timeout_seconds": timeout_seconds,
        })
        outcome = self._outcomes.get((connection_id, tool_name)) or {}
        return (
            outcome.get("ok", True),
            outcome.get("result", {"echo": args}),
            outcome.get("error"),
            outcome.get("latency_ms", 5),
            outcome.get("cost_usd", 0.0002),
        )


# -- available_tools -------------------------------------------------------


def test_available_tools_lists_only_active(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        user_id = _seed_user(session)
        _seed_active_integration(session, user_id=user_id, slug="notion")
        _seed_active_integration(
            session, user_id=user_id, slug="disabled-int",
            integration_status="disabled",
        )
        _seed_active_integration(
            session, user_id=user_id, slug="revoked-conn",
            connection_status="revoked",
        )
    dispatcher = MCPToolDispatcher(database=db, relay_client=_FakeRelay())
    available = dispatcher.available_tools(user_id=user_id)
    slugs = sorted(d.integration_slug for d in available)
    assert slugs == ["notion"]
    assert available[0].tool_name == "search"


def test_available_tools_isolates_users(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        u1 = _seed_user(session)
        u2 = _seed_user(session)
        _seed_active_integration(session, user_id=u1, slug="notion")
        _seed_active_integration(session, user_id=u2, slug="github")
    dispatcher = MCPToolDispatcher(database=db, relay_client=_FakeRelay())
    a = dispatcher.available_tools(user_id=u1)
    b = dispatcher.available_tools(user_id=u2)
    assert {d.integration_slug for d in a} == {"notion"}
    assert {d.integration_slug for d in b} == {"github"}


# -- decide_calls ----------------------------------------------------------


async def test_decide_calls_returns_empty_without_active_connections(tmp_path):
    db = _make_db(tmp_path)
    dispatcher = MCPToolDispatcher(
        database=db, relay_client=_FakeRelay(),
        planner_llm=_StubLLM([]),
    )
    plan = await dispatcher.decide_calls(
        prompt="anything", history=None, available=[],
    )
    assert plan == []


async def test_decide_calls_returns_empty_when_no_planner(tmp_path):
    """No planner_llm injected → dispatcher is a no-op."""
    db = _make_db(tmp_path)
    descs = [MCPToolDescriptor(
        connection_id="c1", integration_slug="notion", integration_id="i1",
        capabilities_hash="h", tool_name="search", description="x",
    )]
    dispatcher = MCPToolDispatcher(database=db, relay_client=_FakeRelay())
    plan = await dispatcher.decide_calls(
        prompt="anything", history=None, available=descs,
    )
    assert plan == []


async def test_decide_calls_picks_correct_tool_from_planner_json(tmp_path):
    db = _make_db(tmp_path)
    descs = [
        MCPToolDescriptor(
            connection_id="c-notion", integration_slug="notion",
            integration_id="i-notion", capabilities_hash="hN",
            tool_name="search", description="search docs",
        ),
        MCPToolDescriptor(
            connection_id="c-github", integration_slug="github",
            integration_id="i-github", capabilities_hash="hG",
            tool_name="list_issues", description="list issues",
        ),
    ]
    llm = _StubLLM([
        '[{"integration_slug": "notion", "tool_name": "search",'
        ' "args": {"q": "design doc"}}]'
    ])
    dispatcher = MCPToolDispatcher(
        database=db, relay_client=_FakeRelay(), planner_llm=llm,
    )
    plan = await dispatcher.decide_calls(
        prompt="find the design doc", history=None, available=descs,
    )
    assert len(plan) == 1
    assert plan[0].connection_id == "c-notion"
    assert plan[0].tool_name == "search"
    assert plan[0].args == {"q": "design doc"}
    assert plan[0].capabilities_hash == "hN"


async def test_decide_calls_drops_invalid_tool_references(tmp_path):
    db = _make_db(tmp_path)
    descs = [MCPToolDescriptor(
        connection_id="c1", integration_slug="notion",
        integration_id="i1", capabilities_hash="h", tool_name="search",
        description="x",
    )]
    llm = _StubLLM([
        '[{"integration_slug": "notion", "tool_name": "search", "args": {}},'
        ' {"integration_slug": "ghost", "tool_name": "x", "args": {}}]'
    ])
    dispatcher = MCPToolDispatcher(
        database=db, relay_client=_FakeRelay(), planner_llm=llm,
    )
    plan = await dispatcher.decide_calls(
        prompt="x", history=None, available=descs,
    )
    assert len(plan) == 1
    assert plan[0].tool_name == "search"


async def test_decide_calls_caps_at_max_tools(tmp_path):
    db = _make_db(tmp_path)
    descs = [
        MCPToolDescriptor(
            connection_id=f"c{i}", integration_slug=f"slug{i}",
            integration_id=f"i{i}", capabilities_hash="h",
            tool_name="t", description="x",
        )
        for i in range(8)
    ]
    plan_json = json.dumps([
        {"integration_slug": f"slug{i}", "tool_name": "t", "args": {}}
        for i in range(8)
    ])
    llm = _StubLLM([plan_json])
    dispatcher = MCPToolDispatcher(
        database=db, relay_client=_FakeRelay(),
        planner_llm=llm, max_tools_per_turn=3,
    )
    plan = await dispatcher.decide_calls(
        prompt="x", history=None, available=descs,
    )
    assert len(plan) == 3


async def test_decide_calls_handles_planner_failure(tmp_path):
    class _BoomLLM:
        async def chat_completions(self, payload):
            raise RuntimeError("planner down")

    db = _make_db(tmp_path)
    descs = [MCPToolDescriptor(
        connection_id="c1", integration_slug="notion",
        integration_id="i1", capabilities_hash="h", tool_name="search",
        description="x",
    )]
    dispatcher = MCPToolDispatcher(
        database=db, relay_client=_FakeRelay(), planner_llm=_BoomLLM(),
    )
    plan = await dispatcher.decide_calls(
        prompt="x", history=None, available=descs,
    )
    assert plan == []


# -- execute_calls ---------------------------------------------------------


async def test_execute_calls_runs_in_parallel_and_writes_audit_rows(tmp_path):
    db = _make_db(tmp_path)
    relay = _FakeRelay()
    dispatcher = MCPToolDispatcher(database=db, relay_client=relay)
    calls = [
        PendingMCPCall(
            connection_id="c1", integration_slug="notion",
            capabilities_hash="h", tool_name="search", args={"q": "x"},
        ),
        PendingMCPCall(
            connection_id="c2", integration_slug="github",
            capabilities_hash="h2", tool_name="list_issues", args={},
        ),
    ]
    results = await dispatcher.execute_calls(
        user_id="u1", calls=calls, conversation_id="conv1", message_id="msg1",
    )
    assert len(results) == 2
    assert all(r.ok for r in results)
    # Both relay calls fired.
    assert {c["tool_name"] for c in relay.calls} == {"search", "list_issues"}
    # Two audit rows persisted.
    with db.sessionmaker() as session:
        rows = session.query(ConsumerMcpToolCall).all()
        assert len(rows) == 2
        names = {r.tool_name for r in rows}
        assert names == {"search", "list_issues"}
        for row in rows:
            assert row.conversation_id == "conv1"
            assert row.message_id == "msg1"


async def test_execute_calls_records_relay_failures_in_audit(tmp_path):
    db = _make_db(tmp_path)
    relay = _FakeRelay(outcomes={
        ("c1", "search"): {"ok": False, "error": "upstream down"},
    })
    dispatcher = MCPToolDispatcher(database=db, relay_client=relay)
    results = await dispatcher.execute_calls(
        user_id="u1",
        calls=[PendingMCPCall(
            connection_id="c1", integration_slug="notion",
            capabilities_hash="h", tool_name="search", args={},
        )],
    )
    assert results[0].ok is False
    assert "upstream down" in (results[0].error or "")
    with db.sessionmaker() as session:
        row = session.query(ConsumerMcpToolCall).first()
        assert row.error == "upstream down"


async def test_execute_calls_total_budget_exceeded_marks_all_failed(tmp_path):
    """When the total budget fires, every call lands a synthetic timeout row."""

    class _SlowRelay:
        def __init__(self):
            self.calls = 0
        async def call(self, **kwargs):
            self.calls += 1
            await asyncio.sleep(0.5)
            return True, {}, None, 500, 0.0

    db = _make_db(tmp_path)
    dispatcher = MCPToolDispatcher(
        database=db, relay_client=_SlowRelay(),
        total_budget_seconds=0.05,
    )
    results = await dispatcher.execute_calls(
        user_id="u1",
        calls=[PendingMCPCall(
            connection_id="c1", integration_slug="x",
            capabilities_hash="", tool_name="t", args={},
        )],
    )
    assert results[0].ok is False
    assert results[0].error == "dispatcher_total_budget_exceeded"
    with db.sessionmaker() as session:
        row = session.query(ConsumerMcpToolCall).first()
        assert row.error == "dispatcher_total_budget_exceeded"


async def test_execute_calls_passes_capabilities_hash_to_relay(tmp_path):
    db = _make_db(tmp_path)
    relay = _FakeRelay()
    dispatcher = MCPToolDispatcher(database=db, relay_client=relay)
    await dispatcher.execute_calls(
        user_id="u1",
        calls=[PendingMCPCall(
            connection_id="c1", integration_slug="notion",
            capabilities_hash="hash-stored", tool_name="t", args={},
        )],
    )
    assert relay.calls[0]["capabilities_hash"] == "hash-stored"
