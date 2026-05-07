"""End-to-end: MCP dispatch lands in envelope metadata, miners never see
MCP env vars.

Two invariants verified here:

  1. When the user has an active :class:`ConsumerMcpConnection`, the
     orchestrator runs :class:`MCPToolDispatcher` and injects the
     results as ``request.metadata.mcp_tool_results``. The miner sees
     pre-computed results — no tool, no token, no URL.
  2. Miner-pod env-injection paths (``infra/miner_runtime/runtime_manager.py``)
     must contain zero ``EIREL_MCP_*`` references — the relay URL,
     token, encryption key etc. are orchestrator-only configuration.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from shared.common.database import Database
from shared.common.models import (
    ConsumerMcpConnection,
    ConsumerMcpToolCall,
    ConsumerUser,
    ManagedDeployment,
    ManagedMinerSubmission,
    McpIntegration,
    ServingDeployment,
    ServingRelease,
)
from shared.safety.token_encryption import build_token_cipher
from orchestration.orchestrator.embedding_client import StubEmbeddingClient
from orchestration.orchestrator.mcp_dispatcher import MCPToolDispatcher
from orchestration.orchestrator.product_orchestrator import ProductOrchestrator
from orchestration.orchestrator.serving_picker import ServingPicker


def _make_db(tmp_path) -> Database:
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'mcp.db'}")
    db.create_all()
    return db


def _seed_serving(session) -> str:
    deployment_id = "serving-mcp"
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


class _StubLLM:
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)

    async def chat_completions(self, payload):
        if not self._replies:
            return {"choices": [{"message": {"content": "[]"}}]}
        return {"choices": [{"message": {"content": self._replies.pop(0)}}]}


class _FakeRelay:
    def __init__(self) -> None:
        self.calls = 0

    async def call(
        self, *, connection_id, tool_name, args,
        capabilities_hash, timeout_seconds,
    ):
        self.calls += 1
        return True, {"hits": ["paris", "lyon"]}, None, 5, 0.0001


def _build_orch_with_mcp(
    db: Database,
    *,
    captured: list[dict[str, Any]],
    dispatcher: MCPToolDispatcher | None = None,
) -> ProductOrchestrator:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode("utf-8")) if request.content else {})
        return httpx.Response(200, json={
            "task_id": "x", "family_id": "general_chat",
            "status": "completed", "output": {"answer": "ack"},
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
        mcp_dispatcher=dispatcher,
    )


# -- Integration tests ----------------------------------------------------


async def test_envelope_carries_mcp_tool_results_when_user_has_connection(tmp_path):
    db = _make_db(tmp_path)
    user_id = str(uuid4())
    cipher = build_token_cipher()
    with db.sessionmaker() as session:
        session.add(ConsumerUser(
            user_id=user_id, auth_subject=f"k:{user_id}", display_name="X",
        ))
        integration = McpIntegration(
            slug="notion", display_name="Notion",
            base_url="https://notion.test/m",
            capabilities_json=[{"name": "search", "description": "search"}],
            capabilities_hash="hh", status="active",
        )
        session.add(integration)
        session.flush()
        session.add(ConsumerMcpConnection(
            user_id=user_id, integration_id=integration.id,
            oauth_access_token_encrypted=cipher.encrypt(b"tok"),
            status="active",
        ))
        _seed_serving(session)
        session.commit()

    relay = _FakeRelay()
    dispatcher = MCPToolDispatcher(
        database=db, relay_client=relay,
        planner_llm=_StubLLM([
            '[{"integration_slug": "notion", "tool_name": "search",'
            ' "args": {"q": "design"}}]'
        ]),
    )
    captured: list[dict[str, Any]] = []
    orch = _build_orch_with_mcp(db, captured=captured, dispatcher=dispatcher)
    result = await orch.invoke(
        user_id=user_id, prompt="find the design doc",
    )
    assert relay.calls == 1
    metadata = captured[-1]["metadata"]
    assert "mcp_tool_results" in metadata
    results = metadata["mcp_tool_results"]
    assert len(results) == 1
    assert results[0]["integration_slug"] == "notion"
    assert results[0]["tool_name"] == "search"
    assert results[0]["ok"] is True
    assert "paris" in results[0]["result_summary"]
    # Audit row landed.
    with db.sessionmaker() as session:
        rows = session.query(ConsumerMcpToolCall).all()
        assert len(rows) == 1
        assert rows[0].conversation_id == result["conversation_id"]


async def test_envelope_omits_mcp_when_user_has_no_connections(tmp_path):
    db = _make_db(tmp_path)
    user_id = str(uuid4())
    with db.sessionmaker() as session:
        session.add(ConsumerUser(
            user_id=user_id, auth_subject=f"k:{user_id}", display_name="X",
        ))
        _seed_serving(session)
        session.commit()

    dispatcher = MCPToolDispatcher(
        database=db, relay_client=_FakeRelay(),
        planner_llm=_StubLLM([]),
    )
    captured: list[dict[str, Any]] = []
    orch = _build_orch_with_mcp(db, captured=captured, dispatcher=dispatcher)
    await orch.invoke(user_id=user_id, prompt="hi")
    metadata = captured[-1]["metadata"]
    # mcp_tool_results is present but empty — the agent can branch on
    # truthiness without optional-key handling.
    assert metadata["mcp_tool_results"] == []


async def test_envelope_omits_mcp_when_no_dispatcher_configured(tmp_path):
    db = _make_db(tmp_path)
    user_id = str(uuid4())
    with db.sessionmaker() as session:
        session.add(ConsumerUser(
            user_id=user_id, auth_subject=f"k:{user_id}", display_name="X",
        ))
        _seed_serving(session)
        session.commit()
    captured: list[dict[str, Any]] = []
    orch = _build_orch_with_mcp(db, captured=captured, dispatcher=None)
    await orch.invoke(user_id=user_id, prompt="hi")
    assert captured[-1]["metadata"]["mcp_tool_results"] == []


# -- Miner isolation --------------------------------------------------------


def test_miner_runtime_manager_has_no_mcp_env_references():
    """``infra/miner_runtime/runtime_manager.py`` must not inject any
    ``EIREL_MCP_*`` env var into miner pods.

    Regression guard. MCP is orchestrator-only — miners never see it.
    A future commit that accidentally adds MCP env injection to the
    miner runtime would break the isolation invariant silently; this
    test catches it at CI time.
    """
    path = Path("/workspace/eirel-ai/infra/miner_runtime/runtime_manager.py")
    if not path.exists():
        # Repo layout shifted; surface as a skip rather than hide drift.
        import pytest
        pytest.skip(f"{path} not found")
    text = path.read_text()
    matches = re.findall(r"EIREL_MCP_[A-Z0-9_]+", text)
    assert matches == [], (
        f"miner runtime injects MCP env vars: {matches}. MCP must stay "
        "orchestrator-only — miners never see it."
    )
