"""Tests for the graph-checkpoint internal router + manifest env injection.

The owner_api tests in this codebase call FastAPI handlers directly
rather than using TestClient — so do these. Auth (the
``require_internal_service_token`` dependency) is exercised via the
real FastAPI app in a focused TestClient test at the bottom.
"""
from __future__ import annotations

import base64
import json
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from shared.common.database import Database
from shared.common.models import (
    ConversationThread,
    GraphCheckpoint,
    ManagedDeployment,
    ManagedMinerSubmission,
)
from control_plane.owner_api.routers.checkpoints import (
    CheckpointWriteRequest,
    delete_thread_checkpoints,
    read_checkpoint_history,
    read_latest_checkpoint,
    write_checkpoint,
)


# -- Fixtures ---------------------------------------------------------------


def _make_db(tmp_path) -> Database:
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'ckpt.db'}")
    db.create_all()
    return db


def _seed_deployment(session, *, deployment_id: str = "deploy-1") -> str:
    """Seed a minimal submission + deployment row pair so the router accepts writes."""
    submission_id = str(uuid4())
    session.add(
        ManagedMinerSubmission(
            id=submission_id,
            miner_hotkey="5HotkeyMiner",
            submission_seq=1,
            family_id="general_chat",
            status="deployed",
            artifact_id=str(uuid4()),
            manifest_json={"runtime": {"kind": "graph"}},
            archive_sha256="0" * 64,
            submission_block=0,
        )
    )
    session.add(
        ManagedDeployment(
            id=deployment_id,
            submission_id=submission_id,
            miner_hotkey="5HotkeyMiner",
            family_id="general_chat",
            deployment_revision=str(uuid4()),
            image_ref="img:test",
            endpoint="http://miner.test:8080",
            status="active",
            health_status="healthy",
            placement_status="placed",
        )
    )
    session.commit()
    return deployment_id


def _make_request(*, db, namespace: str | None = None) -> SimpleNamespace:
    """Fake FastAPI Request carrying app.state.services + the namespace header."""
    services = SimpleNamespace(db=db)
    headers = {}
    if namespace is not None:
        headers["X-Eirel-Checkpoint-Namespace"] = namespace
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(services=services)),
        headers=headers,
    )


def _b64_state(state: dict) -> str:
    return base64.b64encode(json.dumps(state).encode("utf-8")).decode("ascii")


# -- write_checkpoint -------------------------------------------------------


async def test_write_checkpoint_persists_and_anchors_thread(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        deployment_id = _seed_deployment(session)

    req = _make_request(db=db, namespace=f"miner-{deployment_id}")
    body = CheckpointWriteRequest(
        checkpoint_id="cp-1",
        parent_id=None,
        node="planner",
        state=_b64_state({"messages": [{"role": "user", "content": "hi"}]}),
        pending_writes=[],
        metadata={"step": 1},
    )
    response = await write_checkpoint(req, "thread-1", body, _token=None)
    assert response.checkpoint_id == "cp-1"
    assert response.thread_id == "thread-1"
    assert response.node == "planner"

    with db.sessionmaker() as session:
        row = session.query(GraphCheckpoint).first()
        assert row is not None
        assert row.thread_id == "thread-1"
        assert row.deployment_id == deployment_id
        assert row.family_id == "general_chat"
        assert row.checkpoint_namespace == f"miner-{deployment_id}"
        assert row.blob_size_bytes > 0
        anchor = session.get(ConversationThread, "thread-1")
        assert anchor is not None
        assert anchor.last_checkpoint_id == "cp-1"
        assert anchor.deployment_id == deployment_id


async def test_write_checkpoint_is_idempotent_on_replay(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        deployment_id = _seed_deployment(session)

    req = _make_request(db=db, namespace=f"miner-{deployment_id}")
    body = CheckpointWriteRequest(
        checkpoint_id="cp-1",
        parent_id=None,
        node="planner",
        state=_b64_state({"x": 1}),
    )
    first = await write_checkpoint(req, "thread-replay", body, _token=None)
    second = await write_checkpoint(req, "thread-replay", body, _token=None)
    assert first.checkpoint_id == second.checkpoint_id
    with db.sessionmaker() as session:
        rows = session.query(GraphCheckpoint).all()
        assert len(rows) == 1


async def test_write_checkpoint_rejects_oversized_blob(tmp_path):
    from fastapi import HTTPException

    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        deployment_id = _seed_deployment(session)
    req = _make_request(db=db, namespace=f"miner-{deployment_id}")
    huge = _b64_state({"x": "y" * (260 * 1024)})  # 256+ KB
    body = CheckpointWriteRequest(
        checkpoint_id="cp-x", parent_id=None, node="big", state=huge,
    )
    with pytest.raises(HTTPException) as excinfo:
        await write_checkpoint(req, "thread-big", body, _token=None)
    assert excinfo.value.status_code == 413


async def test_write_checkpoint_rejects_missing_namespace_header(tmp_path):
    from fastapi import HTTPException

    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_deployment(session)
    req = _make_request(db=db, namespace=None)
    body = CheckpointWriteRequest(
        checkpoint_id="cp-x", parent_id=None, node="x",
        state=_b64_state({"x": 1}),
    )
    with pytest.raises(HTTPException) as excinfo:
        await write_checkpoint(req, "thread-no-ns", body, _token=None)
    assert excinfo.value.status_code == 400


async def test_write_checkpoint_rejects_unknown_deployment(tmp_path):
    from fastapi import HTTPException

    db = _make_db(tmp_path)
    req = _make_request(db=db, namespace="miner-ghost")
    body = CheckpointWriteRequest(
        checkpoint_id="cp-x", parent_id=None, node="x",
        state=_b64_state({"x": 1}),
    )
    with pytest.raises(HTTPException) as excinfo:
        await write_checkpoint(req, "thread-x", body, _token=None)
    assert excinfo.value.status_code == 404


# -- read_latest / read_history ---------------------------------------------


async def test_read_latest_returns_most_recent(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        deployment_id = _seed_deployment(session)
    req = _make_request(db=db, namespace=f"miner-{deployment_id}")
    for i in range(3):
        body = CheckpointWriteRequest(
            checkpoint_id=f"cp-{i}",
            parent_id=f"cp-{i-1}" if i else None,
            node=f"n{i}",
            state=_b64_state({"step": i}),
        )
        await write_checkpoint(req, "thread-h", body, _token=None)

    latest = await read_latest_checkpoint(req, "thread-h", _token=None)
    assert latest.checkpoint_id == "cp-2"
    decoded = json.loads(base64.b64decode(latest.state.encode("ascii")).decode("utf-8"))
    assert decoded == {"step": 2}


async def test_read_history_orders_newest_first(tmp_path):
    import asyncio

    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        deployment_id = _seed_deployment(session)
    req = _make_request(db=db, namespace=f"miner-{deployment_id}")
    for i in range(3):
        await write_checkpoint(
            req, "thread-h",
            CheckpointWriteRequest(
                checkpoint_id=f"cp-{i}", parent_id=None, node=f"n{i}",
                state=_b64_state({"i": i}),
            ),
            _token=None,
        )
        # Ensure created_at is monotonically distinct on fast hardware —
        # the schema's default=utcnow can collide at sub-millisecond
        # resolution, making ORDER BY created_at non-deterministic.
        await asyncio.sleep(0.01)
    history = await read_checkpoint_history(
        req, "thread-h", limit=10, checkpoint_id=None, _token=None,
    )
    assert [item.checkpoint_id for item in history.items] == ["cp-2", "cp-1", "cp-0"]


async def test_delete_thread_purges_rows_and_anchor(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        deployment_id = _seed_deployment(session)
    req = _make_request(db=db, namespace=f"miner-{deployment_id}")
    for i in range(2):
        await write_checkpoint(
            req, "thread-d",
            CheckpointWriteRequest(
                checkpoint_id=f"cp-{i}", parent_id=None, node=f"n{i}",
                state=_b64_state({"i": i}),
            ),
            _token=None,
        )
    res = await delete_thread_checkpoints(req, "thread-d", _token=None)
    assert res == {"deleted": 2, "thread_id": "thread-d"}
    with db.sessionmaker() as session:
        assert session.query(GraphCheckpoint).count() == 0
        assert session.get(ConversationThread, "thread-d") is None


# -- Auth wiring (real app) -------------------------------------------------


async def test_router_requires_internal_service_token(tmp_path, monkeypatch):
    """Cross-check: the real router rejects unauthenticated calls."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("EIREL_INTERNAL_SERVICE_TOKEN", "the-secret")
    from fastapi import FastAPI
    from control_plane.owner_api.routers.checkpoints import router

    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_deployment(session)

    app = FastAPI()
    # require_internal_service_token reads
    # request.app.state.services.settings.internal_service_token, so
    # mirror that shape.
    app.state.services = SimpleNamespace(
        db=db,
        settings=SimpleNamespace(internal_service_token="the-secret"),
    )
    app.include_router(router)

    body = {
        "checkpoint_id": "cp-1",
        "parent_id": None,
        "node": "x",
        "state": _b64_state({"x": 1}),
    }
    with TestClient(app) as client:
        # No token → 401/403.
        resp = client.post("/v1/internal/checkpoints/thread-auth", json=body)
        assert resp.status_code in (401, 403)
        # Wrong token → still rejected.
        resp = client.post(
            "/v1/internal/checkpoints/thread-auth",
            json=body,
            headers={"Authorization": "Bearer wrong"},
        )
        assert resp.status_code in (401, 403)
        # Right token — fails because no namespace, but auth passed.
        resp = client.post(
            "/v1/internal/checkpoints/thread-auth",
            json=body,
            headers={"Authorization": "Bearer the-secret"},
        )
        assert resp.status_code == 400  # missing namespace header
