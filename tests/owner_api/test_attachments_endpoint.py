"""Tests for the /v1/graph/attachments upload endpoint and orchestrator wiring.

Covers:
  * Upload extracts text + persists ConsumerAttachment row
  * Unknown user_id returns 404
  * Empty file returns 400; oversized file returns 413
  * Cross-user attachment ids are silently dropped on chat invoke
    (security: a user can't reference another user's attachments)
  * ProductOrchestrator hydrates attached_files into request.metadata
    when attachment_ids are passed
"""
from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pytest

from shared.common.database import Database
from shared.common.models import (
    ConsumerAttachment,
    ConsumerUser,
    ManagedDeployment,
    ManagedMinerSubmission,
    ServingDeployment,
    ServingRelease,
)
from orchestration.orchestrator.product_orchestrator import ProductOrchestrator
from orchestration.orchestrator.serving_picker import ServingPicker


_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "attachments"


# -- Fixtures (DB seeding) --------------------------------------------------


def _make_db(tmp_path) -> Database:
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'att.db'}")
    db.create_all()
    return db


def _seed_user(session, user_id: str = "user-1") -> str:
    session.add(ConsumerUser(
        user_id=user_id, auth_subject=f"api-key:{user_id}", display_name="X",
    ))
    session.commit()
    return user_id


def _seed_serving_deployment(session, *, deployment_id: str = "serving-att") -> str:
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


# -- Endpoint tests --------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient against consumer_api with a tmp DB + permissive auth."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("EIREL_CONSUMER_API_KEYS", "")  # disable api-key auth
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'app.db'}")
    monkeypatch.setenv("EIREL_CONSUMER_RATE_LIMIT_REQUESTS", "1000")

    from orchestration.consumer_api.main import app

    with TestClient(app) as client:
        yield client


def test_upload_persists_extracted_text(client):
    db: Database = client.app.state.database
    with db.sessionmaker() as session:
        user_id = _seed_user(session)

    csv_bytes = (_FIXTURES / "sample.csv").read_bytes()
    resp = client.post(
        "/v1/graph/attachments",
        data={"user_id": user_id},
        files={"file": ("sample.csv", csv_bytes, "text/csv")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["filename"] == "sample.csv"
    assert body["extraction_status"] == "ok"
    assert body["extracted_chars"] > 0

    with db.sessionmaker() as session:
        row = session.get(ConsumerAttachment, body["attachment_id"])
        assert row is not None
        assert row.user_id == user_id
        assert "Alice,99" in row.extracted_text
        assert row.extraction_status == "ok"


def test_upload_unknown_user_returns_404(client):
    resp = client.post(
        "/v1/graph/attachments",
        data={"user_id": "ghost"},
        files={"file": ("x.txt", b"hi", "text/plain")},
    )
    assert resp.status_code == 404


def test_upload_empty_file_returns_400(client):
    db: Database = client.app.state.database
    with db.sessionmaker() as session:
        user_id = _seed_user(session)

    resp = client.post(
        "/v1/graph/attachments",
        data={"user_id": user_id},
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert resp.status_code == 400


def test_upload_oversize_file_returns_413(client, monkeypatch):
    # Patch MAX_RAW_BYTES used inside the upload handler so we can
    # exercise the 413 path with a small payload. The handler reads the
    # constant lazily through the module import.
    db: Database = client.app.state.database
    with db.sessionmaker() as session:
        user_id = _seed_user(session)

    monkeypatch.setattr(
        "orchestration.consumer_api.main.MAX_RAW_BYTES", 1024,
    )
    resp = client.post(
        "/v1/graph/attachments",
        data={"user_id": user_id},
        files={"file": ("big.txt", b"x" * 2048, "text/plain")},
    )
    assert resp.status_code == 413
    body = resp.json()
    assert body["size_bytes"] == 2048
    assert "exceeds" in body["detail"]


# -- Orchestrator hydration ------------------------------------------------


def _build_orchestrator(
    db: Database,
    *,
    captured: list[dict[str, Any]],
) -> ProductOrchestrator:
    """ProductOrchestrator with a stub miner pod that records the
    AgentInvocationRequest envelope it receives."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        captured.append(body)
        return httpx.Response(200, json={
            "task_id": body.get("turn_id"),
            "family_id": "general_chat",
            "status": "completed",
            "output": {"answer": "ok"},
            "citations": [],
            "metadata": {"runtime_kind": "graph", "executed_tool_calls": []},
        })

    transport = httpx.MockTransport(handler)
    return ProductOrchestrator(
        database=db,
        serving_picker=ServingPicker(database=db),
        owner_api_url="http://owner-api.test",
        internal_service_token="t",
        transport=transport,
    )


async def test_orchestrator_hydrates_attached_files(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        user_id = _seed_user(session)
        _seed_serving_deployment(session)
        att = ConsumerAttachment(
            user_id=user_id, filename="notes.md", content_type="text/markdown",
            size_bytes=11, extracted_text="hello world",
            extraction_metadata_json={"format": "markdown"},
        )
        session.add(att)
        session.flush()
        attachment_id = att.id
        session.commit()

    captured: list[dict[str, Any]] = []
    orch = _build_orchestrator(db, captured=captured)

    await orch.invoke(
        user_id=user_id,
        prompt="summarize the attached notes",
        attachment_ids=[attachment_id],
    )

    assert captured, "no envelope captured"
    metadata = captured[0]["metadata"]
    files = metadata["attached_files"]
    assert len(files) == 1
    assert files[0]["attachment_id"] == attachment_id
    assert files[0]["filename"] == "notes.md"
    assert files[0]["extracted_text"] == "hello world"
    assert files[0]["extraction_status"] == "ok"


async def test_orchestrator_drops_other_users_attachments(tmp_path):
    """Security: a user cannot reference another user's attachment id."""
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        owner_id = _seed_user(session, user_id="owner")
        attacker_id = _seed_user(session, user_id="attacker")
        _seed_serving_deployment(session)
        att = ConsumerAttachment(
            user_id=owner_id, filename="secret.txt",
            content_type="text/plain", size_bytes=10,
            extracted_text="OWNER ONLY",
        )
        session.add(att)
        session.flush()
        secret_id = att.id
        session.commit()

    captured: list[dict[str, Any]] = []
    orch = _build_orchestrator(db, captured=captured)
    await orch.invoke(
        user_id=attacker_id,
        prompt="show me",
        attachment_ids=[secret_id],
    )
    metadata = captured[0]["metadata"]
    # Attachment is silently dropped — the agent never sees it.
    assert metadata["attached_files"] == []


async def test_orchestrator_preserves_attachment_order(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        user_id = _seed_user(session)
        _seed_serving_deployment(session)
        ids: list[str] = []
        for name in ("first.txt", "second.txt", "third.txt"):
            att = ConsumerAttachment(
                user_id=user_id, filename=name, content_type="text/plain",
                size_bytes=4, extracted_text=name,
            )
            session.add(att)
            session.flush()
            ids.append(att.id)
        session.commit()

    captured: list[dict[str, Any]] = []
    orch = _build_orchestrator(db, captured=captured)
    # Submit in reverse order — orchestrator must preserve caller order,
    # not DB insertion order.
    reverse_ids = list(reversed(ids))
    await orch.invoke(
        user_id=user_id, prompt="x", attachment_ids=reverse_ids,
    )
    files = captured[0]["metadata"]["attached_files"]
    assert [f["attachment_id"] for f in files] == reverse_ids


async def test_orchestrator_skips_unknown_attachment_id(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        user_id = _seed_user(session)
        _seed_serving_deployment(session)

    captured: list[dict[str, Any]] = []
    orch = _build_orchestrator(db, captured=captured)
    await orch.invoke(
        user_id=user_id,
        prompt="x",
        attachment_ids=["does-not-exist", "also-not"],
    )
    metadata = captured[0]["metadata"]
    assert metadata["attached_files"] == []


async def test_orchestrator_attached_files_default_empty(tmp_path):
    """When no attachment_ids are passed, metadata.attached_files is []."""
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        user_id = _seed_user(session)
        _seed_serving_deployment(session)

    captured: list[dict[str, Any]] = []
    orch = _build_orchestrator(db, captured=captured)
    await orch.invoke(user_id=user_id, prompt="hi")
    assert captured[0]["metadata"]["attached_files"] == []
