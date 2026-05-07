"""Tests for ServingPicker (product-mode picker reading ServingDeployment)."""
from __future__ import annotations

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from shared.common.database import Database
from shared.common.models import (
    ManagedDeployment,
    ManagedMinerSubmission,
    ServingDeployment,
    ServingRelease,
)
from orchestration.orchestrator.serving_picker import (
    NoEligibleServingDeploymentError,
    ServingPicker,
)


def _make_db(tmp_path) -> Database:
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'sp.db'}")
    db.create_all()
    return db


def _seed_serving(
    session,
    *,
    deployment_id: str,
    miner_hotkey: str,
    family_id: str = "general_chat",
    status: str = "healthy",
    health: str = "healthy",
    runtime_kind: str = "graph",
    published_at: datetime | None = None,
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
    source_deployment_id = str(uuid4())
    session.add(
        ManagedDeployment(
            id=source_deployment_id,
            submission_id=submission_id,
            miner_hotkey=miner_hotkey,
            family_id=family_id,
            deployment_revision=str(uuid4()),
            image_ref="img:test",
            endpoint=f"http://eval-{deployment_id}.test:8080",
            status="active",
            health_status="healthy",
            placement_status="placed",
        )
    )
    release_id = str(uuid4())
    session.add(ServingRelease(
        id=release_id,
        trigger_type="test",
        status="published",
        published_at=published_at or datetime.utcnow(),
    ))
    session.add(ServingDeployment(
        id=deployment_id,
        release_id=release_id,
        family_id=family_id,
        source_deployment_id=source_deployment_id,
        source_submission_id=submission_id,
        miner_hotkey=miner_hotkey,
        source_deployment_revision=str(uuid4()),
        endpoint=f"http://serving-{deployment_id}.test:8080",
        status=status,
        health_status=health,
        published_at=published_at or datetime.utcnow(),
    ))
    session.commit()
    return deployment_id


def test_serving_picker_returns_healthy_deployment(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_serving(session, deployment_id="serving-1", miner_hotkey="hk1")
    picker = ServingPicker(database=db)
    cand = picker.pick(family_id="general_chat")
    assert cand.deployment_id == "serving-1"
    assert cand.miner_hotkey == "hk1"
    assert cand.runtime_kind == "graph"


def test_serving_picker_skips_unhealthy(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_serving(session, deployment_id="bad", miner_hotkey="hk1", health="degraded")
        _seed_serving(session, deployment_id="good", miner_hotkey="hk2")
    picker = ServingPicker(database=db)
    cand = picker.pick(family_id="general_chat")
    assert cand.deployment_id == "good"


def test_serving_picker_raises_when_no_eligible(tmp_path):
    db = _make_db(tmp_path)
    picker = ServingPicker(database=db)
    with pytest.raises(NoEligibleServingDeploymentError):
        picker.pick(family_id="general_chat")


def test_serving_picker_prefers_most_recently_published(tmp_path):
    """No thread pinning: ranking is by published_at desc (newer wins)."""
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_serving(
            session, deployment_id="old", miner_hotkey="hk1",
            published_at=datetime.utcnow() - timedelta(days=3),
        )
        _seed_serving(
            session, deployment_id="new", miner_hotkey="hk2",
            published_at=datetime.utcnow(),
        )
    picker = ServingPicker(database=db, top_k=1)
    cand = picker.pick(family_id="general_chat")
    assert cand.deployment_id == "new"


def test_serving_picker_no_thread_pinning(tmp_path):
    """Product mode: deployment can change between turns. ServingPicker
    must NOT accept a thread_id arg — verifies the API surface stays
    deliberately distinct from MinerPicker."""
    picker = ServingPicker(database=_make_db(tmp_path))
    with pytest.raises(TypeError):
        picker.pick(family_id="general_chat", thread_id="t1")  # type: ignore[call-arg]
