from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from shared.common.database import Database
from shared.common.models import (
    ManagedDeployment,
    ManagedMinerSubmission,
    SubmissionArtifact,
)
from control_plane.owner_api.routers.health import (
    _format_deployment_info_metrics,
)


def _make_db(tmp_path) -> Database:
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'deployment_info.db'}")
    db.create_all()
    return db


def _seed_deployment(
    session,
    *,
    hotkey: str,
    family_id: str = "general_chat",
    status: str = "deployed_for_eval",
    health: str = "healthy",
    node: str | None = "sky-dev",
    is_active: bool = True,
    retired_at: datetime | None = None,
) -> ManagedDeployment:
    artifact = SubmissionArtifact(
        archive_bytes=b"x",
        sha256="sha-" + uuid4().hex[:8],
        size_bytes=1,
        manifest_json={},
    )
    session.add(artifact)
    session.flush()
    submission = ManagedMinerSubmission(
        miner_hotkey=hotkey,
        submission_seq=1,
        family_id=family_id,
        status="received",
        artifact_id=artifact.id,
        manifest_json={},
        archive_sha256=artifact.sha256,
        submission_block=100,
    )
    session.add(submission)
    session.flush()
    deployment = ManagedDeployment(
        submission_id=submission.id,
        miner_hotkey=hotkey,
        family_id=family_id,
        deployment_revision="rev-" + uuid4().hex[:8],
        image_ref="managed://test",
        endpoint="http://test",
        status=status,
        health_status=health,
        assigned_node_name=node,
        is_active=is_active,
        retired_at=retired_at,
    )
    session.add(deployment)
    session.flush()
    return deployment


def _collect(session) -> list[tuple]:
    return list(
        session.query(
            ManagedDeployment.id,
            ManagedDeployment.submission_id,
            ManagedDeployment.miner_hotkey,
            ManagedDeployment.family_id,
            ManagedDeployment.status,
            ManagedDeployment.health_status,
            ManagedDeployment.assigned_node_name,
            ManagedDeployment.is_active,
        )
        .filter(ManagedDeployment.retired_at.is_(None))
        .all()
    )


def test_deployment_info_emits_one_row_per_active_deployment(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        dep_a = _seed_deployment(session, hotkey="5DqP", node="sky-dev")
        dep_b = _seed_deployment(session, hotkey="5EnoN", node="sky2")
        session.commit()
        rows = _collect(session)

    body = _format_deployment_info_metrics(rows)
    assert "# TYPE eirel_owner_deployment_info gauge" in body
    assert (
        f'deployment_id="{dep_a.id}",submission_id="{dep_a.submission_id}",'
        f'hotkey="5DqP",family="general_chat",status="deployed_for_eval",'
        f'health="healthy",node="sky-dev",is_active="1"' in body
    )
    assert (
        f'deployment_id="{dep_b.id}",submission_id="{dep_b.submission_id}",'
        f'hotkey="5EnoN"' in body
    )
    # Exactly two metric rows plus HELP/TYPE/blank.
    assert body.count("eirel_owner_deployment_info{") == 2


def test_deployment_info_skips_retired(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_deployment(session, hotkey="5Live")
        _seed_deployment(
            session,
            hotkey="5Gone",
            retired_at=datetime.now(UTC).replace(tzinfo=None),
        )
        session.commit()
        rows = _collect(session)

    body = _format_deployment_info_metrics(rows)
    assert '"5Live"' in body
    assert "5Gone" not in body


def test_deployment_info_empty_node_emits_blank_label(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_deployment(session, hotkey="5Pend", status="pending_capacity", node=None)
        session.commit()
        rows = _collect(session)

    body = _format_deployment_info_metrics(rows)
    assert 'node=""' in body
    assert 'status="pending_capacity"' in body


def test_deployment_info_empty_when_no_rows(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        rows = _collect(session)
    assert _format_deployment_info_metrics(rows) == ""
