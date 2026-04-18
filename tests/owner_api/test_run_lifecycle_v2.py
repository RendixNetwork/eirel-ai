from __future__ import annotations

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from shared.common.config import Settings, reset_settings
from shared.common.database import Database
from shared.common.models import (
    DeploymentScoreRecord,
    EpochTargetSnapshot,
    EvaluationRun,
    ManagedDeployment,
    ManagedMinerSubmission,
    MinerEvaluationTask,
    SubmissionArtifact,
)
from shared.common.artifacts import create_artifact_store
from control_plane.owner_api.managed import ManagedOwnerServices
from control_plane.owner_api.deployment.runtime_manager import ManagedDeploymentRuntimeManager
from control_plane.owner_api.evaluation import run_manager as rm_module
from tests.conftest import make_submission_archive, FIXTURES_ROOT
from tests.owner_api._fake_backend import _FakeBackend


def _utcnow():
    from control_plane.owner_api._helpers import utcnow
    return utcnow()


# -- Fake evaluation bundle embedded in run metadata ----------------------
# This satisfies ensure_run_evaluation_bundle's "already built?" check so
# it never tries to load fixture files from disk.

_FAKE_BUNDLE = {
    "kind": "family_evaluation_bundle",
    "family_id": "general_chat",
    "benchmark_version": "test_v1",
    "rubric_version": "test_v1",
    "tasks": [
        {
            "task_id": "test-task-1",
            "family_id": "general_chat",
            "prompt": "Say hello",
            "expected_output": {},
        },
    ],
    "metadata": {
        "dataset_generator": {"version": "test"},
    },
}


def _run_metadata_with_bundle() -> dict:
    return {"evaluation_bundles": {"general_chat": _FAKE_BUNDLE}}


def _make_run(
    session,
    *,
    sequence: int = 1,
    status: str = "open",
    started_at=None,
    ends_at=None,
    with_bundle: bool = False,
) -> EvaluationRun:
    now = _utcnow()
    start = started_at or now
    end = ends_at or (start + timedelta(days=7))
    metadata = _run_metadata_with_bundle() if with_bundle else {}
    run = EvaluationRun(
        id=f"run-{sequence}",
        sequence=sequence,
        status=status,
        benchmark_version="test_v1",
        rubric_version="test_v1",
        judge_model="test-judge",
        min_scores_json={},
        started_at=start,
        ends_at=end,
        metadata_json=metadata,
    )
    session.add(run)
    session.flush()
    if status == "open":
        next_metadata = _run_metadata_with_bundle() if with_bundle else {}
        next_run = EvaluationRun(
            id=f"run-{sequence + 1}",
            sequence=sequence + 1,
            status="scheduled",
            benchmark_version="test_v1",
            rubric_version="test_v1",
            judge_model="test-judge",
            min_scores_json={},
            started_at=end,
            ends_at=end + timedelta(days=7),
            metadata_json=next_metadata,
        )
        session.add(next_run)
        session.flush()
    return run


@pytest.fixture
def services(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("METAGRAPH_SYNC_INTERVAL_SECONDS", "3600")
    monkeypatch.setenv("RESULT_AGGREGATION_INTERVAL_SECONDS", "3600")
    monkeypatch.setenv("USE_REDIS_POOL", "0")
    monkeypatch.setenv("VALIDATOR_EPOCH_QUORUM", "1")
    monkeypatch.setenv("METAGRAPH_SNAPSHOT_PATH", str(tmp_path / "metagraph.json"))
    monkeypatch.setenv("EIREL_PROVIDER_PROXY_URL", "http://provider-proxy.test")
    monkeypatch.setenv("EIREL_PROVIDER_PROXY_TOKEN", "provider-token")
    monkeypatch.setenv("EIREL_INTERNAL_SERVICE_TOKEN", "internal-token")
    monkeypatch.setenv("EIREL_ACTIVE_FAMILIES", "general_chat")
    monkeypatch.setenv("EIREL_LAUNCH_MODE", "1")
    monkeypatch.setenv(
        "EIREL_OWNER_DATASET_ROOT_PATH",
        str(FIXTURES_ROOT / "owner_datasets" / "families"),
    )
    reset_settings()
    settings = Settings()
    db = Database(settings.database_url)
    db.create_all()
    backend = _FakeBackend()
    runtime_mgr = ManagedDeploymentRuntimeManager(backend=backend)
    artifact_store = create_artifact_store(settings)
    svc = ManagedOwnerServices(
        db=db,
        settings=settings,
        runtime_manager=runtime_mgr,
        artifact_store=artifact_store,
    )
    yield svc
    reset_settings()


def _archive(version: str = "1.0.0") -> bytes:
    return make_submission_archive(version, family_id="general_chat")


def _register_neuron(session, hotkey: str) -> None:
    from shared.common.models import RegisteredNeuron
    if session.get(RegisteredNeuron, hotkey) is None:
        session.add(RegisteredNeuron(hotkey=hotkey, uid=0))
        session.flush()


def _submit(services, session, *, hotkey: str, block: int = 100, version: str = "1.0.0"):
    _register_neuron(session, hotkey)
    return services.submissions.create_submission(
        session,
        miner_hotkey=hotkey,
        submission_block=block,
        archive_bytes=_archive(version),
        base_url="http://test",
    )


def _make_scored_deployment(
    session,
    *,
    run_id: str,
    hotkey: str,
    family_id: str = "general_chat",
    raw_score: float = 0.8,
) -> tuple[ManagedMinerSubmission, ManagedDeployment]:
    """Insert a submission + deployment + score record directly in DB."""
    artifact = SubmissionArtifact(
        archive_bytes=b"fake",
        sha256="fakehash" + uuid4().hex[:8],
        size_bytes=4,
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
        introduced_run_id=run_id,
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
        status="deployed_for_eval",
        health_status="healthy",
        health_details_json={},
        placement_status="placed",
        benchmark_version="test_v1",
        rubric_version="test_v1",
        judge_model="test-judge",
    )
    session.add(deployment)
    session.flush()

    score = DeploymentScoreRecord(
        run_id=run_id,
        family_id=family_id,
        deployment_id=deployment.id,
        submission_id=submission.id,
        miner_hotkey=hotkey,
        deployment_revision=deployment.deployment_revision,
        raw_score=raw_score,
        normalized_score=1.0,
        is_eligible=True,
    )
    session.add(score)
    session.flush()

    return submission, deployment


# =========================================================================
# Tests
# =========================================================================


def test_submission_targets_next_run_during_open_run(services):
    with services.db.sessionmaker() as session:
        run1 = _make_run(session)
        session.commit()

    with services.db.sessionmaker() as session:
        target = services.runs.submission_target_run(session)

    assert target.sequence == 2
    assert target.status == "scheduled"
    assert target.id != "run-1"


def test_submission_targets_scheduled_run_before_first_run(services, monkeypatch):
    future = (_utcnow() + timedelta(days=30)).isoformat()
    monkeypatch.setenv("EIREL_FIRST_RUN_START_TIME", future)
    reset_settings()
    # Re-read settings so the services picks up the new value.
    services.settings = Settings()

    with services.db.sessionmaker() as session:
        target = services.runs.submission_target_run(session)

    assert target.sequence == 1
    assert target.status == "scheduled"
    assert target.id == "run-1"


def test_queued_submissions_deploy_when_run_opens(services, monkeypatch):
    with services.db.sessionmaker() as session:
        run1 = _make_run(session, with_bundle=True)
        run1_ends_at = run1.ends_at
        session.commit()

    hk = "5HKQueue" + "Q" * 40
    with services.db.sessionmaker() as session:
        sub, dep = _submit(services, session, hotkey=hk)
    dep_id = dep.id
    assert dep.status == "queued"

    # Advance time past Run 1's end and bypass benchmark loading.
    future = run1_ends_at + timedelta(hours=1)
    monkeypatch.setattr(rm_module, "_utcnow", lambda: future)
    monkeypatch.setattr(
        rm_module.RunManager, "_initialize_run_benchmarks",
        lambda self, session, *, run: None,
    )

    with services.db.sessionmaker() as session:
        current = services.runs.ensure_current_run(session)

    assert current.sequence == 2
    assert current.status == "open"

    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        assert dep.status == "received"
        assert dep.health_status == "starting"


def test_topk_carryover_plus_queued_in_candidate_pool(services):
    with services.db.sessionmaker() as session:
        run1 = _make_run(session)
        session.commit()

    # Miner A: scored winner in Run 1 (carryover candidate)
    scored_hk = "5HKScor" + "A" * 41
    with services.db.sessionmaker() as session:
        _, scored_dep = _make_scored_deployment(
            session, run_id="run-1", hotkey=scored_hk,
        )
        scored_dep_id = scored_dep.id
        session.commit()

    # Miner B: submits during Run 1, targets Run 2 (queued candidate)
    queued_hk = "5HKPool" + "B" * 41
    with services.db.sessionmaker() as session:
        _, queued_dep = _submit(services, session, hotkey=queued_hk)
    queued_dep_id = queued_dep.id

    with services.db.sessionmaker() as session:
        run2 = session.get(EvaluationRun, "run-2")
        pool = services.runs._candidate_pool_deployments(
            session, run=run2, family_id="general_chat",
        )
        pool_ids = {d.id for d in pool}

    assert scored_dep_id in pool_ids, "carryover deployment missing from pool"
    assert queued_dep_id in pool_ids, "queued deployment missing from pool"


def test_snapshot_starts_pending_deployments(services):
    with services.db.sessionmaker() as session:
        run1 = _make_run(session, with_bundle=True)
        session.commit()

    hk = "5HKSnap" + "S" * 41
    with services.db.sessionmaker() as session:
        sub, dep = _submit(services, session, hotkey=hk)

    with services.db.sessionmaker() as session:
        # Start queued deployments so they're in the candidate pool
        run2 = session.get(EvaluationRun, "run-2")
        services.runs.start_queued_deployments(session, run=run2)
        session.commit()

    with services.db.sessionmaker() as session:
        snapshot = services.runs.freeze_run_targets(
            session,
            run_id="run-2",
            family_id="general_chat",
            base_url="http://localhost:8000",
        )

    assert snapshot.status == "pending_deployments"


def test_claim_returns_empty_while_pending_deployments(services):
    with services.db.sessionmaker() as session:
        run1 = _make_run(session, with_bundle=True)
        session.commit()

    hk = "5HKClaim" + "C" * 40
    with services.db.sessionmaker() as session:
        sub, dep = _submit(services, session, hotkey=hk)

    with services.db.sessionmaker() as session:
        run2 = session.get(EvaluationRun, "run-2")
        services.runs.start_queued_deployments(session, run=run2)
        session.commit()

    with services.db.sessionmaker() as session:
        services.runs.freeze_run_targets(
            session,
            run_id="run-2",
            family_id="general_chat",
            base_url="http://localhost:8000",
        )

    with services.db.sessionmaker() as session:
        claimed = services.evaluation_tasks.claim_tasks(
            session,
            run_id="run-2",
            family_id="general_chat",
            validator_hotkey="validator-test-hotkey",
        )

    assert claimed == []


def test_snapshot_opens_when_all_deployments_healthy(services):
    with services.db.sessionmaker() as session:
        run1 = _make_run(session, with_bundle=True)
        session.commit()

    hk = "5HKOpen" + "H" * 41
    with services.db.sessionmaker() as session:
        sub, dep = _submit(services, session, hotkey=hk)
    dep_id = dep.id

    with services.db.sessionmaker() as session:
        run2 = session.get(EvaluationRun, "run-2")
        services.runs.start_queued_deployments(session, run=run2)
        session.commit()

    with services.db.sessionmaker() as session:
        services.runs.freeze_run_targets(
            session,
            run_id="run-2",
            family_id="general_chat",
            base_url="http://localhost:8000",
        )

    # Mark deployment as healthy
    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        dep.health_status = "healthy"
        session.commit()

    with services.db.sessionmaker() as session:
        services.runs.check_and_open_pending_snapshots(session)

    with services.db.sessionmaker() as session:
        snapshot = session.execute(
            EpochTargetSnapshot.__table__.select().where(
                EpochTargetSnapshot.run_id == "run-2",
            )
        ).first()
        # Re-query via ORM
        snapshot = session.get(EpochTargetSnapshot, snapshot.id)
        assert snapshot.status == "open"

    # Now claim_tasks should be able to return tasks (if any were created)
    with services.db.sessionmaker() as session:
        claimed = services.evaluation_tasks.claim_tasks(
            session,
            run_id="run-2",
            family_id="general_chat",
            validator_hotkey="validator-test-hotkey",
        )
        # Tasks were created by freeze_run_targets → initialize_evaluation_tasks
        assert len(claimed) >= 1


def test_snapshot_opens_when_failed_deployments_are_terminal(services):
    with services.db.sessionmaker() as session:
        run1 = _make_run(session, with_bundle=True)
        session.commit()

    # Two miners submit
    hk_ok = "5HKOk00" + "O" * 40
    hk_fail = "5HKFail" + "F" * 41
    with services.db.sessionmaker() as session:
        _, dep_ok = _submit(services, session, hotkey=hk_ok)
    dep_ok_id = dep_ok.id
    with services.db.sessionmaker() as session:
        _, dep_fail = _submit(services, session, hotkey=hk_fail, block=101, version="2.0.0")
    dep_fail_id = dep_fail.id

    with services.db.sessionmaker() as session:
        run2 = session.get(EvaluationRun, "run-2")
        services.runs.start_queued_deployments(session, run=run2)
        session.commit()

    with services.db.sessionmaker() as session:
        services.runs.freeze_run_targets(
            session,
            run_id="run-2",
            family_id="general_chat",
            base_url="http://localhost:8000",
        )

    # One healthy, one build_failed (terminal)
    with services.db.sessionmaker() as session:
        dep_ok = session.get(ManagedDeployment, dep_ok_id)
        dep_ok.health_status = "healthy"
        dep_fail = session.get(ManagedDeployment, dep_fail_id)
        dep_fail.status = "build_failed"
        dep_fail.health_status = "failed"
        session.commit()

    with services.db.sessionmaker() as session:
        services.runs.check_and_open_pending_snapshots(session)

    with services.db.sessionmaker() as session:
        snapshot = session.execute(
            EpochTargetSnapshot.__table__.select().where(
                EpochTargetSnapshot.run_id == "run-2",
            )
        ).first()
        snapshot = session.get(EpochTargetSnapshot, snapshot.id)
        assert snapshot.status == "open"


def test_resubmit_during_run_retires_previous_queues_new(services):
    with services.db.sessionmaker() as session:
        _make_run(session)
        session.commit()

    hk = "5HKResub" + "R" * 40
    with services.db.sessionmaker() as session:
        sub1, dep1 = _submit(services, session, hotkey=hk, version="1.0.0")
    dep1_id = dep1.id
    assert dep1.status == "queued"

    with services.db.sessionmaker() as session:
        sub2, dep2 = _submit(services, session, hotkey=hk, block=101, version="2.0.0")
    dep2_id = dep2.id

    with services.db.sessionmaker() as session:
        old_dep = session.get(ManagedDeployment, dep1_id)
        assert old_dep.status == "retired"

        new_dep = session.get(ManagedDeployment, dep2_id)
        assert new_dep.status == "queued"
        # New submission should also target Run 2
        sub = session.get(ManagedMinerSubmission, new_dep.submission_id)
        assert sub.introduced_run_id == "run-2"
