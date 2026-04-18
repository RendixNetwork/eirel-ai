from __future__ import annotations

import inspect
from datetime import timedelta

import pytest
from sqlalchemy import select

from shared.common.config import Settings, reset_settings
from shared.common.database import Database
from shared.common.models import (
    EpochTargetSnapshot,
    EvaluationRun,
    ManagedDeployment,
    ManagedMinerSubmission,
)
from shared.common.artifacts import create_artifact_store
from control_plane.owner_api.managed import ManagedOwnerServices
from control_plane.owner_api.deployment.runtime_manager import ManagedDeploymentRuntimeManager
from control_plane.owner_api.operations.cleanup_tasks import _reap_pending_runtime_stops
from tests.conftest import make_submission_archive, FIXTURES_ROOT
from tests.owner_api._fake_backend import _FakeBackend


def _utcnow():
    from control_plane.owner_api._helpers import utcnow
    return utcnow()


def _make_run(session, *, sequence: int = 1, status: str = "open") -> EvaluationRun:
    now = _utcnow()
    run = EvaluationRun(
        id=f"run-{sequence}",
        sequence=sequence,
        status=status,
        benchmark_version="test_v1",
        rubric_version="test_v1",
        judge_model="test-judge",
        min_scores_json={},
        started_at=now,
        ends_at=now + timedelta(days=7),
        metadata_json={},
    )
    session.add(run)
    session.flush()
    if status == "open":
        next_run = EvaluationRun(
            id=f"run-{sequence + 1}",
            sequence=sequence + 1,
            status="scheduled",
            benchmark_version="test_v1",
            rubric_version="test_v1",
            judge_model="test-judge",
            min_scores_json={},
            started_at=run.ends_at,
            ends_at=run.ends_at + timedelta(days=7),
            metadata_json={},
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


# -- Fix 7: Duplicate snapshot handled gracefully -------------------------

def test_duplicate_snapshot_handled_gracefully(services, monkeypatch):
    monkeypatch.setattr(
        services.evaluation_tasks, "initialize_evaluation_tasks",
        lambda session, *, run_id, family_id, snapshot: 0,
    )
    with services.db.sessionmaker() as session:
        _make_run(session)
        session.commit()

    with services.db.sessionmaker() as session:
        snap1 = services.runs.freeze_run_targets(
            session,
            run_id="run-1",
            family_id="general_chat",
            base_url="http://test",
        )
    snap1_id = snap1.id

    # Second call for same (run_id, family_id) returns existing snapshot
    with services.db.sessionmaker() as session:
        snap2 = services.runs.freeze_run_targets(
            session,
            run_id="run-1",
            family_id="general_chat",
            base_url="http://test",
        )
    snap2_id = snap2.id

    assert snap1_id == snap2_id

    with services.db.sessionmaker() as session:
        snapshots = session.execute(
            select(EpochTargetSnapshot).where(
                EpochTargetSnapshot.run_id == "run-1",
                EpochTargetSnapshot.family_id == "general_chat",
            )
        ).scalars().all()
        assert len(snapshots) == 1


# -- Fix 8: K8s partial deploy cleanup ------------------------------------

def test_k8s_partial_deploy_cleanup():
    from infra.miner_runtime.runtime_manager import KubernetesMinerRuntimeManager

    source = inspect.getsource(KubernetesMinerRuntimeManager.ensure_runtime)
    assert "partial_deploy_cleanup" in source, (
        "KubernetesMinerRuntimeManager.ensure_runtime must call "
        "stop_runtime with reason 'partial_deploy_cleanup' on failure"
    )
    assert "stop_runtime" in source


# -- Fix 10: Config validation ---------------------------------------------

def test_config_rejects_negative_interval(monkeypatch):
    monkeypatch.setenv("EIREL_OWNER_RUNTIME_REAPER_INTERVAL_SECONDS", "-1")
    reset_settings()
    with pytest.raises(ValueError, match="must be positive"):
        Settings()


def test_config_rejects_zero_cpu(monkeypatch):
    monkeypatch.setenv("EIREL_OWNER_RUNTIME_SUBMISSION_CPU_MILLIS", "0")
    reset_settings()
    with pytest.raises(ValueError, match="must be positive"):
        Settings()


# -- Fix 11: Reaper handles already-deleted deployment ---------------------

async def test_reaper_handles_already_deleted_deployment(services):
    with services.db.sessionmaker() as session:
        _make_run(session)
        session.commit()

    hk = "5HKReaper" + "R" * 39
    with services.db.sessionmaker() as session:
        sub, dep = _submit(services, session, hotkey=hk)
    dep_id = dep.id

    # Mark deployment as needing runtime stop
    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        dep.pending_runtime_stop = True
        session.commit()

    # Simulate container already deleted — backend has no record
    backend = services.runtime_manager.backend
    backend._records.pop(dep_id, None)

    # Reaper should complete without error
    await _reap_pending_runtime_stops(services)

    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        assert dep.pending_runtime_stop is False


# -- Fix 6: FOR UPDATE on latest submission --------------------------------

def test_for_update_on_latest_submission(services):
    from control_plane.owner_api.operations.submission_manager import SubmissionManager

    source = inspect.getsource(SubmissionManager.latest_submission_for_hotkey)
    assert "with_for_update" in source, (
        "latest_submission_for_hotkey must use .with_for_update() "
        "to serialize concurrent submissions"
    )

    # Functional: sequential submissions get different seq values
    with services.db.sessionmaker() as session:
        _make_run(session)
        session.commit()

    hk = "5HKSeq00" + "S" * 40
    with services.db.sessionmaker() as session:
        sub1, _ = _submit(services, session, hotkey=hk, version="1.0.0")
    seq1 = sub1.submission_seq

    with services.db.sessionmaker() as session:
        sub2, _ = _submit(services, session, hotkey=hk, block=101, version="2.0.0")
    seq2 = sub2.submission_seq

    assert seq2 == seq1 + 1
