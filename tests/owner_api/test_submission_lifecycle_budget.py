from __future__ import annotations

from datetime import timedelta

import pytest

from shared.common.config import Settings, reset_settings
from shared.common.database import Database
from shared.common.models import (
    EvaluationRun,
    ManagedDeployment,
    ManagedMinerSubmission,
)
from shared.common.artifacts import create_artifact_store
from control_plane.owner_api.managed import ManagedOwnerServices
from control_plane.owner_api.deployment.runtime_manager import ManagedDeploymentRuntimeManager
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
        # Create next scheduled run so submission_target_run() doesn't
        # call create_run() (which needs evaluation fixtures)
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


def test_new_submission_enters_queued_status(services):
    with services.db.sessionmaker() as session:
        _make_run(session)
        session.commit()
    with services.db.sessionmaker() as session:
        sub, dep = _submit(services, session, hotkey="5HotKey" + "A" * 42)
    assert dep.status == "queued"
    assert dep.placement_status == "queued"
    assert dep.health_status == "queued"


def test_no_runtime_spinup_for_queued_deployment(services):
    with services.db.sessionmaker() as session:
        _make_run(session)
        session.commit()
    with services.db.sessionmaker() as session:
        sub, dep = _submit(services, session, hotkey="5HotKey" + "B" * 42)
    handle = services.runtime_manager.runtime_handle(dep.id)
    assert handle is None


def test_run_open_starts_queued_deployments(services):
    with services.db.sessionmaker() as session:
        run = _make_run(session)
        session.commit()
    with services.db.sessionmaker() as session:
        sub, dep = _submit(services, session, hotkey="5HotKey" + "C" * 42)
    dep_id = dep.id
    assert dep.status == "queued"

    # Submissions during an open run target the next scheduled run (run-2)
    with services.db.sessionmaker() as session:
        run2 = session.get(EvaluationRun, "run-2")
        started = services.runs.start_queued_deployments(session, run=run2)
        session.commit()

    assert dep_id in started
    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        assert dep.status == "received"
        assert dep.placement_status == "pending"
        assert dep.health_status == "starting"


def test_start_queued_only_affects_matching_run(services):
    # Run 1 (open) auto-creates run-2 (scheduled).
    # Submission during run-1 targets run-2.
    with services.db.sessionmaker() as session:
        run1 = _make_run(session, sequence=1)
        session.commit()

    with services.db.sessionmaker() as session:
        sub, dep = _submit(services, session, hotkey="5HotKey" + "D" * 42)
    dep_id = dep.id

    # Starting deployments for run-1 should find nothing (submission targets run-2)
    with services.db.sessionmaker() as session:
        run1 = session.get(EvaluationRun, "run-1")
        started = services.runs.start_queued_deployments(session, run=run1)
        session.commit()

    assert started == []
    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        assert dep.status == "queued"


def test_close_run_keeps_unevaluated_deployments(services):
    """Deployments introduced in a run but never evaluated are kept (N+1 model)."""
    with services.db.sessionmaker() as session:
        run = _make_run(session)
        session.commit()

    dep_ids = []
    with services.db.sessionmaker() as session:
        for i in range(3):
            hk = f"5HK{i:04d}" + "X" * 42
            sub, dep = _submit(services, session, hotkey=hk, block=100 + i, version=f"1.0.{i}")
            dep_ids.append(dep.id)

    # Submissions during open run-1 target run-2
    with services.db.sessionmaker() as session:
        run2 = session.get(EvaluationRun, "run-2")
        services.runs.start_queued_deployments(session, run=run2)
        session.commit()

    with services.db.sessionmaker() as session:
        services.runs.close_run(session, run_id="run-1")
        session.commit()

    with services.db.sessionmaker() as session:
        for dep_id in dep_ids:
            dep = session.get(ManagedDeployment, dep_id)
            assert dep.status != "retired", (
                f"unevaluated deployment {dep_id} should not be retired at run close"
            )


def test_close_run_completes_successfully(services):
    """Run can close cleanly even when deployments are in received status."""
    with services.db.sessionmaker() as session:
        run = _make_run(session)
        session.commit()

    with services.db.sessionmaker() as session:
        sub, dep = _submit(services, session, hotkey="5HKClose" + "C" * 40)

    # Submissions during open run-1 target run-2
    with services.db.sessionmaker() as session:
        run2 = session.get(EvaluationRun, "run-2")
        services.runs.start_queued_deployments(session, run=run2)
        session.commit()

    with services.db.sessionmaker() as session:
        run = services.runs.close_run(session, run_id="run-1")
        session.commit()

    assert run.status == "completed"


def test_managed_miner_submission_rows_never_deleted(services):
    with services.db.sessionmaker() as session:
        _make_run(session)
        session.commit()

    with services.db.sessionmaker() as session:
        sub, dep = _submit(services, session, hotkey="5HKPerm" + "Z" * 41)
    sub_id = sub.id

    # Submissions during open run-1 target run-2
    with services.db.sessionmaker() as session:
        run2 = session.get(EvaluationRun, "run-2")
        services.runs.start_queued_deployments(session, run=run2)
        session.commit()

    with services.db.sessionmaker() as session:
        services.runs.close_run(session, run_id="run-1")
        session.commit()

    with services.db.sessionmaker() as session:
        sub = session.get(ManagedMinerSubmission, sub_id)
        assert sub is not None


def test_resubmit_retires_prior_deployment(services):
    with services.db.sessionmaker() as session:
        _make_run(session)
        session.commit()

    hk = "5HKResub" + "R" * 40
    with services.db.sessionmaker() as session:
        sub1, dep1 = _submit(services, session, hotkey=hk, version="1.0.0")
    dep1_id = dep1.id
    sub1_id = sub1.id

    with services.db.sessionmaker() as session:
        sub2, dep2 = _submit(services, session, hotkey=hk, block=101, version="2.0.0")
    dep2_id = dep2.id

    with services.db.sessionmaker() as session:
        old_dep = session.get(ManagedDeployment, dep1_id)
        assert old_dep.status == "retired"
        old_sub = session.get(ManagedMinerSubmission, sub1_id)
        assert old_sub.status == "retired"
        new_dep = session.get(ManagedDeployment, dep2_id)
        assert new_dep.status == "queued"


def test_rebalance_family_excludes_queued_deployments(services):
    with services.db.sessionmaker() as session:
        _make_run(session)
        session.commit()

    with services.db.sessionmaker() as session:
        sub, dep = _submit(services, session, hotkey="5HKRebal" + "Q" * 40)
    dep_id = dep.id

    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        assert dep.status == "queued"
        assert dep.is_active is False
        assert dep.active_set_rank is None


def test_queued_to_received_transition_valid():
    from control_plane.owner_api.deployment.deployment_manager import (
        _VALID_DEPLOYMENT_TRANSITIONS,
    )

    assert "queued" in _VALID_DEPLOYMENT_TRANSITIONS
    assert "received" in _VALID_DEPLOYMENT_TRANSITIONS["queued"]
    assert "retired" in _VALID_DEPLOYMENT_TRANSITIONS["queued"]


async def test_ensure_current_run_and_reconcile_spins_up_queued_deployment(services):
    with services.db.sessionmaker() as session:
        _make_run(session)
        session.commit()

    with services.db.sessionmaker() as session:
        sub, dep = _submit(services, session, hotkey="5HKWire" + "E" * 41)
    dep_id = dep.id
    assert dep.status == "queued"

    backend = services.runtime_manager.backend
    assert backend.runtime_handle(dep_id) is None

    # Submissions during open run-1 target run-2
    with services.db.sessionmaker() as session:
        run2 = session.get(EvaluationRun, "run-2")
        services.runs.start_queued_deployments(session, run=run2)
        session.commit()

    assert dep_id in services.runs._pending_queued_deployment_ids

    await services.deployments.ensure_current_run_and_reconcile()

    assert services.runs._pending_queued_deployment_ids == []
    assert backend.runtime_handle(dep_id) is not None
    with services.db.sessionmaker() as session:
        refreshed = session.get(ManagedDeployment, dep_id)
        assert refreshed.status == "deployed_for_eval"


async def test_ensure_current_run_and_reconcile_no_queued_is_noop(services):
    with services.db.sessionmaker() as session:
        _make_run(session)
        session.commit()

    backend = services.runtime_manager.backend
    await services.deployments.ensure_current_run_and_reconcile()
    assert services.runs._pending_queued_deployment_ids == []
    assert backend._records == {}
