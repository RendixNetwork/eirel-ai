from __future__ import annotations

from datetime import timedelta

import pytest

from shared.common.config import Settings, reset_settings
from shared.common.database import Database
from shared.common.models import (
    EvaluationRun,
    ManagedDeployment,
)
from shared.common.artifacts import create_artifact_store
from control_plane.owner_api.managed import ManagedOwnerServices
from control_plane.owner_api.deployment.runtime_manager import ManagedDeploymentRuntimeManager
from control_plane.owner_api.operations.cleanup_tasks import (
    _reap_pending_runtime_stops,
    _retry_pending_capacity,
)
from infra.miner_runtime._k8s_helpers import DeploymentStatus, DeploymentStatusCode
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


@pytest.fixture
def make_services(tmp_path, monkeypatch):
    monkeypatch.setenv("EIREL_ACTIVE_FAMILIES", "general_chat")
    monkeypatch.setenv("EIREL_LAUNCH_MODE", "1")

    def factory(*, backend=None):
        if backend is None:
            backend = _FakeBackend()
        reset_settings()
        settings = Settings()
        db = Database(settings.database_url)
        db.create_all()
        runtime_mgr = ManagedDeploymentRuntimeManager(backend=backend)
        artifact_store = create_artifact_store(settings)
        return ManagedOwnerServices(
            db=db,
            settings=settings,
            runtime_manager=runtime_mgr,
            artifact_store=artifact_store,
        )

    yield factory
    reset_settings()


async def _deploy(services, *, hotkey: str, version: str = "1.0.0", block: int = 100) -> str:
    with services.db.sessionmaker() as session:
        _sub, dep = _submit(services, session, hotkey=hotkey, version=version, block=block)
    dep_id = dep.id
    with services.db.sessionmaker() as session:
        run = session.get(EvaluationRun, "run-2")
        services.runs.start_queued_deployments(session, run=run)
        session.commit()
    await services.deployments.ensure_current_run_and_reconcile()
    return dep_id


# -- test_resubmit_sets_pending_runtime_stop_when_backend_raises -----


async def test_resubmit_sets_pending_runtime_stop_when_backend_raises(make_services):
    services = make_services(backend=_FakeBackend(stop_error=RuntimeError("boom")))
    backend = services.runtime_manager.backend

    with services.db.sessionmaker() as session:
        _make_run(session)
        session.commit()

    hk = "5HKStopErr" + "S" * 38
    dep1_id = await _deploy(services, hotkey=hk, version="1.0.0")

    with services.db.sessionmaker() as session:
        dep1 = session.get(ManagedDeployment, dep1_id)
        assert dep1.status == "deployed_for_eval"

    with services.db.sessionmaker() as session:
        _submit(services, session, hotkey=hk, block=101, version="2.0.0")

    with services.db.sessionmaker() as session:
        dep1 = session.get(ManagedDeployment, dep1_id)
        assert dep1.status == "retired"
        assert dep1.pending_runtime_stop is True

    assert dep1_id not in backend.stopped


# -- test_reaper_clears_flag_on_success ------------------------------


async def test_reaper_clears_flag_on_success(make_services):
    services = make_services()
    backend = services.runtime_manager.backend

    with services.db.sessionmaker() as session:
        _make_run(session)
        session.commit()

    hk = "5HKReaper" + "R" * 39
    dep_id = await _deploy(services, hotkey=hk)

    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        dep.pending_runtime_stop = True
        session.commit()

    await _reap_pending_runtime_stops(services)

    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        assert dep.pending_runtime_stop is False
    assert dep_id in backend.stopped


# -- test_reaper_idempotent_when_runtime_handle_already_gone ---------


async def test_reaper_idempotent_when_runtime_handle_already_gone(make_services):
    services = make_services()
    backend = services.runtime_manager.backend

    with services.db.sessionmaker() as session:
        _make_run(session)
        session.commit()

    hk = "5HKGone__" + "G" * 39
    dep_id = await _deploy(services, hotkey=hk)

    backend._records.pop(dep_id, None)

    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        dep.pending_runtime_stop = True
        session.commit()

    await _reap_pending_runtime_stops(services)

    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        assert dep.pending_runtime_stop is False


# -- test_reaper_leaves_flag_set_on_failure --------------------------


async def test_reaper_leaves_flag_set_on_failure(make_services):
    services = make_services(backend=_FakeBackend(stop_error=RuntimeError("transient")))

    with services.db.sessionmaker() as session:
        _make_run(session)
        session.commit()

    hk = "5HKFail__" + "F" * 39
    dep_id = await _deploy(services, hotkey=hk)

    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        dep.pending_runtime_stop = True
        session.commit()

    await _reap_pending_runtime_stops(services)

    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        assert dep.pending_runtime_stop is True


# -- test_pending_capacity_retry_on_tick -----------------------------


async def test_pending_capacity_retry_on_tick(make_services):
    backend = _FakeBackend(available_pods=0)
    services = make_services(backend=backend)

    with services.db.sessionmaker() as session:
        _make_run(session)
        session.commit()

    hk = "5HKCap___" + "C" * 39
    with services.db.sessionmaker() as session:
        _sub, dep = _submit(services, session, hotkey=hk)
    dep_id = dep.id

    with services.db.sessionmaker() as session:
        run = session.get(EvaluationRun, "run-2")
        services.runs.start_queued_deployments(session, run=run)
        session.commit()

    await services.deployments.ensure_current_run_and_reconcile()

    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        assert dep.placement_status == "pending_capacity"

    backend._available_pods = 4

    await _retry_pending_capacity(services)

    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        assert dep.status in ("building", "deployed_for_eval", "active")
        assert dep.placement_status != "pending_capacity"


# -- test_pending_capacity_retry_skips_build_failed ------------------


async def test_pending_capacity_retry_skips_build_failed(make_services):
    services = make_services()

    with services.db.sessionmaker() as session:
        _make_run(session)
        session.commit()

    hk = "5HKBldFl_" + "B" * 39
    with services.db.sessionmaker() as session:
        _sub, dep = _submit(services, session, hotkey=hk)
    dep_id = dep.id

    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        dep.status = "build_failed"
        dep.placement_status = "pending_capacity"
        dep.health_status = "unhealthy"
        session.commit()

    await _retry_pending_capacity(services)

    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        assert dep.status == "build_failed"
        assert dep.placement_status == "pending_capacity"


# -- test_pending_unschedulable_recovers_when_capacity_returns ---------


async def test_pending_unschedulable_recovers_when_capacity_returns(make_services):
    backend = _FakeBackend(
        status_override=DeploymentStatus(
            code=DeploymentStatusCode.PENDING_UNSCHEDULABLE,
            ready_replicas=0,
            desired_replicas=1,
            message="unschedulable",
            last_pod_phase="Pending",
        ),
    )
    services = make_services(backend=backend)

    with services.db.sessionmaker() as session:
        _make_run(session)
        session.commit()

    hk = "5HKUnsRe_" + "X" * 39
    dep_id = await _deploy(services, hotkey=hk)

    await services.deployments.recover_or_demote_deployment(
        deployment_id=dep_id, reason="k8s_unschedulable",
    )

    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        assert dep.placement_status == "pending_capacity"
        assert dep.health_status == "unhealthy"

    backend._records.pop(dep_id, None)

    backend.status_override = DeploymentStatus(
        code=DeploymentStatusCode.READY,
        ready_replicas=1,
        desired_replicas=1,
        message="ok",
        last_pod_phase="Running",
    )
    backend._available_pods = 4

    await _retry_pending_capacity(services)

    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        assert dep.status in ("building", "deployed_for_eval", "active")
        assert dep.placement_status != "pending_capacity"


# -- test_retry_picks_up_queued_deployments_after_run_open -------------


async def test_retry_picks_up_queued_deployments_after_run_open(make_services):
    """Deployment submitted AFTER the current run opened must get built
    by the retry loop.

    Regression guard for the operator footgun we hit twice during the
    multi-miner E2E: ``start_queued_deployments`` only fires on
    run-open events, so a miner submitting during an open run would
    sit in ``standby_cold / queued`` indefinitely until the next
    rollover.  ``_retry_pending_capacity`` is the scheduled sweep —
    it has to include ``placement_status='queued'`` or queued
    deployments never get un-queued.
    """
    services = make_services()

    with services.db.sessionmaker() as session:
        _make_run(session)
        session.commit()

    hk = "5HKQueRe_" + "Q" * 39
    with services.db.sessionmaker() as session:
        _sub, dep = _submit(services, session, hotkey=hk)
    dep_id = dep.id

    # Simulate "miner submitted after run-open": skip the normal
    # start_queued_deployments that would have fired at open time.
    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        assert dep.placement_status == "queued", (
            "fresh submissions must land in queued so this test exercises the gap"
        )

    await _retry_pending_capacity(services)

    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        assert dep.status in ("building", "deployed_for_eval", "active")
        assert dep.placement_status != "queued"
