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


def _status(code: DeploymentStatusCode) -> DeploymentStatus:
    return DeploymentStatus(
        code=code,
        ready_replicas=1 if code == DeploymentStatusCode.READY else 0,
        desired_replicas=1,
        message=str(code),
        last_pod_phase=None,
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
    introduced_run_id = _sub.introduced_run_id
    with services.db.sessionmaker() as session:
        run = session.get(EvaluationRun, introduced_run_id)
        services.runs.start_queued_deployments(session, run=run)
        session.commit()
    await services.deployments.ensure_current_run_and_reconcile()
    return dep_id


# -- test_recover_ready_is_noop ----------------------------------------


async def test_recover_ready_is_noop(make_services):
    backend = _FakeBackend(status_override=_status(DeploymentStatusCode.READY))
    services = make_services(backend=backend)

    with services.db.sessionmaker() as session:
        _make_run(session)
        session.commit()

    hk = "5HKReady_" + "R" * 39
    dep_id = await _deploy(services, hotkey=hk)

    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        dep.health_status = "unhealthy"
        original_details = dict(dep.health_details_json)
        session.commit()

    await services.deployments.recover_or_demote_deployment(
        deployment_id=dep_id, reason="test",
    )

    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        assert dep.health_details_json.get("restart_attempts", 0) == original_details.get("restart_attempts", 0)
        assert dep.placement_status != "pending_capacity"


# -- test_recover_pending_starting_is_noop -----------------------------


async def test_recover_pending_starting_is_noop(make_services):
    backend = _FakeBackend(status_override=_status(DeploymentStatusCode.PENDING_STARTING))
    services = make_services(backend=backend)

    with services.db.sessionmaker() as session:
        _make_run(session)
        session.commit()

    hk = "5HKPendS_" + "P" * 39
    dep_id = await _deploy(services, hotkey=hk)

    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        dep.health_status = "unhealthy"
        original_details = dict(dep.health_details_json)
        session.commit()

    await services.deployments.recover_or_demote_deployment(
        deployment_id=dep_id, reason="test",
    )

    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        assert dep.health_details_json.get("restart_attempts", 0) == original_details.get("restart_attempts", 0)
        assert dep.placement_status != "pending_capacity"


# -- test_recover_pending_unschedulable_moves_to_pending_capacity ------


async def test_recover_pending_unschedulable_moves_to_pending_capacity(make_services):
    backend = _FakeBackend(status_override=_status(DeploymentStatusCode.PENDING_UNSCHEDULABLE))
    services = make_services(backend=backend)

    with services.db.sessionmaker() as session:
        _make_run(session)
        session.commit()

    hk = "5HKUnscd_" + "U" * 39
    dep_id = await _deploy(services, hotkey=hk)

    await services.deployments.recover_or_demote_deployment(
        deployment_id=dep_id, reason="test_unschedulable",
    )

    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        assert dep.placement_status == "pending_capacity"
        assert dep.health_status == "unhealthy"


# -- test_recover_pending_unschedulable_does_not_burn_restart_budget ----


async def test_recover_pending_unschedulable_does_not_burn_restart_budget(make_services):
    backend = _FakeBackend(status_override=_status(DeploymentStatusCode.PENDING_UNSCHEDULABLE))
    services = make_services(backend=backend)

    with services.db.sessionmaker() as session:
        _make_run(session)
        session.commit()

    hk = "5HKUBudg_" + "B" * 39
    dep_id = await _deploy(services, hotkey=hk)

    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        original_restart_attempts = int(dep.health_details_json.get("restart_attempts", 0))

    await services.deployments.recover_or_demote_deployment(
        deployment_id=dep_id, reason="test_unschedulable",
    )

    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        assert int(dep.health_details_json.get("restart_attempts", 0)) == original_restart_attempts
        assert dep.placement_status == "pending_capacity"
        assert dep.health_status == "unhealthy"


# -- test_recover_crashloop_increments_restart_attempts ----------------


async def test_recover_crashloop_increments_restart_attempts(make_services):
    backend = _FakeBackend(status_override=_status(DeploymentStatusCode.CRASHLOOP))
    services = make_services(backend=backend)

    with services.db.sessionmaker() as session:
        _make_run(session)
        session.commit()

    hk = "5HKCrash_" + "C" * 39
    dep_id = await _deploy(services, hotkey=hk)

    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        before = int(dep.health_details_json.get("restart_attempts", 0))

    await services.deployments.recover_or_demote_deployment(
        deployment_id=dep_id, reason="test_crashloop",
    )

    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        after = int(dep.health_details_json.get("restart_attempts", 0))
        assert after == before + 1


# -- test_recover_crashloop_demotes_after_budget_exhausted -------------


async def test_recover_crashloop_demotes_after_budget_exhausted(make_services):
    backend = _FakeBackend(status_override=_status(DeploymentStatusCode.CRASHLOOP))
    services = make_services(backend=backend)
    budget = services.settings.owner_runtime_restart_budget

    with services.db.sessionmaker() as session:
        _make_run(session)
        session.commit()

    hk = "5HKDemot_" + "D" * 39
    dep_id = await _deploy(services, hotkey=hk)

    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        dep.health_details_json = {
            **dep.health_details_json,
            "restart_attempts": budget - 1,
        }
        session.commit()

    await services.deployments.recover_or_demote_deployment(
        deployment_id=dep_id, reason="test_crashloop_rebuild",
    )

    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        assert int(dep.health_details_json.get("restart_attempts", 0)) == budget

    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        dep.health_details_json = {
            **dep.health_details_json,
            "restart_attempts": budget,
        }
        session.commit()

    await services.deployments.recover_or_demote_deployment(
        deployment_id=dep_id, reason="test_crashloop_demote",
    )

    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        assert dep.health_status == "unhealthy"


# -- test_recover_missing_triggers_rebuild -----------------------------


async def test_recover_missing_triggers_rebuild(make_services):
    backend = _FakeBackend(status_override=_status(DeploymentStatusCode.MISSING))
    services = make_services(backend=backend)

    with services.db.sessionmaker() as session:
        _make_run(session)
        session.commit()

    hk = "5HKMiss__" + "M" * 39
    dep_id = await _deploy(services, hotkey=hk)

    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        before = int(dep.health_details_json.get("restart_attempts", 0))

    await services.deployments.recover_or_demote_deployment(
        deployment_id=dep_id, reason="test_missing",
    )

    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        after = int(dep.health_details_json.get("restart_attempts", 0))
        assert after == before + 1


# -- test_recover_unknown_falls_back_to_legacy_path --------------------


async def test_recover_unknown_falls_back_to_legacy_path(make_services):
    backend = _FakeBackend()
    services = make_services(backend=backend)

    with services.db.sessionmaker() as session:
        _make_run(session)
        session.commit()

    hk = "5HKUnkn__" + "U" * 39
    dep_id = await _deploy(services, hotkey=hk)

    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        before_attempts = int(dep.health_details_json.get("restart_attempts", 0))

    await services.deployments.recover_or_demote_deployment(
        deployment_id=dep_id, reason="test_unknown_legacy",
    )

    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        after_attempts = int(dep.health_details_json.get("restart_attempts", 0))
        assert after_attempts == before_attempts + 1
        assert dep.placement_status != "pending_capacity"
