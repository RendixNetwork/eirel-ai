from __future__ import annotations

from datetime import timedelta

import pytest

from shared.common.config import Settings, reset_settings
from shared.common.database import Database
from shared.common.models import (
    DeploymentScoreRecord,
    EvaluationRun,
    ManagedDeployment,
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

    def factory(*, backend=None, restart_budget: int | None = None):
        if restart_budget is not None:
            monkeypatch.setenv("OWNER_RUNTIME_RESTART_BUDGET", str(restart_budget))
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
        sub, dep = _submit(services, session, hotkey=hotkey, version=version, block=block)
        target_run_id = sub.introduced_run_id
    dep_id = dep.id
    with services.db.sessionmaker() as session:
        run = session.get(EvaluationRun, target_run_id)
        services.runs.start_queued_deployments(session, run=run)
        session.commit()
    await services.deployments.ensure_current_run_and_reconcile()
    return dep_id


def _add_score_record(session, *, deployment: ManagedDeployment, run_id: str = "run-1"):
    record = DeploymentScoreRecord(
        run_id=run_id,
        family_id=deployment.family_id,
        deployment_id=deployment.id,
        submission_id=deployment.submission_id,
        miner_hotkey=deployment.miner_hotkey,
        deployment_revision=deployment.deployment_revision,
        raw_score=0.8,
        normalized_score=0.8,
        is_eligible=True,
    )
    session.add(record)
    session.flush()
    return record


# -- test_reconcile_family_drains_unhealthy_active -------------------


async def test_reconcile_family_drains_unhealthy_active(make_services):
    services = make_services(restart_budget=0)

    with services.db.sessionmaker() as session:
        _make_run(session)
        session.commit()

    hk = "5HKDrain_" + "D" * 39
    dep_id = await _deploy(services, hotkey=hk)

    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        assert dep.status == "deployed_for_eval"
        assert dep.health_status == "healthy"
        dep.is_active = True
        dep.health_status = "unhealthy"
        session.commit()

    await services.reconcile_family_deployments(family_id="general_chat")

    with services.db.sessionmaker() as session:
        dep = session.get(ManagedDeployment, dep_id)
        assert dep.is_active is False
        assert dep.status in ("unhealthy", "standby_cold")


# -- test_reconcile_family_promotes_standby_when_active_retires ------


async def test_reconcile_family_promotes_standby_when_active_retires(make_services):
    services = make_services()

    with services.db.sessionmaker() as session:
        _make_run(session)
        session.commit()

    hk_a = "5HKActiv_" + "A" * 39
    hk_b = "5HKStdby_" + "B" * 39
    dep_a_id = await _deploy(services, hotkey=hk_a, version="1.0.0", block=100)
    dep_b_id = await _deploy(services, hotkey=hk_b, version="1.0.0", block=101)

    with services.db.sessionmaker() as session:
        dep_a = session.get(ManagedDeployment, dep_a_id)
        dep_b = session.get(ManagedDeployment, dep_b_id)
        _add_score_record(session, deployment=dep_a)
        _add_score_record(session, deployment=dep_b)
        dep_a.is_active = True
        dep_a.status = "active"
        dep_b.is_active = False
        session.commit()

    with services.db.sessionmaker() as session:
        dep_a = session.get(ManagedDeployment, dep_a_id)
        dep_a.status = "retired"
        dep_a.is_active = False
        dep_a.health_status = "retired"
        dep_a.retired_at = _utcnow()
        session.commit()

    await services.reconcile_family_deployments(family_id="general_chat")

    with services.db.sessionmaker() as session:
        dep_b = session.get(ManagedDeployment, dep_b_id)
        assert dep_b.is_active is True
