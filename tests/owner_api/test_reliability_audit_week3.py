from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime, timedelta
from types import ModuleType, SimpleNamespace
from uuid import uuid4

import pytest

from shared.common.config import Settings, reset_settings
from shared.common.database import Database
from shared.common.models import (
    EpochTargetSnapshot,
    ManagedDeployment,
    ManagedMinerSubmission,
    SubmissionArtifact,
)
from shared.common.circuit_breaker import CircuitBreaker, CircuitOpenError
from infra.miner_runtime._k8s_helpers import DeploymentStatus, DeploymentStatusCode
from infra.miner_runtime.runtime_manager import KubernetesMinerRuntimeManager
from tests.conftest import FIXTURES_ROOT


# -- Fake kubernetes SDK objects -----------------------------------------------
# Reuse the same pattern as test_k8s_backend_readonly.py so both files can
# coexist without conflicting sys.modules registrations.

class _FakeApiException(Exception):
    def __init__(self, status: int) -> None:
        self.status = status
        super().__init__(f"({status})")


_k8s_mod = ModuleType("kubernetes")
_k8s_client_mod = ModuleType("kubernetes.client")
_k8s_exc_mod = ModuleType("kubernetes.client.exceptions")
_k8s_exc_mod.ApiException = _FakeApiException  # type: ignore[attr-defined]
_k8s_client_mod.exceptions = _k8s_exc_mod  # type: ignore[attr-defined]
_k8s_mod.client = _k8s_client_mod  # type: ignore[attr-defined]
sys.modules.setdefault("kubernetes", _k8s_mod)
sys.modules.setdefault("kubernetes.client", _k8s_client_mod)
sys.modules.setdefault("kubernetes.client.exceptions", _k8s_exc_mod)
_FakeApiException = sys.modules["kubernetes.client.exceptions"].ApiException  # type: ignore[attr-defined]


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)


# -- K8s test helpers ----------------------------------------------------------


def _make_deployment(*, replicas: int = 1, ready_replicas: int | None = None):
    return _ns(
        spec=_ns(replicas=replicas),
        status=_ns(ready_replicas=ready_replicas),
    )


def _make_pod(
    *,
    phase: str = "Pending",
    conditions: list | None = None,
    container_statuses: list | None = None,
):
    return _ns(
        status=_ns(
            phase=phase,
            conditions=conditions,
            container_statuses=container_statuses,
        ),
    )


def _make_container_status(waiting_reason: str | None = None, waiting_message: str | None = None):
    if waiting_reason is None:
        return _ns(state=None)
    return _ns(
        state=_ns(
            waiting=_ns(reason=waiting_reason, message=waiting_message),
        ),
    )


class _FakeCoreV1Api:
    def __init__(self, *, nodes: list | None = None, pods: list | None = None) -> None:
        self._nodes = nodes or []
        self._pods = pods or []

    def list_node(self, label_selector=None):
        return _ns(items=self._nodes)

    def list_namespaced_pod(self, namespace, label_selector=None):
        return _ns(items=self._pods)


class _FakeAppsV1Api:
    def __init__(self, *, deployment=None, raise_404: bool = False) -> None:
        self._deployment = deployment
        self._raise_404 = raise_404

    def read_namespaced_deployment(self, name, namespace):
        if self._raise_404:
            raise _FakeApiException(404)
        return self._deployment


class _FakeNetworkingV1Api:
    pass


def _make_manager(
    *,
    core: _FakeCoreV1Api | None = None,
    apps: _FakeAppsV1Api | None = None,
    netv1: _FakeNetworkingV1Api | None = None,
    namespace: str = "eirel-miners",
    service_domain: str = "svc.cluster.local",
) -> KubernetesMinerRuntimeManager:
    mgr = object.__new__(KubernetesMinerRuntimeManager)
    mgr._records = {}
    mgr._core = core or _FakeCoreV1Api()
    mgr._apps = apps or _FakeAppsV1Api()
    mgr._netv1 = netv1 or _FakeNetworkingV1Api()
    mgr._namespace = namespace
    mgr._system_namespace = "eirel-system"
    mgr._runtime_image = "registry.eirel.internal/eirel-miner-runtime:latest"
    mgr._shared_secret_name = "eirel-runtime-shared"
    mgr._service_domain = service_domain
    mgr._health_timeout_seconds = 30.0
    mgr._probe_period_seconds = 5
    return mgr


# -- Snapshot test helpers -----------------------------------------------------


def _utcnow():
    from control_plane.owner_api._helpers import utcnow
    return utcnow()


def _make_artifact(session) -> SubmissionArtifact:
    artifact = SubmissionArtifact(
        archive_bytes=b"fake-archive",
        sha256="a" * 64,
        size_bytes=12,
        manifest_json={},
    )
    session.add(artifact)
    session.flush()
    return artifact


def _make_submission(session, *, artifact_id: str, hotkey: str = "5HKTest" + "A" * 42) -> ManagedMinerSubmission:
    sub = ManagedMinerSubmission(
        miner_hotkey=hotkey,
        submission_seq=1,
        family_id="analyst",
        status="active",
        artifact_id=artifact_id,
        manifest_json={},
        archive_sha256="a" * 64,
        submission_block=100,
    )
    session.add(sub)
    session.flush()
    return sub


def _make_test_deployment(
    session,
    *,
    submission_id: str,
    health_status: str = "healthy",
    status: str = "deployed_for_eval",
) -> ManagedDeployment:
    dep = ManagedDeployment(
        submission_id=submission_id,
        miner_hotkey="5HKTest" + "A" * 42,
        family_id="analyst",
        deployment_revision=str(uuid4())[:12],
        image_ref="registry.test/img:v1",
        endpoint="http://test:8080",
        status=status,
        health_status=health_status,
    )
    session.add(dep)
    session.flush()
    return dep


def _make_snapshot(
    session,
    *,
    run_id: str,
    deployment_ids: list[str],
    created_at: datetime | None = None,
) -> EpochTargetSnapshot:
    members = [{"metadata": {"deployment_id": did}} for did in deployment_ids]
    snap = EpochTargetSnapshot(
        run_id=run_id,
        family_id="analyst",
        benchmark_version="test_v1",
        rubric_version="test_v1",
        judge_model="test-judge",
        status="pending_deployments",
        members_json=members,
    )
    if created_at is not None:
        snap.created_at = created_at
    session.add(snap)
    session.flush()
    return snap


@pytest.fixture
def snapshot_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("METAGRAPH_SYNC_INTERVAL_SECONDS", "3600")
    monkeypatch.setenv("RESULT_AGGREGATION_INTERVAL_SECONDS", "3600")
    monkeypatch.setenv("USE_REDIS_POOL", "0")
    monkeypatch.setenv("METAGRAPH_SNAPSHOT_PATH", str(tmp_path / "metagraph.json"))
    monkeypatch.setenv("EIREL_PROVIDER_PROXY_URL", "http://provider-proxy.test")
    monkeypatch.setenv("EIREL_PROVIDER_PROXY_TOKEN", "provider-token")
    monkeypatch.setenv("EIREL_INTERNAL_SERVICE_TOKEN", "internal-token")
    monkeypatch.setenv("EIREL_ACTIVE_FAMILIES", "analyst")
    monkeypatch.setenv("EIREL_LAUNCH_MODE", "1")
    # Dataset root is set up by the autouse conftest fixture.
    reset_settings()
    settings = Settings()
    db = Database(settings.database_url)
    db.create_all()
    yield db, settings
    reset_settings()


def _make_run_manager(settings):
    from control_plane.owner_api.evaluation.run_manager import RunManager

    owner = SimpleNamespace(settings=settings)
    mgr = object.__new__(RunManager)
    mgr._owner = owner
    mgr._last_closed_run_retired_ids = []
    mgr._pending_queued_deployment_ids = []
    return mgr


# -- Test 1: K8s API timeout --------------------------------------------------


async def test_k8s_api_timeout_raises():
    mgr = _make_manager()
    mgr._K8S_API_TIMEOUT = 0.1  # 100ms timeout for testing

    def _slow_call():
        time.sleep(5)

    with pytest.raises(asyncio.TimeoutError):
        await mgr._k8s_call(_slow_call)


# -- Test 2: ImagePullBackOff detection ----------------------------------------


async def test_image_pull_backoff_detected():
    dep = _make_deployment(replicas=1, ready_replicas=0)
    pod = _make_pod(
        phase="Pending",
        container_statuses=[
            _make_container_status("ImagePullBackOff", "pull access denied"),
        ],
    )
    mgr = _make_manager(
        apps=_FakeAppsV1Api(deployment=dep),
        core=_FakeCoreV1Api(pods=[pod]),
    )
    status = await mgr.deployment_status("abc123")
    assert status.code == DeploymentStatusCode.CRASHLOOP
    assert "image pull failed" in status.message


# -- Test 3: Circuit breaker → 503 --------------------------------------------


async def test_circuit_open_returns_503():
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=60.0)

    async def _failing():
        raise RuntimeError("downstream down")

    for _ in range(2):
        with pytest.raises(RuntimeError, match="downstream down"):
            await cb.call("test-endpoint", _failing)

    with pytest.raises(CircuitOpenError) as exc_info:
        await cb.call("test-endpoint", _failing)
    assert exc_info.value.retry_after > 0

    # Verify the error carries the attributes a handler needs for 503
    err = exc_info.value
    assert isinstance(err, RuntimeError)
    assert err.key == "test-endpoint"


# -- Test 4: Snapshot timeout opens with partial healthy -----------------------


def test_snapshot_timeout_opens_with_partial_healthy(snapshot_db):
    db, settings = snapshot_db
    run_mgr = _make_run_manager(settings)
    now = _utcnow()

    with db.sessionmaker() as session:
        artifact = _make_artifact(session)
        sub = _make_submission(session, artifact_id=artifact.id)
        dep_healthy = _make_test_deployment(
            session, submission_id=sub.id, health_status="healthy",
        )
        dep_starting = _make_test_deployment(
            session, submission_id=sub.id, health_status="starting",
        )
        snap = _make_snapshot(
            session,
            run_id="run-timeout-partial",
            deployment_ids=[dep_healthy.id, dep_starting.id],
            created_at=now - timedelta(minutes=20),
        )
        snap_id = snap.id
        session.commit()

    with db.sessionmaker() as session:
        run_mgr.check_and_open_pending_snapshots(session)

    with db.sessionmaker() as session:
        snap = session.get(EpochTargetSnapshot, snap_id)
        assert snap.status == "open"


# -- Test 5: Snapshot blocks with zero healthy ---------------------------------


def test_snapshot_blocks_with_zero_healthy(snapshot_db):
    db, settings = snapshot_db
    run_mgr = _make_run_manager(settings)
    now = _utcnow()

    with db.sessionmaker() as session:
        artifact = _make_artifact(session)
        sub = _make_submission(session, artifact_id=artifact.id)
        dep1 = _make_test_deployment(
            session, submission_id=sub.id, health_status="starting",
        )
        dep2 = _make_test_deployment(
            session, submission_id=sub.id, health_status="starting",
        )
        snap = _make_snapshot(
            session,
            run_id="run-timeout-zero",
            deployment_ids=[dep1.id, dep2.id],
            created_at=now - timedelta(minutes=20),
        )
        snap_id = snap.id
        session.commit()

    with db.sessionmaker() as session:
        run_mgr.check_and_open_pending_snapshots(session)

    with db.sessionmaker() as session:
        snap = session.get(EpochTargetSnapshot, snap_id)
        assert snap.status == "pending_deployments"


# -- Test 6: Snapshot opens immediately when all ready -------------------------


def test_snapshot_opens_immediately_when_all_ready(snapshot_db):
    db, settings = snapshot_db
    run_mgr = _make_run_manager(settings)

    with db.sessionmaker() as session:
        artifact = _make_artifact(session)
        sub = _make_submission(session, artifact_id=artifact.id)
        dep1 = _make_test_deployment(
            session, submission_id=sub.id, health_status="healthy",
        )
        dep2 = _make_test_deployment(
            session, submission_id=sub.id, health_status="healthy",
        )
        snap = _make_snapshot(
            session,
            run_id="run-all-ready",
            deployment_ids=[dep1.id, dep2.id],
            # No created_at override — defaults to now, so NOT timed out
        )
        snap_id = snap.id
        session.commit()

    with db.sessionmaker() as session:
        run_mgr.check_and_open_pending_snapshots(session)

    with db.sessionmaker() as session:
        snap = session.get(EpochTargetSnapshot, snap_id)
        assert snap.status == "open"


# -- Test 7: Config setting exists and works -----------------------------------


def test_config_validates_positive_timeout(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///unused.db")
    monkeypatch.setenv("METAGRAPH_SYNC_INTERVAL_SECONDS", "3600")
    monkeypatch.setenv("RESULT_AGGREGATION_INTERVAL_SECONDS", "3600")
    monkeypatch.setenv("USE_REDIS_POOL", "0")
    monkeypatch.setenv("EIREL_PROVIDER_PROXY_URL", "http://provider-proxy.test")
    monkeypatch.setenv("EIREL_PROVIDER_PROXY_TOKEN", "provider-token")
    monkeypatch.setenv("EIREL_INTERNAL_SERVICE_TOKEN", "internal-token")
    monkeypatch.setenv("EIREL_SNAPSHOT_READINESS_TIMEOUT_MINUTES", "30")
    reset_settings()
    settings = Settings()
    assert settings.snapshot_readiness_timeout_minutes == 30
    reset_settings()
