from __future__ import annotations

import asyncio
import io
import sys
import tarfile
import time as _time_mod
from types import ModuleType, SimpleNamespace

import pytest

from infra.miner_runtime._k8s_helpers import (
    DeploymentStatus,
    DeploymentStatusCode,
)
from infra.miner_runtime.runtime_manager import (
    KubernetesMinerRuntimeManager,
    RuntimeManagerError,
)


# -- Fake kubernetes SDK -------------------------------------------------------


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


# -- Fake K8s Cluster ----------------------------------------------------------


class _FakeK8sCluster:
    def __init__(self) -> None:
        self.created: list[tuple[str, str, dict]] = []
        self.replaced: list[tuple[str, str, dict]] = []
        self.deleted: list[tuple[str, str, str]] = []
        self._status_sequence: list[DeploymentStatus] = []
        self._existing: set[tuple[str, str]] = set()
        self._delete_404: set[tuple[str, str]] = set()
        self._deployments_for_list: list = []

    def next_status(self) -> DeploymentStatus:
        if self._status_sequence:
            return self._status_sequence.pop(0)
        return DeploymentStatus(
            code=DeploymentStatusCode.PENDING_STARTING,
            ready_replicas=0,
            desired_replicas=1,
            message="default",
            last_pod_phase=None,
        )

    def inject_create_409(self, kind: str, name: str) -> None:
        self._existing.add((kind, name))

    def inject_delete_404(self, kind: str, name: str) -> None:
        self._delete_404.add((kind, name))


# -- Fake API Clients ----------------------------------------------------------


class _FakeCoreV1Api:
    def __init__(self, cluster: _FakeK8sCluster) -> None:
        self._cluster = cluster

    def create_namespaced_config_map(self, *, namespace, body):
        name = body.get("metadata", {}).get("name", "")
        if ("ConfigMap", name) in self._cluster._existing:
            raise _FakeApiException(409)
        self._cluster.created.append(("ConfigMap", namespace, body))

    def replace_namespaced_config_map(self, *, name, namespace, body):
        self._cluster.replaced.append(("ConfigMap", namespace, body))

    def create_namespaced_service(self, *, namespace, body):
        name = body.get("metadata", {}).get("name", "")
        if ("Service", name) in self._cluster._existing:
            raise _FakeApiException(409)
        self._cluster.created.append(("Service", namespace, body))

    def replace_namespaced_service(self, *, name, namespace, body):
        self._cluster.replaced.append(("Service", namespace, body))

    def delete_namespaced_config_map(self, *, name, namespace, **kw):
        if ("ConfigMap", name) in self._cluster._delete_404:
            raise _FakeApiException(404)
        self._cluster.deleted.append(("ConfigMap", namespace, name))

    def delete_namespaced_service(self, *, name, namespace, **kw):
        if ("Service", name) in self._cluster._delete_404:
            raise _FakeApiException(404)
        self._cluster.deleted.append(("Service", namespace, name))


class _FakeAppsV1Api:
    def __init__(self, cluster: _FakeK8sCluster) -> None:
        self._cluster = cluster

    def create_namespaced_deployment(self, *, namespace, body):
        name = body.get("metadata", {}).get("name", "")
        if ("Deployment", name) in self._cluster._existing:
            raise _FakeApiException(409)
        self._cluster.created.append(("Deployment", namespace, body))

    def replace_namespaced_deployment(self, *, name, namespace, body):
        self._cluster.replaced.append(("Deployment", namespace, body))

    def delete_namespaced_deployment(self, *, name, namespace, **kw):
        if ("Deployment", name) in self._cluster._delete_404:
            raise _FakeApiException(404)
        self._cluster.deleted.append(("Deployment", namespace, name))

    def list_namespaced_deployment(self, *, namespace, label_selector=None):
        return SimpleNamespace(items=self._cluster._deployments_for_list)


class _FakeNetworkingV1Api:
    def __init__(self, cluster: _FakeK8sCluster) -> None:
        self._cluster = cluster

    def create_namespaced_network_policy(self, *, namespace, body):
        name = body.get("metadata", {}).get("name", "")
        if ("NetworkPolicy", name) in self._cluster._existing:
            raise _FakeApiException(409)
        self._cluster.created.append(("NetworkPolicy", namespace, body))

    def replace_namespaced_network_policy(self, *, name, namespace, body):
        self._cluster.replaced.append(("NetworkPolicy", namespace, body))

    def delete_namespaced_network_policy(self, *, name, namespace, **kw):
        if ("NetworkPolicy", name) in self._cluster._delete_404:
            raise _FakeApiException(404)
        self._cluster.deleted.append(("NetworkPolicy", namespace, name))


# -- Test helpers ---------------------------------------------------------------


def _make_manager(cluster: _FakeK8sCluster) -> KubernetesMinerRuntimeManager:
    mgr = object.__new__(KubernetesMinerRuntimeManager)
    mgr._records = {}
    mgr._core = _FakeCoreV1Api(cluster)
    mgr._apps = _FakeAppsV1Api(cluster)
    mgr._netv1 = _FakeNetworkingV1Api(cluster)
    mgr._namespace = "eirel-miners"
    mgr._system_namespace = "eirel-system"
    mgr._runtime_image = "registry.eirel.internal/miner-runtime:v1"
    mgr._shared_secret_name = "eirel-runtime-shared"
    mgr._service_domain = "svc.cluster.local"
    mgr._health_timeout_seconds = 60.0
    mgr._probe_period_seconds = 5
    return mgr


def _make_archive(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _manifest(port: int = 8080):
    return SimpleNamespace(
        runtime=SimpleNamespace(port=port, health_path="/healthz"),
        family_id="analyst",
        hotkey="5GrwvaEF",
    )


def _make_fake_time():
    state = {"t": 1000.0}

    def monotonic():
        return state["t"]

    async def sleep(seconds):
        state["t"] += seconds

    return monotonic, sleep


def _patch_time(monkeypatch):
    fake_monotonic, fake_sleep = _make_fake_time()
    monkeypatch.setattr(_time_mod, "monotonic", fake_monotonic)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)


async def _coro(value):
    return value


def _ensure_kwargs(submission_id: str = "sub-1", **overrides):
    defaults = {
        "submission_id": submission_id,
        "archive_bytes": _make_archive({"app.py": b"print('hello')"}),
        "manifest": _manifest(),
        "assigned_node_name": None,
        "requested_cpu_millis": 500,
        "requested_memory_bytes": 256 * 1024 * 1024,
    }
    defaults.update(overrides)
    return defaults


_READY = DeploymentStatus(
    code=DeploymentStatusCode.READY,
    ready_replicas=1,
    desired_replicas=1,
    message="ok",
    last_pod_phase="Running",
)

_PENDING = DeploymentStatus(
    code=DeploymentStatusCode.PENDING_STARTING,
    ready_replicas=0,
    desired_replicas=1,
    message="starting",
    last_pod_phase=None,
)

_CRASHLOOP = DeploymentStatus(
    code=DeploymentStatusCode.CRASHLOOP,
    ready_replicas=0,
    desired_replicas=1,
    message="CrashLoopBackOff",
    last_pod_phase="Running",
)


# -- ensure_runtime tests -----------------------------------------------------


async def test_ensure_runtime_creates_all_four_resources(monkeypatch):
    cluster = _FakeK8sCluster()
    cluster._status_sequence = [_READY]
    mgr = _make_manager(cluster)
    monkeypatch.setattr(mgr, "deployment_status", lambda sid: _coro(cluster.next_status()))
    _patch_time(monkeypatch)

    handle = await mgr.ensure_runtime(**_ensure_kwargs())

    created_kinds = [kind for kind, _, _ in cluster.created]
    assert "ConfigMap" in created_kinds
    assert "Deployment" in created_kinds
    assert "Service" in created_kinds
    assert "NetworkPolicy" in created_kinds
    for _, ns, _ in cluster.created:
        assert ns == "eirel-miners"
    assert handle.state == "healthy"
    assert "sub-1" in handle.endpoint_url


async def test_ensure_runtime_waits_for_ready_replicas(monkeypatch):
    cluster = _FakeK8sCluster()
    cluster._status_sequence = [_PENDING, _PENDING, _READY]
    mgr = _make_manager(cluster)
    monkeypatch.setattr(mgr, "deployment_status", lambda sid: _coro(cluster.next_status()))
    _patch_time(monkeypatch)

    handle = await mgr.ensure_runtime(**_ensure_kwargs())

    assert handle.state == "healthy"
    assert len(cluster._status_sequence) == 0


async def test_ensure_runtime_raises_on_crashloop(monkeypatch):
    cluster = _FakeK8sCluster()
    cluster._status_sequence = [_PENDING, _CRASHLOOP]
    mgr = _make_manager(cluster)
    monkeypatch.setattr(mgr, "deployment_status", lambda sid: _coro(cluster.next_status()))
    _patch_time(monkeypatch)

    with pytest.raises(RuntimeManagerError, match="CrashLoopBackOff"):
        await mgr.ensure_runtime(**_ensure_kwargs())


async def test_ensure_runtime_times_out_on_pending_forever(monkeypatch):
    cluster = _FakeK8sCluster()
    cluster._status_sequence = [_PENDING] * 100
    mgr = _make_manager(cluster)
    mgr._health_timeout_seconds = 0.1
    monkeypatch.setattr(mgr, "deployment_status", lambda sid: _coro(cluster.next_status()))
    _patch_time(monkeypatch)

    with pytest.raises(RuntimeManagerError, match="did not become ready"):
        await mgr.ensure_runtime(**_ensure_kwargs())


async def test_ensure_runtime_replaces_existing_deployment_on_409(monkeypatch):
    cluster = _FakeK8sCluster()
    cluster.inject_create_409("Deployment", "miner-sub-1")
    cluster._status_sequence = [_READY]
    mgr = _make_manager(cluster)
    monkeypatch.setattr(mgr, "deployment_status", lambda sid: _coro(cluster.next_status()))
    _patch_time(monkeypatch)

    handle = await mgr.ensure_runtime(**_ensure_kwargs())

    created_kinds = [kind for kind, _, _ in cluster.created]
    replaced_kinds = [kind for kind, _, _ in cluster.replaced]
    assert "Deployment" not in created_kinds
    assert "Deployment" in replaced_kinds
    assert handle.state == "healthy"


async def test_ensure_runtime_rejects_archive_over_900kib(monkeypatch):
    cluster = _FakeK8sCluster()
    mgr = _make_manager(cluster)
    _patch_time(monkeypatch)

    large_content = b"x" * (901 * 1024)
    archive = _make_archive({"big_file.bin": large_content})

    with pytest.raises(RuntimeManagerError, match="too large"):
        await mgr.ensure_runtime(**_ensure_kwargs(archive_bytes=archive))

    assert len(cluster.created) == 0


# -- stop_runtime tests --------------------------------------------------------


async def test_stop_runtime_deletes_all_four_resources():
    cluster = _FakeK8sCluster()
    mgr = _make_manager(cluster)
    mgr._records["sub-1"] = SimpleNamespace(handle=SimpleNamespace())

    await mgr.stop_runtime("sub-1", reason="test")

    deleted_kinds = [kind for kind, _, _ in cluster.deleted]
    assert "Deployment" in deleted_kinds
    assert "Service" in deleted_kinds
    assert "NetworkPolicy" in deleted_kinds
    assert "ConfigMap" in deleted_kinds
    assert len(cluster.deleted) == 4
    assert "sub-1" not in mgr._records


async def test_stop_runtime_treats_404_as_success():
    cluster = _FakeK8sCluster()
    cluster.inject_delete_404("Deployment", "miner-sub-1")
    mgr = _make_manager(cluster)
    mgr._records["sub-1"] = SimpleNamespace(handle=SimpleNamespace())

    await mgr.stop_runtime("sub-1", reason="test")

    deleted_kinds = [kind for kind, _, _ in cluster.deleted]
    assert "Service" in deleted_kinds
    assert "NetworkPolicy" in deleted_kinds
    assert "ConfigMap" in deleted_kinds
    assert "sub-1" not in mgr._records


# -- reconcile_active_submissions tests ----------------------------------------


async def test_reconcile_deletes_orphan_deployments():
    cluster = _FakeK8sCluster()
    cluster._deployments_for_list = [
        SimpleNamespace(
            metadata=SimpleNamespace(
                name="miner-stale-1",
                labels={"eirel.dev/submission-id": "stale-1"},
            ),
        ),
        SimpleNamespace(
            metadata=SimpleNamespace(
                name="miner-kept-1",
                labels={"eirel.dev/submission-id": "kept-1"},
            ),
        ),
    ]
    mgr = _make_manager(cluster)

    await mgr.reconcile_active_submissions({"kept-1"})

    deleted_names = [name for _, _, name in cluster.deleted]
    assert "miner-stale-1" in deleted_names
    assert "miner-kept-1" not in deleted_names


async def test_reconcile_preserves_active_deployments():
    cluster = _FakeK8sCluster()
    cluster._deployments_for_list = [
        SimpleNamespace(
            metadata=SimpleNamespace(
                name="miner-active-1",
                labels={"eirel.dev/submission-id": "active-1"},
            ),
        ),
    ]
    mgr = _make_manager(cluster)
    mgr._records["active-1"] = SimpleNamespace(handle=SimpleNamespace())
    mgr._records["stale-record"] = SimpleNamespace(handle=SimpleNamespace())

    await mgr.reconcile_active_submissions({"active-1"})

    assert len(cluster.deleted) == 0
    assert "active-1" in mgr._records
    assert "stale-record" not in mgr._records
