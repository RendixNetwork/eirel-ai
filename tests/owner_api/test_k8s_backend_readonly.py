from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from infra.miner_runtime._k8s_helpers import DeploymentStatusCode
from infra.miner_runtime.runtime_manager import (
    KubernetesMinerRuntimeManager,
    RuntimeNodeInfo,
)


# -- Fake kubernetes SDK objects -----------------------------------------------


class _FakeApiException(Exception):
    def __init__(self, status: int) -> None:
        self.status = status
        super().__init__(f"({status})")


# Register a minimal fake kubernetes package in sys.modules so that
# ``from kubernetes.client.exceptions import ApiException`` inside
# deployment_status resolves without installing the real SDK.
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


def _make_node(
    name: str,
    *,
    labels: dict[str, str] | None = None,
    allocatable: dict[str, str] | None = None,
    conditions: list | None = None,
    unschedulable: bool = False,
):
    return _ns(
        metadata=_ns(name=name, labels=labels or {}),
        spec=_ns(unschedulable=unschedulable),
        status=_ns(
            allocatable=allocatable or {},
            conditions=conditions or [],
        ),
    )


def _make_condition(type_: str, status: str):
    return _ns(type=type_, status=status)


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


def _make_pod_condition(type_: str, status: str, reason: str | None = None, message: str | None = None):
    return _ns(type=type_, status=status, reason=reason, message=message)


def _make_container_status(waiting_reason: str | None = None, waiting_message: str | None = None):
    if waiting_reason is None:
        return _ns(state=None)
    return _ns(
        state=_ns(
            waiting=_ns(reason=waiting_reason, message=waiting_message),
        ),
    )


# -- Fake kubernetes API clients -----------------------------------------------


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


# -- Helper to build a manager without real kubernetes --------------------------


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
    mgr._runtime_image = "registry.eirel.internal/miner-runtime:v1"
    mgr._shared_secret_name = "eirel-runtime-shared"
    mgr._service_domain = service_domain
    mgr._health_timeout_seconds = 30.0
    mgr._probe_period_seconds = 5
    return mgr


# -- list_runtime_nodes tests -------------------------------------------------


async def test_list_runtime_nodes_parses_allocatable_and_labels():
    node = _make_node(
        "gpu-worker-1",
        labels={"eirel.dev/runtime-pool": "true", "eirel.dev/runtime-class": "miner"},
        allocatable={"cpu": "4", "memory": "8Gi", "pods": "110"},
        conditions=[_make_condition("Ready", "True")],
    )
    mgr = _make_manager(core=_FakeCoreV1Api(nodes=[node]))
    nodes = await mgr.list_runtime_nodes()
    assert len(nodes) == 1
    n = nodes[0]
    assert n.node_name == "gpu-worker-1"
    assert n.allocatable_cpu_millis == 4000
    assert n.allocatable_memory_bytes == 8 * 2**30
    assert n.allocatable_pod_count == 110
    assert n.labels["eirel.dev/runtime-pool"] == "true"
    assert n.ready is True
    assert n.schedulable is True
    assert n.metadata == {"backend": "kubernetes", "ssh_host": ""}


async def test_list_runtime_nodes_includes_not_ready_but_flags_them():
    node = _make_node(
        "sick-node",
        allocatable={"cpu": "2", "memory": "4Gi", "pods": "50"},
        conditions=[_make_condition("Ready", "False")],
    )
    mgr = _make_manager(core=_FakeCoreV1Api(nodes=[node]))
    nodes = await mgr.list_runtime_nodes()
    assert len(nodes) == 1
    assert nodes[0].ready is False


async def test_list_runtime_nodes_marks_cordoned_as_not_schedulable():
    node = _make_node(
        "cordoned-node",
        allocatable={"cpu": "2", "memory": "4Gi", "pods": "50"},
        conditions=[_make_condition("Ready", "True")],
        unschedulable=True,
    )
    mgr = _make_manager(core=_FakeCoreV1Api(nodes=[node]))
    nodes = await mgr.list_runtime_nodes()
    assert len(nodes) == 1
    assert nodes[0].schedulable is False
    assert nodes[0].ready is True


# -- deployment_status tests ---------------------------------------------------


async def test_deployment_status_missing_when_api_returns_404():
    mgr = _make_manager(apps=_FakeAppsV1Api(raise_404=True))
    status = await mgr.deployment_status("abc123")
    assert status.code == DeploymentStatusCode.MISSING
    assert status.ready_replicas == 0
    assert status.message == "not found"


async def test_deployment_status_ready_when_replicas_match():
    dep = _make_deployment(replicas=1, ready_replicas=1)
    mgr = _make_manager(apps=_FakeAppsV1Api(deployment=dep))
    status = await mgr.deployment_status("abc123")
    assert status.code == DeploymentStatusCode.READY
    assert status.ready_replicas == 1
    assert status.desired_replicas == 1
    assert status.last_pod_phase == "Running"


async def test_deployment_status_unschedulable_when_pod_condition_shows():
    dep = _make_deployment(replicas=1, ready_replicas=0)
    pod = _make_pod(
        phase="Pending",
        conditions=[
            _make_pod_condition("PodScheduled", "False", reason="Unschedulable", message="no capacity"),
        ],
    )
    mgr = _make_manager(
        apps=_FakeAppsV1Api(deployment=dep),
        core=_FakeCoreV1Api(pods=[pod]),
    )
    status = await mgr.deployment_status("abc123")
    assert status.code == DeploymentStatusCode.PENDING_UNSCHEDULABLE
    assert status.message == "no capacity"
    assert status.last_pod_phase == "Pending"


async def test_deployment_status_crashloop_when_container_state_shows():
    dep = _make_deployment(replicas=1, ready_replicas=0)
    pod = _make_pod(
        phase="Running",
        container_statuses=[
            _make_container_status("CrashLoopBackOff", "back-off 5m"),
        ],
    )
    mgr = _make_manager(
        apps=_FakeAppsV1Api(deployment=dep),
        core=_FakeCoreV1Api(pods=[pod]),
    )
    status = await mgr.deployment_status("abc123")
    assert status.code == DeploymentStatusCode.CRASHLOOP
    assert "back-off" in status.message


async def test_deployment_status_pending_starting_when_no_pods_yet():
    dep = _make_deployment(replicas=1, ready_replicas=0)
    mgr = _make_manager(
        apps=_FakeAppsV1Api(deployment=dep),
        core=_FakeCoreV1Api(pods=[]),
    )
    status = await mgr.deployment_status("abc123")
    assert status.code == DeploymentStatusCode.PENDING_STARTING
    assert status.message == "no pods yet"
    assert status.last_pod_phase is None


async def test_deployment_status_pending_starting_when_pod_phase_pending():
    dep = _make_deployment(replicas=1, ready_replicas=0)
    pod = _make_pod(phase="Pending")
    mgr = _make_manager(
        apps=_FakeAppsV1Api(deployment=dep),
        core=_FakeCoreV1Api(pods=[pod]),
    )
    status = await mgr.deployment_status("abc123")
    assert status.code == DeploymentStatusCode.PENDING_STARTING
    assert "Pending" in status.message


# -- recover_runtime_handle tests ----------------------------------------------


async def test_recover_runtime_handle_returns_service_dns_for_ready():
    dep = _make_deployment(replicas=1, ready_replicas=1)
    mgr = _make_manager(apps=_FakeAppsV1Api(deployment=dep))
    manifest = _ns(runtime=_ns(port=8080))
    handle = await mgr.recover_runtime_handle(submission_id="sub42", manifest=manifest)
    assert handle is not None
    assert handle.endpoint_url == "http://miner-sub42.eirel-miners.svc.cluster.local"
    assert handle.state == "healthy"
    assert handle.container_name == "miner-sub42"
    assert mgr.runtime_handle("sub42") is handle


async def test_recover_runtime_handle_returns_none_for_missing():
    mgr = _make_manager(apps=_FakeAppsV1Api(raise_404=True))
    manifest = _ns(runtime=_ns(port=8080))
    handle = await mgr.recover_runtime_handle(submission_id="gone", manifest=manifest)
    assert handle is None


async def test_recover_runtime_handle_returns_unhealthy_for_crashloop():
    dep = _make_deployment(replicas=1, ready_replicas=0)
    pod = _make_pod(
        phase="Running",
        container_statuses=[
            _make_container_status("CrashLoopBackOff", "back-off 5m"),
        ],
    )
    mgr = _make_manager(
        apps=_FakeAppsV1Api(deployment=dep),
        core=_FakeCoreV1Api(pods=[pod]),
    )
    manifest = _ns(runtime=_ns(port=9000))
    handle = await mgr.recover_runtime_handle(submission_id="crash1", manifest=manifest)
    assert handle is not None
    assert handle.state == "unhealthy"
    assert handle.endpoint_url == "http://miner-crash1.eirel-miners.svc.cluster.local"
    assert mgr.runtime_handle("crash1") is None
