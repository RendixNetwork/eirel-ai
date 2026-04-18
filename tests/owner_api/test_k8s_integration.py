from __future__ import annotations

import os
import sys

import pytest

if os.environ.get("EIREL_K8S_INTEGRATION") != "1":
    # pytest.skip(allow_module_level=True) raises Skipped (a BaseException).
    # Pytest's collection catches it; a bare `python -c "import ..."` does
    # not.  Only skip when pytest is driving the import.
    if any("pytest" in a for a in sys.argv[:2]):
        pytest.skip(
            "set EIREL_K8S_INTEGRATION=1 to run k3d integration tests",
            allow_module_level=True,
        )

import asyncio
import contextlib
import io
import json
import socket
import subprocess
import tarfile
import time
from types import SimpleNamespace

import httpx

from infra.miner_runtime._k8s_helpers import DeploymentStatusCode
from infra.miner_runtime.runtime_manager import KubernetesMinerRuntimeManager


_NAMESPACE = "eirel-miners"
_SYSTEM_NAMESPACE = "eirel-system"
_HEALTH_TIMEOUT = 120
_KUBECONFIG = os.environ.get(
    "EIREL_OWNER_KUBECONFIG_PATH",
    os.path.expanduser("~/.kube/config"),
)
_RUNTIME_IMAGE = os.environ.get("EIREL_OWNER_RUNTIME_IMAGE", "miner-runtime:v1")


# -- Helpers -------------------------------------------------------------------


def _kubectl(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["kubectl", *args],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"kubectl {' '.join(args)} failed (rc={result.returncode}): "
            f"{result.stderr.strip()}"
        )
    return result.stdout.strip()


def _resource_exists(kind: str, name: str) -> bool:
    result = subprocess.run(
        ["kubectl", "get", kind, name, "-n", _NAMESPACE],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.returncode == 0


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def _port_forward(svc_name: str, svc_port: int):
    local_port = _find_free_port()
    proc = subprocess.Popen(
        [
            "kubectl", "port-forward",
            f"svc/{svc_name}", f"{local_port}:{svc_port}",
            "-n", _NAMESPACE,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(3)
    try:
        yield local_port
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _make_test_archive() -> bytes:
    app_code = (
        "from http.server import HTTPServer, BaseHTTPRequestHandler\n"
        "class H(BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        '        if self.path == "/healthz":\n'
        "            self.send_response(200)\n"
        "            self.end_headers()\n"
        '            self.wfile.write(b"ok")\n'
        "        else:\n"
        "            self.send_response(404)\n"
        "            self.end_headers()\n"
        "    def log_message(self, *a): pass\n"
        'HTTPServer(("0.0.0.0", 8080), H).serve_forever()\n'
    )
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = app_code.encode()
        info = tarfile.TarInfo(name="app.py")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _make_manifest() -> SimpleNamespace:
    return SimpleNamespace(
        runtime=SimpleNamespace(
            port=8080,
            health_path="/healthz",
            invoke_path="/v1/chat/completions",
        ),
        family_id="analyst",
        hotkey="test-hotkey",
    )


def _apply_raw_deployment(submission_id: str, *, cpu_request: str = "100m") -> None:
    dep_name = f"miner-{submission_id}"
    body = f"""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {dep_name}
  namespace: {_NAMESPACE}
  labels:
    eirel.dev/submission-id: "{submission_id}"
spec:
  replicas: 1
  selector:
    matchLabels:
      app: {dep_name}
  template:
    metadata:
      labels:
        app: {dep_name}
        eirel.dev/submission-id: "{submission_id}"
    spec:
      nodeSelector:
        eirel.dev/runtime-pool: "true"
        eirel.dev/runtime-class: "miner"
      containers:
      - name: {dep_name}
        image: busybox
        command: ["sleep", "3600"]
        resources:
          requests:
            cpu: "{cpu_request}"
"""
    subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=body.encode(),
        check=True,
        capture_output=True,
        timeout=30,
    )


# -- Session fixtures ----------------------------------------------------------


@pytest.fixture(scope="session")
def k8s_namespaces():
    for ns in [_NAMESPACE, _SYSTEM_NAMESPACE]:
        subprocess.run(
            f"kubectl create ns {ns} --dry-run=client -o yaml | kubectl apply -f -",
            shell=True,
            check=True,
            capture_output=True,
            timeout=30,
        )
    _kubectl("label", "ns", _NAMESPACE, f"name={_NAMESPACE}", "--overwrite")
    _kubectl(
        "label", "ns", _SYSTEM_NAMESPACE,
        f"name={_SYSTEM_NAMESPACE}", "--overwrite",
    )


@pytest.fixture(scope="session")
def runtime_manager(k8s_namespaces):
    return KubernetesMinerRuntimeManager(
        kubeconfig_path=_KUBECONFIG,
        namespace=_NAMESPACE,
        system_namespace=_SYSTEM_NAMESPACE,
        runtime_image=_RUNTIME_IMAGE,
        shared_secret_name="eirel-runtime-shared",
        service_domain="svc.cluster.local",
        health_timeout_seconds=_HEALTH_TIMEOUT,
    )


# -- Per-test cleanup ----------------------------------------------------------


@pytest.fixture
async def cleanup(runtime_manager):
    tracked: list[str] = []
    yield tracked
    for sid in tracked:
        dep_name = f"miner-{sid}"
        try:
            await runtime_manager.stop_runtime(sid, reason="test-cleanup")
        except Exception:
            pass
        for kind in ["deployment", "service", "networkpolicy"]:
            subprocess.run(
                [
                    "kubectl", "delete", kind, dep_name,
                    "-n", _NAMESPACE, "--ignore-not-found=true",
                ],
                capture_output=True,
                timeout=30,
            )
        subprocess.run(
            [
                "kubectl", "delete", "configmap", f"{dep_name}-code",
                "-n", _NAMESPACE, "--ignore-not-found=true",
            ],
            capture_output=True,
            timeout=30,
        )


# -- Tests ---------------------------------------------------------------------


async def test_end_to_end_submission_deploys_pod(runtime_manager, cleanup):
    sid = "integ-e2e-001"
    cleanup.append(sid)

    handle = await runtime_manager.ensure_runtime(
        submission_id=sid,
        archive_bytes=_make_test_archive(),
        manifest=_make_manifest(),
        assigned_node_name=None,
        requested_cpu_millis=100,
        requested_memory_bytes=64 * 1024 * 1024,
    )
    assert handle.submission_id == sid
    assert handle.state == "healthy"

    dep_name = f"miner-{sid}"
    assert _resource_exists("deployment", dep_name)
    assert _resource_exists("service", dep_name)
    assert _resource_exists("networkpolicy", dep_name)
    assert _resource_exists("configmap", f"{dep_name}-code")

    with _port_forward(dep_name, 8080) as local_port:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"http://localhost:{local_port}/healthz", timeout=10,
            )
            assert resp.status_code == 200

    await runtime_manager.stop_runtime(sid, reason="test-complete")
    cleanup.remove(sid)


async def test_drain_reschedules_pod_transparently(runtime_manager, cleanup):
    sid = "integ-drain-001"
    cleanup.append(sid)

    nodes = await runtime_manager.list_runtime_nodes()
    assert len(nodes) >= 2, "drain test requires at least 2 agent nodes"
    target_node = nodes[0].node_name

    for n in nodes:
        _kubectl("uncordon", n.node_name, check=False)

    await runtime_manager.ensure_runtime(
        submission_id=sid,
        archive_bytes=_make_test_archive(),
        manifest=_make_manifest(),
        assigned_node_name=target_node,
        requested_cpu_millis=100,
        requested_memory_bytes=64 * 1024 * 1024,
    )

    dep_name = f"miner-{sid}"
    original_node = _kubectl(
        "get", "pods", "-n", _NAMESPACE,
        "-l", f"app={dep_name}",
        "-o", "jsonpath={.items[0].spec.nodeName}",
    )
    assert original_node, "pod must be scheduled on a node"

    original_handle = runtime_manager.runtime_handle(sid)
    assert original_handle is not None

    drained_node = original_node
    try:
        _kubectl(
            "drain", drained_node,
            "--ignore-daemonsets",
            "--delete-emptydir-data",
            "--timeout=60s",
        )

        deadline = time.monotonic() + 90
        new_node = None
        while time.monotonic() < deadline:
            status = await runtime_manager.deployment_status(sid)
            if status.code == DeploymentStatusCode.READY:
                pods_json = _kubectl(
                    "get", "pods", "-n", _NAMESPACE,
                    "-l", f"app={dep_name}", "-o", "json",
                    check=False,
                )
                if pods_json:
                    for pod in json.loads(pods_json).get("items", []):
                        node = pod.get("spec", {}).get("nodeName", "")
                        phase = pod.get("status", {}).get("phase", "")
                        if (
                            node
                            and node != drained_node
                            and phase == "Running"
                        ):
                            new_node = node
                            break
                if new_node:
                    break
            await asyncio.sleep(3)

        assert new_node is not None, (
            f"pod did not reschedule away from {drained_node} within 90s"
        )
        assert new_node != drained_node

        with _port_forward(dep_name, 8080) as local_port:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"http://localhost:{local_port}/healthz", timeout=10,
                )
                assert resp.status_code == 200

        post_drain_handle = runtime_manager.runtime_handle(sid)
        assert post_drain_handle is not None
        assert post_drain_handle.submission_id == sid

    finally:
        _kubectl("uncordon", drained_node)


async def test_stop_runtime_deletes_all_resources(runtime_manager, cleanup):
    sid = "integ-stop-001"
    cleanup.append(sid)

    await runtime_manager.ensure_runtime(
        submission_id=sid,
        archive_bytes=_make_test_archive(),
        manifest=_make_manifest(),
        assigned_node_name=None,
        requested_cpu_millis=100,
        requested_memory_bytes=64 * 1024 * 1024,
    )

    dep_name = f"miner-{sid}"
    assert _resource_exists("deployment", dep_name)
    assert _resource_exists("service", dep_name)
    assert _resource_exists("networkpolicy", dep_name)
    assert _resource_exists("configmap", f"{dep_name}-code")

    await runtime_manager.stop_runtime(sid, reason="test-deletion")
    cleanup.remove(sid)

    await asyncio.sleep(3)

    assert not _resource_exists("deployment", dep_name)
    assert not _resource_exists("service", dep_name)
    assert not _resource_exists("networkpolicy", dep_name)
    assert not _resource_exists("configmap", f"{dep_name}-code")


async def test_reconcile_deletes_orphan_deployment(runtime_manager, cleanup):
    orphan_id = "orphan-1"
    cleanup.append(orphan_id)

    _apply_raw_deployment(orphan_id, cpu_request="50m")

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if _resource_exists("deployment", f"miner-{orphan_id}"):
            break
        await asyncio.sleep(1)
    assert _resource_exists("deployment", f"miner-{orphan_id}")

    await runtime_manager.reconcile_active_submissions(set())

    await asyncio.sleep(5)

    assert not _resource_exists("deployment", f"miner-{orphan_id}")
    cleanup.remove(orphan_id)


async def test_pending_unschedulable_surfaces_from_deployment_status(
    runtime_manager, cleanup,
):
    sid = "unsched-1"
    cleanup.append(sid)

    _apply_raw_deployment(sid, cpu_request="9999")

    deadline = time.monotonic() + 30
    status = None
    while time.monotonic() < deadline:
        status = await runtime_manager.deployment_status(sid)
        if status.code == DeploymentStatusCode.PENDING_UNSCHEDULABLE:
            break
        await asyncio.sleep(2)

    assert status is not None
    assert status.code == DeploymentStatusCode.PENDING_UNSCHEDULABLE
