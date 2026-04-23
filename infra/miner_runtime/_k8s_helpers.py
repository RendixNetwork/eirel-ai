from __future__ import annotations

import asyncio
import base64
import io
import posixpath
import re
import tarfile
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


def parse_cpu_to_millis(value: str | int) -> int:
    if value is None or value == "":
        return 0
    if isinstance(value, int):
        return max(0, value * 1000)
    s = str(value).strip()
    if not s:
        return 0
    if s.endswith("m"):
        return max(0, int(s[:-1]))
    return max(0, int(float(s) * 1000))


_BINARY_SUFFIXES: dict[str, int] = {
    "Ki": 2**10,
    "Mi": 2**20,
    "Gi": 2**30,
    "Ti": 2**40,
}
_DECIMAL_SUFFIXES: dict[str, int] = {
    "K": 10**3,
    "M": 10**6,
    "G": 10**9,
    "T": 10**12,
}
_MEM_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*(Ki|Mi|Gi|Ti|K|M|G|T)?$")


def parse_memory_to_bytes(value: str | int) -> int:
    if value is None or value == "":
        return 0
    if isinstance(value, int):
        return max(0, value)
    s = str(value).strip()
    if not s:
        return 0
    m = _MEM_RE.match(s)
    if not m:
        return max(0, int(s))
    numeric = float(m.group(1))
    suffix = m.group(2) or ""
    if suffix in _BINARY_SUFFIXES:
        return max(0, int(numeric * _BINARY_SUFFIXES[suffix]))
    if suffix in _DECIMAL_SUFFIXES:
        return max(0, int(numeric * _DECIMAL_SUFFIXES[suffix]))
    return max(0, int(numeric))


class DeploymentStatusCode(StrEnum):
    READY = "ready"
    PENDING_STARTING = "pending_starting"
    PENDING_UNSCHEDULABLE = "pending_unschedulable"
    CRASHLOOP = "crashloop"
    MISSING = "missing"
    UNKNOWN = "unknown"


@dataclass(slots=True, frozen=True)
class DeploymentStatus:
    code: DeploymentStatusCode
    ready_replicas: int
    desired_replicas: int
    message: str
    last_pod_phase: str | None


# -- Phase 4 helpers -----------------------------------------------------------

_CONFIGMAP_MAX_BYTES = 900 * 1024

_ARCHIVE_EXCLUDE_DIRS = {"__pycache__", ".git", ".venv", "venv", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules"}
_ARCHIVE_EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".pyd", ".swp", ".swo"}
_ARCHIVE_EXCLUDE_NAMES = {".DS_Store", "Thumbs.db"}


def _should_skip_archive_entry(normalized_path: str) -> bool:
    parts = normalized_path.split("/")
    if any(part in _ARCHIVE_EXCLUDE_DIRS for part in parts):
        return True
    leaf = parts[-1] if parts else ""
    if leaf in _ARCHIVE_EXCLUDE_NAMES:
        return True
    for suffix in _ARCHIVE_EXCLUDE_SUFFIXES:
        if leaf.endswith(suffix):
            return True
    return False


def _extract_archive_to_dict(archive_bytes: bytes) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.isdir():
                continue
            raw = member.name.replace("\\", "/")
            normalized = posixpath.normpath(raw)
            if normalized == ".":
                continue
            if normalized.startswith("/") or normalized.startswith(".."):
                raise ValueError(
                    f"archive entry would escape root: {member.name}"
                )
            if _should_skip_archive_entry(normalized):
                continue
            f = tar.extractfile(member)
            if f is not None:
                files[normalized] = f.read()
    return files


def _check_configmap_size(files: dict[str, bytes]) -> None:
    total = sum(len(v) for v in files.values())
    if total > _CONFIGMAP_MAX_BYTES:
        from .runtime_manager import RuntimeManagerError

        raise RuntimeManagerError(
            f"archive too large for ConfigMap: {total} bytes "
            f"(limit {_CONFIGMAP_MAX_BYTES} bytes)"
        )


async def _create_or_replace(create_fn, replace_fn, *, namespace, name, body, _call=None) -> None:
    from kubernetes.client.exceptions import ApiException

    _do = _call or (lambda fn, **kw: asyncio.to_thread(fn, **kw))
    try:
        await _do(create_fn, namespace=namespace, body=body)
    except ApiException as exc:
        if exc.status == 409:
            await _do(replace_fn, name=name, namespace=namespace, body=body)
        else:
            raise


def _build_code_configmap(
    *,
    name: str,
    submission_id: str,
    files: dict[str, bytes],
) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": name,
            "labels": {"eirel.dev/submission-id": submission_id},
        },
        "binaryData": {
            path: base64.b64encode(content).decode()
            for path, content in files.items()
        },
    }


def _build_k8s_deployment(
    *,
    name: str,
    submission_id: str,
    manifest,
    image: str,
    shared_secret_name: str,
    code_configmap_name: str,
    assigned_node_name: str | None,
    requested_cpu_millis: int,
    requested_memory_bytes: int,
    health_path: str,
    port: int,
    probe_period_seconds: int,
    deployment_id: str | None = None,
    internal_service_token: str = "",
    provider_proxy_url: str = "",
    provider_proxy_token: str = "",
) -> dict[str, Any]:
    from .runtime_manager import _deployment_manifest_common

    manifests = _deployment_manifest_common(
        deployment_name=name,
        service_name=name,
        submission_id=submission_id,
        deployment_id=deployment_id,
        artifact_url=image,
        manifest=manifest,
        internal_service_token=internal_service_token,
        provider_proxy_url=provider_proxy_url,
        provider_proxy_token=provider_proxy_token,
        assigned_node_name=assigned_node_name,
        requested_cpu_millis=requested_cpu_millis,
        requested_memory_bytes=requested_memory_bytes,
        readiness_probe_path=health_path,
        liveness_probe_path=health_path,
        probe_period_seconds=probe_period_seconds,
        shared_secret_name=shared_secret_name,
        code_configmap_name=code_configmap_name,
    )
    deployment = next(m for m in manifests if m["kind"] == "Deployment")

    deployment["metadata"].setdefault("labels", {})
    deployment["metadata"]["labels"]["eirel.dev/submission-id"] = submission_id

    pod_labels = deployment["spec"]["template"]["metadata"]["labels"]
    pod_labels["eirel.dev/submission-id"] = submission_id
    family_id = getattr(manifest, "family_id", None)
    if family_id:
        pod_labels["eirel.dev/family-id"] = str(family_id)
    hotkey = getattr(manifest, "hotkey", None)
    if hotkey:
        pod_labels["eirel.dev/hotkey"] = str(hotkey)

    pod_spec = deployment["spec"]["template"]["spec"]
    pod_spec["terminationGracePeriodSeconds"] = 15

    container = pod_spec["containers"][0]
    for p in container.get("ports", []):
        p["name"] = "http"

    return deployment


def _build_k8s_service(*, name: str, port: int) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": name},
        "spec": {
            "type": "ClusterIP",
            "selector": {"app": name},
            "ports": [{"port": port, "targetPort": port}],
        },
    }


def _build_network_policy(
    *,
    name: str,
    submission_id: str,
    system_namespace: str,
    port: int,
    control_plane_namespace: str = "eirel-control-plane",
    host_ip: str = "",
) -> dict[str, Any]:
    ingress_from: list[dict[str, Any]] = [
        {
            "namespaceSelector": {
                "matchLabels": {"name": system_namespace},
            },
        },
        {
            "namespaceSelector": {
                "matchLabels": {"name": control_plane_namespace},
            },
        },
    ]
    egress_rules: list[dict[str, Any]] = []
    if host_ip:
        egress_rules.append({
            "to": [{"ipBlock": {"cidr": f"{host_ip}/32"}}],
            "ports": [
                {"port": 18092, "protocol": "TCP"},
                {"port": 18085, "protocol": "TCP"},
                {"port": 18086, "protocol": "TCP"},
                {"port": 18087, "protocol": "TCP"},
                {"port": 18091, "protocol": "TCP"},
            ],
        })
    else:
        egress_rules.append({
            "to": [
                {
                    "namespaceSelector": {
                        "matchLabels": {"name": system_namespace},
                    },
                    "podSelector": {
                        "matchLabels": {"app": "provider-proxy"},
                    },
                },
            ],
            "ports": [{"port": 8092}],
        })
    egress_rules.append({
        "to": [
            {
                "namespaceSelector": {},
                "podSelector": {
                    "matchLabels": {"k8s-app": "kube-dns"},
                },
            },
        ],
        "ports": [{"port": 53, "protocol": "UDP"}],
    })
    egress_rules.append({
        "to": [{"ipBlock": {"cidr": "0.0.0.0/0"}}],
        "ports": [{"port": 443, "protocol": "TCP"}],
    })
    return {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "NetworkPolicy",
        "metadata": {
            "name": name,
            "labels": {"eirel.dev/submission-id": submission_id},
        },
        "spec": {
            "podSelector": {"matchLabels": {"app": name}},
            "policyTypes": ["Ingress", "Egress"],
            "ingress": [
                {
                    "from": ingress_from,
                },
            ],
            "egress": egress_rules,
        },
    }
