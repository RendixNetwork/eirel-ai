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


# -- K8s deployment helpers ---------------------------------------------------

_CONFIGMAP_MAX_BYTES = 900 * 1024

# Single ConfigMap key holding the whole cleaned submission as a gzipped
# tar. We deliberately do NOT map one file → one ConfigMap key: k8s
# ConfigMap keys must match ``[-._a-zA-Z0-9]+`` and are length-capped, so
# any submission with subdirectories (``pkg/mod.py``), unusual characters,
# or deep paths makes the ConfigMap API reject the whole object with 422.
# One opaque archive key sidesteps every key-charset / key-length / nested
# -path failure mode permanently; an init container unpacks it into a
# shared emptyDir before the runtime container starts.
_CODE_ARCHIVE_KEY = "code.tar.gz"
# Where the ConfigMap (archive) is mounted, and where the init container
# unpacks it for the runtime container to import from.
_CODE_ARCHIVE_MOUNT = "/submission-archive"
_CODE_EXTRACT_MOUNT = "/submission"
_CODE_EXTRACT_VOLUME = "submission-code-extracted"
_CODE_ARCHIVE_VOLUME = "submission-code"

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


def _repack_clean_archive(files: dict[str, bytes]) -> bytes:
    """Re-tar the cleaned file set into one deterministic gzipped archive.

    ``files`` has already been path-validated and junk-filtered by
    :func:`_extract_archive_to_dict`. We repack (rather than forwarding the
    miner's original tar) so the archive shipped to the pod contains
    exactly the sanitized tree — no ``__pycache__``/``.git`` bloat, no
    path-escape members, no preserved ownership/mtime. Deterministic
    output (sorted names, zeroed mtime/uid/gid) keeps the ConfigMap stable
    across re-deploys so unchanged code doesn't churn the object.
    """
    raw = io.BytesIO()
    # mtime=0 → reproducible gzip header; sorted names → stable tar order.
    with tarfile.open(fileobj=raw, mode="w:gz", compresslevel=9) as tar:
        for path in sorted(files):
            data = files[path]
            info = tarfile.TarInfo(name=path)
            info.size = len(data)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    return raw.getvalue()


def _check_archive_configmap_size(archive_bytes: bytes) -> None:
    """Reject archives that won't fit in a ConfigMap.

    etcd stores the base64 of ``binaryData`` values, so the on-wire size
    is ~4/3 the raw archive. Bound the base64 size, not the raw size, so
    the check matches what the API server actually enforces (~1 MiB).
    """
    encoded = (len(archive_bytes) + 2) // 3 * 4
    if encoded > _CONFIGMAP_MAX_BYTES:
        from .runtime_manager import RuntimeManagerError

        raise RuntimeManagerError(
            f"submission archive too large for ConfigMap: {encoded} bytes "
            f"base64 (limit {_CONFIGMAP_MAX_BYTES} bytes); raw archive "
            f"{len(archive_bytes)} bytes"
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
    archive_bytes: bytes,
) -> dict[str, Any]:
    """One ConfigMap, one key (``code.tar.gz``) holding the whole archive.
    """
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": name,
            "labels": {"eirel.dev/submission-id": submission_id},
        },
        "binaryData": {
            _CODE_ARCHIVE_KEY: base64.b64encode(archive_bytes).decode(),
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
    # Deliberately NO blanket 0.0.0.0/0:443 rule. Miners must route LLM
    # traffic through provider-proxy (chutes-only) and retrieval through
    # the owner-operated tool services — no direct outbound HTTPS to
    # api.openai.com, api.anthropic.com, or anywhere else. This preserves
    # the baseline-as-reference property (miners run open-source models
    # through the proxy, they can't clone GPT-5 to win).
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
