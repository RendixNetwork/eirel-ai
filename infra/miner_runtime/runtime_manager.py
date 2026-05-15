from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._k8s_helpers import (
    DeploymentStatus,
    DeploymentStatusCode,
    _CODE_ARCHIVE_KEY,
    _CODE_ARCHIVE_MOUNT,
    _CODE_ARCHIVE_VOLUME,
    _CODE_EXTRACT_MOUNT,
    _CODE_EXTRACT_VOLUME,
    _build_code_configmap,
    _build_k8s_deployment,
    _build_k8s_service,
    _build_network_policy,
    _check_archive_configmap_size,
    _create_or_replace,
    _extract_archive_to_dict,
    _repack_clean_archive,
)

logger = logging.getLogger(__name__)


class RuntimeManagerError(RuntimeError):
    pass


@dataclass(slots=True)
class MinerRuntimeHandle:
    submission_id: str
    endpoint_url: str
    container_name: str
    host_port: int
    state: str


@dataclass(slots=True)
class RuntimeNodeInfo:
    node_name: str
    labels: dict[str, str]
    ready: bool
    schedulable: bool
    allocatable_cpu_millis: int
    allocatable_memory_bytes: int
    allocatable_pod_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


async def _run_command(command: list[str], *, check: bool = True) -> str:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if check and process.returncode != 0:
        raise RuntimeManagerError(
            f"command failed with exit code {process.returncode}: {stderr.decode().strip()}"
        )
    return stdout.decode().strip()


class MinerRuntimeManager:
    def __init__(self) -> None:
        self._records: dict[str, Any] = {}

    async def ensure_runtime(self, **kwargs):  # pragma: no cover - interface only
        raise NotImplementedError

    async def stop_runtime(self, submission_id: str, *, reason: str, soft: bool = False):  # pragma: no cover - interface only
        raise NotImplementedError

    async def reconcile_active_submissions(self, active_submission_ids: set[str]):  # pragma: no cover - interface only
        raise NotImplementedError

    async def list_runtime_nodes(self) -> list[RuntimeNodeInfo]:  # pragma: no cover - interface only
        raise NotImplementedError

    def runtime_handle(self, deployment_id: str) -> MinerRuntimeHandle | None:
        record = self._records.get(deployment_id)
        return None if record is None else record.handle

    async def recover_runtime_handle(self, *, submission_id: str, manifest):  # pragma: no cover - interface only
        raise NotImplementedError

    async def deployment_status(self, submission_id: str) -> DeploymentStatus:
        return DeploymentStatus(
            code=DeploymentStatusCode.UNKNOWN,
            ready_replicas=0,
            desired_replicas=0,
            message="not supported by this backend",
            last_pod_phase=None,
        )

    async def is_container_running(self, submission_id: str) -> bool | None:
        """Check if the container for a submission is running.

        Returns True/False, or None if the check is not supported by this backend.
        """
        return None  # Not supported by default


def _millicores_to_str(value: int) -> str:
    return f"{max(0, int(value))}m"


def _bytes_to_mi(value: int) -> str:
    mib = max(0, int(value)) // (1024 * 1024)
    return f"{mib}Mi"


def _deployment_manifest_common(
    *,
    deployment_name: str,
    service_name: str,
    submission_id: str,
    artifact_url: str,
    manifest,
    internal_service_token: str,
    provider_proxy_url: str,
    provider_proxy_token: str,
    assigned_node_name: str | None,
    requested_cpu_millis: int,
    requested_memory_bytes: int,
    deployment_id: str | None = None,
    readiness_probe_path: str = "/healthz",
    liveness_probe_path: str = "/healthz",
    probe_period_seconds: int = 5,
    shared_secret_name: str | None = None,
    code_configmap_name: str | None = None,
    system_namespace: str = "eirel-system",
    control_plane_namespace: str = "eirel-control-plane",
    emit_network_policy: bool = False,
) -> list[dict[str, Any]]:
    runtime = getattr(manifest, "runtime", None)
    port = int(getattr(runtime, "port", 8080) or 8080)
    health_path = str(getattr(runtime, "health_path", "/healthz") or "/healthz")
    invoke_path = str(getattr(runtime, "invoke_path", "/v1/chat/completions") or "/v1/chat/completions")
    container: dict[str, Any] = {
        "name": deployment_name,
        "image": artifact_url,
        # ``eirel-miner-runtime:latest`` is a rolling tag — we push over
        # it each time the SDK changes.  With the k8s default ``IfNotPresent``,
        # kubelet keeps the first-pulled digest indefinitely and new
        # pods silently run an old SDK while we think they're fresh.
        # ``Always`` costs one manifest round-trip per pod create (the
        # layers are cached on-node anyway) and guarantees every new
        # pod sees the current image.
        "imagePullPolicy": "Always",
        "ports": [{"containerPort": port}],
        "resources": {
            "requests": {
                "cpu": _millicores_to_str(requested_cpu_millis),
                "memory": _bytes_to_mi(requested_memory_bytes),
            },
            "limits": {
                "cpu": _millicores_to_str(requested_cpu_millis),
                "memory": _bytes_to_mi(requested_memory_bytes),
            },
        },
        "readinessProbe": {
            "httpGet": {"path": readiness_probe_path, "port": port},
            "periodSeconds": probe_period_seconds,
        },
        "livenessProbe": {
            "httpGet": {"path": liveness_probe_path, "port": port},
            "periodSeconds": probe_period_seconds,
            "initialDelaySeconds": 60,
            "failureThreshold": 4,
        },
    }
    sdk_runtime = getattr(manifest, "sdk_runtime", None)
    inference = getattr(manifest, "inference", None)
    package_mode = getattr(sdk_runtime, "package_mode", "package") or "package"
    dependency_group = getattr(sdk_runtime, "dependency_group", None) or ""
    entry_module = getattr(sdk_runtime, "entry_module", None) or "app"
    app_object = getattr(sdk_runtime, "app_object", None) or "app"
    providers = getattr(inference, "providers", None) or []
    provider = providers[0] if providers else "openai"
    model = getattr(inference, "model", None) or "gpt-4.1-mini"
    miner_env = [
        {"name": "MINER_SUBMISSION_ID", "value": submission_id},
        {"name": "MINER_HEALTH_PATH", "value": health_path},
        {"name": "MINER_INVOKE_PATH", "value": invoke_path},
        {"name": "MINER_RUNTIME_PORT", "value": str(port)},
        {"name": "MINER_PACKAGE_MODE", "value": str(package_mode)},
        {"name": "MINER_DEPENDENCY_GROUP", "value": str(dependency_group)},
        {"name": "MINER_ENTRY_MODULE", "value": str(entry_module)},
        {"name": "MINER_APP_OBJECT", "value": str(app_object)},
        {"name": "MINER_PROVIDER", "value": str(provider)},
        {"name": "MINER_MODEL", "value": str(model)},
        {"name": "EIREL_DISABLE_REQUEST_AUTH", "value": "1"},
        {"name": "EIREL_RUN_BUDGET_USD", "value": "30.0"},
        # SDK in-memory quota knobs. These are defense-in-depth — the
        # subnet provider-proxy is the authoritative spend gate. The
        # defaults in the SDK (24 req / 60k tok) assume single-task
        # usage; production miners serve many tasks per run across many
        # runs, so we raise both limits and rely on the proxy for real
        # budget enforcement.
        {"name": "EIREL_PROVIDER_MAX_REQUESTS", "value": "1000"},
        {"name": "EIREL_PROVIDER_MAX_TOTAL_TOKENS", "value": "2000000"},
        {"name": "EIREL_PROVIDER_MAX_WALL_CLOCK_SECONDS", "value": "3600"},
        # Per-request timeout must exceed typical Chutes first-call
        # latency (up to ~60s for some models) but stay under the
        # owner-api proxy budget (150s) and validator httpx budget
        # (180s) so the whole chain drains in order on a slow call.
        {"name": "EIREL_PROVIDER_PER_REQUEST_TIMEOUT_SECONDS", "value": "120"},
    ]
    # Cost-attribution key used in X-Eirel-Job-Id on provider-proxy calls.
    # Must match the key owner-api uses when querying
    # ``/v1/jobs/{key}/cost`` in ScoringManager.fetch_deployment_cost —
    # otherwise llm_cost_usd stays 0 on every score record.  Prefer
    # deployment_id (the owner-api's natural key); fall back to
    # submission_id for older callers that don't pass it.
    _job_id = f"miner-{deployment_id}" if deployment_id else f"miner-{submission_id}"
    miner_env.append({"name": "EIREL_PROVIDER_PROXY_JOB_ID", "value": _job_id})
    # Provider proxy URL/token must be EIREL_-prefixed for the miner SDK's
    # MinerProviderConfig.validate_for_runtime check. The shared Secret today
    # only carries unprefixed PROVIDER_PROXY_TOKEN and no URL, so inject both
    # explicitly from the values owner-api already has in settings.
    miner_env.append({"name": "EIREL_PROVIDER_PROXY_URL", "value": provider_proxy_url})
    miner_env.append({"name": "EIREL_PROVIDER_PROXY_TOKEN", "value": provider_proxy_token})
    # Tool service URLs for the miner SDK (web_search / sandbox). The
    # shared Secret injected below provides the *_TOKEN counterparts
    # but no URLs, so without these the miner instantiates tool clients
    # with base_url="", fails silently on every call, and never records
    # citations. Names match what the general_chat agent's
    # _service_client() calls read via os.getenv.
    miner_env.append({
        "name": "EIREL_WEB_SEARCH_URL",
        "value": os.getenv("EIREL_WEB_SEARCH_TOOL_URL", ""),
    })
    miner_env.append({
        "name": "EIREL_SANDBOX_URL",
        "value": os.getenv("EIREL_SANDBOX_TOOL_URL", os.getenv("EIREL_SANDBOX_TOOL_SERVICE_URL", "")),
    })
    miner_env.append({
        "name": "EIREL_URL_FETCH_URL",
        "value": os.getenv("EIREL_URL_FETCH_TOOL_URL", ""),
    })
    miner_env.append({
        "name": "EIREL_RAG_URL",
        "value": os.getenv("EIREL_RAG_TOOL_URL", ""),
    })
    # Graph-runtime miners post their checkpoints back through
    # eirel-ai's internal checkpoints API, keyed by a per-miner
    # namespace. The namespace prefix matches what the checkpoints
    # router expects (``miner-{deployment_id}``). Skipped for
    # ``base_agent`` miners — they never call the SDK's
    # PostgresCheckpointer, so the env vars would be inert anyway.
    runtime_kind = str(getattr(runtime, "kind", "base_agent") or "base_agent")
    if runtime_kind == "graph" and deployment_id:
        backend_url = os.getenv(
            "EIREL_CHECKPOINT_BACKEND_URL",
            os.getenv("OWNER_API_URL", ""),
        )
        miner_env.append(
            {"name": "EIREL_CHECKPOINT_BACKEND_URL", "value": backend_url}
        )
        miner_env.append(
            {
                "name": "EIREL_CHECKPOINT_NAMESPACE",
                "value": f"miner-{deployment_id}",
            }
        )
        miner_env.append(
            {
                "name": "EIREL_CHECKPOINT_BACKEND_TOKEN",
                "value": internal_service_token,
            }
        )
        # Resume-token signing secret. Graph agents use this to mint the
        # token they return on Interrupt; the validator hands it back on
        # the next turn and the SDK decodes it to find the checkpoint.
        # Reuse the existing miner-internal secret rather than introducing
        # a new key material — owner-api already owns and rotates it.
        miner_env.append(
            {
                "name": "EIREL_RESUME_TOKEN_SECRET",
                "value": os.getenv(
                    "EIREL_RESUME_TOKEN_SECRET", internal_service_token
                ),
            }
        )
    if shared_secret_name is not None:
        container["envFrom"] = [{"secretRef": {"name": shared_secret_name}}]
        container["env"] = miner_env
    else:
        container["env"] = [
            {"name": "OWNER_API_URL", "value": str(artifact_url)},
            {"name": "INTERNAL_SERVICE_TOKEN", "value": internal_service_token},
            {"name": "PROVIDER_PROXY_URL", "value": provider_proxy_url},
            {"name": "PROVIDER_PROXY_TOKEN", "value": provider_proxy_token},
            {"name": "MINER_SUBMISSION_ID", "value": submission_id},
            {"name": "MINER_HEALTH_PATH", "value": health_path},
            {"name": "MINER_INVOKE_PATH", "value": invoke_path},
            {"name": "EIREL_PROVIDER_PROXY_JOB_ID", "value": _job_id},
            {"name": "SANDBOX_TOOL_URL", "value": os.getenv("EIREL_SANDBOX_TOOL_SERVICE_URL", "http://sandbox-tool-service:8091")},
            {"name": "SANDBOX_TOOL_TOKEN", "value": os.getenv("EIREL_SANDBOX_TOOL_SERVICE_TOKEN", "")},
            {"name": "EIREL_URL_FETCH_URL", "value": os.getenv("EIREL_URL_FETCH_TOOL_URL", "http://url-fetch-tool-service:8087")},
            {"name": "EIREL_RAG_URL", "value": os.getenv("EIREL_RAG_TOOL_URL", "http://rag-tool-service:8088")},
        ]
        # Mirror the graph-runtime checkpoint env into the no-secret
        # branch so K8s pods deployed without a shared Secret still see
        # the checkpoint backend config when ``runtime.kind == graph``.
        if runtime_kind == "graph" and deployment_id:
            backend_url = os.getenv(
                "EIREL_CHECKPOINT_BACKEND_URL",
                os.getenv("OWNER_API_URL", ""),
            )
            container["env"].extend(
                [
                    {"name": "EIREL_CHECKPOINT_BACKEND_URL", "value": backend_url},
                    {
                        "name": "EIREL_CHECKPOINT_NAMESPACE",
                        "value": f"miner-{deployment_id}",
                    },
                    {
                        "name": "EIREL_CHECKPOINT_BACKEND_TOKEN",
                        "value": internal_service_token,
                    },
                    {
                        "name": "EIREL_RESUME_TOKEN_SECRET",
                        "value": os.getenv(
                            "EIREL_RESUME_TOKEN_SECRET",
                            internal_service_token,
                        ),
                    },
                ]
            )
    pod_spec: dict[str, Any] = {
        "containers": [container],
        "restartPolicy": "Always",
    }
    if code_configmap_name is not None:
        # The runtime container imports code from an emptyDir
        # (``/submission``) that an init container fills by unpacking the
        # single-key archive ConfigMap. The runtime container never sees
        # the raw ConfigMap, so the legacy one-file-per-key layout (and
        # its 422 key-charset failures) is gone for good.
        container["volumeMounts"] = [
            {"name": _CODE_EXTRACT_VOLUME, "mountPath": _CODE_EXTRACT_MOUNT},
        ]
        # Python 3.12 ``tarfile`` ``data`` filter: blocks path traversal,
        # absolute paths, links escaping the tree, and device/special
        # files — safe extraction of an untrusted miner archive. owner-api
        # also validates the archive at build time (defense in depth).
        extract_script = (
            "import tarfile, pathlib; "
            f"dest = pathlib.Path({_CODE_EXTRACT_MOUNT!r}); "
            "dest.mkdir(parents=True, exist_ok=True); "
            "tar = tarfile.open("
            f"{_CODE_ARCHIVE_MOUNT!r} + '/' + {_CODE_ARCHIVE_KEY!r}); "
            "tar.extractall(dest, filter='data'); "
            "tar.close()"
        )
        pod_spec["initContainers"] = [
            {
                "name": "extract-submission-code",
                "image": artifact_url,
                "imagePullPolicy": "Always",
                "command": ["python", "-c", extract_script],
                "volumeMounts": [
                    {
                        "name": _CODE_ARCHIVE_VOLUME,
                        "mountPath": _CODE_ARCHIVE_MOUNT,
                        "readOnly": True,
                    },
                    {
                        "name": _CODE_EXTRACT_VOLUME,
                        "mountPath": _CODE_EXTRACT_MOUNT,
                    },
                ],
            }
        ]
        pod_spec["volumes"] = [
            {
                "name": _CODE_ARCHIVE_VOLUME,
                "configMap": {"name": code_configmap_name},
            },
            {
                "name": _CODE_EXTRACT_VOLUME,
                "emptyDir": {},
            },
        ]
    pod_spec["nodeSelector"] = {
        "eirel.dev/runtime-pool": "true",
        "eirel.dev/runtime-class": "miner",
    }
    if assigned_node_name:
        pod_spec["affinity"] = {
            "nodeAffinity": {
                "preferredDuringSchedulingIgnoredDuringExecution": [
                    {
                        "weight": 100,
                        "preference": {
                            "matchExpressions": [
                                {
                                    "key": "kubernetes.io/hostname",
                                    "operator": "In",
                                    "values": [assigned_node_name],
                                },
                            ],
                        },
                    },
                ],
            },
        }
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": deployment_name},
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": {"app": deployment_name}},
            "template": {
                "metadata": {"labels": {"app": deployment_name}},
                "spec": pod_spec,
            },
        },
    }
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": service_name},
        "spec": {
            "selector": {"app": deployment_name},
            "ports": [{"port": port, "targetPort": port}],
        },
    }
    result: list[dict[str, Any]] = [deployment, service]
    if emit_network_policy:
        network_policy = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {"name": deployment_name},
            "spec": {
                "podSelector": {"matchLabels": {"app": deployment_name}},
                "policyTypes": ["Ingress", "Egress"],
                "ingress": [
                    {
                        "from": [
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
                        ],
                    },
                ],
                "egress": [
                    {
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
                    },
                    {
                        "to": [
                            {
                                "namespaceSelector": {},
                                "podSelector": {
                                    "matchLabels": {"k8s-app": "kube-dns"},
                                },
                            },
                        ],
                        "ports": [{"port": 53, "protocol": "UDP"}],
                    },
                ],
            },
        }
        result.append(network_policy)
    return result


class DockerMinerRuntimeManager(MinerRuntimeManager):
    def __init__(
        self,
        *,
        docker_binary: str,
        runtime_image: str,
        sdk_root: str,
        work_root: str,
        endpoint_host: str,
        bind_host: str,
        health_timeout_seconds: float,
        docker_network: str | None = None,
    ) -> None:
        super().__init__()
        self.docker_binary = docker_binary
        self.runtime_image = runtime_image
        self.sdk_root = sdk_root
        self.work_root = work_root
        self.endpoint_host = endpoint_host
        self.bind_host = bind_host
        self.health_timeout_seconds = health_timeout_seconds
        self.docker_network = docker_network
        self._base_image_ready: bool = False
        self._base_image_lock = asyncio.Lock()
        self._base_image_tag: str = ""

    async def _ensure_base_image(self) -> None:
        """Build the SDK base runtime image once (lazy, on first use)."""
        async with self._base_image_lock:
            if self._base_image_ready:
                return
            sdk_dockerfile = Path(self.sdk_root) / "Dockerfile.runtime"
            if sdk_dockerfile.exists():
                tag = f"{self.runtime_image}-base"
                await _run_command([
                    self.docker_binary, "build",
                    "-f", str(sdk_dockerfile),
                    str(Path(self.sdk_root)),
                    "-t", tag,
                ])
                self._base_image_tag = tag
            else:
                self._base_image_tag = "python:3.12-slim"
            self._base_image_ready = True

    async def list_runtime_nodes(self) -> list[RuntimeNodeInfo]:
        import os as _os
        cpu_count = _os.cpu_count() or 4
        try:
            mem_bytes = _os.sysconf("SC_PAGE_SIZE") * _os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError):
            mem_bytes = 16 * 1024 ** 3
        return [
            RuntimeNodeInfo(
                node_name="docker-local",
                labels={
                    "eirel.dev/runtime-pool": "true",
                    "eirel.dev/runtime-class": "miner",
                },
                ready=True,
                schedulable=True,
                allocatable_cpu_millis=cpu_count * 1000,
                allocatable_memory_bytes=mem_bytes,
                allocatable_pod_count=256,
                metadata={"backend": "docker"},
            ),
        ]

    def _deployment_manifest(self, **kwargs) -> list[dict[str, Any]]:
        return _deployment_manifest_common(**kwargs)

    async def _wait_for_health(self, handle: MinerRuntimeHandle, manifest) -> None:
        """Poll the container's health endpoint until it responds or timeout."""
        import httpx as _httpx

        health_path = str(
            getattr(getattr(manifest, "runtime", None), "health_path", "/healthz") or "/healthz"
        )
        url = f"{handle.endpoint_url}{health_path}"
        deadline = time.monotonic() + self.health_timeout_seconds
        attempt = 0
        last_error = ""
        while time.monotonic() < deadline:
            # First check the container is still running
            try:
                inspect_out = await _run_command(
                    [self.docker_binary, "inspect", "-f", "{{.State.Running}}", handle.container_name],
                    check=False,
                )
                if inspect_out.strip().lower() != "true":
                    logs = await _run_command(
                        [self.docker_binary, "logs", "--tail", "30", handle.container_name],
                        check=False,
                    )
                    raise RuntimeManagerError(
                        f"container {handle.container_name} exited before becoming healthy. "
                        f"Last logs:\n{logs}"
                    )
            except RuntimeManagerError:
                raise
            except Exception:
                pass  # inspect failed, container may still be starting
            # Try the health endpoint via httpx
            try:
                async with _httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(url)
                    if resp.status_code < 500:
                        logger.info(
                            "container %s healthy after %d attempts (%.1fs)",
                            handle.container_name, attempt + 1,
                            self.health_timeout_seconds - (deadline - time.monotonic()),
                        )
                        return
            except Exception as exc:
                last_error = str(exc)
            attempt += 1
            await asyncio.sleep(min(2.0, 0.5 * (attempt ** 0.5)))
        raise RuntimeManagerError(
            f"container {handle.container_name} did not become healthy within "
            f"{self.health_timeout_seconds}s ({attempt} attempts). Last error: {last_error}"
        )

    async def _is_healthy(self, handle: MinerRuntimeHandle, manifest) -> bool:
        """Check if the container is running and its health endpoint responds."""
        import httpx as _httpx

        try:
            inspect_out = await _run_command(
                [self.docker_binary, "inspect", "-f", "{{.State.Running}}", handle.container_name],
                check=False,
            )
            if inspect_out.strip().lower() != "true":
                return False
        except Exception:
            return False
        health_path = str(
            getattr(getattr(manifest, "runtime", None), "health_path", "/healthz") or "/healthz"
        )
        url = f"{handle.endpoint_url}{health_path}"
        try:
            async with _httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
                return resp.status_code < 500
        except Exception:
            return False

    async def ensure_runtime(self, **kwargs) -> MinerRuntimeHandle:
        import io as _io
        import tarfile as _tarfile

        submission_id = str(kwargs["submission_id"])
        archive_bytes = kwargs["archive_bytes"]
        manifest = kwargs["manifest"]
        owner_api_url = kwargs["owner_api_url"]
        internal_service_token = kwargs["internal_service_token"]
        provider_proxy_url = kwargs["provider_proxy_url"]
        provider_proxy_token = kwargs["provider_proxy_token"]
        run_budget_usd: str = str(kwargs.get("run_budget_usd", "30.0"))
        assigned_node_name = kwargs.get("assigned_node_name")
        requested_cpu_millis = int(kwargs.get("requested_cpu_millis", 0) or 0)
        requested_memory_bytes = int(kwargs.get("requested_memory_bytes", 0) or 0)
        del owner_api_url

        container_name = f"eirel-miner-{submission_id}"
        port = int(getattr(getattr(manifest, "runtime", None), "port", 8080) or 8080)

        # Extract submission archive to work directory
        work_dir = Path(self.work_root) / submission_id
        work_dir.mkdir(parents=True, exist_ok=True)
        with _tarfile.open(fileobj=_io.BytesIO(archive_bytes), mode="r:gz") as tar:
            tar.extractall(path=str(work_dir))

        # Ensure the SDK base image is built (lazy, once per process)
        await self._ensure_base_image()

        # Remove any existing container with the same name (from previous deployments)
        await _run_command(
            [self.docker_binary, "rm", "-f", container_name], check=False,
        )
        # Resolve service URLs: if on a named docker network, use the
        # compose service name directly; otherwise rewrite for host access.
        if self.docker_network:
            resolved_provider_proxy_url = provider_proxy_url
        else:
            resolved_provider_proxy_url = provider_proxy_url.replace(
                "provider-proxy:8092", "host.docker.internal:18092"
            )
        run_command = [
            self.docker_binary,
            "run",
            "-d",
            "--name",
            container_name,
            "--add-host", "host.docker.internal:host-gateway",
            "-p",
            f"{self.bind_host}::{port}",
            "-v",
            f"{str(work_dir)}:/submission",
            "-e",
            f"MINER_PACKAGE_MODE={getattr(getattr(manifest, 'sdk_runtime', None), 'package_mode', 'package') or 'package'}",
            "-e",
            f"MINER_DEPENDENCY_GROUP={getattr(getattr(manifest, 'sdk_runtime', None), 'dependency_group', 'providers') or 'providers'}",
            "-e",
            f"MINER_ENTRY_MODULE={getattr(getattr(manifest, 'sdk_runtime', None), 'entry_module', 'app') or 'app'}",
            "-e",
            f"MINER_APP_OBJECT={getattr(getattr(manifest, 'sdk_runtime', None), 'app_object', 'app') or 'app'}",
            "-e",
            f"MINER_RUNTIME_PORT={port}",
            "-e",
            f"MINER_PROVIDER={(getattr(getattr(manifest, 'inference', None), 'providers', None) or ['openai'])[0]}",
            "-e",
            f"MINER_MODEL={getattr(getattr(manifest, 'inference', None), 'model', 'gpt-4.1-mini')}",
            "-e",
            f"PROVIDER_PROXY_URL={resolved_provider_proxy_url}",
            "-e",
            f"EIREL_PROVIDER_PROXY_URL={resolved_provider_proxy_url}",
            "-e",
            f"PROVIDER_PROXY_TOKEN={provider_proxy_token}",
            "-e",
            f"EIREL_PROVIDER_PROXY_TOKEN={provider_proxy_token}",
            "-e",
            f"INTERNAL_SERVICE_TOKEN={internal_service_token}",
            "-e",
            f"EIREL_PROVIDER_PROXY_JOB_ID=miner-{kwargs.get('deployment_id') or submission_id}",
            "-e",
            f"EIREL_RUN_BUDGET_USD={run_budget_usd}",
            "-e",
            f"MINER_SUBMISSION_ID={submission_id}",
            self._base_image_tag,
        ]
        # If a docker network is configured, attach the miner container to it
        # so compose services (owner-api, validator-engine) can reach it directly.
        if self.docker_network:
            run_command.insert(run_command.index("--name"), "--network")
            run_command.insert(run_command.index("--name"), self.docker_network)
        await _run_command(run_command)
        # Build the endpoint URL.  When on a named docker network the
        # container is reachable by name on its internal port; otherwise fall
        # back to the host-mapped port.
        if self.docker_network:
            endpoint_url = f"http://{container_name}:{port}"
            endpoint_port = port
        else:
            port_command = [self.docker_binary, "port", container_name, f"{port}/tcp"]
            port_output = await _run_command(port_command)
            endpoint_port = int(str(port_output).split(":")[-1].strip())
            endpoint_url = f"http://{self.endpoint_host}:{endpoint_port}"
        handle = MinerRuntimeHandle(
            submission_id=submission_id,
            endpoint_url=endpoint_url,
            container_name=container_name,
            host_port=endpoint_port,
            state="starting",
        )
        self._records[submission_id] = type("Record", (), {"handle": handle})()
        try:
            await self._wait_for_health(handle, manifest)
        except Exception:
            try:
                await self.stop_runtime(submission_id, reason="health_check_failed")
            except Exception as cleanup_exc:
                logger.warning("cleanup after health failure failed: %s", cleanup_exc)
            raise
        handle.state = "healthy"
        return handle

    async def stop_runtime(self, submission_id: str, *, reason: str, soft: bool = False):
        container_name = f"eirel-miner-{submission_id}"
        logger.info("stopping container %s (reason=%s, soft=%s)", container_name, reason, soft)
        if soft:
            # Graceful stop (SIGTERM, then wait)
            await _run_command(
                [self.docker_binary, "stop", "-t", "10", container_name], check=False,
            )
        else:
            await _run_command(
                [self.docker_binary, "rm", "-f", container_name], check=False,
            )
        self._records.pop(submission_id, None)

    async def reconcile_active_submissions(self, active_submission_ids: set[str]):
        # Step 1: Clean up known orphans from _records
        orphaned = {
            sid for sid in self._records
            if sid not in active_submission_ids
        }
        for sid in orphaned:
            container_name = f"eirel-miner-{sid}"
            logger.info("reconcile: stopping orphaned container %s", container_name)
            await _run_command(
                [self.docker_binary, "rm", "-f", container_name], check=False,
            )
        self._records = {
            submission_id: record
            for submission_id, record in self._records.items()
            if submission_id in active_submission_ids
        }

        # Step 2: Scan Docker for eirel-miner-* containers unknown to _records
        # (e.g. left over from a previous owner-api process)
        try:
            ps_output = await _run_command(
                [self.docker_binary, "ps", "-a",
                 "--filter", "name=eirel-miner-",
                 "--format", "{{.Names}}"],
                check=False,
            )
            for name in (ps_output.strip().split("\n") if ps_output.strip() else []):
                if not name.startswith("eirel-miner-"):
                    continue
                sid = name.removeprefix("eirel-miner-")
                if sid not in active_submission_ids:
                    logger.info("reconcile: stopping Docker-orphaned container %s", name)
                    await _run_command(
                        [self.docker_binary, "rm", "-f", name], check=False,
                    )
        except Exception:
            logger.exception("reconcile: Docker container scan failed")

    async def is_container_running(self, submission_id: str) -> bool | None:
        container_name = f"eirel-miner-{submission_id}"
        try:
            inspect_out = await _run_command(
                [self.docker_binary, "inspect", "-f", "{{.State.Running}}", container_name],
                check=False,
            )
            return inspect_out.strip().lower() == "true"
        except Exception:
            return False

    async def recover_runtime_handle(self, *, submission_id: str, manifest):
        container_name = f"eirel-miner-{submission_id}"
        # Check container exists and is running
        try:
            inspect_out = await _run_command(
                [self.docker_binary, "inspect", "-f", "{{.State.Running}}", container_name],
                check=False,
            )
        except Exception:
            logger.info("recover: container %s not found", container_name)
            return None
        is_running = inspect_out.strip().lower() == "true"
        if not is_running:
            logger.info("recover: container %s exists but is not running", container_name)
            return None
        port = int(getattr(getattr(manifest, "runtime", None), "port", 8080) or 8080)
        if self.docker_network:
            endpoint_url = f"http://{container_name}:{port}"
            endpoint_port = port
        else:
            try:
                port_output = await _run_command(
                    [self.docker_binary, "port", container_name, f"{port}/tcp"],
                )
                endpoint_port = int(str(port_output).split(":")[-1].strip())
            except Exception:
                logger.warning("recover: could not resolve port for %s", container_name)
                return None
            endpoint_url = f"http://{self.endpoint_host}:{endpoint_port}"
        handle = MinerRuntimeHandle(
            submission_id=submission_id,
            endpoint_url=endpoint_url,
            container_name=container_name,
            host_port=endpoint_port,
            state="recovering",
        )
        healthy = await self._is_healthy(handle, manifest)
        handle.state = "healthy" if healthy else "unhealthy"
        if healthy:
            self._records[submission_id] = type("Record", (), {"handle": handle})()
            logger.info("recover: successfully recovered handle for %s at %s", container_name, endpoint_url)
        else:
            logger.warning("recover: container %s is running but unhealthy", container_name)
        return handle


class KubernetesMinerRuntimeManager(MinerRuntimeManager):

    _K8S_API_TIMEOUT = 30.0  # seconds per individual K8s API call

    async def _k8s_call(self, fn, *args, **kwargs):
        return await asyncio.wait_for(
            asyncio.to_thread(fn, *args, **kwargs),
            timeout=self._K8S_API_TIMEOUT,
        )

    def __init__(
        self,
        *,
        kubeconfig_path: str | None,
        namespace: str,
        system_namespace: str,
        runtime_image: str,
        shared_secret_name: str,
        service_domain: str,
        health_timeout_seconds: float,
        probe_period_seconds: int = 5,
        control_plane_namespace: str = "eirel-control-plane",
    ) -> None:
        super().__init__()
        try:
            from kubernetes import client, config
        except ImportError as exc:
            raise RuntimeError(
                "kubernetes package is required for the Kubernetes backend "
                "(pip install kubernetes)"
            ) from exc
        if kubeconfig_path:
            config.load_kube_config(config_file=kubeconfig_path)
        else:
            config.load_incluster_config()
        logger.info(
            "k8s client: namespace=%s, system_namespace=%s, control_plane_namespace=%s",
            namespace, system_namespace, control_plane_namespace,
        )
        self._apps = client.AppsV1Api()
        self._core = client.CoreV1Api()
        self._netv1 = client.NetworkingV1Api()
        self._namespace = namespace
        self._system_namespace = system_namespace
        self._control_plane_namespace = control_plane_namespace
        self._runtime_image = runtime_image
        self._shared_secret_name = shared_secret_name
        self._service_domain = service_domain
        self._health_timeout_seconds = health_timeout_seconds
        self._probe_period_seconds = probe_period_seconds

    async def list_runtime_nodes(self) -> list[RuntimeNodeInfo]:
        from ._k8s_helpers import parse_cpu_to_millis, parse_memory_to_bytes

        resp = await self._k8s_call(
            self._core.list_node,
            label_selector="eirel.dev/runtime-pool=true,eirel.dev/runtime-class=miner",
        )
        out: list[RuntimeNodeInfo] = []
        for n in resp.items:
            alloc = n.status.allocatable or {}
            conds = {c.type: c.status for c in (n.status.conditions or [])}
            # Pick the reachable IP so the Prometheus http_sd endpoint can
            # scrape node_exporter on k3s-managed miner hosts.  Prefer
            # ExternalIP when present (e.g. cloud-provisioned nodes), else
            # fall back to InternalIP (VPS-style setups where the routable
            # address is the InternalIP).
            addresses = getattr(n.status, "addresses", None) or []
            node_ip = next(
                (a.address for a in addresses if a.type == "ExternalIP"), ""
            ) or next(
                (a.address for a in addresses if a.type == "InternalIP"), ""
            )
            out.append(RuntimeNodeInfo(
                node_name=n.metadata.name,
                labels=dict(n.metadata.labels or {}),
                ready=conds.get("Ready") == "True",
                schedulable=not bool(n.spec.unschedulable),
                allocatable_cpu_millis=parse_cpu_to_millis(alloc.get("cpu", "0")),
                allocatable_memory_bytes=parse_memory_to_bytes(alloc.get("memory", "0")),
                allocatable_pod_count=int(alloc.get("pods", 0)),
                metadata={"backend": "kubernetes", "ssh_host": node_ip},
            ))
        return out

    async def deployment_status(self, submission_id: str) -> DeploymentStatus:
        from kubernetes.client.exceptions import ApiException

        dep_name = f"miner-{submission_id}"
        try:
            dep = await self._k8s_call(
                self._apps.read_namespaced_deployment,
                name=dep_name,
                namespace=self._namespace,
            )
        except ApiException as exc:
            if exc.status == 404:
                return DeploymentStatus(
                    code=DeploymentStatusCode.MISSING,
                    ready_replicas=0,
                    desired_replicas=0,
                    message="not found",
                    last_pod_phase=None,
                )
            raise
        ready = int(dep.status.ready_replicas or 0)
        desired = int(dep.spec.replicas or 0)
        if ready >= desired and desired > 0:
            return DeploymentStatus(
                code=DeploymentStatusCode.READY,
                ready_replicas=ready,
                desired_replicas=desired,
                message="ok",
                last_pod_phase="Running",
            )
        pods = await self._k8s_call(
            self._core.list_namespaced_pod,
            namespace=self._namespace,
            label_selector=f"app={dep_name}",
        )
        if not pods.items:
            return DeploymentStatus(
                code=DeploymentStatusCode.PENDING_STARTING,
                ready_replicas=ready,
                desired_replicas=desired,
                message="no pods yet",
                last_pod_phase=None,
            )
        pod = pods.items[0]
        phase = pod.status.phase
        for cond in (pod.status.conditions or []):
            if (
                cond.type == "PodScheduled"
                and cond.status == "False"
                and cond.reason == "Unschedulable"
            ):
                return DeploymentStatus(
                    code=DeploymentStatusCode.PENDING_UNSCHEDULABLE,
                    ready_replicas=ready,
                    desired_replicas=desired,
                    message=cond.message or "unschedulable",
                    last_pod_phase=phase,
                )
        for cs in (pod.status.container_statuses or []):
            if cs.state and cs.state.waiting:
                reason = cs.state.waiting.reason or ""
                if reason == "CrashLoopBackOff":
                    return DeploymentStatus(
                        code=DeploymentStatusCode.CRASHLOOP,
                        ready_replicas=ready,
                        desired_replicas=desired,
                        message=cs.state.waiting.message or "crashloop",
                        last_pod_phase=phase,
                    )
                if reason in ("ImagePullBackOff", "ErrImagePull"):
                    return DeploymentStatus(
                        code=DeploymentStatusCode.CRASHLOOP,
                        ready_replicas=ready,
                        desired_replicas=desired,
                        message=f"image pull failed: {cs.state.waiting.message or reason}",
                        last_pod_phase=phase,
                    )
        return DeploymentStatus(
            code=DeploymentStatusCode.PENDING_STARTING,
            ready_replicas=ready,
            desired_replicas=desired,
            message=f"phase={phase}",
            last_pod_phase=phase,
        )

    async def recover_runtime_handle(self, *, submission_id: str, manifest):
        status = await self.deployment_status(submission_id)
        if status.code == DeploymentStatusCode.MISSING:
            return None
        port = int(
            getattr(getattr(manifest, "runtime", None), "port", 8080) or 8080
        )
        endpoint_url = (
            f"http://miner-{submission_id}.{self._namespace}.{self._service_domain}:{port}"
        )
        handle_state = (
            "healthy" if status.code == DeploymentStatusCode.READY else "unhealthy"
        )
        handle = MinerRuntimeHandle(
            submission_id=submission_id,
            endpoint_url=endpoint_url,
            container_name=f"miner-{submission_id}",
            host_port=port,
            state=handle_state,
        )
        if status.code == DeploymentStatusCode.READY:
            self._records[submission_id] = type("Record", (), {"handle": handle})()
        return handle

    async def ensure_runtime(self, **kwargs) -> MinerRuntimeHandle:
        submission_id = str(kwargs["submission_id"])
        deployment_id = str(kwargs.get("deployment_id") or submission_id)
        archive_bytes: bytes = kwargs["archive_bytes"]
        manifest = kwargs["manifest"]
        assigned_node_name = kwargs.get("assigned_node_name")
        requested_cpu_millis = int(kwargs.get("requested_cpu_millis", 0) or 0)
        requested_memory_bytes = int(kwargs.get("requested_memory_bytes", 0) or 0)

        dep_name = f"miner-{submission_id}"
        port = int(
            getattr(getattr(manifest, "runtime", None), "port", 8080) or 8080
        )
        health_path = str(
            getattr(getattr(manifest, "runtime", None), "health_path", "/healthz")
            or "/healthz"
        )

        # Validate + junk-filter (raises on path escape), then repack the
        # cleaned tree into one deterministic archive. The pod's init
        # container unpacks this single key, so nested paths and odd
        # filenames can never break ConfigMap creation again.
        code_files = _extract_archive_to_dict(archive_bytes)
        clean_archive = _repack_clean_archive(code_files)
        _check_archive_configmap_size(clean_archive)

        cm_body = _build_code_configmap(
            name=f"{dep_name}-code",
            submission_id=submission_id,
            archive_bytes=clean_archive,
        )
        try:
            await _create_or_replace(
                self._core.create_namespaced_config_map,
                self._core.replace_namespaced_config_map,
                namespace=self._namespace,
                name=f"{dep_name}-code",
                body=cm_body,
                _call=self._k8s_call,
            )

            deployment_body = _build_k8s_deployment(
                name=dep_name,
                submission_id=submission_id,
                deployment_id=deployment_id,
                manifest=manifest,
                image=self._runtime_image,
                shared_secret_name=self._shared_secret_name,
                code_configmap_name=f"{dep_name}-code",
                assigned_node_name=assigned_node_name,
                requested_cpu_millis=requested_cpu_millis,
                requested_memory_bytes=requested_memory_bytes,
                health_path=health_path,
                port=port,
                probe_period_seconds=self._probe_period_seconds,
                internal_service_token=str(kwargs.get("internal_service_token", "")),
                provider_proxy_url=str(kwargs.get("provider_proxy_url", "")),
                provider_proxy_token=str(kwargs.get("provider_proxy_token", "")),
            )
            await _create_or_replace(
                self._apps.create_namespaced_deployment,
                self._apps.replace_namespaced_deployment,
                namespace=self._namespace,
                name=dep_name,
                body=deployment_body,
                _call=self._k8s_call,
            )

            svc_body = _build_k8s_service(name=dep_name, port=port)
            await _create_or_replace(
                self._core.create_namespaced_service,
                self._core.replace_namespaced_service,
                namespace=self._namespace,
                name=dep_name,
                body=svc_body,
                _call=self._k8s_call,
            )

            np_body = _build_network_policy(
                name=dep_name,
                submission_id=submission_id,
                system_namespace=self._system_namespace,
                control_plane_namespace=self._control_plane_namespace,
                port=port,
                host_ip=os.getenv("SERVER_A_HOST_IP", ""),
            )
            await _create_or_replace(
                self._netv1.create_namespaced_network_policy,
                self._netv1.replace_namespaced_network_policy,
                namespace=self._namespace,
                name=dep_name,
                body=np_body,
                _call=self._k8s_call,
            )
        except Exception:
            try:
                await self.stop_runtime(submission_id, reason="partial_deploy_cleanup")
            except Exception as cleanup_exc:
                logger.warning("cleanup after partial deploy failed: %s", cleanup_exc)
            raise

        deadline = time.monotonic() + self._health_timeout_seconds
        while time.monotonic() < deadline:
            status = await self.deployment_status(submission_id)
            if status.code == DeploymentStatusCode.READY:
                break
            if status.code == DeploymentStatusCode.CRASHLOOP:
                raise RuntimeManagerError(
                    f"{dep_name} entered CrashLoopBackOff: {status.message}"
                )
            await asyncio.sleep(2.0)
        else:
            raise RuntimeManagerError(
                f"{dep_name} did not become ready within "
                f"{self._health_timeout_seconds}s"
            )

        endpoint_url = (
            f"http://{dep_name}.{self._namespace}.{self._service_domain}:{port}"
        )
        handle = MinerRuntimeHandle(
            submission_id=submission_id,
            endpoint_url=endpoint_url,
            container_name=dep_name,
            host_port=port,
            state="healthy",
        )
        self._records[submission_id] = type("Record", (), {"handle": handle})()
        return handle

    async def stop_runtime(
        self, submission_id: str, *, reason: str, soft: bool = False,
    ) -> None:
        from kubernetes.client.exceptions import ApiException

        dep_name = f"miner-{submission_id}"
        cm_name = f"{dep_name}-code"
        delete_calls: list[tuple[Any, str, str]] = [
            (self._apps.delete_namespaced_deployment, dep_name, self._namespace),
            (self._core.delete_namespaced_service, dep_name, self._namespace),
            (self._netv1.delete_namespaced_network_policy, dep_name, self._namespace),
            (self._core.delete_namespaced_config_map, cm_name, self._namespace),
        ]
        for delete_fn, name, ns in delete_calls:
            try:
                await self._k8s_call(
                    delete_fn,
                    name=name,
                    namespace=ns,
                    propagation_policy="Background",
                )
            except ApiException as exc:
                if exc.status != 404:
                    logger.warning("k8s delete %s failed: %s", name, exc)
        self._records.pop(submission_id, None)

    async def reconcile_active_submissions(
        self, active_submission_ids: set[str],
    ) -> None:
        resp = await self._k8s_call(
            self._apps.list_namespaced_deployment,
            namespace=self._namespace,
            label_selector="eirel.dev/submission-id",
        )
        for dep in resp.items:
            sub_id = (dep.metadata.labels or {}).get(
                "eirel.dev/submission-id", "",
            )
            if sub_id and sub_id not in active_submission_ids:
                logger.info(
                    "reconcile: deleting stale deployment %s",
                    dep.metadata.name,
                )
                await self.stop_runtime(sub_id, reason="reconcile")
        self._records = {
            sid: rec
            for sid, rec in self._records.items()
            if sid in active_submission_ids
        }

