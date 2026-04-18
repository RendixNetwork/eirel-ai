from __future__ import annotations

"""Baremetal Docker runtime manager.

Deploys miner agent containers on remote baremetal servers via SSH.
A pre-built base image is transferred once per node; submission code is
synced via rsync and volume-mounted into the container at runtime.
"""

import asyncio
import io
import json
import logging
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from control_plane.owner_api.baremetal_inventory import BaremetalNode, parse_ansible_inventory
from infra.miner_runtime.remote_command import (
    RemoteCommandError,
    rsync_to_remote,
    run_remote_command,
    run_remote_docker,
    scp_to_remote,
)
from infra.miner_runtime.runtime_manager import (
    MinerRuntimeHandle,
    MinerRuntimeManager,
    RuntimeManagerError,
    RuntimeNodeInfo,
    _run_command,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BaremetalRecord:
    handle: MinerRuntimeHandle
    node: BaremetalNode


class BaremetalDockerMinerRuntimeManager(MinerRuntimeManager):
    """Manages Docker containers on remote baremetal servers via SSH."""

    def __init__(
        self,
        *,
        inventory_path: str,
        docker_binary: str,
        runtime_image: str,
        sdk_root: str,
        work_root: str,
        health_timeout_seconds: float,
        storage_root: str = "/var/lib/eirel",
        provider_proxy_url_override: str = "",
        research_tool_url_override: str = "",
    ) -> None:
        super().__init__()
        self.inventory_path = inventory_path
        self.docker_binary = docker_binary
        self.runtime_image = runtime_image
        self.sdk_root = sdk_root
        self.work_root = work_root
        self.health_timeout_seconds = health_timeout_seconds
        self.storage_root = storage_root
        self._provider_proxy_url_override = provider_proxy_url_override
        self._research_tool_url_override = research_tool_url_override
        self._node_map: dict[str, BaremetalNode] = {}
        self._base_image_ready: bool = False
        self._base_image_lock = asyncio.Lock()
        self._base_image_tag: str = ""
        self._base_image_pushed_to: set[str] = set()

    def _refresh_node_map(self) -> list[BaremetalNode]:
        nodes = parse_ansible_inventory(self.inventory_path)
        self._node_map = {n.node_name: n for n in nodes}
        return nodes

    async def _ensure_base_image(self) -> None:
        """Build the SDK base runtime image locally once (lazy, on first use)."""
        async with self._base_image_lock:
            if self._base_image_ready:
                return
            image_name = self.runtime_image.split(":")[0] if ":" in self.runtime_image else self.runtime_image
            sdk_dockerfile = Path(self.sdk_root) / "Dockerfile.runtime"
            if sdk_dockerfile.exists():
                tag = f"{image_name}-base"
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

    async def _ensure_base_image_on_node(self, node: BaremetalNode) -> None:
        """Transfer the base image to a remote node (once per node)."""
        if node.node_name in self._base_image_pushed_to:
            return
        with tempfile.NamedTemporaryFile(suffix=".tar", delete=True) as tmp:
            tarball_path = tmp.name
        await _run_command([self.docker_binary, "save", "-o", tarball_path, self._base_image_tag])
        try:
            remote_image_path = f"{self.storage_root}/images/base-runtime.tar"
            await run_remote_command(
                ssh_host=node.ssh_host,
                ssh_user=node.ssh_user,
                ssh_key_path=node.ssh_key_path,
                ssh_port=node.ssh_port,
                command=["mkdir", "-p", f"{self.storage_root}/images"],
                check=False,
            )
            await scp_to_remote(
                node=node,
                local_path=tarball_path,
                remote_path=remote_image_path,
                timeout_seconds=300.0,
            )
            await run_remote_docker(
                node=node,
                docker_args=["load", "-i", remote_image_path],
                timeout_seconds=120.0,
            )
        finally:
            Path(tarball_path).unlink(missing_ok=True)
        self._base_image_pushed_to.add(node.node_name)

    async def _sync_submission_to_node(
        self, node: BaremetalNode, work_dir: Path, submission_id: str,
    ) -> str:
        """Sync extracted submission code to the remote node. Returns remote path."""
        remote_dir = f"{self.storage_root}/submissions/{submission_id}"
        await run_remote_command(
            ssh_host=node.ssh_host,
            ssh_user=node.ssh_user,
            ssh_key_path=node.ssh_key_path,
            ssh_port=node.ssh_port,
            command=["mkdir", "-p", remote_dir],
            check=False,
        )
        await rsync_to_remote(
            node=node,
            local_dir=str(work_dir),
            remote_dir=remote_dir,
        )
        return remote_dir

    async def _wait_for_health(
        self, handle: MinerRuntimeHandle, manifest: Any, node: BaremetalNode,
    ) -> None:
        """Poll the container's health endpoint on a remote node until it responds."""
        health_path = str(
            getattr(getattr(manifest, "runtime", None), "health_path", "/healthz") or "/healthz"
        )
        url = f"{handle.endpoint_url}{health_path}"
        deadline = time.monotonic() + self.health_timeout_seconds
        attempt = 0
        last_error = ""
        while time.monotonic() < deadline:
            try:
                inspect_out = await run_remote_docker(
                    node=node,
                    docker_args=["inspect", "-f", "{{.State.Running}}", handle.container_name],
                    check=False,
                    timeout_seconds=10.0,
                )
                if inspect_out.strip().lower() != "true":
                    logs = await run_remote_docker(
                        node=node,
                        docker_args=["logs", "--tail", "30", handle.container_name],
                        check=False,
                        timeout_seconds=10.0,
                    )
                    raise RuntimeManagerError(
                        f"container {handle.container_name} exited before becoming healthy. "
                        f"Last logs:\n{logs}"
                    )
            except RuntimeManagerError:
                raise
            except Exception:
                pass
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
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

    async def _is_healthy(
        self, handle: MinerRuntimeHandle, manifest: Any, node: BaremetalNode,
    ) -> bool:
        """Check if a remote container is running and its health endpoint responds."""
        try:
            inspect_out = await run_remote_docker(
                node=node,
                docker_args=["inspect", "-f", "{{.State.Running}}", handle.container_name],
                check=False,
                timeout_seconds=10.0,
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
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
                return resp.status_code < 500
        except Exception:
            return False

    # ------------------------------------------------------------------
    # MinerRuntimeManager interface
    # ------------------------------------------------------------------

    async def list_runtime_nodes(self) -> list[RuntimeNodeInfo]:
        nodes = self._refresh_node_map()
        if not nodes:
            return []

        async def _probe(node: BaremetalNode) -> RuntimeNodeInfo:
            try:
                raw = await run_remote_command(
                    ssh_host=node.ssh_host,
                    ssh_user=node.ssh_user,
                    ssh_key_path=node.ssh_key_path,
                    ssh_port=node.ssh_port,
                    command=["cat", f"{self.storage_root}/node-status.json"],
                    timeout_seconds=10.0,
                )
                status = json.loads(raw)
                cpu_count = int(status.get("cpu_count", 0))
                mem_bytes = int(status.get("mem_total_bytes", 0))
                disk_bytes = int(status.get("disk_avail_bytes", 0))
                drain_active = bool(status.get("drain_active", False))
                pod_capacity = min(
                    cpu_count,
                    mem_bytes // (512 * 1024 * 1024) if mem_bytes > 0 else 0,
                    disk_bytes // (10 * 1024 * 1024 * 1024) if disk_bytes > 0 else 0,
                )
                return RuntimeNodeInfo(
                    node_name=node.node_name,
                    labels={
                        "eirel.dev/runtime-pool": "true",
                        "eirel.dev/runtime-class": "miner",
                    },
                    ready=True,
                    schedulable=not drain_active,
                    allocatable_cpu_millis=cpu_count * 1000,
                    allocatable_memory_bytes=mem_bytes,
                    allocatable_pod_count=pod_capacity,
                    metadata={
                        "backend": "baremetal",
                        "ssh_host": node.ssh_host,
                        "disk_avail_bytes": disk_bytes,
                    },
                )
            except Exception as exc:
                logger.warning("failed to probe node %s: %s", node.node_name, exc)
                return RuntimeNodeInfo(
                    node_name=node.node_name,
                    labels={
                        "eirel.dev/runtime-pool": "true",
                        "eirel.dev/runtime-class": "miner",
                    },
                    ready=False,
                    schedulable=False,
                    allocatable_cpu_millis=0,
                    allocatable_memory_bytes=0,
                    allocatable_pod_count=0,
                    metadata={"backend": "baremetal", "error": str(exc)},
                )

        return list(await asyncio.gather(*[_probe(n) for n in nodes]))

    async def ensure_runtime(self, **kwargs: Any) -> MinerRuntimeHandle:
        submission_id = str(kwargs["submission_id"])
        archive_bytes: bytes = kwargs["archive_bytes"]
        manifest = kwargs["manifest"]
        provider_proxy_url: str = self._provider_proxy_url_override or kwargs.get("provider_proxy_url", "")
        provider_proxy_token: str = kwargs.get("provider_proxy_token", "")
        research_tool_url: str = self._research_tool_url_override or kwargs.get("research_tool_url", "")
        research_tool_token: str = kwargs.get("research_tool_token", "")
        internal_service_token: str = kwargs.get("internal_service_token", "")
        assigned_node_name: str | None = kwargs.get("assigned_node_name")
        requested_cpu_millis = int(kwargs.get("requested_cpu_millis", 0) or 0)
        requested_memory_bytes = int(kwargs.get("requested_memory_bytes", 0) or 0)

        if not assigned_node_name or assigned_node_name not in self._node_map:
            self._refresh_node_map()
        node = self._node_map.get(assigned_node_name or "")
        if node is None:
            raise RuntimeError(f"baremetal node {assigned_node_name!r} not found in inventory")

        container_name = f"eirel-miner-{submission_id}"
        port = int(getattr(getattr(manifest, "runtime", None), "port", 8080) or 8080)

        # 1. Extract submission archive locally
        work_dir = Path(self.work_root) / submission_id
        work_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
            tar.extractall(path=str(work_dir))

        # 2. Ensure base image is built locally and pushed to the remote node
        await self._ensure_base_image()
        await self._ensure_base_image_on_node(node)

        # 3. Sync submission code to remote node
        remote_submission_dir = await self._sync_submission_to_node(node, work_dir, submission_id)

        # 4. Run container on remote node (volume-mount submission code)
        await run_remote_docker(
            node=node,
            docker_args=["rm", "-f", container_name],
            check=False,
        )

        sdk_runtime = getattr(manifest, "sdk_runtime", None)
        inference = getattr(manifest, "inference", None)
        cpu_limit = max(0.25, requested_cpu_millis / 1000) if requested_cpu_millis > 0 else 1.0
        mem_mb = max(128, requested_memory_bytes // (1024 * 1024)) if requested_memory_bytes > 0 else 512
        run_args = [
            "run", "-d",
            "--name", container_name,
            f"--cpus={cpu_limit}",
            f"--memory={mem_mb}m",
            "-p", f"0:{port}",
            "-v", f"{remote_submission_dir}:/submission",
            "-e", f"MINER_PACKAGE_MODE={getattr(sdk_runtime, 'package_mode', 'package')}",
            "-e", f"MINER_DEPENDENCY_GROUP={getattr(sdk_runtime, 'dependency_group', 'providers')}",
            "-e", f"MINER_ENTRY_MODULE={getattr(sdk_runtime, 'entry_module', 'app')}",
            "-e", f"MINER_APP_OBJECT={getattr(sdk_runtime, 'app_object', 'app')}",
            "-e", f"MINER_RUNTIME_PORT={port}",
            "-e", f"MINER_PROVIDER={(getattr(inference, 'providers', None) or ['openai'])[0]}",
            "-e", f"MINER_MODEL={getattr(inference, 'model', 'gpt-4.1-mini')}",
            "-e", f"PROVIDER_PROXY_URL={provider_proxy_url}",
            "-e", f"PROVIDER_PROXY_TOKEN={provider_proxy_token}",
            "-e", f"RESEARCH_TOOL_URL={research_tool_url}",
            "-e", f"EIREL_RESEARCH_TOOL_URL={research_tool_url}",
            "-e", f"RESEARCH_TOOL_TOKEN={research_tool_token}",
            "-e", f"EIREL_RESEARCH_TOOL_TOKEN={research_tool_token}",
            "-e", f"EIREL_RESEARCH_TOOL_JOB_ID=miner-{submission_id}",
            "-e", f"INTERNAL_SERVICE_TOKEN={internal_service_token}",
            "-e", f"MINER_SUBMISSION_ID={submission_id}",
            self._base_image_tag,
        ]
        await run_remote_docker(node=node, docker_args=run_args)

        # 5. Get assigned port
        port_output = await run_remote_docker(
            node=node,
            docker_args=["port", container_name, f"{port}/tcp"],
        )
        endpoint_port = int(str(port_output).split(":")[-1].strip())

        handle = MinerRuntimeHandle(
            submission_id=submission_id,
            endpoint_url=f"http://{node.ssh_host}:{endpoint_port}",
            container_name=container_name,
            host_port=endpoint_port,
            state="starting",
        )
        await self._wait_for_health(handle, manifest, node)
        handle.state = "healthy"
        self._records[submission_id] = BaremetalRecord(handle=handle, node=node)
        return handle

    async def stop_runtime(
        self, submission_id: str, *, reason: str, soft: bool = False
    ) -> None:
        record = self._records.pop(submission_id, None)
        if record is None:
            return
        node = record.node
        container_name = f"eirel-miner-{submission_id}"
        try:
            await run_remote_docker(
                node=node,
                docker_args=["stop", container_name],
                check=False,
                timeout_seconds=30.0,
            )
            await run_remote_docker(
                node=node,
                docker_args=["rm", "-f", container_name],
                check=False,
                timeout_seconds=15.0,
            )
        except Exception as exc:
            logger.warning("failed to stop container %s on %s: %s", container_name, node.node_name, exc)

    async def is_container_running(self, submission_id: str) -> bool | None:
        record = self._records.get(submission_id)
        if record is None:
            return None
        try:
            output = await run_remote_docker(
                node=record.node,
                docker_args=[
                    "inspect", "-f", "{{.State.Running}}",
                    f"eirel-miner-{submission_id}",
                ],
                check=False,
                timeout_seconds=10.0,
            )
            return output.strip().lower() == "true"
        except Exception:
            return False

    async def reconcile_active_submissions(self, active_submission_ids: set[str]) -> None:
        # Step 1: Clean up known stale records AND stop their containers
        orphaned = {
            sid: record for sid, record in self._records.items()
            if sid not in active_submission_ids
        }
        for sid, record in orphaned.items():
            container_name = f"eirel-miner-{sid}"
            logger.info("reconcile: stopping stale container %s on %s", container_name, record.node.node_name)
            try:
                await run_remote_docker(
                    node=record.node,
                    docker_args=["rm", "-f", container_name],
                    check=False,
                    timeout_seconds=15.0,
                )
            except Exception:
                logger.exception("reconcile: failed to stop stale container %s", container_name)
        self._records = {
            sid: record for sid, record in self._records.items()
            if sid in active_submission_ids
        }

        # Step 2: Scan all nodes for orphaned eirel-miner-* containers
        nodes = self._refresh_node_map()

        async def _scan_node(node: BaremetalNode) -> None:
            try:
                ps_output = await run_remote_docker(
                    node=node,
                    docker_args=[
                        "ps", "-a",
                        "--filter", "name=eirel-miner-",
                        "--format", "{{.Names}}",
                    ],
                    check=False,
                    timeout_seconds=15.0,
                )
                for name in (ps_output.strip().split("\n") if ps_output.strip() else []):
                    if not name.startswith("eirel-miner-"):
                        continue
                    sid = name.removeprefix("eirel-miner-")
                    if sid not in active_submission_ids:
                        logger.info(
                            "reconcile: stopping orphaned container %s on %s",
                            name, node.node_name,
                        )
                        await run_remote_docker(
                            node=node,
                            docker_args=["rm", "-f", name],
                            check=False,
                            timeout_seconds=15.0,
                        )
            except Exception:
                logger.exception("reconcile: scan failed on node %s", node.node_name)

        await asyncio.gather(*[_scan_node(n) for n in nodes])

    async def recover_runtime_handle(
        self, *, submission_id: str, manifest: Any
    ) -> MinerRuntimeHandle | None:
        container_name = f"eirel-miner-{submission_id}"
        port = int(getattr(getattr(manifest, "runtime", None), "port", 8080) or 8080)
        nodes = self._refresh_node_map()
        for node in nodes:
            try:
                await run_remote_docker(
                    node=node,
                    docker_args=["inspect", container_name],
                    timeout_seconds=10.0,
                )
                port_output = await run_remote_docker(
                    node=node,
                    docker_args=["port", container_name, f"{port}/tcp"],
                    timeout_seconds=10.0,
                )
                endpoint_port = int(str(port_output).split(":")[-1].strip())
                handle = MinerRuntimeHandle(
                    submission_id=submission_id,
                    endpoint_url=f"http://{node.ssh_host}:{endpoint_port}",
                    container_name=container_name,
                    host_port=endpoint_port,
                    state="healthy",
                )
                self._records[submission_id] = BaremetalRecord(handle=handle, node=node)
                return handle
            except Exception:
                continue
        return None
