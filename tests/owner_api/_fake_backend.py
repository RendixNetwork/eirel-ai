from __future__ import annotations

from infra.miner_runtime.runtime_manager import MinerRuntimeHandle, RuntimeNodeInfo
from infra.miner_runtime._k8s_helpers import DeploymentStatus, DeploymentStatusCode


class _FakeBackend:
    def __init__(
        self,
        *,
        available_pods: int | None = None,
        stop_error: Exception | None = None,
        ensure_error: Exception | None = None,
        status_override: DeploymentStatus | None = None,
    ) -> None:
        self._records: dict[str, MinerRuntimeHandle] = {}
        self.stopped: list[str] = []
        self._available_pods = available_pods
        self._stop_error = stop_error
        self._ensure_error = ensure_error
        self.status_override: DeploymentStatus | None = status_override

    async def ensure_runtime(self, **kwargs) -> MinerRuntimeHandle:
        if self._ensure_error is not None:
            raise self._ensure_error
        sid = kwargs.get("submission_id", "unknown")
        handle = MinerRuntimeHandle(
            submission_id=sid,
            endpoint_url=f"http://runtime.local/{sid}",
            container_name=f"container-{sid}",
            host_port=18080,
            state="healthy",
        )
        self._records[sid] = type("Record", (), {"handle": handle})()
        return handle

    async def stop_runtime(self, deployment_id: str, *, reason: str, soft: bool = False) -> None:
        if self._stop_error is not None:
            raise self._stop_error
        self.stopped.append(deployment_id)
        self._records.pop(deployment_id, None)

    async def reconcile_active_submissions(self, active_ids: set[str]) -> None:
        pass

    async def list_runtime_nodes(self) -> list[RuntimeNodeInfo]:
        pods = 16 if self._available_pods is None else self._available_pods
        return [
            RuntimeNodeInfo(
                node_name="test-node-1",
                labels={
                    "eirel.dev/runtime-pool": "true",
                    "eirel.dev/runtime-class": "miner",
                },
                ready=True,
                schedulable=True,
                allocatable_cpu_millis=16000,
                allocatable_memory_bytes=64 * 1024**3,
                allocatable_pod_count=pods,
                metadata={"backend": "fake"},
            )
        ]

    async def recover_runtime_handle(self, *, submission_id: str, manifest) -> MinerRuntimeHandle | None:
        record = self._records.get(submission_id)
        return None if record is None else record.handle

    async def deployment_status(self, submission_id: str) -> DeploymentStatus:
        if self.status_override is not None:
            return self.status_override
        return DeploymentStatus(
            code=DeploymentStatusCode.UNKNOWN,
            ready_replicas=0,
            desired_replicas=0,
            message="not supported by this backend",
            last_pod_phase=None,
        )

    async def is_container_running(self, submission_id: str) -> bool | None:
        return None

    def runtime_handle(self, deployment_id: str) -> MinerRuntimeHandle | None:
        record = self._records.get(deployment_id)
        return None if record is None else record.handle
