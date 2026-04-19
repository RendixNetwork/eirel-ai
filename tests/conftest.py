from __future__ import annotations

import io
import tarfile
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
import sys

import pytest
import bittensor as bt
from httpx import ASGITransport, AsyncClient

ROOT = Path(__file__).resolve().parents[1]
SDK_ROOT = Path(__file__).resolve().parents[2] / "eirel"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))

from eirel.schemas import AgentInvocationRequest, AgentInvocationResponse, ArtifactReference
from shared.common.config import reset_settings
from shared.common.security import sha256_hex
from control_plane.owner_api.app import app
from infra.miner_runtime.runtime_manager import MinerRuntimeHandle, RuntimeNodeInfo


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    """Ensure each test gets fresh settings from current env vars."""
    reset_settings()
    yield
    reset_settings()


def make_submission_archive(
    version: str = "1.0.0",
    *,
    family_id: str = "general_chat",
    invoke_path: str = "/invoke",
    protocol: str = "openai_chat_completions_subset_v1",
) -> bytes:
    manifest = f"""
schema_version: 1
agent:
  name: demo-agent
  version: {version}
family_id: {family_id}
build:
  context: .
  dockerfile: Dockerfile
runtime:
  port: 8080
  health_path: /healthz
  invoke_path: {invoke_path}
capabilities:
  - {family_id}
resources:
  cpu: 1
  memory_gb: 2
  gpu: false
timeout_seconds: 30
inference:
  providers:
    - openai
  requires_subnet_provider_proxy: true
  provider_mode: proxy
  protocol: {protocol}
""".strip()
    dockerfile = "FROM python:3.12-slim\nCMD [\"python\",\"-m\",\"http.server\",\"8080\"]\n"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        manifest_bytes = manifest.encode()
        manifest_info = tarfile.TarInfo("submission.yaml")
        manifest_info.size = len(manifest_bytes)
        archive.addfile(manifest_info, io.BytesIO(manifest_bytes))
        dockerfile_bytes = dockerfile.encode()
        dockerfile_info = tarfile.TarInfo("Dockerfile")
        dockerfile_info.size = len(dockerfile_bytes)
        archive.addfile(dockerfile_info, io.BytesIO(dockerfile_bytes))
    return buffer.getvalue()


FIXTURES_ROOT = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(autouse=True)
def test_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv(
        "DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
    )
    monkeypatch.setenv("METAGRAPH_SYNC_INTERVAL_SECONDS", "3600")
    monkeypatch.setenv("RESULT_AGGREGATION_INTERVAL_SECONDS", "3600")
    monkeypatch.setenv("USE_REDIS_POOL", "0")
    monkeypatch.setenv("VALIDATOR_EPOCH_QUORUM", "1")
    monkeypatch.setenv("METAGRAPH_SNAPSHOT_PATH", str(tmp_path / "metagraph.json"))
    monkeypatch.setenv("EIREL_PROVIDER_PROXY_URL", "http://provider-proxy.test")
    monkeypatch.setenv("EIREL_PROVIDER_PROXY_TOKEN", "provider-token")
    monkeypatch.setenv("EIREL_INTERNAL_SERVICE_TOKEN", "internal-token")
    # Force the lightweight docker backend during tests so the app
    # startup doesn't try to load in-cluster kubeconfig (which fails in
    # CI). Runtime tests swap in FakeRuntimeManager anyway.
    monkeypatch.setenv("OWNER_RUNTIME_BACKEND", "docker")
    # Private evaluation fixtures are not committed under data/ once the
    # repo is published. Tests use local copies under tests/fixtures/.
    monkeypatch.setenv(
        "EIREL_OWNER_DATASET_ROOT_PATH",
        str(FIXTURES_ROOT / "owner_datasets" / "families"),
    )


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    async with app.router.lifespan_context(app):
        class FakeRuntimeManager:
            def __init__(self):
                self.handles: dict[str, MinerRuntimeHandle] = {}

            async def ensure_runtime(
                self,
                *,
                deployment_id: str,
                submission_id: str,
                archive_sha256: str,
                archive_bytes: bytes,
                manifest,
                owner_api_url: str,
                internal_service_token: str,
                provider_proxy_url: str,
                provider_proxy_token: str,
                assigned_node_name: str | None = None,
                requested_cpu_millis: int = 0,
                requested_memory_bytes: int = 0,
                research_tool_url: str = "",
                research_tool_token: str = "",
            ) -> MinerRuntimeHandle:
                del submission_id, archive_sha256, archive_bytes, manifest, owner_api_url, internal_service_token, provider_proxy_url, provider_proxy_token, assigned_node_name, requested_cpu_millis, requested_memory_bytes, research_tool_url, research_tool_token
                handle = MinerRuntimeHandle(
                    submission_id=deployment_id,
                    endpoint_url=f"http://runtime.local/{deployment_id}",
                    container_name=f"container-{deployment_id}",
                    host_port=18080,
                    state="healthy",
                )
                self.handles[deployment_id] = handle
                return handle

            async def stop_runtime(self, deployment_id: str, *, reason: str, soft: bool = False) -> None:
                del reason, soft
                self.handles.pop(deployment_id, None)

            async def reconcile_active_deployments(self, active_deployment_ids: set[str]) -> None:
                self.handles = {
                    deployment_id: handle
                    for deployment_id, handle in self.handles.items()
                    if deployment_id in active_deployment_ids
                }

            async def list_runtime_nodes(self) -> list[RuntimeNodeInfo]:
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
                        allocatable_memory_bytes=64 * 1024 * 1024 * 1024,
                        allocatable_pod_count=16,
                        metadata={"backend": "fake"},
                    )
                ]

            @property
            def backend(self):
                return self

            async def recover_runtime_handle(self, *, submission_id: str, manifest) -> MinerRuntimeHandle | None:
                return self.handles.get(submission_id)

            def runtime_handle(self, deployment_id: str):
                return self.handles.get(deployment_id)

            async def invoke_runtime(self, *, deployment_id: str, manifest, request: AgentInvocationRequest) -> AgentInvocationResponse:
                del manifest
                handle = self.handles[deployment_id]
                content = request.subtask
                artifacts: list[ArtifactReference] = []
                return AgentInvocationResponse(
                    task_id=request.task_id,
                    family_id=request.family_id,
                    output={"content": content},
                    artifacts=artifacts,
                    latency_ms=1,
                    metadata={"runtime_endpoint_url": handle.endpoint_url},
                )

        app.state.services.runtime_manager = FakeRuntimeManager()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as http_client:
            yield http_client


def signed_headers(
    signer,
    *,
    method: str,
    path: str,
    body: bytes,
) -> dict[str, str]:
    return signer.signed_headers(method, path, sha256_hex(body))


@pytest.fixture
def identities():
    def make():
        mnemonic = bt.Keypair.generate_mnemonic()
        keypair = bt.Keypair.create_from_mnemonic(mnemonic)
        from shared.common.bittensor_signing import LoadedSigner

        return {"mnemonic": mnemonic, "keypair": keypair, "signer": LoadedSigner(keypair)}

    return {
        "miner": make(),
        "validator-1": make(),
        "validator-2": make(),
        "validator-3": make(),
        "validator-4": make(),
    }
