from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

from shared.common.manifest import SubmissionManifest
from eirel.groups import ensure_active_family_id
from eirel.schemas import AgentInvocationRequest, AgentInvocationResponse
from shared.core.protocol import normalize_chat_completion_response
from infra.miner_runtime.runtime_manager import (
    DockerMinerRuntimeManager,
    KubernetesMinerRuntimeManager,
    MinerRuntimeHandle,
    MinerRuntimeManager,
    RuntimeNodeInfo,
    RuntimeManagerError,
)


MEDIA_ARTIFACT_FIELDS = {
    "image": "asset_uri",
    "audio": "audio_uri",
    "video": "video_uri",
}


@dataclass(slots=True)
class ManagedDeploymentRuntimeManager:
    backend: MinerRuntimeManager

    async def ensure_runtime(
        self,
        *,
        deployment_id: str,
        submission_id: str,
        archive_sha256: str,
        archive_bytes: bytes,
        manifest: SubmissionManifest,
        owner_api_url: str,
        internal_service_token: str,
        provider_proxy_url: str,
        provider_proxy_token: str,
        assigned_node_name: str | None = None,
        requested_cpu_millis: int = 0,
        requested_memory_bytes: int = 0,
        research_tool_url: str = "",
        research_tool_token: str = "",
        run_budget_usd: float = 30.0,
    ) -> MinerRuntimeHandle:
        return await self.backend.ensure_runtime(
            deployment_id=deployment_id,
            submission_id=submission_id,
            archive_sha256=archive_sha256,
            archive_bytes=archive_bytes,
            manifest=manifest,
            owner_api_url=owner_api_url,
            internal_service_token=internal_service_token,
            provider_proxy_url=provider_proxy_url,
            provider_proxy_token=provider_proxy_token,
            assigned_node_name=assigned_node_name,
            requested_cpu_millis=requested_cpu_millis,
            requested_memory_bytes=requested_memory_bytes,
            research_tool_url=research_tool_url,
            research_tool_token=research_tool_token,
            run_budget_usd=run_budget_usd,
        )

    async def stop_runtime(
        self,
        deployment_id: str,
        *,
        reason: str,
        soft: bool = False,
    ) -> None:
        await self.backend.stop_runtime(deployment_id, reason=reason, soft=soft)

    async def reconcile_active_deployments(self, active_deployment_ids: set[str]) -> None:
        await self.backend.reconcile_active_submissions(active_deployment_ids)

    async def list_runtime_nodes(self) -> list[RuntimeNodeInfo]:
        return await self.backend.list_runtime_nodes()

    def runtime_handle(self, deployment_id: str) -> MinerRuntimeHandle | None:
        records = getattr(self.backend, "_records", {})
        record = records.get(deployment_id)
        return None if record is None else record.handle

    async def invoke_runtime(
        self,
        *,
        deployment_id: str,
        manifest: SubmissionManifest,
        request: AgentInvocationRequest,
    ) -> AgentInvocationResponse:
        handle = self.runtime_handle(deployment_id)
        if handle is None:
            handle = await self.backend.recover_runtime_handle(
                submission_id=deployment_id,
                manifest=manifest,
            )
        if handle is None:
            raise RuntimeManagerError(f"runtime for deployment {deployment_id} is not active")
        if handle.state == "unhealthy":
            raise RuntimeManagerError(f"runtime for deployment {deployment_id} is unhealthy")
        family_id = ensure_active_family_id(str(request.family_id))
        if manifest.runtime.invoke_path == "/v1/agent/infer":
            async with httpx.AsyncClient(timeout=90.0) as client:
                response = await client.post(
                    f"{handle.endpoint_url}{manifest.runtime.invoke_path}",
                    json=request.model_dump(mode="json"),
                )
                response.raise_for_status()
                typed_response = AgentInvocationResponse.model_validate(response.json())
            self._validate_runtime_response(
                family_id=family_id,
                response=typed_response,
            )
            return typed_response

        if family_id == "media":
            raise RuntimeManagerError(
                "media family deployments must expose /v1/agent/infer with typed artifact outputs"
            )

        payload = self._chat_payload(request=request)
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                f"{handle.endpoint_url}{manifest.runtime.invoke_path}",
                json=payload,
            )
            response.raise_for_status()
            raw_payload = response.json()
        normalized = normalize_chat_completion_response(raw_payload)
        output = self._normalize_output(family_id=family_id, normalized=normalized)
        metadata = {
            "runtime_endpoint_url": handle.endpoint_url,
            "runtime_invoke_path": manifest.runtime.invoke_path,
        }
        if normalized.tool_calls:
            metadata["tool_calls"] = [
                {
                    "id": item.id,
                    "name": item.name,
                    "arguments": item.arguments,
                }
                for item in normalized.tool_calls
            ]
        return AgentInvocationResponse(
            task_id=request.task_id,
            family_id=request.family_id,
            status="completed",
            output=output,
            citations=self._extract_citations(normalized.content),
            latency_ms=0,
            metadata={**metadata, "compatibility_mode": "chat_completions"},
        )

    def _chat_payload(self, *, request: AgentInvocationRequest) -> dict[str, Any]:
        messages = []
        if request.primary_goal:
            messages.append({"role": "system", "content": request.primary_goal})
        for item in request.context_history:
            messages.append({"role": item.role, "content": item.content})
        messages.append({"role": "user", "content": request.subtask})
        return {
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 800,
        }

    def _normalize_output(self, *, family_id: str, normalized) -> dict[str, Any]:
        content = normalized.content or ""
        if family_id == "analyst":
            return {"summary": content}
        if family_id == "builder":
            return {"code": content}
        if family_id == "verifier":
            return {"analysis": content}
        return {"content": content}

    def _expected_media_artifact_kind(self, *, response: AgentInvocationResponse) -> str | None:
        artifacts = response.artifacts or []
        for artifact in artifacts:
            if artifact.kind in MEDIA_ARTIFACT_FIELDS:
                return artifact.kind
        output = response.output or {}
        for kind, field in MEDIA_ARTIFACT_FIELDS.items():
            if str(output.get(field) or "").strip():
                return kind
        return None

    def _extract_citations(self, content: str | None) -> list[str]:
        if not content:
            return []
        citations: list[str] = []
        for token in content.split():
            if token.startswith("http://") or token.startswith("https://"):
                citations.append(token.rstrip(".,);]"))
        return citations

    def _validate_runtime_response(
        self,
        *,
        family_id: str,
        response: AgentInvocationResponse,
    ) -> None:
        output = response.output or {}
        if family_id == "media":
            artifacts = response.artifacts or []
            required_kind = self._expected_media_artifact_kind(response=response)
            if required_kind is None:
                raise RuntimeManagerError(
                    "media family deployments must return an image, audio, or video artifact reference"
                )
            output_field = MEDIA_ARTIFACT_FIELDS[required_kind]
            has_required_artifact = any(item.kind == required_kind and item.uri for item in artifacts)
            if not has_required_artifact and not str(output.get(output_field) or "").strip():
                raise RuntimeManagerError(
                    f"media family deployments must return a {required_kind} artifact reference"
                )
            return
        if family_id == "verifier":
            if not any(str(output.get(field) or "").strip() for field in ("analysis", "answer", "summary", "content")):
                raise RuntimeManagerError(
                    "verifier family deployments must return grounded analysis text"
                )
