from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

logger = logging.getLogger(__name__)

from shared.common.manifest import SubmissionManifest
from shared.common.models import ManagedDeployment, ManagedMinerSubmission, ServingDeployment
from eirel.schemas import AgentInvocationRequest, AgentInvocationResponse
from control_plane.owner_api.dependencies import (
    ensure_candidate_runtime_available,
    ensure_serving_runtime_available,
    require_internal_service_token,
    validator_dependency,
)
from control_plane.owner_api.managed import ManagedOwnerServices

router = APIRouter(tags=["runtime"])


@router.get("/runtime/{deployment_id}/healthz")
async def runtime_health(
    request: Request,
    deployment_id: str,
    _token: None = Depends(require_internal_service_token),
) -> dict[str, Any]:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        deployment = session.get(ManagedDeployment, deployment_id)
        if deployment is None:
            raise HTTPException(status_code=404, detail="deployment not found")
        handle = services.runtime_manager.runtime_handle(deployment_id)
        return {
            "status": deployment.health_status,
            "family_id": deployment.family_id,
            "deployment_revision": deployment.deployment_revision,
            "is_active": deployment.is_active,
            "runtime_endpoint_url": handle.endpoint_url if handle else None,
        }


@router.post("/runtime/{deployment_id}/v1/agent/infer", response_model=AgentInvocationResponse)
async def runtime_infer(
    request: Request,
    deployment_id: str,
    payload: AgentInvocationRequest,
    _token: None = Depends(require_internal_service_token),
) -> AgentInvocationResponse:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        deployment = session.get(ManagedDeployment, deployment_id)
        if deployment is None:
            raise HTTPException(status_code=404, detail="deployment not found")
        submission = session.get(ManagedMinerSubmission, deployment.submission_id)
        if submission is None:
            raise HTTPException(status_code=404, detail="submission not found")
        manifest = SubmissionManifest.model_validate(submission.manifest_json)
        if deployment.status == "retired":
            raise HTTPException(status_code=409, detail="deployment retired")
        if deployment.health_status != "healthy":
            raise HTTPException(status_code=503, detail="deployment unhealthy")
        if deployment.placement_status not in {"running", "ready"}:
            raise HTTPException(status_code=409, detail="deployment runtime not ready")
        candidate_registry = services.get_candidate_registry(session)
        candidate_ids = {
            str((item.metadata or {}).get("deployment_id"))
            for item in candidate_registry.get(deployment.family_id, [])
        }
        if deployment.id not in candidate_ids:
            raise HTTPException(status_code=409, detail="deployment is not callable on the consumer path")
    await ensure_candidate_runtime_available(
        services=services,
        deployment_id=deployment_id,
    )
    try:
        response = await services.runtime_manager.invoke_runtime(
            deployment_id=deployment_id,
            manifest=manifest,
            request=payload,
        )
    except Exception as exc:
        logger.warning("runtime invocation failed for %s: %s", deployment_id, exc)
        raise HTTPException(status_code=502, detail="runtime invocation failed") from exc
    response.latency_ms = deployment.latency_ms_p50
    response.metadata = {
        **response.metadata,
        "managed_execution": True,
        "deployment_id": deployment.id,
        "deployment_revision": deployment.deployment_revision,
    }
    return response


@router.get("/runtime/serving/{serving_deployment_id}/healthz")
async def serving_runtime_health(
    request: Request,
    serving_deployment_id: str,
    _token: None = Depends(require_internal_service_token),
) -> dict[str, Any]:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        serving = session.get(ServingDeployment, serving_deployment_id)
        if serving is None:
            raise HTTPException(status_code=404, detail="serving deployment not found")
        handle = services.runtime_manager.runtime_handle(serving_deployment_id)
        return {
            "status": serving.health_status,
            "family_id": serving.family_id,
            "source_deployment_revision": serving.source_deployment_revision,
            "runtime_endpoint_url": handle.endpoint_url if handle else None,
        }


@router.post(
    "/runtime/serving/{serving_deployment_id}/v1/agent/infer",
    response_model=AgentInvocationResponse,
)
async def serving_runtime_infer(
    request: Request,
    serving_deployment_id: str,
    payload: AgentInvocationRequest,
    _token: None = Depends(require_internal_service_token),
) -> AgentInvocationResponse:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        serving = session.get(ServingDeployment, serving_deployment_id)
        if serving is None:
            raise HTTPException(status_code=404, detail="serving deployment not found")
        if serving.status != "healthy" or serving.health_status != "healthy":
            raise HTTPException(status_code=409, detail="serving deployment is not healthy")
        source = session.get(ManagedDeployment, serving.source_deployment_id)
        if source is None:
            raise HTTPException(status_code=404, detail="source deployment not found")
        submission = session.get(ManagedMinerSubmission, serving.source_submission_id)
        if submission is None:
            raise HTTPException(status_code=404, detail="submission not found")
        manifest = SubmissionManifest.model_validate(submission.manifest_json)
    await ensure_serving_runtime_available(
        services=services,
        serving_deployment_id=serving_deployment_id,
    )
    try:
        response = await services.runtime_manager.invoke_runtime(
            deployment_id=serving_deployment_id,
            manifest=manifest,
            request=payload,
        )
    except Exception as exc:
        logger.warning("runtime invocation failed for %s: %s", serving_deployment_id, exc)
        raise HTTPException(status_code=502, detail="runtime invocation failed") from exc
    response.metadata = {
        **response.metadata,
        "managed_execution": True,
        "serving_deployment_id": serving.id,
        "source_deployment_id": serving.source_deployment_id,
        "deployment_revision": serving.source_deployment_revision,
        "serving_release_id": serving.release_id,
    }
    return response


async def _validator_run_infer_impl(
    request: Request,
    *,
    run_id: str,
    deployment_id: str,
    payload: AgentInvocationRequest,
) -> AgentInvocationResponse:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        snapshot, member = services.resolve_run_member(
            session,
            run_id=run_id,
            deployment_id=deployment_id,
        )
        if snapshot is None or member is None:
            raise HTTPException(status_code=404, detail="run deployment not found")
        if snapshot.status != "open":
            raise HTTPException(status_code=409, detail="run snapshot is closed")
        deployment = session.get(ManagedDeployment, deployment_id)
        if deployment is None:
            raise HTTPException(status_code=404, detail="deployment not found")
        submission = session.get(ManagedMinerSubmission, deployment.submission_id)
        if submission is None:
            raise HTTPException(status_code=404, detail="submission not found")
        manifest = SubmissionManifest.model_validate(submission.manifest_json)
    await ensure_candidate_runtime_available(
        services=services,
        deployment_id=deployment_id,
    )
    try:
        response = await services.runtime_manager.invoke_runtime(
            deployment_id=deployment_id,
            manifest=manifest,
            request=payload,
        )
    except Exception as exc:
        logger.warning("runtime invocation failed for %s: %s", deployment_id, exc)
        raise HTTPException(status_code=502, detail="runtime invocation failed") from exc
    response.latency_ms = deployment.latency_ms_p50
    response.metadata = {
        **response.metadata,
        "managed_execution": True,
        "deployment_id": deployment.id,
        "deployment_revision": deployment.deployment_revision,
        "run_id": run_id,
        "validator_pinned": True,
    }
    return response


@router.get("/v1/validator/runs/{run_id}/deployments/{deployment_id}/healthz")
async def validator_runtime_health(
    request: Request,
    run_id: str,
    deployment_id: str,
    validator_hotkey: str = Depends(validator_dependency),
) -> dict[str, Any]:
    del validator_hotkey
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        snapshot, member = services.resolve_run_member(
            session,
            run_id=run_id,
            deployment_id=deployment_id,
        )
        if snapshot is None or member is None:
            raise HTTPException(status_code=404, detail="run deployment not found")
        if snapshot.status != "open":
            raise HTTPException(status_code=409, detail="run snapshot is closed")
        deployment = session.get(ManagedDeployment, deployment_id)
        if deployment is None:
            raise HTTPException(status_code=404, detail="deployment not found")
        handle = services.runtime_manager.runtime_handle(deployment_id)
        return {
            "status": deployment.health_status,
            "family_id": deployment.family_id,
            "deployment_revision": deployment.deployment_revision,
            "run_id": snapshot.run_id,
            "snapshot_status": snapshot.status,
            "runtime_endpoint_url": handle.endpoint_url if handle else None,
        }


@router.get("/v1/validator/epochs/{epoch_id}/deployments/{deployment_id}/healthz")
async def validator_runtime_health_epoch_alias(
    request: Request,
    epoch_id: str,
    deployment_id: str,
    validator_hotkey: str = Depends(validator_dependency),
) -> dict[str, Any]:
    del validator_hotkey
    return await validator_runtime_health(request, run_id=epoch_id, deployment_id=deployment_id)


@router.get("/v1/internal/runs/{run_id}/deployments/{deployment_id}/healthz")
async def internal_runtime_health(
    request: Request,
    run_id: str,
    deployment_id: str,
) -> dict[str, Any]:
    require_internal_service_token(request)
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        snapshot, member = services.resolve_run_member(
            session,
            run_id=run_id,
            deployment_id=deployment_id,
        )
        if snapshot is None or member is None:
            raise HTTPException(status_code=404, detail="run deployment not found")
        if snapshot.status != "open":
            raise HTTPException(status_code=409, detail="run snapshot is closed")
        deployment = session.get(ManagedDeployment, deployment_id)
        if deployment is None:
            raise HTTPException(status_code=404, detail="deployment not found")
        handle = services.runtime_manager.runtime_handle(deployment_id)
        return {
            "status": deployment.health_status,
            "family_id": deployment.family_id,
            "deployment_revision": deployment.deployment_revision,
            "run_id": snapshot.run_id,
            "snapshot_status": snapshot.status,
            "runtime_endpoint_url": handle.endpoint_url if handle else None,
        }


@router.get("/v1/internal/epochs/{epoch_id}/deployments/{deployment_id}/healthz")
async def internal_runtime_health_epoch_alias(
    request: Request,
    epoch_id: str,
    deployment_id: str,
) -> dict[str, Any]:
    return await internal_runtime_health(request, run_id=epoch_id, deployment_id=deployment_id)


@router.post(
    "/v1/validator/runs/{run_id}/deployments/{deployment_id}/v1/agent/infer",
    response_model=AgentInvocationResponse,
)
async def validator_runtime_infer(
    request: Request,
    run_id: str,
    deployment_id: str,
    payload: AgentInvocationRequest,
    validator_hotkey: str = Depends(validator_dependency),
) -> AgentInvocationResponse:
    del validator_hotkey
    return await _validator_run_infer_impl(
        request,
        run_id=run_id,
        deployment_id=deployment_id,
        payload=payload,
    )


@router.post(
    "/v1/validator/runs/{run_id}/deployments/{deployment_id}/infer",
    response_model=AgentInvocationResponse,
)
async def validator_runtime_infer_alias(
    request: Request,
    run_id: str,
    deployment_id: str,
    payload: AgentInvocationRequest,
    validator_hotkey: str = Depends(validator_dependency),
) -> AgentInvocationResponse:
    del validator_hotkey
    return await _validator_run_infer_impl(
        request,
        run_id=run_id,
        deployment_id=deployment_id,
        payload=payload,
    )


@router.post(
    "/v1/validator/epochs/{epoch_id}/deployments/{deployment_id}/v1/agent/infer",
    response_model=AgentInvocationResponse,
)
async def validator_runtime_infer_epoch_alias(
    request: Request,
    epoch_id: str,
    deployment_id: str,
    payload: AgentInvocationRequest,
    validator_hotkey: str = Depends(validator_dependency),
) -> AgentInvocationResponse:
    del validator_hotkey
    return await _validator_run_infer_impl(
        request,
        run_id=epoch_id,
        deployment_id=deployment_id,
        payload=payload,
    )


@router.post(
    "/v1/internal/runs/{run_id}/deployments/{deployment_id}/v1/agent/infer",
    response_model=AgentInvocationResponse,
)
async def internal_runtime_infer(
    request: Request,
    run_id: str,
    deployment_id: str,
    payload: AgentInvocationRequest,
) -> AgentInvocationResponse:
    require_internal_service_token(request)
    return await _validator_run_infer_impl(
        request,
        run_id=run_id,
        deployment_id=deployment_id,
        payload=payload,
    )


@router.post(
    "/v1/internal/epochs/{epoch_id}/deployments/{deployment_id}/v1/agent/infer",
    response_model=AgentInvocationResponse,
)
async def internal_runtime_infer_epoch_alias(
    request: Request,
    epoch_id: str,
    deployment_id: str,
    payload: AgentInvocationRequest,
) -> AgentInvocationResponse:
    require_internal_service_token(request)
    return await _validator_run_infer_impl(
        request,
        run_id=epoch_id,
        deployment_id=deployment_id,
        payload=payload,
    )
