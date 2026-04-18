from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from control_plane.owner_api.dependencies import require_internal_service_token, workflow_episode_http_error
from control_plane.owner_api.managed import ManagedOwnerServices
from control_plane.owner_api.schemas import (
    WorkflowEpisodeAdminFinalizeRequest,
    WorkflowEpisodeCancelRequest,
    WorkflowEpisodeCompleteRequest,
    WorkflowEpisodeDeadLetterRequest,
    WorkflowEpisodeHeartbeatRequest,
    WorkflowEpisodeLeaseRequest,
    WorkflowEpisodeRegisterPayload,
    WorkflowEpisodeRequeueRequest,
    WorkflowEpisodeSelectionUpdateRequest,
    WorkflowEpisodeUpload,
)

router = APIRouter(tags=["workflows"])


@router.get("/v1/workflow-specs")
async def workflow_specs(request: Request) -> list[dict[str, Any]]:
    services: ManagedOwnerServices = request.app.state.services
    return [services.workflow_spec_payload(item) for item in services.list_workflow_specs()]


@router.get("/v1/workflow-corpus")
async def workflow_corpus_metadata(request: Request) -> dict[str, Any]:
    services: ManagedOwnerServices = request.app.state.services
    return services.workflow_corpus_public_payload()


@router.get("/v1/workflow-composition/registry")
async def workflow_composition_registry(request: Request) -> dict[str, Any]:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        return services.workflow_composition_registry(session)


@router.get("/v1/workflow-specs/{workflow_spec_id}")
async def workflow_spec(request: Request, workflow_spec_id: str) -> dict[str, Any]:
    services: ManagedOwnerServices = request.app.state.services
    try:
        spec = services.get_workflow_spec(workflow_spec_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return services.workflow_spec_payload(spec)


@router.post("/v1/internal/workflow-episodes")
async def store_workflow_episode(
    request: Request,
    payload: WorkflowEpisodeUpload,
) -> dict[str, Any]:
    require_internal_service_token(request)
    services: ManagedOwnerServices = request.app.state.services
    if payload.episode.episode_id != payload.result.episode_id:
        raise HTTPException(status_code=400, detail="episode/result mismatch")
    with services.db.sessionmaker() as session:
        record = services.store_workflow_episode(
            session,
            episode=payload.episode,
            result=payload.result,
        )
        session.commit()
        session.refresh(record)
        return services.workflow_episode_payload(record)


@router.post("/v1/internal/workflow-episodes/register")
async def register_workflow_episode(
    request: Request,
    payload: WorkflowEpisodeRegisterPayload,
) -> dict[str, Any]:
    require_internal_service_token(request)
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        record = services.register_workflow_episode(session, episode=payload.episode)
        session.commit()
        session.refresh(record)
        return services.workflow_episode_payload(record)


@router.get("/v1/internal/workflow-episodes")
async def list_workflow_episodes(
    request: Request,
    run_id: str | None = None,
    workflow_spec_id: str | None = None,
    queue_state: str | None = None,
    retryable_only: bool = False,
    dead_lettered_only: bool = False,
    stale_only: bool = False,
    lease_owner: str | None = None,
    task_id: str | None = None,
) -> list[dict[str, Any]]:
    require_internal_service_token(request)
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        return [
            services.workflow_episode_payload(record)
            for record in services.list_workflow_episodes(
                session,
                run_id=run_id,
                workflow_spec_id=workflow_spec_id,
                queue_state=queue_state,
                retryable_only=retryable_only,
                dead_lettered_only=dead_lettered_only,
                stale_only=stale_only,
                lease_owner=lease_owner,
                task_id=task_id,
            )
        ]


@router.get("/v1/internal/workflow-episodes/{episode_id}")
async def get_workflow_episode(
    request: Request,
    episode_id: str,
) -> dict[str, Any]:
    require_internal_service_token(request)
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        record = services.workflow_episode_record(session, episode_id=episode_id)
        if record is None:
            raise HTTPException(status_code=404, detail="workflow episode not found")
        return services.workflow_episode_payload(record)


@router.get("/v1/internal/workflow-episodes/{episode_id}/trace")
async def get_workflow_episode_trace(
    request: Request,
    episode_id: str,
) -> dict[str, Any]:
    require_internal_service_token(request)
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        record = services.workflow_episode_record(session, episode_id=episode_id)
        if record is None:
            raise HTTPException(status_code=404, detail="workflow episode not found")
        return services.workflow_episode_trace_payload(record)


@router.post("/v1/internal/workflow-episodes/{episode_id}/lease")
async def lease_workflow_episode(
    request: Request,
    episode_id: str,
    payload: WorkflowEpisodeLeaseRequest,
) -> dict[str, Any]:
    require_internal_service_token(request)
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        try:
            record = services.lease_workflow_episode(
                session,
                episode_id=episode_id,
                worker_id=payload.worker_id,
                lease_seconds=payload.lease_seconds,
                active_node_id=payload.active_node_id,
                active_role_id=payload.active_role_id,
            )
        except ValueError as exc:
            raise workflow_episode_http_error(exc) from exc
        session.commit()
        session.refresh(record)
        return services.workflow_episode_payload(record)


@router.post("/v1/internal/workflow-episodes/{episode_id}/heartbeat")
async def heartbeat_workflow_episode(
    request: Request,
    episode_id: str,
    payload: WorkflowEpisodeHeartbeatRequest,
) -> dict[str, Any]:
    require_internal_service_token(request)
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        try:
            record = services.heartbeat_workflow_episode(
                session,
                episode_id=episode_id,
                worker_id=payload.worker_id,
                lease_seconds=payload.lease_seconds,
                queue_state=payload.queue_state,
                active_node_id=payload.active_node_id,
                active_role_id=payload.active_role_id,
                checkpoint_state=payload.checkpoint_state,
                runtime_state_patch=payload.runtime_state_patch,
                resume_tokens=payload.resume_tokens,
                deferred_node_ids=payload.deferred_node_ids,
                metadata_patch=payload.metadata_patch,
            )
        except ValueError as exc:
            raise workflow_episode_http_error(exc) from exc
        session.commit()
        session.refresh(record)
        return services.workflow_episode_payload(record)


@router.post("/v1/internal/workflow-episodes/{episode_id}/complete")
async def complete_workflow_episode(
    request: Request,
    episode_id: str,
    payload: WorkflowEpisodeCompleteRequest,
) -> dict[str, Any]:
    require_internal_service_token(request)
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        try:
            record = services.finalize_workflow_episode(
                session,
                episode_id=episode_id,
                status=payload.status,
                worker_id=payload.worker_id,
                error_text=payload.error_text,
                final_outcome_score=payload.final_outcome_score,
            )
        except ValueError as exc:
            raise workflow_episode_http_error(exc) from exc
        session.commit()
        session.refresh(record)
        return services.workflow_episode_payload(record)


@router.post("/v1/internal/workflow-episodes/{episode_id}/cancel")
async def cancel_workflow_episode(
    request: Request,
    episode_id: str,
    payload: WorkflowEpisodeCancelRequest,
) -> dict[str, Any]:
    require_internal_service_token(request)
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        try:
            record = services.cancel_workflow_episode(
                session,
                episode_id=episode_id,
                reason=payload.reason,
                requested_by=payload.requested_by,
                cancellation_source=payload.cancellation_source,
            )
        except ValueError as exc:
            raise workflow_episode_http_error(exc) from exc
        session.commit()
        session.refresh(record)
        return services.workflow_episode_payload(record)


@router.post("/v1/internal/workflow-episodes/{episode_id}/update-selection")
async def update_workflow_episode_selection(
    request: Request,
    episode_id: str,
    payload: WorkflowEpisodeSelectionUpdateRequest,
) -> dict[str, Any]:
    require_internal_service_token(request)
    if payload.episode.episode_id != episode_id:
        raise HTTPException(status_code=400, detail="episode mismatch")
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        try:
            record = services.update_workflow_episode_selection(
                session,
                episode_id=episode_id,
                episode=payload.episode,
                metadata_patch=payload.metadata_patch,
            )
        except ValueError as exc:
            raise workflow_episode_http_error(exc) from exc
        session.commit()
        session.refresh(record)
        return services.workflow_episode_payload(record)


@router.post("/v1/internal/workflow-episodes/{episode_id}/requeue")
async def requeue_workflow_episode(
    request: Request,
    episode_id: str,
    payload: WorkflowEpisodeRequeueRequest,
) -> dict[str, Any]:
    require_internal_service_token(request)
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        try:
            record = services.requeue_workflow_episode(
                session,
                episode_id=episode_id,
                reason=payload.reason,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        session.commit()
        session.refresh(record)
        return services.workflow_episode_payload(record)


@router.post("/v1/internal/workflow-episodes/{episode_id}/dead-letter")
async def dead_letter_workflow_episode(
    request: Request,
    episode_id: str,
    payload: WorkflowEpisodeDeadLetterRequest,
) -> dict[str, Any]:
    require_internal_service_token(request)
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        try:
            record = services.dead_letter_workflow_episode(
                session,
                episode_id=episode_id,
                reason=payload.reason,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        session.commit()
        session.refresh(record)
        return services.workflow_episode_payload(record)


@router.post("/v1/internal/workflow-episodes/{episode_id}/admin-complete")
async def admin_complete_workflow_episode(
    request: Request,
    episode_id: str,
    payload: WorkflowEpisodeAdminFinalizeRequest,
) -> dict[str, Any]:
    require_internal_service_token(request)
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        try:
            record = services.admin_finalize_workflow_episode(
                session,
                episode_id=episode_id,
                status=payload.status,
                error_text=payload.error_text,
                final_outcome_score=payload.final_outcome_score,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        session.commit()
        session.refresh(record)
        return services.workflow_episode_payload(record)


@router.post("/v1/internal/workflow-episodes/recover-expired-leases")
async def recover_expired_workflow_episode_leases(request: Request) -> dict[str, Any]:
    require_internal_service_token(request)
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        recovered = services.recover_expired_workflow_episode_leases(session)
        session.commit()
        return {
            **recovered,
            "recovered_count": len(recovered["recovered_episode_ids"]),
            "retried_count": len(recovered["retried_episode_ids"]),
            "dead_lettered_count": len(recovered["dead_lettered_episode_ids"]),
        }
