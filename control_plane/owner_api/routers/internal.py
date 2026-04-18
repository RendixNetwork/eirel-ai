from __future__ import annotations

from base64 import b64decode
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from shared.common.artifacts import ArtifactStore, ArtifactStoreError, sha256_hex
from shared.common.models import ManagedArtifact, ManagedDeployment, ManagedMinerSubmission, SubmissionArtifact
from control_plane.owner_api.dependencies import require_internal_service_token
from control_plane.owner_api.managed import ManagedOwnerServices
from control_plane.owner_api.schemas import InternalArtifactUpload

router = APIRouter(tags=["internal"])


@router.get("/v1/internal/submissions/{submission_id}/artifact")
async def internal_download_artifact(
    request: Request,
    submission_id: str,
):
    require_internal_service_token(request)
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        submission = session.get(ManagedMinerSubmission, submission_id)
        if submission is None:
            raise HTTPException(status_code=404, detail="submission not found")
        artifact = session.get(SubmissionArtifact, submission.artifact_id)
        if artifact is None:
            raise HTTPException(status_code=404, detail="artifact not found")
        return Response(
            content=artifact.archive_bytes,
            media_type="application/gzip",
        )


@router.get("/v1/internal/deployments/{deployment_id}/artifact")
async def internal_download_deployment_artifact(
    request: Request,
    deployment_id: str,
):
    require_internal_service_token(request)
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        deployment = session.get(ManagedDeployment, deployment_id)
        if deployment is None:
            raise HTTPException(status_code=404, detail="deployment not found")
        submission = session.get(ManagedMinerSubmission, deployment.submission_id)
        if submission is None:
            raise HTTPException(status_code=404, detail="submission not found")
        artifact = session.get(SubmissionArtifact, submission.artifact_id)
        if artifact is None:
            raise HTTPException(status_code=404, detail="artifact not found")
        return Response(content=artifact.archive_bytes, media_type="application/gzip")


@router.get("/v1/internal/registry")
async def internal_registry(
    request: Request,
    _token: None = Depends(require_internal_service_token),
) -> dict[str, list[dict[str, Any]]]:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        snapshot = services.get_serving_registry(session)
        return {
            family_id: [item.model_dump(mode="json") for item in entries]
            for family_id, entries in snapshot.items()
        }


@router.get("/v1/internal/candidate-registry")
async def internal_candidate_registry(
    request: Request,
    _token: None = Depends(require_internal_service_token),
) -> dict[str, list[dict[str, Any]]]:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        snapshot = services.get_candidate_registry(session)
        return {
            family_id: [item.model_dump(mode="json") for item in entries]
            for family_id, entries in snapshot.items()
        }


@router.get("/v1/internal/workflow-composition/registry")
async def internal_workflow_composition_registry(
    request: Request,
    _token: None = Depends(require_internal_service_token),
) -> dict[str, Any]:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        return services.workflow_composition_registry(session)


@router.post("/v1/internal/artifacts")
async def create_internal_artifact(
    request: Request,
    payload: InternalArtifactUpload,
) -> dict[str, Any]:
    require_internal_service_token(request)
    services: ManagedOwnerServices = request.app.state.services
    artifact_store: ArtifactStore = request.app.state.artifact_store
    content = b64decode(payload.content_base64.encode())
    digest = sha256_hex(content)
    owner_key = payload.deployment_id or payload.submission_id or "shared"
    storage_key = (
        f"{payload.family_id}/{payload.artifact_kind}/{owner_key}/{digest}.bin"
    )
    try:
        stored = artifact_store.put_bytes(storage_key=storage_key, content=content)
    except ArtifactStoreError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    with services.db.sessionmaker() as session:
        record = ManagedArtifact(
            deployment_id=payload.deployment_id,
            submission_id=payload.submission_id,
            family_id=payload.family_id,
            artifact_kind=payload.artifact_kind,
            storage_key=stored.storage_key,
            storage_uri=stored.storage_uri,
            mime_type=payload.mime_type,
            sha256=stored.sha256,
            size_bytes=stored.size_bytes,
            metadata_json=payload.metadata,
            retained_for_run_id=payload.retained_for_run_id,
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return {
            "artifact_id": record.id,
            "storage_uri": record.storage_uri,
            "size_bytes": record.size_bytes,
            "sha256": record.sha256,
            "download_url": f"/v1/internal/artifacts/{record.id}",
        }


@router.get("/v1/internal/artifacts/{artifact_id}")
async def read_internal_artifact(request: Request, artifact_id: str):
    require_internal_service_token(request)
    services: ManagedOwnerServices = request.app.state.services
    artifact_store: ArtifactStore = request.app.state.artifact_store
    with services.db.sessionmaker() as session:
        artifact = session.get(ManagedArtifact, artifact_id)
        if artifact is None:
            raise HTTPException(status_code=404, detail="artifact not found")
        try:
            content = artifact_store.get_bytes(storage_key=artifact.storage_key)
        except ArtifactStoreError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return Response(content=content, media_type=artifact.mime_type or "application/octet-stream")
