from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, Response, UploadFile
from sqlalchemy import select

from shared.common.http_control import SlidingWindowRateLimiter
from shared.common.models import (
    ManagedDeployment,
    ManagedMinerSubmission,
    SubmissionArtifact,
)
from control_plane.owner_api.dependencies import require_internal_service_token, signature_dependency
from control_plane.owner_api.managed import ManagedOwnerServices, fixed_family_weight

_security_logger = logging.getLogger('eirel.security')

router = APIRouter(tags=["submissions"])


@router.post("/v1/submissions")
async def create_submission(
    request: Request,
    archive: UploadFile = File(...),
    extrinsic_hash: str | None = None,
    block_hash: str | None = None,
    hotkey: str = Depends(signature_dependency),
) -> dict[str, Any]:
    limiter: SlidingWindowRateLimiter = request.app.state.submission_rate_limiter
    try:
        limiter.check(hotkey)
    except HTTPException:
        _security_logger.warning(
            'rate_limited hotkey=%s endpoint=/v1/submissions',
            hotkey,
        )
        raise
    services: ManagedOwnerServices = request.app.state.services
    submission_block = int(time.time())
    max_archive_bytes = 200 * 1024 * 1024  # 200 MB
    archive_bytes = await archive.read(max_archive_bytes + 1)
    if len(archive_bytes) > max_archive_bytes:
        raise HTTPException(status_code=413, detail="archive exceeds 200 MB limit")
    with services.db.sessionmaker() as session:
        try:
            submission, deployment = services.create_submission(
                session,
                miner_hotkey=hotkey,
                submission_block=submission_block,
                archive_bytes=archive_bytes,
                base_url=str(request.base_url).rstrip("/"),
                extrinsic_hash=extrinsic_hash,
                block_hash=block_hash,
            )
        except ValueError as exc:
            _security_logger.warning(
                'submission_rejected hotkey=%s reason=%s',
                hotkey, str(exc),
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if submission.family_id != "general_chat":
            raise HTTPException(
                status_code=400,
                detail=(
                    f"family_id={submission.family_id!r} is not supported — "
                    "only 'general_chat' submissions are accepted at this time"
                ),
            )
        await services.reconcile_family_deployments(family_id=submission.family_id)
        retired_dep = session.execute(
            select(ManagedDeployment)
            .where(ManagedDeployment.miner_hotkey == hotkey)
            .where(ManagedDeployment.status == "retired")
            .where(ManagedDeployment.pending_runtime_stop.is_(True))
            .where(ManagedDeployment.family_id != submission.family_id)
            .limit(1)
        ).scalar_one_or_none()
        if retired_dep is not None:
            await services.reconcile_family_deployments(family_id=retired_dep.family_id)
        with services.db.sessionmaker() as refresh_session:
            submission = refresh_session.get(ManagedMinerSubmission, submission.id)
            deployment = services.get_deployment_for_submission(refresh_session, submission.id) if submission else None
        return services.submission_payload(submission, deployment)


@router.get("/v1/submissions/current")
async def current_submission(
    request: Request,
    hotkey: str = Depends(signature_dependency),
) -> dict[str, Any] | None:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        submission = services.latest_submission_for_hotkey(session, miner_hotkey=hotkey)
        if submission is None:
            return None
        deployment = services.get_deployment_for_submission(session, submission.id)
        return services.submission_payload(
            submission,
            deployment,
            latest_scorecard_summary=services.latest_scorecard_summary(
                session,
                deployment_id=deployment.id if deployment is not None else None,
                submission_id=submission.id,
            ),
        )


@router.get("/v1/submissions/pool")
async def submission_pool(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    _token: None = Depends(require_internal_service_token),
) -> list[dict[str, Any]]:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        submissions = list(
            session.execute(
                select(ManagedMinerSubmission)
                .order_by(ManagedMinerSubmission.created_at.desc())
                .offset(offset)
                .limit(limit)
            ).scalars()
        )
        payload: list[dict[str, Any]] = []
        for submission in submissions:
            deployment = services.get_deployment_for_submission(session, submission.id)
            payload.append(
                {
                    "submission": services.submission_payload(
                        submission,
                        deployment,
                        latest_scorecard_summary=services.latest_scorecard_summary(
                            session,
                            deployment_id=deployment.id if deployment is not None else None,
                            submission_id=submission.id,
                        ),
                    ),
                    "artifact_download_url": f"/v1/submissions/{submission.id}/artifact",
                    "family_id": submission.family_id,
                    "deployment_status": deployment.status if deployment else None,
                }
            )
        return payload


@router.get("/v1/submissions/{submission_id}")
async def get_submission(
    request: Request,
    submission_id: str,
    hotkey: str = Depends(signature_dependency),
) -> dict[str, Any]:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        submission = session.get(ManagedMinerSubmission, submission_id)
        if submission is None:
            raise HTTPException(status_code=404, detail="submission not found")
        if submission.miner_hotkey != hotkey:
            raise HTTPException(status_code=403, detail="access denied")
        deployment = services.get_deployment_for_submission(session, submission.id)
        return services.submission_payload(
            submission,
            deployment,
            latest_scorecard_summary=services.latest_scorecard_summary(
                session,
                deployment_id=deployment.id if deployment is not None else None,
                submission_id=submission.id,
            ),
        )


@router.get("/v1/env-windows")
async def env_windows() -> list[dict[str, Any]]:
    return [
        {
            "capability": family_id,
            "family_id": family_id,
            "enabled": True,
            "min_completeness": 1.0,
            "weight": fixed_family_weight(family_id),
            "evaluation_split": "managed_run",
            "latest_window_membership_version": f"{family_id}:managed:v1",
            "tasks": [],
        }
        for family_id in ("general_chat",)
    ]


@router.get("/v1/submissions/{submission_id}/artifact")
async def download_artifact(
    request: Request,
    submission_id: str,
    hotkey: str = Depends(signature_dependency),
):
    # Own-hotkey only. Validators never download miner source — they
    # invoke the owner-api-managed pod over HTTP via the endpoint URL
    # surfaced in TaskClaimItem.miners. Owner-api itself uses the
    # internal-token-gated /v1/internal/submissions/{id}/artifact path
    # to fetch bytes for runtime image builds; no other consumer needs
    # the public artifact, so we deny everything except the submitter.
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        submission = session.get(ManagedMinerSubmission, submission_id)
        if submission is None:
            raise HTTPException(status_code=404, detail="submission not found")
        if submission.miner_hotkey != hotkey:
            raise HTTPException(status_code=403, detail="artifact access denied")
        artifact = session.get(SubmissionArtifact, submission.artifact_id)
        if artifact is None:
            raise HTTPException(status_code=404, detail="artifact not found")
        return Response(
            content=artifact.archive_bytes,
            media_type="application/gzip",
        )


@router.get("/v1/submissions/{submission_id}/scorecards")
async def submission_scorecards(
    request: Request,
    submission_id: str,
    limit: int = 20,
    hotkey: str = Depends(signature_dependency),
) -> list[dict[str, Any]]:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        submission = session.get(ManagedMinerSubmission, submission_id)
        if submission is None:
            raise HTTPException(status_code=404, detail="submission not found")
        if submission.miner_hotkey != hotkey:
            raise HTTPException(status_code=403, detail="scorecard access denied")
        return services.submission_scorecards(session, submission_id=submission_id, limit=limit)


@router.get("/v1/submissions/{submission_id}/progress")
async def submission_progress(
    request: Request,
    submission_id: str,
    hotkey: str = Depends(signature_dependency),
) -> dict[str, Any]:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        submission = session.get(ManagedMinerSubmission, submission_id)
        if submission is None:
            raise HTTPException(status_code=404, detail="submission not found")
        if submission.miner_hotkey != hotkey:
            raise HTTPException(status_code=403, detail="access denied")
        return services.submission_progress_payload(
            session,
            submission=submission,
        )


@router.get("/v1/submissions/{submission_id}/canonical")
async def submission_canonical(
    request: Request,
    submission_id: str,
    hotkey: str = Depends(signature_dependency),
) -> dict[str, Any]:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        submission = session.get(ManagedMinerSubmission, submission_id)
        if submission is None:
            raise HTTPException(status_code=404, detail="submission not found")
        if submission.miner_hotkey != hotkey:
            raise HTTPException(status_code=403, detail="access denied")
        return services.submission_canonical_payload(
            session,
            submission=submission,
        )
