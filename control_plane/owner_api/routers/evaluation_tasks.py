from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from control_plane.owner_api.dependencies import validator_dependency
from control_plane.owner_api.managed import ManagedOwnerServices
from control_plane.owner_api.schemas import (
    EvaluationStatusResponse,
    TaskClaimRequest,
    TaskClaimResponse,
    TaskResultResponse,
    TaskResultSubmission,
)


router = APIRouter(tags=["evaluation-tasks"])


@router.post("/v1/families/{family_id}/tasks/claim", response_model=TaskClaimResponse)
async def claim_tasks(
    request: Request,
    family_id: str,
    payload: TaskClaimRequest,
    validator_hotkey: str = Depends(validator_dependency),
) -> TaskClaimResponse:
    services: ManagedOwnerServices = request.app.state.services

    # If a run transition occurred (run expired), this stops retired
    # containers and reconciles carry-over deployments before we proceed.
    await services.deployments.ensure_current_run_and_reconcile()

    with services.db.sessionmaker() as session:
        run = services.ensure_current_run(session)
        run_id = payload.run_id or run.id

        # If the run is not yet open (e.g. scheduled for the future),
        # there are no tasks to claim — validators should keep polling.
        if run.status != "open":
            return TaskClaimResponse(tasks=[], remaining_task_count=0)

        # Auto-freeze targets (creates tasks) if not already done for this run+family
        services.freeze_run_targets(
            session,
            run_id=run_id,
            family_id=family_id,
            base_url=str(request.base_url).rstrip("/"),
        )

    # Ensure deployments are running for pinned targets (outside DB session
    # so the snapshot is committed and visible to reconciliation).
    try:
        await services.reconcile_family_deployments(family_id=family_id)
    except Exception:
        pass  # best-effort; validators will retry on next poll

    with services.db.sessionmaker() as session:
        claimed = services.evaluation_tasks.claim_tasks(
            session,
            run_id=run_id,
            family_id=family_id,
            validator_hotkey=validator_hotkey,
            batch_size=payload.batch_size,
        )
        if not claimed:
            return TaskClaimResponse(tasks=[], remaining_task_count=0)

        items = services.evaluation_tasks.build_claim_items(
            session,
            claimed_tasks=claimed,
            run_id=run_id,
            family_id=family_id,
        )
        session.commit()

        # Count remaining pending tasks
        status = services.evaluation_tasks.evaluation_status(
            session, run_id=run_id, family_id=family_id,
        )
        return TaskClaimResponse(
            tasks=items,
            remaining_task_count=status["pending_tasks"] + status["claimed_tasks"],
        )


@router.post(
    "/v1/families/{family_id}/task-evaluations/{task_evaluation_id}/result",
    response_model=TaskResultResponse,
)
async def submit_task_result(
    request: Request,
    family_id: str,
    task_evaluation_id: str,
    payload: TaskResultSubmission,
    validator_hotkey: str = Depends(validator_dependency),
) -> TaskResultResponse:
    del family_id
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        result = services.evaluation_tasks.submit_task_result(
            session,
            task_evaluation_id=task_evaluation_id,
            validator_hotkey=validator_hotkey,
            baseline_response=payload.baseline_response,
            miner_results=payload.miner_results,
        )
        session.commit()

        if result["status"].startswith("rejected"):
            raise HTTPException(
                status_code=409 if result["status"] == "rejected_not_claimed" else 403,
                detail=result["status"],
            )

        return TaskResultResponse(
            task_evaluation_id=task_evaluation_id,
            status=result["status"],
            family_evaluation_complete=result["family_evaluation_complete"],
            remaining_task_count=result["remaining_task_count"],
        )


@router.get(
    "/v1/families/{family_id}/tasks/status",
    response_model=EvaluationStatusResponse,
)
async def evaluation_status(
    request: Request,
    family_id: str,
    run_id: str | None = None,
    validator_hotkey: str = Depends(validator_dependency),
) -> EvaluationStatusResponse:
    del validator_hotkey
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        if run_id is None:
            run = services.ensure_current_run(session)
            run_id = run.id
        status = services.evaluation_tasks.evaluation_status(
            session, run_id=run_id, family_id=family_id,
        )
        return EvaluationStatusResponse(**status)
