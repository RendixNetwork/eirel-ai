from __future__ import annotations

from typing import Any

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from shared.common.models import MinerEvaluationTask
from control_plane.owner_api.dependencies import validator_dependency
from control_plane.owner_api.managed import ManagedOwnerServices
from control_plane.owner_api.schemas import (
    EvaluationStatusResponse,
    TaskClaimRequest,
    TaskClaimResponse,
    TaskResultResponse,
    TaskResultSubmission,
)


class JudgeProxyRequest(BaseModel):
    miner_response: dict[str, object] = Field(default_factory=dict)


class JudgeProxyResponse(BaseModel):
    task_score: float
    judge_output: dict[str, object]

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
    "/v1/families/{family_id}/tasks/{task_assignment_id}/result",
    response_model=TaskResultResponse,
)
async def submit_task_result(
    request: Request,
    family_id: str,
    task_assignment_id: str,
    payload: TaskResultSubmission,
    validator_hotkey: str = Depends(validator_dependency),
) -> TaskResultResponse:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        result = services.evaluation_tasks.submit_task_result(
            session,
            task_assignment_id=task_assignment_id,
            validator_hotkey=validator_hotkey,
            miner_response=payload.miner_response,
            judge_output=payload.judge_output,
            task_score=payload.task_score,
            task_status=payload.task_status,
            metadata=payload.metadata,
        )
        session.commit()

        if result["status"].startswith("rejected"):
            raise HTTPException(
                status_code=409 if result["status"] == "rejected_not_claimed" else 403,
                detail=result["status"],
            )

        return TaskResultResponse(
            task_assignment_id=task_assignment_id,
            status=result["status"],
            miner_evaluation_complete=result["miner_evaluation_complete"],
            family_evaluation_complete=result["family_evaluation_complete"],
            remaining_task_count=result["remaining_task_count"],
        )


@router.post(
    "/v1/families/{family_id}/tasks/{task_assignment_id}/judge",
    response_model=JudgeProxyResponse,
)
async def judge_task_proxy(
    request: Request,
    family_id: str,
    task_assignment_id: str,
    payload: JudgeProxyRequest,
    validator_hotkey: str = Depends(validator_dependency),
) -> JudgeProxyResponse:
    """C4: run the judge server-side for a task the validator has claimed.

    Validators no longer receive ``expected_output`` (the grading rubric)
    in the claim payload. Instead, after invoking the miner they POST the
    miner response here and receive back ``{task_score, judge_output}``.
    The rubric never leaves the owner process, so a validator that is
    also a miner cannot learn the rubric by claiming tasks in bulk.

    Gated on the validator signature; the caller must be the validator
    that currently holds the claim on ``task_assignment_id``.
    """
    services: ManagedOwnerServices = request.app.state.services

    # Guard: the caller must hold the active claim on this assignment.
    with services.db.sessionmaker() as session:
        task = session.get(MinerEvaluationTask, task_assignment_id)
        if task is None:
            raise HTTPException(status_code=404, detail="task_assignment_not_found")
        if task.family_id != family_id:
            raise HTTPException(status_code=404, detail="task_assignment_family_mismatch")
        if task.claimed_by_validator != validator_hotkey:
            raise HTTPException(status_code=403, detail="not_claimed_by_caller")
        if task.status != "claimed":
            raise HTTPException(status_code=409, detail=f"task_in_status_{task.status}")

    # Run the judge server-side. ``run_judge_for_assignment`` is a sync
    # method that calls the sync ``JudgeServiceClient``; offload it so we
    # don't block the async event loop.
    def _invoke() -> dict[str, object]:
        with services.db.sessionmaker() as session:
            return services.evaluation_tasks.run_judge_for_assignment(
                session,
                task_assignment_id=task_assignment_id,
                miner_response=payload.miner_response,
            )

    try:
        result = await asyncio.to_thread(_invoke)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"judge_proxy_failed: {exc}")
    return JudgeProxyResponse(
        task_score=float(result.get("task_score", 0.0)),
        judge_output=dict(result.get("judge_output", {})),
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
