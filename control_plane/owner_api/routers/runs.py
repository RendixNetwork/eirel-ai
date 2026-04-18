from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request
from sqlalchemy import select

from shared.common.models import RunFamilyResult
from control_plane.owner_api.dependencies import require_internal_service_token
from control_plane.owner_api.managed import ManagedOwnerServices

router = APIRouter(tags=["runs"])


def serialize_run_record(
    run,
    *,
    family_results,
    family_score_summaries=None,
) -> dict[str, Any]:
    metadata = dict(run.metadata_json or {})
    return {
        "run_id": run.id,
        "sequence": run.sequence,
        "status": run.status,
        "benchmark_version": run.benchmark_version,
        "rubric_version": run.rubric_version,
        "judge_model": run.judge_model,
        "started_at": run.started_at.isoformat(),
        "ends_at": run.ends_at.isoformat(),
        "closed_at": run.closed_at.isoformat() if run.closed_at else None,
        "min_scores": run.min_scores_json,
        "evaluation_bundle_summaries": dict(metadata.get("evaluation_bundle_summaries", {}) or {}),
        "evaluation_bundle_artifacts": dict(metadata.get("evaluation_bundle_artifacts", {}) or {}),
        "family_score_summaries": dict(family_score_summaries or {}),
        "family_results": {
            item.family_id: {
                "family_id": item.family_id,
                "has_winner": item.has_winner,
                "winner_deployment_id": item.winner_deployment_id,
                "winner_submission_id": item.winner_submission_id,
                "winner_hotkey": item.winner_hotkey,
                "best_raw_score": item.best_raw_score,
                "min_score": item.min_score,
                "top_deployment_ids": item.top_deployment_ids_json,
                "metadata": item.metadata_json,
            }
            for item in family_results
        },
    }


@router.get("/v1/runs/current")
async def current_run(request: Request) -> dict[str, Any]:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        run = services.ensure_current_run(session)
        results = list(
            session.execute(
                select(RunFamilyResult).where(RunFamilyResult.run_id == run.id)
            ).scalars()
        )
        return serialize_run_record(
            run,
            family_results=results,
            family_score_summaries=services.run_family_score_summaries(session, run_id=run.id),
        )


@router.get("/v1/runs")
async def list_runs(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[dict[str, Any]]:
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        results_by_run: dict[str, list[RunFamilyResult]] = {}
        for item in session.execute(select(RunFamilyResult)).scalars():
            results_by_run.setdefault(item.run_id, []).append(item)
        runs = services.list_runs(session)[offset:offset + limit]
        return [
            serialize_run_record(
                run,
                family_results=results_by_run.get(run.id, []),
                family_score_summaries=services.run_family_score_summaries(session, run_id=run.id),
            )
            for run in runs
        ]


@router.post("/v1/operators/runs/rollover")
async def operator_rollover_run(request: Request) -> dict[str, Any]:
    require_internal_service_token(request)
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        current = services.ensure_current_run(session)
        current.ends_at = current.started_at
        current.updated_at = current.started_at
        session.commit()
    # ensure_current_run_and_reconcile will detect the run transition,
    # stop retired deployment containers, and reconcile carry-overs.
    next_run = await services.deployments.ensure_current_run_and_reconcile()
    return {
        "status": "rolled_over",
        "run_id": next_run.id,
        "sequence": next_run.sequence,
        "started_at": next_run.started_at.isoformat(),
        "ends_at": next_run.ends_at.isoformat(),
    }
