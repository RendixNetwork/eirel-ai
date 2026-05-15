from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from eirel.groups import ensure_active_family_id
from control_plane.owner_api.dependencies import require_internal_service_token, validator_dependency
from control_plane.owner_api.managed import ManagedOwnerServices
from control_plane.owner_api.schemas import (
    AggregateRunScoreResponse,
    FamilyWinnerWeight,
    RunTargetResponse,
    WeightsResponse,
)
from control_plane.owner_api._helpers import (
    _active_families,
    _parse_family_weights,
    _strip_sensitive_bundle_metadata,
)

router = APIRouter(tags=["scoring"])


@router.get("/v1/families/{family_id}/targets", response_model=RunTargetResponse)
async def family_targets(
    request: Request,
    family_id: str,
    run_id: str | None = None,
    epoch_id: str | None = None,
    benchmark_version: str = "family_benchmark_v2",
    rubric_version: str = "family_rubric_v2",
    judge_model: str = "local-rubric-judge",
    validator_hotkey: str = Depends(validator_dependency),
) -> RunTargetResponse:
    del validator_hotkey
    family_id = ensure_active_family_id(family_id)
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        resolved_run_id = run_id or epoch_id
        if resolved_run_id is None:
            resolved_run_id = services.ensure_current_run(session).id
        bundle = services.run_evaluation_bundle(session, run_id=resolved_run_id, family_id=family_id) or {}
        effective_benchmark_version = str(bundle.get("benchmark_version") or benchmark_version)
        effective_rubric_version = str(bundle.get("rubric_version") or rubric_version)
        snapshot = services.freeze_run_targets(
            session,
            run_id=resolved_run_id,
            family_id=family_id,
            base_url=str(request.base_url).rstrip("/"),
            benchmark_version=effective_benchmark_version,
            rubric_version=effective_rubric_version,
            judge_model=judge_model,
        )
        resolved_bundle = services.run_evaluation_bundle(
            session,
            run_id=snapshot.run_id,
            family_id=snapshot.family_id,
        )
        evaluation_bundle_artifact = services.run_evaluation_bundle_artifact(
            session,
            run_id=snapshot.run_id,
            family_id=snapshot.family_id,
        )
        resolved_bundle = resolved_bundle if isinstance(resolved_bundle, dict) else None
        if resolved_bundle is not None:
            # Anti-gaming: strip hidden-fixture markers and seed/topic provenance
            # before returning to validators (who may also be miners).
            resolved_bundle = _strip_sensitive_bundle_metadata(resolved_bundle)
        retrieval_environment = (
            dict(resolved_bundle.get("retrieval_environment") or {})
            if isinstance((resolved_bundle or {}).get("retrieval_environment"), dict)
            else None
        )
        return RunTargetResponse(
            run_id=snapshot.run_id,
            family_id=snapshot.family_id,
            benchmark_version=(
                (resolved_bundle or {}).get("benchmark_version", snapshot.benchmark_version)
                if isinstance(resolved_bundle, dict)
                else snapshot.benchmark_version
            ),
            rubric_version=snapshot.rubric_version,
            judge_model=snapshot.judge_model,
            status=snapshot.status,
            members=list(snapshot.members_json),
            evaluation_bundle=resolved_bundle,
            evaluation_bundle_artifact=evaluation_bundle_artifact,
            retrieval_environment=retrieval_environment,
            allowed_tool_policy=(resolved_bundle or {}).get("allowed_tool_policy") if isinstance(resolved_bundle, dict) else None,
            policy_version=(resolved_bundle or {}).get("policy_version") if isinstance(resolved_bundle, dict) else None,
        )


@router.get("/v1/families/{family_id}/scorecards")
async def family_scorecards(
    request: Request,
    family_id: str,
    run_id: str | None = None,
    epoch_id: str | None = None,
    validator_hotkey: str = Depends(validator_dependency),
) -> list[dict[str, Any]]:
    del validator_hotkey
    family_id = ensure_active_family_id(family_id)
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        resolved_run_id = run_id or epoch_id or services.ensure_current_run(session).id
        return services.family_scorecards(session, family_id=family_id, run_id=resolved_run_id)


@router.get(
    "/v1/families/{family_id}/aggregate",
    response_model=AggregateRunScoreResponse,
)
async def aggregate_family_scores(
    request: Request,
    family_id: str,
    run_id: str | None = None,
    epoch_id: str | None = None,
    validator_hotkey: str = Depends(validator_dependency),
) -> AggregateRunScoreResponse:
    del validator_hotkey
    family_id = ensure_active_family_id(family_id)
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        payload = services.aggregate_status_payload(
            session,
            run_id=run_id or epoch_id or services.ensure_current_run(session).id,
            family_id=family_id,
        )
    return AggregateRunScoreResponse(**{**payload, "family_id": family_id})


@router.get("/v1/weights", response_model=WeightsResponse)
async def get_weights(
    request: Request,
    run_id: str | None = None,
) -> WeightsResponse:
    """Return the current weight table for validators to set on-chain.

    Returns ``{hotkey: weight}`` for each active family's winner.
    Each family winner gets that family's full weight allocation.
    If a family has no winner (gate not passed), the family weight
    is not assigned (validators should map unassigned weight to UID 0).

    Validators poll this every ~180 blocks, resolve hotkey→UID via
    their own metagraph sync, and call ``subtensor.set_weights()``.
    """
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        from shared.common.models import EvaluationRun, RunFamilyResult
        from sqlalchemy import select

        if run_id:
            target_run = session.get(EvaluationRun, run_id)
        else:
            target_run = session.execute(
                select(EvaluationRun)
                .where(EvaluationRun.status == "completed")
                .order_by(EvaluationRun.sequence.desc())
                .limit(1)
            ).scalar_one_or_none()

        active_families = _active_families(services.settings)
        family_weights_map = _parse_family_weights(services.settings)

        # No completed run yet → nothing to publish.  Signalling ``ready=True``
        # here would cause the validator's weight-setting loop to dump the
        # whole family allocation to UID 0, consuming a ~100-block chain
        # rate-limit window before any real winner exists.
        if target_run is None:
            return WeightsResponse(
                ready=False,
                weights={},
                family_winners=[],
            )

        family_winners: list[FamilyWinnerWeight] = []
        weights: dict[str, float] = {}

        for family_id_iter in active_families:
            family_weight = family_weights_map.get(family_id_iter, 0.0)
            if family_weight <= 0:
                continue

            result = session.execute(
                select(RunFamilyResult)
                .where(RunFamilyResult.run_id == target_run.id)
                .where(RunFamilyResult.family_id == family_id_iter)
                .limit(1)
            ).scalar_one_or_none()

            if result is None or not result.has_winner or not result.winner_hotkey:
                # No winner — validator will assign this weight to UID 0
                family_winners.append(FamilyWinnerWeight(
                    family_id=family_id_iter,
                    family_weight=family_weight,
                ))
                continue

            official_score = float(
                (result.metadata_json or {}).get("winner_official_family_score", result.best_raw_score)
                or result.best_raw_score
            )

            family_winners.append(FamilyWinnerWeight(
                family_id=family_id_iter,
                winner_hotkey=result.winner_hotkey,
                family_weight=family_weight,
                official_family_score=official_score,
            ))
            weights[result.winner_hotkey] = family_weight

        return WeightsResponse(
            run_id=target_run.id,
            weights=weights,
            family_winners=family_winners,
            ready=True,
        )


