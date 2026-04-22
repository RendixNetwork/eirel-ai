from __future__ import annotations

from collections import defaultdict
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from eirel.groups import LAUNCH_FAMILIES, ensure_family_id
from shared.common.models import (
    DeploymentScoreRecord,
    EvaluationRun,
    ManagedDeployment,
    ManagedMinerSubmission,
    MinerEvaluationSummary,
    MinerEvaluationTask,
    RegisteredNeuron,
    RunFamilyResult,
    ValidatorRecord,
)
from control_plane.owner_api._helpers import _parse_family_weights, _strip_sensitive_task_metadata

from .schemas import (
    FamiliesResponse,
    FamilySummary,
    JudgeDimensions,
    LeaderboardEntry,
    LeaderboardResponse,
    MinerMetrics,
    MinerProfileResponse,
    MinerRunSummary,
    MinerRunsResponse,
    ModeLiteral,
    OverviewResponse,
    RunDetailResponse,
    RunListResponse,
    RunSummary,
    TaskEvaluation,
    TrendLiteral,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_FAMILY_LABELS: dict[str, str] = {
    "general_chat": "General Chat",
}


def shorten_hotkey(hk: str) -> str:
    if len(hk) <= 7:
        return hk
    return f"{hk[:4]}...{hk[-3:]}"


def _compute_trend(rank: int, previous_rank: int | None) -> TrendLiteral:
    if previous_rank is None:
        return "new"
    if rank < previous_rank:
        return "up"
    if rank > previous_rank:
        return "down"
    return "stable"


def _latest_run(session: Session) -> EvaluationRun | None:
    stmt = select(EvaluationRun).order_by(EvaluationRun.sequence.desc()).limit(1)
    return session.execute(stmt).scalar_one_or_none()


def _current_or_latest_run(session: Session) -> EvaluationRun | None:
    """Prefer an open run; fall back to the most recent closed run."""
    open_stmt = (
        select(EvaluationRun)
        .where(EvaluationRun.status == "open")
        .order_by(EvaluationRun.sequence.desc())
        .limit(1)
    )
    run = session.execute(open_stmt).scalar_one_or_none()
    if run is not None:
        return run
    return _latest_run(session)


def _as_mode(value: Any) -> ModeLiteral | None:
    return "instant" if value == "instant" else ("thinking" if value == "thinking" else None)


# ---------------------------------------------------------------------------
# Per-miner metric aggregation from MinerEvaluationTask
# ---------------------------------------------------------------------------


def _metrics_for_tasks(
    session: Session,
    *,
    run_id: str,
    family_id: str,
    hotkey: str | None,
) -> dict[str, MinerMetrics]:
    """
    Return per-miner metrics aggregated from MinerEvaluationTask rows.
    If ``hotkey`` is given, restrict to that miner (dict has 0 or 1 entries).
    """
    stmt = select(
        MinerEvaluationTask.miner_hotkey,
        MinerEvaluationTask.judge_output_json,
        MinerEvaluationTask.result_metadata_json,
    ).where(
        MinerEvaluationTask.run_id == run_id,
        MinerEvaluationTask.family_id == family_id,
        MinerEvaluationTask.status == "evaluated",
    )
    if hotkey is not None:
        stmt = stmt.where(MinerEvaluationTask.miner_hotkey == hotkey)

    quality_sum: dict[str, float] = defaultdict(float)
    latency_sum: dict[str, float] = defaultdict(float)
    trace_pass_count: dict[str, int] = defaultdict(int)
    honeytoken_count: dict[str, int] = defaultdict(int)
    task_count: dict[str, int] = defaultdict(int)

    for hk, judge_output, result_metadata in session.execute(stmt).all():
        task_count[hk] += 1
        jo = judge_output or {}
        rm = result_metadata or {}
        conversation = (rm or {}).get("conversation_score") or {}
        conv_meta = conversation.get("metadata") or {}

        quality = jo.get("score")
        if isinstance(quality, (int, float)):
            quality_sum[hk] += float(quality)

        latency = conversation.get("latency")
        if isinstance(latency, (int, float)):
            latency_sum[hk] += float(latency)

        if conv_meta.get("trace_gate_passed"):
            trace_pass_count[hk] += 1
        if conv_meta.get("honeytoken_cited"):
            honeytoken_count[hk] += 1

    out: dict[str, MinerMetrics] = {}
    for hk, n in task_count.items():
        if n == 0:
            continue
        out[hk] = MinerMetrics(
            quality_mean=quality_sum[hk] / n if hk in quality_sum else None,
            latency_mean=latency_sum[hk] / n if hk in latency_sum else None,
            trace_gate_pass_rate=trace_pass_count[hk] / n,
            honeytoken_count=honeytoken_count[hk],
        )
    return out


def _merge_summary_metrics(
    metrics: MinerMetrics,
    summary: MinerEvaluationSummary | None,
) -> MinerMetrics:
    """Overlay post-close aggregates (instant_mean/thinking_mean/blended/cost_efficiency) from summary."""
    if summary is None:
        return metrics
    rollout = dict(summary.rollout_metadata_json or {})
    gc = rollout.get("general_chat") or {}
    return metrics.model_copy(update={
        "instant_mean": gc.get("instant_mean"),
        "thinking_mean": gc.get("thinking_mean"),
        "blended": gc.get("blended"),
        "cost_efficiency": gc.get("cost_efficiency"),
    })


# ---------------------------------------------------------------------------
# /overview
# ---------------------------------------------------------------------------


def fetch_overview(session: Session, *, services: Any) -> OverviewResponse:
    total_miners = session.scalar(
        select(func.count(func.distinct(ManagedDeployment.miner_hotkey))).where(
            ManagedDeployment.is_active.is_(True),
        )
    ) or 0

    active_validators = session.scalar(
        select(func.count(ValidatorRecord.hotkey)).where(
            ValidatorRecord.is_active.is_(True),
        )
    ) or 0

    active_families = [f.strip() for f in services.settings.active_families.split(",") if f.strip()]
    total_families = len(active_families) or len(LAUNCH_FAMILIES)

    current = _current_or_latest_run(session)

    sub_counts = dict(
        session.execute(
            select(ManagedMinerSubmission.status, func.count(ManagedMinerSubmission.id))
            .group_by(ManagedMinerSubmission.status)
        ).all()
    )
    # Real submission lifecycle values (written by deployment_manager):
    #   received → pending_capacity → building → deployed_for_eval → retired
    #   build_failed is a terminal error branch.
    _QUEUED = ("received", "pending_capacity", "building")
    _EVALUATING = ("deployed_for_eval",)
    queued = sum(int(sub_counts.get(s, 0)) for s in _QUEUED)
    evaluating = sum(int(sub_counts.get(s, 0)) for s in _EVALUATING)
    retired = int(sub_counts.get("retired", 0))
    build_failed = int(sub_counts.get("build_failed", 0))
    completed = retired + build_failed

    started_iso = current.started_at.isoformat() if current and current.started_at else None
    ends_iso = current.ends_at.isoformat() if current and current.ends_at else None

    return OverviewResponse(
        total_miners=int(total_miners),
        active_validators=int(active_validators),
        total_families=total_families,
        netuid=int(getattr(services.settings, "bittensor_netuid", 0) or 0),
        network=str(getattr(services.settings, "bittensor_network", "") or ""),
        current_run_id=current.id if current else None,
        current_run_sequence=current.sequence if current else None,
        current_run_status=current.status if current else None,
        current_run_started_at=started_iso,
        current_run_ends_at=ends_iso,
        queued_submissions=queued,
        evaluating_submissions=evaluating,
        completed_submissions=completed,
        retired_submissions=retired,
        build_failed_submissions=build_failed,
    )


# ---------------------------------------------------------------------------
# /families
# ---------------------------------------------------------------------------


def fetch_families(session: Session, *, services: Any) -> FamiliesResponse:
    del session
    active = {f.strip() for f in services.settings.active_families.split(",") if f.strip()}
    weights = _parse_family_weights(services.settings)
    items: list[FamilySummary] = []
    for fam in weights:
        items.append(
            FamilySummary(
                id=fam,
                label=_FAMILY_LABELS.get(fam, fam.replace("_", " ").title()),
                weight=float(weights[fam]),
                active=fam in active,
            )
        )
    items.sort(key=lambda f: (-f.weight, f.id))
    return FamiliesResponse(families=items)


# ---------------------------------------------------------------------------
# /leaderboard
# ---------------------------------------------------------------------------


def _collect_single_run_rows(
    session: Session,
    *,
    run: EvaluationRun,
    family_id: str,
) -> list[dict[str, Any]]:
    if run.status == "open":
        rows = session.execute(
            select(
                MinerEvaluationTask.miner_hotkey,
                func.avg(MinerEvaluationTask.task_score).label("raw_score"),
                func.count(MinerEvaluationTask.id).label("task_count"),
            )
            .where(
                MinerEvaluationTask.run_id == run.id,
                MinerEvaluationTask.family_id == family_id,
                MinerEvaluationTask.status == "evaluated",
            )
            .group_by(MinerEvaluationTask.miner_hotkey)
        ).all()
        return [
            {
                "miner_hotkey": r.miner_hotkey,
                "raw_score": float(r.raw_score or 0.0),
                "normalized_score": None,
                "is_running": True,
            }
            for r in rows
            if (r.task_count or 0) > 0
        ]

    records = session.execute(
        select(DeploymentScoreRecord).where(
            DeploymentScoreRecord.run_id == run.id,
            DeploymentScoreRecord.family_id == family_id,
        )
    ).scalars().all()
    return [
        {
            "miner_hotkey": rec.miner_hotkey,
            "raw_score": float(rec.raw_score),
            "normalized_score": float(rec.normalized_score),
            "is_running": False,
        }
        for rec in records
    ]


def _rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda r: (-(r["raw_score"] or 0.0), r["miner_hotkey"]))
    for idx, row in enumerate(sorted_rows, start=1):
        row["rank"] = idx
    return sorted_rows


def fetch_runs(session: Session, *, services: Any) -> RunListResponse:
    del services
    runs = session.execute(
        select(EvaluationRun).order_by(EvaluationRun.sequence.desc())
    ).scalars().all()
    items = [
        RunSummary(
            id=r.id,
            sequence=r.sequence,
            status=r.status,
            started_at=r.started_at.isoformat() if r.started_at else None,
            ends_at=r.ends_at.isoformat() if r.ends_at else None,
            closed_at=r.closed_at.isoformat() if r.closed_at else None,
        )
        for r in runs
    ]
    return RunListResponse(runs=items)


def fetch_leaderboard(
    session: Session,
    *,
    services: Any,
    family_id: str,
    run_id: str | None,
    limit: int,
    offset: int,
) -> LeaderboardResponse:
    family_id = ensure_family_id(family_id)

    if run_id is None or run_id == "latest":
        target = _current_or_latest_run(session)
    else:
        target = session.execute(
            select(EvaluationRun).where(EvaluationRun.id == run_id)
        ).scalar_one_or_none()

    if target is None:
        return LeaderboardResponse(
            run_id=None, run_sequence=None, run_status=None, family_id=family_id,
            total=0, entries=[],
        )

    rows = _collect_single_run_rows(session, run=target, family_id=family_id)
    prior_run = session.execute(
        select(EvaluationRun)
        .where(EvaluationRun.sequence < target.sequence)
        .order_by(EvaluationRun.sequence.desc())
        .limit(1)
    ).scalar_one_or_none()
    prev_rows = (
        _collect_single_run_rows(session, run=prior_run, family_id=family_id)
        if prior_run is not None else []
    )

    ranked = _rank_rows(rows)
    prev_ranks = {r["miner_hotkey"]: r.get("rank") for r in _rank_rows(prev_rows)}

    metrics_by_hk = _metrics_for_tasks(
        session, run_id=target.id, family_id=family_id, hotkey=None,
    )
    summaries_by_hk: dict[str, MinerEvaluationSummary] = {
        s.miner_hotkey: s
        for s in session.execute(
            select(MinerEvaluationSummary).where(
                MinerEvaluationSummary.run_id == target.id,
                MinerEvaluationSummary.family_id == family_id,
            )
        ).scalars()
    }
    for hk, summary in summaries_by_hk.items():
        metrics_by_hk[hk] = _merge_summary_metrics(
            metrics_by_hk.get(hk, MinerMetrics()), summary,
        )

    rfr = session.execute(
        select(RunFamilyResult).where(
            RunFamilyResult.run_id == target.id,
            RunFamilyResult.family_id == family_id,
        )
    ).scalar_one_or_none()
    winner_hk = rfr.winner_hotkey if (rfr and rfr.has_winner) else None

    win_counts = dict(
        session.execute(
            select(
                RunFamilyResult.winner_hotkey,
                func.count(RunFamilyResult.id),
            )
            .where(
                RunFamilyResult.family_id == family_id,
                RunFamilyResult.has_winner.is_(True),
            )
            .group_by(RunFamilyResult.winner_hotkey)
        ).all()
    )

    epoch_counts = dict(
        session.execute(
            select(
                DeploymentScoreRecord.miner_hotkey,
                func.count(func.distinct(DeploymentScoreRecord.run_id)),
            )
            .where(DeploymentScoreRecord.family_id == family_id)
            .group_by(DeploymentScoreRecord.miner_hotkey)
        ).all()
    )

    entries: list[LeaderboardEntry] = []
    for row in ranked:
        hk = row["miner_hotkey"]
        prev = prev_ranks.get(hk)
        entries.append(
            LeaderboardEntry(
                rank=row["rank"],
                hotkey=hk,
                hotkey_short=shorten_hotkey(hk),
                raw_score=row["raw_score"],
                normalized_score=row.get("normalized_score"),
                is_serving_winner=(hk == winner_hk),
                is_running=bool(row.get("is_running")),
                trend=_compute_trend(row["rank"], prev),
                previous_rank=prev,
                metrics=metrics_by_hk.get(hk, MinerMetrics()),
                epochs_participated=int(epoch_counts.get(hk, 0) or 0),
                win_count=int(win_counts.get(hk, 0) or 0),
            )
        )

    total = len(entries)
    entries = entries[offset:offset + limit]
    return LeaderboardResponse(
        run_id=target.id,
        run_sequence=target.sequence,
        run_status=target.status,
        family_id=family_id,
        total=total,
        entries=entries,
    )


# ---------------------------------------------------------------------------
# /miners/{hotkey}
# ---------------------------------------------------------------------------


def fetch_miner_profile(
    session: Session,
    *,
    services: Any,
    hotkey: str,
    family_id: str,
) -> MinerProfileResponse:
    family_id = ensure_family_id(family_id)

    latest = _latest_run(session)
    current_rank: int | None = None
    current_score: float | None = None
    current_weight: float | None = None
    latest_metrics = MinerMetrics()

    if latest is not None:
        rows = _collect_single_run_rows(session, run=latest, family_id=family_id)
        ranked = _rank_rows(rows)
        for r in ranked:
            if r["miner_hotkey"] == hotkey:
                current_rank = r["rank"]
                current_score = r["raw_score"]
                current_weight = r.get("normalized_score")
                break

        m = _metrics_for_tasks(
            session, run_id=latest.id, family_id=family_id, hotkey=hotkey,
        ).get(hotkey, MinerMetrics())
        summary = session.execute(
            select(MinerEvaluationSummary).where(
                MinerEvaluationSummary.run_id == latest.id,
                MinerEvaluationSummary.family_id == family_id,
                MinerEvaluationSummary.miner_hotkey == hotkey,
            )
        ).scalar_one_or_none()
        latest_metrics = _merge_summary_metrics(m, summary)

    lifetime_wins = session.scalar(
        select(func.count(RunFamilyResult.id)).where(
            RunFamilyResult.family_id == family_id,
            RunFamilyResult.winner_hotkey == hotkey,
            RunFamilyResult.has_winner.is_(True),
        )
    ) or 0

    epochs_participated = session.scalar(
        select(func.count(func.distinct(DeploymentScoreRecord.run_id))).where(
            DeploymentScoreRecord.family_id == family_id,
            DeploymentScoreRecord.miner_hotkey == hotkey,
        )
    ) or 0

    first_seen = session.scalar(
        select(func.min(ManagedMinerSubmission.created_at)).where(
            ManagedMinerSubmission.miner_hotkey == hotkey,
            ManagedMinerSubmission.family_id == family_id,
        )
    )

    neuron = session.get(RegisteredNeuron, hotkey)

    return MinerProfileResponse(
        hotkey=hotkey,
        hotkey_short=shorten_hotkey(hotkey),
        uid=neuron.uid if neuron else None,
        family_id=family_id,
        current_rank=current_rank,
        current_score=current_score,
        current_weight=current_weight,
        lifetime_wins=int(lifetime_wins),
        epochs_participated=int(epochs_participated),
        first_seen_at=first_seen.isoformat() if first_seen else None,
        latest_metrics=latest_metrics,
    )


# ---------------------------------------------------------------------------
# /miners/{hotkey}/runs
# ---------------------------------------------------------------------------


def fetch_miner_runs(
    session: Session,
    *,
    services: Any,
    hotkey: str,
    family_id: str,
    limit: int,
    offset: int,
) -> MinerRunsResponse:
    family_id = ensure_family_id(family_id)

    pair_rows = session.execute(
        select(DeploymentScoreRecord, EvaluationRun)
        .join(EvaluationRun, EvaluationRun.id == DeploymentScoreRecord.run_id)
        .where(
            DeploymentScoreRecord.miner_hotkey == hotkey,
            DeploymentScoreRecord.family_id == family_id,
        )
        .order_by(EvaluationRun.sequence.desc())
    ).all()

    latest = _latest_run(session)
    include_open = latest is not None and latest.status == "open" and all(
        er.id != latest.id for (_rec, er) in pair_rows
    )

    winners_by_run = dict(
        session.execute(
            select(RunFamilyResult.run_id, RunFamilyResult.winner_hotkey)
            .where(
                RunFamilyResult.family_id == family_id,
                RunFamilyResult.has_winner.is_(True),
            )
        ).all()
    )

    summaries: list[MinerRunSummary] = []

    if include_open:
        rows = _collect_single_run_rows(session, run=latest, family_id=family_id)
        ranked = _rank_rows(rows)
        rank = next((r["rank"] for r in ranked if r["miner_hotkey"] == hotkey), None)
        raw = next((r["raw_score"] for r in ranked if r["miner_hotkey"] == hotkey), None)
        if rank is not None:
            summaries.append(
                MinerRunSummary(
                    run_id=latest.id,
                    epoch_sequence=latest.sequence,
                    status=latest.status,
                    started_at=latest.started_at.isoformat(),
                    closed_at=latest.closed_at.isoformat() if latest.closed_at else None,
                    rank=rank,
                    raw_score=raw,
                    normalized_score=None,
                    was_winner=False,
                )
            )

    for rec, run in pair_rows:
        higher = session.scalar(
            select(func.count(DeploymentScoreRecord.id)).where(
                DeploymentScoreRecord.run_id == run.id,
                DeploymentScoreRecord.family_id == family_id,
                DeploymentScoreRecord.normalized_score > rec.normalized_score,
            )
        ) or 0
        rank = int(higher) + 1
        summaries.append(
            MinerRunSummary(
                run_id=run.id,
                epoch_sequence=run.sequence,
                status=run.status,
                started_at=run.started_at.isoformat(),
                closed_at=run.closed_at.isoformat() if run.closed_at else None,
                rank=rank,
                raw_score=float(rec.raw_score),
                normalized_score=float(rec.normalized_score),
                was_winner=(winners_by_run.get(run.id) == hotkey),
            )
        )

    paginated = summaries[offset:offset + limit]
    return MinerRunsResponse(
        miner_hotkey=hotkey,
        family_id=family_id,
        runs=paginated,
    )


# ---------------------------------------------------------------------------
# /miners/{hotkey}/runs/{run_id}
# ---------------------------------------------------------------------------


def _task_evaluation_from_row(
    row: MinerEvaluationTask,
    *,
    bundle_task: dict[str, Any],
) -> TaskEvaluation:
    jo = row.judge_output_json or {}
    rm = row.result_metadata_json or {}
    conversation = (rm or {}).get("conversation_score") or {}
    conv_meta = conversation.get("metadata") or {}

    dims_raw = jo.get("dimension_scores") or {}
    dims = JudgeDimensions(
        goal_fulfillment=dims_raw.get("goal_fulfillment"),
        correctness=dims_raw.get("correctness"),
        grounding=dims_raw.get("grounding"),
        conversation_coherence=dims_raw.get("conversation_coherence"),
    )

    prompt_val = bundle_task.get("prompt")
    # Prompts can technically be dicts in some bundle versions; flatten to str
    # so the frontend always gets a consistent type.
    if isinstance(prompt_val, dict):
        prompt_val = prompt_val.get("text") or prompt_val.get("prompt") or str(prompt_val)

    meta = dict(bundle_task.get("metadata") or {})
    category = bundle_task.get("category") or meta.get("category")
    difficulty = bundle_task.get("difficulty") or meta.get("difficulty")
    mode = _as_mode(bundle_task.get("mode") or meta.get("mode"))

    latency_ms = conv_meta.get("latency_ms")

    return TaskEvaluation(
        task_id=row.task_id,
        task_index=row.task_index,
        mode=mode,
        category=category,
        difficulty=difficulty,
        validator_hotkey=row.claimed_by_validator,
        task_score=row.task_score,
        task_status=row.task_status,
        evaluated_at=row.evaluated_at.isoformat() if row.evaluated_at else None,
        prompt=prompt_val if isinstance(prompt_val, str) else None,
        miner_response=row.miner_response_json,
        quality_score=jo.get("score") if isinstance(jo.get("score"), (int, float)) else None,
        dimension_scores=dims,
        latency_score=conversation.get("latency") if isinstance(conversation.get("latency"), (int, float)) else None,
        latency_ms=int(latency_ms) if isinstance(latency_ms, (int, float)) else None,
        trace_gate_passed=conv_meta.get("trace_gate_passed"),
        honeytoken_cited=conv_meta.get("honeytoken_cited"),
        judge_rationale=jo.get("rationale"),
    )


def fetch_run_detail(
    session: Session,
    *,
    services: Any,
    hotkey: str,
    family_id: str,
    run_id: str,
) -> RunDetailResponse:
    family_id = ensure_family_id(family_id)

    run = session.get(EvaluationRun, run_id)
    if run is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")

    summary = session.execute(
        select(MinerEvaluationSummary).where(
            MinerEvaluationSummary.run_id == run_id,
            MinerEvaluationSummary.family_id == family_id,
            MinerEvaluationSummary.miner_hotkey == hotkey,
        )
    ).scalar_one_or_none()

    task_rows = session.execute(
        select(MinerEvaluationTask)
        .where(
            MinerEvaluationTask.run_id == run_id,
            MinerEvaluationTask.family_id == family_id,
            MinerEvaluationTask.miner_hotkey == hotkey,
        )
        .order_by(MinerEvaluationTask.task_index.asc())
    ).scalars().all()

    bundle = services.runs.run_evaluation_bundle(session, run_id=run_id, family_id=family_id)
    tasks_by_id: dict[str, dict[str, Any]] = {}
    if isinstance(bundle, dict):
        for task_def in bundle.get("tasks") or []:
            tid = (task_def or {}).get("task_id")
            if tid:
                tasks_by_id[tid] = _strip_sensitive_task_metadata(task_def)

    tasks = [
        _task_evaluation_from_row(row, bundle_task=tasks_by_id.get(row.task_id, {}))
        for row in task_rows
    ]

    live_metrics = _metrics_for_tasks(
        session, run_id=run_id, family_id=family_id, hotkey=hotkey,
    ).get(hotkey, MinerMetrics())
    metrics = _merge_summary_metrics(live_metrics, summary)

    return RunDetailResponse(
        run_id=run.id,
        epoch_sequence=run.sequence,
        status=run.status,
        official_score=summary.official_family_score if summary else None,
        metrics=metrics,
        tasks=tasks,
    )
