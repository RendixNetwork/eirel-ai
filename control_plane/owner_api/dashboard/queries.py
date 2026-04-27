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
    RegisteredNeuron,
    RunFamilyResult,
    TaskMinerResult,
    ValidatorRecord,
)
from control_plane.owner_api._helpers import _parse_family_weights, _strip_sensitive_task_metadata

from .schemas import (
    CitationRef,
    FamiliesResponse,
    FamilySummary,
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
# Per-miner metric aggregation from TaskMinerResult
# ---------------------------------------------------------------------------


def _metrics_for_tasks(
    session: Session,
    *,
    run_id: str,
    family_id: str,
    hotkey: str | None,
) -> dict[str, MinerMetrics]:
    """
    Return per-miner metrics aggregated from TaskMinerResult rows.
    If ``hotkey`` is given, restrict to that miner (dict has 0 or 1 entries).
    """
    stmt = select(
        TaskMinerResult.miner_hotkey,
        TaskMinerResult.agreement_verdict,
        TaskMinerResult.agreement_score,
    ).where(
        TaskMinerResult.run_id == run_id,
        TaskMinerResult.family_id == family_id,
    )
    if hotkey is not None:
        stmt = stmt.where(TaskMinerResult.miner_hotkey == hotkey)

    task_count: dict[str, int] = defaultdict(int)
    score_sum: dict[str, float] = defaultdict(float)  # sum over non-error rows
    verdict_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for hk, verdict, score in session.execute(stmt).all():
        task_count[hk] += 1
        v = verdict or "error"
        verdict_counts[hk][v] += 1
        if v != "error" and isinstance(score, (int, float)):
            score_sum[hk] += float(score)

    out: dict[str, MinerMetrics] = {}
    for hk, n in task_count.items():
        if n == 0:
            continue
        counts = verdict_counts[hk]
        completed = sum(c for v, c in counts.items() if v != "error")
        mean_agreement = score_sum[hk] / completed if completed else None
        error_rate = counts.get("error", 0) / n
        out[hk] = MinerMetrics(
            mean_agreement=mean_agreement,
            matches_count=counts.get("matches", 0),
            partially_matches_count=counts.get("partially_matches", 0),
            not_applicable_count=counts.get("not_applicable", 0),
            contradicts_count=counts.get("contradicts", 0),
            error_rate=error_rate,
            reliable=error_rate <= 0.30,
        )
    return out


def _merge_summary_metrics(
    metrics: MinerMetrics,
    summary: MinerEvaluationSummary | None,
) -> MinerMetrics:
    """Overlay post-close aggregates from the MinerEvaluationSummary row.

    ``rollout_metadata_json`` is the dict emitted by ``MinerRollup.to_metadata()``
    so we prefer those values (they're authoritative once the run closes)
    over the per-row averages computed above.
    """
    if summary is None:
        return metrics
    rollout = dict(summary.rollout_metadata_json or {})
    updates: dict = {}
    if "mean_agreement" in rollout:
        updates["mean_agreement"] = rollout.get("mean_agreement")
    if "error_rate" in rollout:
        updates["error_rate"] = rollout.get("error_rate")
    if "reliable" in rollout:
        updates["reliable"] = rollout.get("reliable")
    counts = rollout.get("verdict_counts") or {}
    if isinstance(counts, dict):
        if "matches" in counts:
            updates["matches_count"] = int(counts.get("matches") or 0)
        if "partially_matches" in counts:
            updates["partially_matches_count"] = int(counts.get("partially_matches") or 0)
        if "not_applicable" in counts:
            updates["not_applicable_count"] = int(counts.get("not_applicable") or 0)
        if "contradicts" in counts:
            updates["contradicts_count"] = int(counts.get("contradicts") or 0)
    return metrics.model_copy(update=updates) if updates else metrics


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
                TaskMinerResult.miner_hotkey,
                func.avg(TaskMinerResult.agreement_score).label("raw_score"),
                func.count(TaskMinerResult.id).label("task_count"),
            )
            .where(
                TaskMinerResult.run_id == run.id,
                TaskMinerResult.family_id == family_id,
            )
            .group_by(TaskMinerResult.miner_hotkey)
        ).all()
        # Resolve each live miner's currently-active deployment so the
        # leaderboard can surface the agent name + artifact sha for the
        # submission that is actually being benchmarked right now.
        live_hotkeys = [r.miner_hotkey for r in rows if (r.task_count or 0) > 0]
        submission_by_hk: dict[str, str] = {}
        if live_hotkeys:
            for dep in session.execute(
                select(ManagedDeployment)
                .where(
                    ManagedDeployment.family_id == family_id,
                    ManagedDeployment.miner_hotkey.in_(live_hotkeys),
                    ManagedDeployment.status != "retired",
                )
                .order_by(ManagedDeployment.created_at.desc())
            ).scalars():
                submission_by_hk.setdefault(dep.miner_hotkey, dep.submission_id)
        return [
            {
                "miner_hotkey": r.miner_hotkey,
                "raw_score": float(r.raw_score or 0.0),
                "normalized_score": None,
                "is_running": True,
                "submission_id": submission_by_hk.get(r.miner_hotkey),
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
            "submission_id": rec.submission_id,
        }
        for rec in records
    ]


def _resolve_submission_metadata(
    session: Session, submission_ids: list[str]
) -> dict[str, dict[str, Any]]:
    """Batch-load agent name/version + archive sha256 by submission id.

    Surfaced on the leaderboard so miners can spot-check that the artifact
    the subnet is running for them matches their local sha (reproducibility
    + tamper detection).
    """
    out: dict[str, dict[str, Any]] = {}
    if not submission_ids:
        return out
    for sub in session.execute(
        select(ManagedMinerSubmission).where(
            ManagedMinerSubmission.id.in_(submission_ids)
        )
    ).scalars():
        manifest = sub.manifest_json or {}
        agent = manifest.get("agent") or {}
        out[sub.id] = {
            "agent_name": agent.get("name") if isinstance(agent, dict) else None,
            "agent_version": agent.get("version") if isinstance(agent, dict) else None,
            "artifact_sha256": sub.archive_sha256,
        }
    return out


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

    submission_ids = [
        row["submission_id"] for row in ranked if row.get("submission_id")
    ]
    submission_meta = _resolve_submission_metadata(session, submission_ids)

    entries: list[LeaderboardEntry] = []
    for row in ranked:
        hk = row["miner_hotkey"]
        prev = prev_ranks.get(hk)
        meta = submission_meta.get(row.get("submission_id") or "", {})
        entries.append(
            LeaderboardEntry(
                rank=row["rank"],
                hotkey=hk,
                hotkey_short=shorten_hotkey(hk),
                agent_name=meta.get("agent_name"),
                agent_version=meta.get("agent_version"),
                artifact_sha256=meta.get("artifact_sha256"),
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
    row: TaskMinerResult,
    *,
    bundle_task: dict[str, Any],
    baseline_response_json: dict[str, Any] | None = None,
) -> TaskEvaluation:
    jo = row.judge_output_json or {}

    prompt_val = bundle_task.get("prompt")
    if isinstance(prompt_val, dict):
        prompt_val = prompt_val.get("text") or prompt_val.get("prompt") or str(prompt_val)

    meta = dict(bundle_task.get("metadata") or {})
    category = bundle_task.get("category") or meta.get("category")
    difficulty = bundle_task.get("difficulty") or meta.get("difficulty")
    mode = _as_mode(bundle_task.get("mode") or meta.get("mode"))
    web_search = bool(bundle_task.get("web_search") or meta.get("web_search") or False)

    status = "completed" if row.agreement_verdict != "error" else "failed"

    miner_citations = [
        CitationRef(url=str(c.get("url") or ""), title=c.get("title"))
        for c in (row.miner_citations_json or [])
        if isinstance(c, dict) and c.get("url")
    ]
    baseline_citations: list[CitationRef] = []
    baseline_text: str | None = None
    if isinstance(baseline_response_json, dict):
        for c in baseline_response_json.get("citations") or []:
            if isinstance(c, dict) and c.get("url"):
                baseline_citations.append(
                    CitationRef(url=str(c.get("url") or ""), title=c.get("title"))
                )
        raw_text = baseline_response_json.get("response_text")
        if isinstance(raw_text, str) and raw_text.strip():
            baseline_text = raw_text

    return TaskEvaluation(
        task_id=row.task_id,
        mode=mode,
        category=category,
        difficulty=difficulty,
        web_search=web_search,
        task_status=status,
        evaluated_at=row.created_at.isoformat() if row.created_at else None,
        prompt=prompt_val if isinstance(prompt_val, str) else None,
        miner_response=row.miner_response_json,
        baseline_response_text=baseline_text,
        agreement_verdict=row.agreement_verdict,
        agreement_score=(
            float(row.agreement_score) if row.agreement_score is not None else None
        ),
        miner_citations=miner_citations,
        baseline_citations=baseline_citations,
        latency_ms=(
            int(row.miner_latency_seconds * 1000)
            if row.miner_latency_seconds
            else None
        ),
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
        select(TaskMinerResult)
        .where(
            TaskMinerResult.run_id == run_id,
            TaskMinerResult.family_id == family_id,
            TaskMinerResult.miner_hotkey == hotkey,
        )
        .order_by(TaskMinerResult.created_at.asc())
    ).scalars().all()

    bundle = services.runs.run_evaluation_bundle(session, run_id=run_id, family_id=family_id)
    tasks_by_id: dict[str, dict[str, Any]] = {}
    if isinstance(bundle, dict):
        for task_def in bundle.get("tasks") or []:
            tid = (task_def or {}).get("task_id")
            if tid:
                tasks_by_id[tid] = _strip_sensitive_task_metadata(task_def)

    # Fetch baseline responses alongside the task evals so we can surface
    # OpenAI's citations next to the miner's on each task row.
    from shared.common.models import TaskEvaluation as TaskEvaluationRow
    baseline_rows = session.execute(
        select(TaskEvaluationRow.task_id, TaskEvaluationRow.baseline_response_json)
        .where(
            TaskEvaluationRow.run_id == run_id,
            TaskEvaluationRow.family_id == family_id,
        )
    ).all()
    baseline_by_task: dict[str, dict[str, Any] | None] = {
        task_id: baseline for (task_id, baseline) in baseline_rows
    }

    tasks = [
        _task_evaluation_from_row(
            row,
            bundle_task=tasks_by_id.get(row.task_id, {}),
            baseline_response_json=baseline_by_task.get(row.task_id),
        )
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
