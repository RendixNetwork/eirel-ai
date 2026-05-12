from __future__ import annotations

from collections import defaultdict
from typing import Any

from sqlalchemy import case, func, select
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
    PairwiseBreakdown,
    QueuedSubmission,
    QueuedSubmissionsResponse,
    RunDetailResponse,
    RunListResponse,
    RunSummary,
    TaskEvaluation,
    TrendLiteral,
    ValidatorRunCost,
    ValidatorRunCostsResponse,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_FAMILY_LABELS: dict[str, str] = {
    "general_chat": "General Chat",
}


def _normalize_winner(raw: Any) -> str | None:
    """Coerce a judge ``winner`` to canonical ``"A" | "B" | "tie"`` or None."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if s.lower() == "tie":
        return "tie"
    s_upper = s.upper()
    return s_upper if s_upper in ("A", "B") else None


def _normalize_category_scores(raw: Any) -> dict[str, dict[str, int]] | None:
    if not isinstance(raw, dict):
        return None
    out: dict[str, dict[str, int]] = {}
    for k, v in raw.items():
        if isinstance(v, dict) and "A" in v and "B" in v:
            try:
                out[str(k)] = {"A": int(v["A"]), "B": int(v["B"])}
            except (TypeError, ValueError):
                continue
    return out or None


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


# Verdict buckets that mean "the verdict judge said the response passed".
# Used by ``_effective_verdict`` to decide whether a zero ``final_task_score``
# should reclassify the row as ``gate_knockout``.
_PASS_VERDICTS: frozenset[str] = frozenset({"matches", "partially_matches", "not_applicable"})


def _effective_verdict(verdict: str | None, final_task_score: float | None) -> str:
    """Reclassify a pass-bucket verdict to ``gate_knockout`` when the
    multi-metric composite zeroed the task.

    Rule: a task is in its pass bucket (matches / partially_matches /
    not_applicable) only if the composite gates (tool_attestation,
    hallucination_knockout, grounded_gate, safety_gate, cost_attestation,
    safety_attestation) didn't zero it. When ``final_task_score == 0``
    and the verdict was a pass type, the row becomes ``gate_knockout``
    so the displayed bucket counts and ``mean_agreement`` reflect the
    same reality as the canonical score.

    Errors stay errors. Contradicts / latency_violation stay as-is.
    Legacy rows with ``final_task_score IS NULL`` keep their verdict
    (no gate signal available).
    """
    v = verdict or "error"
    if v not in _PASS_VERDICTS:
        return v
    if final_task_score is None:
        return v
    if final_task_score <= 0.0:
        return "gate_knockout"
    return v


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
        TaskMinerResult.pairwise_preference_score,
        TaskMinerResult.grounded_correctness,
        TaskMinerResult.retrieval_quality,
        TaskMinerResult.tool_routing,
        TaskMinerResult.instruction_safety,
        TaskMinerResult.latency_cost,
        TaskMinerResult.computation_correctness,
        TaskMinerResult.final_task_score,
    ).where(
        TaskMinerResult.run_id == run_id,
        TaskMinerResult.family_id == family_id,
    )
    if hotkey is not None:
        stmt = stmt.where(TaskMinerResult.miner_hotkey == hotkey)

    task_count: dict[str, int] = defaultdict(int)
    score_sum: dict[str, float] = defaultdict(float)  # sum over non-error rows
    verdict_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # Multi-metric per-dimension running sums + counts.
    dim_keys = (
        "pairwise", "grounded", "retrieval", "tool_routing",
        "safety", "latency_cost", "computation_correctness", "task_score",
    )
    dim_sum: dict[str, dict[str, float]] = defaultdict(
        lambda: {k: 0.0 for k in dim_keys}
    )
    dim_count: dict[str, dict[str, int]] = defaultdict(
        lambda: {k: 0 for k in dim_keys}
    )

    for row in session.execute(stmt).all():
        hk = row[0]
        verdict = row[1]
        score = row[2]
        (pairwise_v, grounded_v, retrieval_v, tool_v, safety_v,
         latency_cost_v, computation_v, final_v) = row[3:11]
        task_count[hk] += 1
        # Gate-aware: a pass-bucket verdict whose multi-metric
        # composite was zeroed by a gate becomes "gate_knockout" — so
        # the displayed bucket totals and mean_agreement reflect the
        # same reality as the canonical score.
        v = _effective_verdict(verdict, final_v)
        verdict_counts[hk][v] += 1
        # mean_agreement sums effective agreement_score: gate-knocked
        # rows contribute 0 (matching the reality the gate established),
        # error rows are excluded from both numerator and denominator.
        if v == "gate_knockout":
            pass  # contributes 0 to the sum; still counts toward completed
        elif v != "error" and isinstance(score, (int, float)):
            score_sum[hk] += float(score)
        # Per-dimension accumulation — only if the column has a real
        # number (None means N/A for this task type).
        for key, val in (
            ("pairwise", pairwise_v),
            ("grounded", grounded_v),
            ("retrieval", retrieval_v),
            ("tool_routing", tool_v),
            ("safety", safety_v),
            ("latency_cost", latency_cost_v),
            ("computation_correctness", computation_v),
            ("task_score", final_v),
        ):
            if isinstance(val, (int, float)):
                dim_sum[hk][key] += float(val)
                dim_count[hk][key] += 1

    out: dict[str, MinerMetrics] = {}
    for hk, n in task_count.items():
        if n == 0:
            continue
        counts = verdict_counts[hk]
        completed = sum(c for v, c in counts.items() if v != "error")
        mean_agreement = score_sum[hk] / completed if completed else None
        error_rate = counts.get("error", 0) / n
        sums = dim_sum[hk]
        cts = dim_count[hk]

        def _mean(key: str) -> float | None:
            return sums[key] / cts[key] if cts[key] > 0 else None

        out[hk] = MinerMetrics(
            mean_task_score=_mean("task_score"),
            mean_agreement=mean_agreement,
            mean_pairwise_preference=_mean("pairwise"),
            mean_grounded_correctness=_mean("grounded"),
            mean_retrieval_quality=_mean("retrieval"),
            mean_tool_routing=_mean("tool_routing"),
            mean_instruction_safety=_mean("safety"),
            mean_latency_cost=_mean("latency_cost"),
            mean_computation_correctness=_mean("computation_correctness"),
            tasks_with_pairwise=cts["pairwise"],
            tasks_with_grounded=cts["grounded"],
            tasks_with_retrieval=cts["retrieval"],
            tasks_with_tool_routing=cts["tool_routing"],
            tasks_with_safety=cts["safety"],
            tasks_with_latency_cost=cts["latency_cost"],
            tasks_with_computation_correctness=cts["computation_correctness"],
            matches_count=counts.get("matches", 0),
            partially_matches_count=counts.get("partially_matches", 0),
            not_applicable_count=counts.get("not_applicable", 0),
            contradicts_count=counts.get("contradicts", 0),
            gate_knockout_count=counts.get("gate_knockout", 0),
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
        if "gate_knockout" in counts:
            updates["gate_knockout_count"] = int(counts.get("gate_knockout") or 0)
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
        # Headline raw_score is the mean of per-task ``final_task_score``
        # (multi-metric weighted sum, post re-normalization). Falls back
        # to ``agreement_score`` (legacy pairwise verdict mapping) when
        # ``final_task_score`` is NULL — that happens for rows from the
        # old single-pairwise pipeline before multi-metric scoring landed.
        rows = session.execute(
            select(
                TaskMinerResult.miner_hotkey,
                func.avg(
                    func.coalesce(
                        TaskMinerResult.final_task_score,
                        TaskMinerResult.agreement_score,
                    )
                ).label("raw_score"),
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
# /submissions/queued
# ---------------------------------------------------------------------------


def fetch_queued_submissions(
    session: Session,
    *,
    services: Any,
    limit: int = 200,
) -> QueuedSubmissionsResponse:
    """List submissions that have not yet produced a leaderboard score.

    Includes pre-evaluation states (``queued``, ``building``,
    ``evaluating``) plus ``build_failed`` so miners can see why their
    submission stalled. Excludes ``retired`` and any submission whose
    deployment has already produced a DeploymentScoreRecord — those are
    on the leaderboard already.
    """
    del services
    scored_ids = {
        sid for sid, in session.execute(
            select(DeploymentScoreRecord.submission_id).distinct()
        ).all()
    }

    rows = session.execute(
        select(ManagedMinerSubmission)
        .where(ManagedMinerSubmission.status != "retired")
        .order_by(
            ManagedMinerSubmission.submission_block.desc(),
            ManagedMinerSubmission.created_at.desc(),
        )
        .limit(limit * 4)  # over-fetch then drop scored ones; simpler than a NOT IN.
    ).scalars().all()

    out: list[QueuedSubmission] = []
    for sub in rows:
        if sub.id in scored_ids:
            continue
        manifest = sub.manifest_json or {}
        agent = manifest.get("agent") or {}
        out.append(
            QueuedSubmission(
                submission_id=sub.id,
                hotkey=sub.miner_hotkey,
                hotkey_short=shorten_hotkey(sub.miner_hotkey),
                family_id=sub.family_id,
                agent_name=agent.get("name") if isinstance(agent, dict) else None,
                agent_version=agent.get("version") if isinstance(agent, dict) else None,
                artifact_sha256=sub.archive_sha256,
                status=sub.status,
                submitted_at=sub.created_at.isoformat() if sub.created_at else None,
                submission_block=int(sub.submission_block) if sub.submission_block else None,
            )
        )
        if len(out) >= limit:
            break
    return QueuedSubmissionsResponse(total=len(out), submissions=out)


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
    latest_metrics = MinerMetrics()

    if latest is not None:
        rows = _collect_single_run_rows(session, run=latest, family_id=family_id)
        ranked = _rank_rows(rows)
        for r in ranked:
            if r["miner_hotkey"] == hotkey:
                current_rank = r["rank"]
                current_score = r["raw_score"]
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

    # DSR rows for an open run are stale-by-construction: they're written
    # at run-close, but partial-aggregation paths can persist them while
    # the run is still open. The leaderboard SQL is the source of truth
    # for any open run, so drop any open-status pairs and let the
    # ``include_open`` branch below re-add the live entry.
    pair_rows = [(rec, er) for (rec, er) in pair_rows if er.status != "open"]

    latest = _latest_run(session)
    include_open = latest is not None and latest.status == "open"

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
                    was_winner=False,
                )
            )

    for rec, run in pair_rows:
        higher = session.scalar(
            select(func.count(DeploymentScoreRecord.id)).where(
                DeploymentScoreRecord.run_id == run.id,
                DeploymentScoreRecord.family_id == family_id,
                DeploymentScoreRecord.raw_score > rec.raw_score,
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

    # Multi-turn fixtures carry a ``turns`` array of {user, assistant?}.
    # Surface the user-prompt sequence for the dashboard so the
    # conversation context is visible alongside the final answer (which
    # is what the judge actually scored). Single-turn tasks leave
    # ``turns`` unset and the row falls back to ``prompt``.
    raw_turns = bundle_task.get("turns") or []
    user_turns: list[str] = []
    if isinstance(raw_turns, list):
        for turn in raw_turns:
            if isinstance(turn, dict):
                u = turn.get("user")
            else:
                u = getattr(turn, "user", None)
            if isinstance(u, str) and u:
                user_turns.append(u)
    turn_count = len(user_turns) or 1

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

    def _opt_score(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    applied_weights_raw = getattr(row, "applied_weights_json", None)
    applied_weights = (
        dict(applied_weights_raw) if isinstance(applied_weights_raw, dict) else None
    )
    applicable_metrics_raw = getattr(row, "applicable_metrics_json", None)
    applicable_metrics = (
        list(applicable_metrics_raw)
        if isinstance(applicable_metrics_raw, list) else None
    )

    # Pairwise breakdown — single judge call per task with a randomized
    # miner_position (A or B chosen uniformly per task). Lives in
    # ``judge_output_json.metadata`` as flat keys: ``miner_position``,
    # ``winner``, ``confidence``, ``reason``, ``category_scores``.
    # Legacy rows that used the swap-and-average path stored ``call1``
    # and ``call2`` here instead — for those we surface call1's data so
    # the panel still renders something useful (the row's
    # ``pairwise_preference_score`` was computed from both, so showing
    # one slice is informational only).
    pairwise_breakdown: PairwiseBreakdown | None = None
    judge_meta = jo.get("metadata") if isinstance(jo, dict) else None
    if isinstance(judge_meta, dict):
        ppref = judge_meta.get("pairwise_preference_score")
        # New shape: flat keys at the metadata root.
        winner = _normalize_winner(judge_meta.get("winner"))
        miner_position_raw = str(judge_meta.get("miner_position") or "").strip().upper()
        miner_position: Any = (
            miner_position_raw if miner_position_raw in ("A", "B") else None
        )
        confidence = judge_meta.get("confidence")
        reason = judge_meta.get("reason")
        cat_scores = _normalize_category_scores(judge_meta.get("category_scores"))
        # Legacy fallback: if the new flat keys aren't present, look
        # under ``call1`` (older swap-and-average rows).
        if winner is None and miner_position is None:
            legacy = judge_meta.get("call1")
            if isinstance(legacy, dict):
                winner = _normalize_winner(legacy.get("winner"))
                lp = str(legacy.get("miner_position") or "").strip().upper()
                miner_position = lp if lp in ("A", "B") else None
                confidence = legacy.get("confidence")
                reason = legacy.get("reason")
                cat_scores = _normalize_category_scores(legacy.get("category_scores"))
        if (
            winner is not None
            or miner_position is not None
            or ppref is not None
            or reason
        ):
            try:
                conf_f: float | None = float(confidence) if confidence is not None else None
            except (TypeError, ValueError):
                conf_f = None
            pairwise_breakdown = PairwiseBreakdown(
                final_score=_opt_score(ppref),
                miner_position=miner_position,
                winner=winner,  # type: ignore[arg-type]
                confidence=conf_f,
                reason=str(reason) if isinstance(reason, str) else None,
                category_scores=cat_scores,
            )

    # Composite + EvalJudge surfacing — these fields live on
    # ``judge_output_json.metadata`` (written by the validator's
    # engine.py judge call site). Legacy rows just pass ``None`` through.
    jm = judge_meta if isinstance(judge_meta, dict) else {}
    composite_score_val = _opt_score(jm.get("composite_score"))
    composite_knockout_reason = jm.get("composite_knockout_reason")
    weighted_sum_score_val = _opt_score(jm.get("weighted_sum_score"))
    eval_outcome = jm.get("eval_outcome")
    eval_failure_mode = jm.get("eval_failure_mode")
    eval_guidance = jm.get("eval_guidance")
    oracle_status = jm.get("oracle_status")
    oracle_disagreement_note = jm.get("oracle_disagreement_note")
    vendor_status_raw = jm.get("vendor_status")
    vendor_status = (
        {str(k): str(v) for k, v in vendor_status_raw.items()}
        if isinstance(vendor_status_raw, dict) else None
    )
    ledger_tools_raw = jm.get("ledger_tools")
    ledger_tools = (
        [str(t) for t in ledger_tools_raw if isinstance(t, str)]
        if isinstance(ledger_tools_raw, list) else []
    )
    oracle_source = bundle_task.get("oracle_source") or meta.get("oracle_source")
    capability = meta.get("capability")
    domain = meta.get("domain")

    return TaskEvaluation(
        task_id=row.task_id,
        mode=mode,
        category=category,
        difficulty=difficulty,
        web_search=web_search,
        task_status=status,
        evaluated_at=row.created_at.isoformat() if row.created_at else None,
        prompt=prompt_val if isinstance(prompt_val, str) else None,
        turn_count=turn_count,
        user_turns=user_turns,
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
        # Multi-metric breakdown — None for legacy rows.
        task_type=getattr(row, "task_type", None),
        pairwise_preference_score=_opt_score(getattr(row, "pairwise_preference_score", None)),
        grounded_correctness=_opt_score(getattr(row, "grounded_correctness", None)),
        retrieval_quality=_opt_score(getattr(row, "retrieval_quality", None)),
        tool_routing=_opt_score(getattr(row, "tool_routing", None)),
        instruction_safety=_opt_score(getattr(row, "instruction_safety", None)),
        latency_cost=_opt_score(getattr(row, "latency_cost", None)),
        computation_correctness=_opt_score(getattr(row, "computation_correctness", None)),
        final_task_score=_opt_score(getattr(row, "final_task_score", None)),
        applied_weights=applied_weights,
        applicable_metrics=applicable_metrics,
        pairwise_breakdown=pairwise_breakdown,
        oracle_source=oracle_source if oracle_source in ("three_oracle", "deterministic") else None,
        oracle_status=oracle_status if oracle_status in ("consensus", "majority", "disputed", "deterministic") else None,
        oracle_disagreement_note=oracle_disagreement_note if isinstance(oracle_disagreement_note, str) else None,
        vendor_status=vendor_status,
        composite_score=composite_score_val,
        composite_knockout_reason=composite_knockout_reason if isinstance(composite_knockout_reason, str) else None,
        weighted_sum_score=weighted_sum_score_val,
        eval_outcome=eval_outcome if eval_outcome in ("correct", "partial", "wrong", "hallucinated", "refused", "disputed") else None,
        eval_failure_mode=eval_failure_mode if isinstance(eval_failure_mode, str) else None,
        eval_guidance=eval_guidance if isinstance(eval_guidance, str) else None,
        ledger_tools=ledger_tools,
        capability=capability if isinstance(capability, str) else None,
        domain=domain if isinstance(domain, str) else None,
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

    # Resolve the miner's submission for this run via DeploymentScoreRecord.
    # The viewer / archive download endpoints verify the same (run_id,
    # submission_id) pair before exposing source publicly.
    submission_id_for_run = session.execute(
        select(DeploymentScoreRecord.submission_id)
        .where(
            DeploymentScoreRecord.run_id == run_id,
            DeploymentScoreRecord.family_id == family_id,
            DeploymentScoreRecord.miner_hotkey == hotkey,
        )
        .order_by(DeploymentScoreRecord.created_at.desc())
        .limit(1)
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
        total_tasks=len(tasks_by_id) or len(tasks),
        submission_id=submission_id_for_run,
        tasks=tasks,
    )


# ---------------------------------------------------------------------------
# Validator-cost aggregation
# ---------------------------------------------------------------------------


def validator_run_costs(
    session: Session, *, run_id: str,
) -> ValidatorRunCostsResponse:
    """Aggregate validator-paid spend per (run, validator).

    Two SQL aggregates joined in Python:
      * ``task_evaluations`` grouped by ``claimed_by_validator`` →
        oracle_cost_usd sum + tasks_claimed / tasks_evaluated counts.
      * ``task_miner_results`` joined to ``task_evaluations`` and
        grouped by ``claimed_by_validator`` → judge_cost_usd sum.

    Validators that haven't claimed anything yet are absent from the
    response. Result rows are sorted by total_cost descending so the
    highest-spend validator is first — useful when piping to a
    dashboard table that truncates.
    """
    from shared.common.models import TaskEvaluation as TaskEvaluationRow

    # Per-validator task counts + oracle cost sum.
    eval_rows = session.execute(
        select(
            TaskEvaluationRow.claimed_by_validator,
            func.count(TaskEvaluationRow.id).label("tasks_claimed"),
            func.coalesce(
                func.sum(case(
                    (TaskEvaluationRow.status == "evaluated", 1),
                    else_=0,
                )),
                0,
            ).label("tasks_evaluated"),
            func.coalesce(
                func.sum(TaskEvaluationRow.oracle_cost_usd), 0.0,
            ).label("oracle_cost_usd"),
        ).where(
            TaskEvaluationRow.run_id == run_id,
            TaskEvaluationRow.claimed_by_validator.is_not(None),
        ).group_by(TaskEvaluationRow.claimed_by_validator)
    ).all()

    # Per-validator judge cost: sum across miner-result rows whose
    # parent task was claimed by V. Filter to evaluated tasks only —
    # judge cost is meaningless for tasks that errored before
    # judging. The join collapses to a single sum per validator.
    judge_rows = session.execute(
        select(
            TaskEvaluationRow.claimed_by_validator,
            func.coalesce(
                func.sum(TaskMinerResult.judge_cost_usd), 0.0,
            ).label("judge_cost_usd"),
        )
        .join(
            TaskMinerResult,
            TaskMinerResult.task_evaluation_id == TaskEvaluationRow.id,
        )
        .where(
            TaskEvaluationRow.run_id == run_id,
            TaskEvaluationRow.claimed_by_validator.is_not(None),
        )
        .group_by(TaskEvaluationRow.claimed_by_validator)
    ).all()
    judge_by_validator: dict[str, float] = {
        row.claimed_by_validator: float(row.judge_cost_usd or 0.0)
        for row in judge_rows
    }

    validators: list[ValidatorRunCost] = []
    total_oracle = 0.0
    total_judge = 0.0
    for row in eval_rows:
        hotkey = row.claimed_by_validator
        oracle = float(row.oracle_cost_usd or 0.0)
        judge = judge_by_validator.get(hotkey, 0.0)
        validators.append(ValidatorRunCost(
            validator_hotkey=hotkey,
            tasks_claimed=int(row.tasks_claimed or 0),
            tasks_evaluated=int(row.tasks_evaluated or 0),
            oracle_cost_usd=round(oracle, 6),
            judge_cost_usd=round(judge, 6),
            total_cost_usd=round(oracle + judge, 6),
        ))
        total_oracle += oracle
        total_judge += judge

    validators.sort(key=lambda v: v.total_cost_usd, reverse=True)
    return ValidatorRunCostsResponse(
        run_id=run_id,
        validators=validators,
        total_oracle_cost_usd=round(total_oracle, 6),
        total_judge_cost_usd=round(total_judge, 6),
        total_cost_usd=round(total_oracle + total_judge, 6),
    )
