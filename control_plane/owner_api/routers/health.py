from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Request
from starlette.responses import Response
from sqlalchemy import func, text

from shared.common.models import (
    AggregateFamilyScoreSnapshot,
    DeploymentScoreRecord,
    EpochTargetSnapshot,
    EvaluationRun,
    ManagedDeployment,
    ManagedMinerSubmission,
    RunFamilyResult,
    RuntimeNodeSnapshot,
    ServingDeployment,
    ServingRelease,
    TaskEvaluation,
    TaskMinerResult,
    WorkflowEpisodeRecord,
)
from control_plane.owner_api.managed import ManagedOwnerServices

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz(request: Request) -> dict[str, Any]:
    services: ManagedOwnerServices = request.app.state.services
    checks: dict[str, str] = {}
    overall = "ok"
    try:
        with services.db.sessionmaker() as session:
            session.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = f"degraded: {exc}"
        overall = "degraded"
    replay_protector = getattr(request.app.state, "replay_protector", None)
    if replay_protector is not None and hasattr(replay_protector, "client"):
        try:
            await replay_protector.client.ping()
            checks["redis"] = "ok"
        except Exception as exc:
            checks["redis"] = f"degraded: {exc}"
            overall = "degraded"
    return {"status": overall, "mode": "managed-execution", "checks": checks}


_metrics_cache: dict[str, Any] = {"body": None, "expires_at": 0.0}
_METRICS_CACHE_TTL = 5.0


@router.get("/metrics")
async def metrics(request: Request) -> Response:
    now = time.monotonic()
    if _metrics_cache["body"] is not None and now < _metrics_cache["expires_at"]:
        return Response(
            content=_metrics_cache["body"],
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )
    services: ManagedOwnerServices = request.app.state.services
    with services.db.sessionmaker() as session:
        submission_count = session.query(ManagedMinerSubmission).count()
        deployment_count = session.query(ManagedDeployment).count()
        active_count = session.query(ManagedDeployment).filter_by(is_active=True).count()
        serving_release_count = session.query(ServingRelease).count()
        serving_deployment_count = session.query(ServingDeployment).count()
        current_serving_count = len(services.current_serving_fleet(session))
        unhealthy_count = session.query(ManagedDeployment).filter_by(health_status="unhealthy").count()
        open_run_count = session.query(EpochTargetSnapshot).filter_by(status="open").count()
        aggregate_pending = session.query(AggregateFamilyScoreSnapshot).filter_by(status="pending").count()
        validator_submission_count = session.query(TaskMinerResult).count()
        runtime_capacity = services.runtime_capacity_summary(session)
        workflow_episode_records = list(session.query(WorkflowEpisodeRecord).all())
        workflow_runtime_health = services._workflow_runtime_health_summary(
            session,
            workflow_episode_records=workflow_episode_records,
        )

        # ── Labeled breakdowns ───────────────────────────────────────────
        submission_by_status_family = (
            session.query(
                ManagedMinerSubmission.status,
                ManagedMinerSubmission.family_id,
                func.count(),
            )
            .group_by(ManagedMinerSubmission.status, ManagedMinerSubmission.family_id)
            .all()
        )

        deployment_by_status_health_family = (
            session.query(
                ManagedDeployment.status,
                ManagedDeployment.health_status,
                ManagedDeployment.family_id,
                func.count(),
            )
            .filter(ManagedDeployment.retired_at.is_(None))
            .group_by(
                ManagedDeployment.status,
                ManagedDeployment.health_status,
                ManagedDeployment.family_id,
            )
            .all()
        )

        deployment_by_node = (
            session.query(
                ManagedDeployment.assigned_node_name,
                func.count(),
                func.sum(ManagedDeployment.assigned_cpu_millis),
                func.sum(ManagedDeployment.assigned_memory_bytes),
            )
            .filter(
                ManagedDeployment.retired_at.is_(None),
                ManagedDeployment.assigned_node_name.isnot(None),
            )
            .group_by(ManagedDeployment.assigned_node_name)
            .all()
        )

        deployment_info_rows = (
            session.query(
                ManagedDeployment.id,
                ManagedDeployment.submission_id,
                ManagedDeployment.miner_hotkey,
                ManagedDeployment.family_id,
                ManagedDeployment.status,
                ManagedDeployment.health_status,
                ManagedDeployment.assigned_node_name,
                ManagedDeployment.is_active,
            )
            .filter(ManagedDeployment.retired_at.is_(None))
            .all()
        )

        # ── Baremetal / runtime node metrics ─────────────────────────────
        runtime_nodes = list(session.query(RuntimeNodeSnapshot).all())

        # ── Per-miner scorecard rows for the last 2 runs per family ──────
        scorecard_rows = _collect_scorecard_rows(session)

        # ── Per-task live-run rows (bounded to the current open run) ─────
        live_run_rows = _collect_live_run_rows(session)

        # ── Per-validator cost rows for the open run ─────────────────────
        validator_cost_rows = _collect_validator_cost_rows(session)

    submission_labeled = _format_submission_metrics(submission_by_status_family)
    deployment_labeled = _format_deployment_metrics(deployment_by_status_health_family)
    node_deployment_lines = _format_node_deployment_metrics(deployment_by_node)
    deployment_info_lines = _format_deployment_info_metrics(deployment_info_rows)
    node_lines = _format_node_metrics(runtime_nodes)
    scorecard_lines = _format_scorecard_metrics(scorecard_rows)
    live_run_lines = _format_live_run_metrics(live_run_rows)
    validator_cost_lines = _format_validator_cost_metrics(validator_cost_rows)

    body = (
        # ── Service up marker ────────────────────────────────────────
        "# TYPE eirel_owner_api_up gauge\n"
        "eirel_owner_api_up 1\n"
        # ── Submission totals ────────────────────────────────────────
        "# TYPE eirel_owner_submissions_total gauge\n"
        f"eirel_owner_submissions_total {submission_count}\n"
        + submission_labeled
        # ── Deployment totals ────────────────────────────────────────
        + "# TYPE eirel_owner_deployments_total gauge\n"
        f"eirel_owner_deployments_total {deployment_count}\n"
        "# TYPE eirel_owner_active_deployments_total gauge\n"
        f"eirel_owner_active_deployments_total {active_count}\n"
        "# TYPE eirel_owner_unhealthy_deployments_total gauge\n"
        f"eirel_owner_unhealthy_deployments_total {unhealthy_count}\n"
        + deployment_labeled
        # ── Serving ──────────────────────────────────────────────────
        + "# TYPE eirel_owner_serving_releases_total gauge\n"
        f"eirel_owner_serving_releases_total {serving_release_count}\n"
        "# TYPE eirel_owner_serving_deployments_total gauge\n"
        f"eirel_owner_serving_deployments_total {serving_deployment_count}\n"
        "# TYPE eirel_owner_current_serving_fleet_total gauge\n"
        f"eirel_owner_current_serving_fleet_total {current_serving_count}\n"
        # ── Epochs / scoring ─────────────────────────────────────────
        "# TYPE eirel_owner_open_run_snapshots gauge\n"
        f"eirel_owner_open_run_snapshots {open_run_count}\n"
        "# TYPE eirel_owner_pending_aggregate_snapshots gauge\n"
        f"eirel_owner_pending_aggregate_snapshots {aggregate_pending}\n"
        "# TYPE eirel_owner_validator_score_submissions_total gauge\n"
        f"eirel_owner_validator_score_submissions_total {validator_submission_count}\n"
        # ── Runtime capacity ─────────────────────────────────────────
        "# TYPE eirel_owner_runtime_verified_nodes gauge\n"
        f"eirel_owner_runtime_verified_nodes {runtime_capacity['verified_node_count']}\n"
        "# TYPE eirel_owner_runtime_pending_candidate_capacity_total gauge\n"
        f"eirel_owner_runtime_pending_candidate_capacity_total {runtime_capacity['pending_candidate_count']}\n"
        "# TYPE eirel_owner_runtime_pending_serving_capacity_total gauge\n"
        f"eirel_owner_runtime_pending_serving_capacity_total {runtime_capacity['pending_serving_count']}\n"
        "# TYPE eirel_owner_runtime_reserved_cpu_millis gauge\n"
        f"eirel_owner_runtime_reserved_cpu_millis {runtime_capacity['total_reserved_cpu_millis']}\n"
        "# TYPE eirel_owner_runtime_reserved_memory_bytes gauge\n"
        f"eirel_owner_runtime_reserved_memory_bytes {runtime_capacity['total_reserved_memory_bytes']}\n"
        # ── Per-node deployment reservation ──────────────────────────
        + node_deployment_lines
        # ── Per-deployment info (for pod/node join) ──────────────────
        + deployment_info_lines
        # ── Workflow episodes ────────────────────────────────────────
        + "# TYPE eirel_owner_workflow_episode_total gauge\n"
        f"eirel_owner_workflow_episode_total {len(workflow_episode_records)}\n"
        "# TYPE eirel_owner_workflow_episode_queued_total gauge\n"
        f"eirel_owner_workflow_episode_queued_total {workflow_runtime_health['lifecycle_counts']['queued']}\n"
        "# TYPE eirel_owner_workflow_episode_executing_total gauge\n"
        f"eirel_owner_workflow_episode_executing_total {workflow_runtime_health['lifecycle_counts']['executing']}\n"
        "# TYPE eirel_owner_workflow_episode_deferred_total gauge\n"
        f"eirel_owner_workflow_episode_deferred_total {workflow_runtime_health['lifecycle_counts']['deferred']}\n"
        "# TYPE eirel_owner_workflow_episode_retry_wait_total gauge\n"
        f"eirel_owner_workflow_episode_retry_wait_total {workflow_runtime_health['retry_wait_episode_count']}\n"
        "# TYPE eirel_owner_workflow_episode_retryable_total gauge\n"
        f"eirel_owner_workflow_episode_retryable_total {workflow_runtime_health['retryable_episode_count']}\n"
        "# TYPE eirel_owner_workflow_episode_dead_lettered_total gauge\n"
        f"eirel_owner_workflow_episode_dead_lettered_total {workflow_runtime_health['dead_lettered_episode_count']}\n"
        "# TYPE eirel_owner_workflow_episode_stale_total gauge\n"
        f"eirel_owner_workflow_episode_stale_total {workflow_runtime_health['stale_episode_count']}\n"
        "# TYPE eirel_owner_workflow_episode_task_backed_total gauge\n"
        f"eirel_owner_workflow_episode_task_backed_total {workflow_runtime_health['task_backed_episode_count']}\n"
        "# TYPE eirel_owner_workflow_episode_internal_total gauge\n"
        f"eirel_owner_workflow_episode_internal_total {workflow_runtime_health['internal_episode_count']}\n"
        # ── Node fleet ───────────────────────────────────────────────
        + node_lines
        # ── Per-miner scorecard leaderboard ──────────────────────────
        + scorecard_lines
        # ── Per-task live progress for the current open run ──────────
        + live_run_lines
        # ── Per-validator validator-paid cost (open run + most-recent) ─
        + validator_cost_lines
    )
    _metrics_cache["body"] = body
    _metrics_cache["expires_at"] = time.monotonic() + _METRICS_CACHE_TTL
    return Response(content=body, media_type="text/plain; version=0.0.4; charset=utf-8")


def _format_submission_metrics(rows: list[tuple[str, str, int]]) -> str:
    """Format per-status, per-family submission gauge lines."""
    if not rows:
        return ""
    lines = [
        "# HELP eirel_owner_submissions Submissions by status and family.",
        "# TYPE eirel_owner_submissions gauge",
    ]
    for status, family, count in rows:
        lines.append(f'eirel_owner_submissions{{status="{status}",family="{family}"}} {count}')
    lines.append("")
    return "\n".join(lines) + "\n"


def _format_deployment_metrics(rows: list[tuple[str, str, str, int]]) -> str:
    """Format per-status, per-health, per-family deployment gauge lines."""
    if not rows:
        return ""
    lines = [
        "# HELP eirel_owner_deployments Active deployments by status, health, and family.",
        "# TYPE eirel_owner_deployments gauge",
    ]
    for status, health, family, count in rows:
        lines.append(
            f'eirel_owner_deployments{{status="{status}",health="{health}",family="{family}"}} {count}'
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def _format_node_deployment_metrics(rows: list[tuple[str, int, int, int]]) -> str:
    """Format per-node deployment count and resource reservation."""
    if not rows:
        return ""
    lines = [
        "# HELP eirel_owner_node_deployments Active deployments on a runtime node.",
        "# TYPE eirel_owner_node_deployments gauge",
    ]
    for node, count, _cpu, _mem in rows:
        lines.append(f'eirel_owner_node_deployments{{node="{node}"}} {count}')

    lines.append("# HELP eirel_owner_node_reserved_cpu_millis CPU millicores reserved by deployments on a node.")
    lines.append("# TYPE eirel_owner_node_reserved_cpu_millis gauge")
    for node, _count, cpu, _mem in rows:
        lines.append(f'eirel_owner_node_reserved_cpu_millis{{node="{node}"}} {cpu or 0}')

    lines.append("# HELP eirel_owner_node_reserved_memory_bytes Memory bytes reserved by deployments on a node.")
    lines.append("# TYPE eirel_owner_node_reserved_memory_bytes gauge")
    for node, _count, _cpu, mem in rows:
        lines.append(f'eirel_owner_node_reserved_memory_bytes{{node="{node}"}} {mem or 0}')

    lines.append("")
    return "\n".join(lines) + "\n"


def _format_deployment_info_metrics(
    rows: list[tuple[str, str, str, str, str, str, str | None, bool]],
) -> str:
    """Per-deployment identity gauge for submission→pod→node joins.

    Emitted value is always 1; the carrier is the label set.  Joined against
    ``kube_pod_info`` (submission_id extracted from pod name) to attach the
    pod name and k8s-visible node to each owner-api-tracked deployment.
    """
    if not rows:
        return ""
    lines = [
        "# HELP eirel_owner_deployment_info Per-deployment identity row "
        "(value=1) carrying labels for submission→pod→node joins.",
        "# TYPE eirel_owner_deployment_info gauge",
    ]
    for (
        deployment_id,
        submission_id,
        hotkey,
        family,
        status,
        health,
        node,
        is_active,
    ) in rows:
        lines.append(
            f'eirel_owner_deployment_info{{'
            f'deployment_id="{deployment_id}",'
            f'submission_id="{submission_id}",'
            f'hotkey="{hotkey}",'
            f'family="{family}",'
            f'status="{status}",'
            f'health="{health}",'
            f'node="{node or ""}",'
            f'is_active="{int(bool(is_active))}"'
            f"}} 1"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def _format_node_metrics(nodes: list[RuntimeNodeSnapshot]) -> str:
    """Format per-node Prometheus gauge lines for baremetal/runtime nodes."""
    if not nodes:
        return (
            "# HELP eirel_owner_node_count Total runtime nodes known to the owner API.\n"
            "# TYPE eirel_owner_node_count gauge\n"
            "eirel_owner_node_count 0\n"
        )

    lines: list[str] = [
        "# HELP eirel_owner_node_count Total runtime nodes known to the owner API.",
        "# TYPE eirel_owner_node_count gauge",
        f"eirel_owner_node_count {len(nodes)}",
        "# HELP eirel_owner_node_ready Whether the node is reachable and reporting status.",
        "# TYPE eirel_owner_node_ready gauge",
    ]
    for n in nodes:
        lines.append(f'eirel_owner_node_ready{{node="{n.node_name}"}} {int(n.ready)}')

    lines.append("# HELP eirel_owner_node_schedulable Whether the node accepts new containers.")
    lines.append("# TYPE eirel_owner_node_schedulable gauge")
    for n in nodes:
        lines.append(f'eirel_owner_node_schedulable{{node="{n.node_name}"}} {int(n.schedulable)}')

    lines.append("# HELP eirel_owner_node_cpu_millis Allocatable CPU in millicores.")
    lines.append("# TYPE eirel_owner_node_cpu_millis gauge")
    for n in nodes:
        lines.append(f'eirel_owner_node_cpu_millis{{node="{n.node_name}"}} {n.allocatable_cpu_millis}')

    lines.append("# HELP eirel_owner_node_memory_bytes Allocatable memory in bytes.")
    lines.append("# TYPE eirel_owner_node_memory_bytes gauge")
    for n in nodes:
        lines.append(f'eirel_owner_node_memory_bytes{{node="{n.node_name}"}} {n.allocatable_memory_bytes}')

    lines.append("# HELP eirel_owner_node_pod_capacity Maximum container slots on the node.")
    lines.append("# TYPE eirel_owner_node_pod_capacity gauge")
    for n in nodes:
        lines.append(f'eirel_owner_node_pod_capacity{{node="{n.node_name}"}} {n.allocatable_pod_count}')

    lines.append("# HELP eirel_owner_node_disk_avail_bytes Available disk in bytes (baremetal only).")
    lines.append("# TYPE eirel_owner_node_disk_avail_bytes gauge")
    for n in nodes:
        meta = n.metadata_json or {}
        disk = int(meta.get("disk_avail_bytes", 0))
        lines.append(f'eirel_owner_node_disk_avail_bytes{{node="{n.node_name}"}} {disk}')

    lines.append("")
    return "\n".join(lines)


# -- Scorecard leaderboard ---------------------------------------------------


_SCORECARD_RUNS_PER_FAMILY = 2


def _collect_scorecard_rows(session) -> list[dict[str, Any]]:
    """Return per-miner scorecard rows for the latest N runs per family.

    Caps cardinality by: (a) only DeploymentScoreRecord rows (no zero-fill for
    missing miners), (b) at most ``_SCORECARD_RUNS_PER_FAMILY`` most-recent
    runs per family, chosen by EvaluationRun.sequence.
    """
    pair_rows = (
        session.query(
            DeploymentScoreRecord.family_id,
            DeploymentScoreRecord.run_id,
            EvaluationRun.sequence,
        )
        .join(EvaluationRun, EvaluationRun.id == DeploymentScoreRecord.run_id)
        .distinct()
        .all()
    )
    runs_by_family: dict[str, list[tuple[int, str]]] = {}
    for family_id, run_id, sequence in pair_rows:
        runs_by_family.setdefault(family_id, []).append((sequence, run_id))

    latest_runs_by_family: dict[str, list[str]] = {}
    for family_id, pairs in runs_by_family.items():
        pairs.sort(reverse=True)
        latest_runs_by_family[family_id] = [
            run_id for _seq, run_id in pairs[:_SCORECARD_RUNS_PER_FAMILY]
        ]

    if not latest_runs_by_family:
        return []

    run_ids = [rid for ids in latest_runs_by_family.values() for rid in ids]
    score_rows = (
        session.query(DeploymentScoreRecord)
        .filter(DeploymentScoreRecord.run_id.in_(run_ids))
        .all()
    )
    winner_rows = (
        session.query(RunFamilyResult.run_id, RunFamilyResult.family_id, RunFamilyResult.winner_hotkey)
        .filter(RunFamilyResult.run_id.in_(run_ids))
        .all()
    )
    winner_by_run_family: dict[tuple[str, str], str | None] = {
        (run_id, family_id): winner
        for run_id, family_id, winner in winner_rows
    }
    # Per-(run, family, miner) counts derived from TaskMinerResult. Under
    # the pairwise redesign every row represents a completed judgment, so
    # total == evaluated; errored judgments are still rows with verdict=error.
    task_rows = (
        session.query(
            TaskMinerResult.run_id,
            TaskMinerResult.family_id,
            TaskMinerResult.miner_hotkey,
            func.count(),
        )
        .filter(TaskMinerResult.run_id.in_(run_ids))
        .group_by(
            TaskMinerResult.run_id,
            TaskMinerResult.family_id,
            TaskMinerResult.miner_hotkey,
        )
        .all()
    )
    task_totals: dict[tuple[str, str, str], dict[str, int]] = {}
    for run_id, family_id, hotkey, count in task_rows:
        task_totals[(run_id, family_id, hotkey)] = {"evaluated": count, "total": count}

    allowed_pairs = {
        (run_id, family_id)
        for family_id, ids in latest_runs_by_family.items()
        for run_id in ids
    }
    by_run_family: dict[tuple[str, str], list[DeploymentScoreRecord]] = {}
    for record in score_rows:
        key = (record.run_id, record.family_id)
        if key not in allowed_pairs:
            continue
        by_run_family.setdefault(key, []).append(record)

    rows: list[dict[str, Any]] = []
    for (run_id, family_id), records in by_run_family.items():
        ordered = sorted(records, key=lambda r: r.raw_score, reverse=True)
        winner = winner_by_run_family.get((run_id, family_id))
        for rank, record in enumerate(ordered, start=1):
            tasks = task_totals.get((run_id, family_id, record.miner_hotkey), {"evaluated": 0, "total": 0})
            rows.append(
                {
                    "family_id": family_id,
                    "run_id": run_id,
                    "hotkey": record.miner_hotkey,
                    "raw_score": float(record.raw_score or 0.0),
                    "normalized_score": float(record.normalized_score or 0.0),
                    "rank": rank,
                    "tasks_completed": tasks["evaluated"],
                    "tasks_total": tasks["total"],
                    "llm_cost_usd": float(record.llm_cost_usd or 0.0),
                    "tool_cost_usd": float(record.tool_cost_usd or 0.0),
                    "is_winner": 1 if (winner and record.miner_hotkey == winner) else 0,
                }
            )
    return rows


def _format_scorecard_metrics(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    lines: list[str] = []
    metric_defs = [
        (
            "eirel_owner_scorecard_raw_score",
            "Raw score for a miner in a given run/family.",
            "raw_score",
        ),
        (
            "eirel_owner_scorecard_normalized_score",
            "Normalized score for a miner in a given run/family.",
            "normalized_score",
        ),
        (
            "eirel_owner_scorecard_rank",
            "Rank (1 = top) of a miner by raw_score within run/family.",
            "rank",
        ),
        (
            "eirel_owner_scorecard_tasks_completed",
            "Count of evaluated tasks for a miner in the given run.",
            "tasks_completed",
        ),
        (
            "eirel_owner_scorecard_tasks_total",
            "Total tasks assigned to a miner in the given run.",
            "tasks_total",
        ),
        (
            "eirel_owner_scorecard_llm_cost_usd",
            "LLM spend (USD) attributed to a miner in the given run.",
            "llm_cost_usd",
        ),
        (
            "eirel_owner_scorecard_tool_cost_usd",
            "Tool spend (USD) attributed to a miner in the given run.",
            "tool_cost_usd",
        ),
        (
            "eirel_owner_scorecard_is_winner",
            "1 if the miner was the published winner of the run, 0 otherwise.",
            "is_winner",
        ),
    ]
    for metric_name, help_text, row_key in metric_defs:
        lines.append(f"# HELP {metric_name} {help_text}")
        lines.append(f"# TYPE {metric_name} gauge")
        for row in rows:
            value = row[row_key]
            formatted = f"{value:.6f}" if isinstance(value, float) else str(value)
            lines.append(
                f'{metric_name}{{family="{row["family_id"]}",'
                f'run_id="{row["run_id"]}",'
                f'hotkey="{row["hotkey"]}"}} {formatted}'
            )
    lines.append("")
    return "\n".join(lines)


# -- Live-run task progress ---------------------------------------------------


_LIVE_RUN_STATUSES = ("pending", "claimed", "evaluated")


def _collect_live_run_rows(session) -> dict[str, Any]:
    """Per-task status/score snapshot for the single current open run.

    Bounded cardinality by design: only one run (the open one) is emitted,
    so the series count is ``|miners| * |tasks|`` at most — e.g. ~60 for
    general_chat's 3-miner × 20-task benchmark.
    """
    open_run = (
        session.query(EvaluationRun)
        .filter_by(status="open")
        .order_by(EvaluationRun.sequence.desc())
        .first()
    )
    if open_run is None:
        return {"run_id": None, "tasks": [], "progress": {}}

    tasks = (
        session.query(TaskMinerResult)
        .filter_by(run_id=open_run.id)
        .all()
    )
    rows: list[dict[str, Any]] = []
    progress: dict[tuple[str, str, str], int] = {}
    for task in tasks:
        # Under the pairwise schema every TaskMinerResult row represents a
        # landed judgment. A verdict of "error" is the analog of old "failed";
        # everything else is "evaluated".
        status = "evaluated" if task.agreement_verdict != "error" else "failed"
        response = (task.miner_response_json or {}).get("response") or {}
        output = response.get("output") or {}
        out_meta = output.get("metadata") or {}
        tool_calls = output.get("tool_calls") or []
        web_search_used = any(
            isinstance(tc, dict) and tc.get("tool_name") == "web_search"
            for tc in tool_calls
        )
        rows.append(
            {
                "run_id": open_run.id,
                "family_id": task.family_id,
                "hotkey": task.miner_hotkey,
                "task_id": task.task_id,
                "task_index": 0,
                "status": status,
                "score": float(task.agreement_score or 0.0),
                "latency_ms": (
                    int(task.miner_latency_seconds * 1000)
                    if task.miner_latency_seconds
                    else 0
                ),
                "web_search_enabled": bool(out_meta.get("web_search_enabled", False)),
                "web_search_used": 1 if web_search_used else 0,
                "tool_call_count": len(tool_calls),
                "is_evaluated": 1 if status == "evaluated" else 0,
            }
        )
        key = (task.family_id, task.miner_hotkey, status)
        progress[key] = progress.get(key, 0) + 1

    return {"run_id": open_run.id, "tasks": rows, "progress": progress}


def _format_live_run_metrics(data: dict[str, Any]) -> str:
    run_id = data.get("run_id")
    rows: list[dict[str, Any]] = data.get("tasks") or []
    progress: dict[tuple[str, str, str], int] = data.get("progress") or {}
    if not run_id:
        return ""
    lines: list[str] = []

    # eirel_owner_task_status: one row per task with status baked into the
    # label, value=1.  Grafana filters `status="..."` to pick a state.
    lines.append("# HELP eirel_owner_task_status Status of a task in the current open run (value=1).")
    lines.append("# TYPE eirel_owner_task_status gauge")
    for row in rows:
        lines.append(
            f'eirel_owner_task_status{{run_id="{row["run_id"]}",'
            f'family="{row["family_id"]}",'
            f'hotkey="{row["hotkey"]}",'
            f'task_id="{row["task_id"]}",'
            f'status="{row["status"]}"}} 1'
        )

    # eirel_owner_task_score: only emit for evaluated tasks so the timeline
    # panel doesn't need to drop zeros from the render.
    lines.append("# HELP eirel_owner_task_score Task score (0-1) for evaluated tasks in the current open run.")
    lines.append("# TYPE eirel_owner_task_score gauge")
    for row in rows:
        if row["is_evaluated"]:
            lines.append(
                f'eirel_owner_task_score{{run_id="{row["run_id"]}",'
                f'family="{row["family_id"]}",'
                f'hotkey="{row["hotkey"]}",'
                f'task_id="{row["task_id"]}"}} {row["score"]:.6f}'
            )

    lines.append("# HELP eirel_owner_task_latency_ms Miner response latency (ms) for evaluated tasks.")
    lines.append("# TYPE eirel_owner_task_latency_ms gauge")
    for row in rows:
        if row["is_evaluated"] and row["latency_ms"] > 0:
            lines.append(
                f'eirel_owner_task_latency_ms{{run_id="{row["run_id"]}",'
                f'family="{row["family_id"]}",'
                f'hotkey="{row["hotkey"]}",'
                f'task_id="{row["task_id"]}"}} {row["latency_ms"]}'
            )

    lines.append("# HELP eirel_owner_task_web_search_used 1 if the miner invoked web_search on this task, else 0.")
    lines.append("# TYPE eirel_owner_task_web_search_used gauge")
    for row in rows:
        if row["is_evaluated"]:
            lines.append(
                f'eirel_owner_task_web_search_used{{run_id="{row["run_id"]}",'
                f'family="{row["family_id"]}",'
                f'hotkey="{row["hotkey"]}",'
                f'task_id="{row["task_id"]}"}} {row["web_search_used"]}'
            )

    # Aggregate progress bars.
    lines.append("# HELP eirel_owner_task_progress Task count per (run,family,hotkey,state) for the current open run.")
    lines.append("# TYPE eirel_owner_task_progress gauge")
    for (family_id, hotkey, status), count in progress.items():
        lines.append(
            f'eirel_owner_task_progress{{run_id="{run_id}",'
            f'family="{family_id}",'
            f'hotkey="{hotkey}",'
            f'state="{status}"}} {count}'
        )

    lines.append("")
    return "\n".join(lines) + "\n"


# -- Validator cost roll-up ---------------------------------------------------


_VALIDATOR_COST_RUNS = 2  # current open + 1 most-recent closed


def _collect_validator_cost_rows(session) -> list[dict[str, Any]]:
    """Per-(run, validator) cost roll-up bounded to the latest 2 runs.

    Series cardinality = ``|validators| × 2 runs``, which keeps the
    Prometheus footprint constant even as runs accumulate. Each row
    covers ``oracle_cost_usd`` (validator's grounding spend) and
    ``judge_cost_usd`` (validator's eiretes-judge spend) — the two
    components a validator actually pays out of pocket.
    """
    runs = (
        session.query(EvaluationRun.id, EvaluationRun.sequence)
        .order_by(EvaluationRun.sequence.desc())
        .limit(_VALIDATOR_COST_RUNS)
        .all()
    )
    if not runs:
        return []
    run_ids = [run_id for run_id, _seq in runs]

    # Oracle cost + claim counts grouped by (run, validator).
    oracle_rows = (
        session.query(
            TaskEvaluation.run_id,
            TaskEvaluation.claimed_by_validator,
            func.count(TaskEvaluation.id),
            func.coalesce(func.sum(TaskEvaluation.oracle_cost_usd), 0.0),
        )
        .filter(
            TaskEvaluation.run_id.in_(run_ids),
            TaskEvaluation.claimed_by_validator.is_not(None),
        )
        .group_by(TaskEvaluation.run_id, TaskEvaluation.claimed_by_validator)
        .all()
    )

    # Judge cost grouped by (run, validator) — joined through
    # ``task_evaluations`` to attribute each miner-result row to the
    # claiming validator.
    judge_rows = (
        session.query(
            TaskEvaluation.run_id,
            TaskEvaluation.claimed_by_validator,
            func.coalesce(func.sum(TaskMinerResult.judge_cost_usd), 0.0),
        )
        .join(
            TaskMinerResult,
            TaskMinerResult.task_evaluation_id == TaskEvaluation.id,
        )
        .filter(
            TaskEvaluation.run_id.in_(run_ids),
            TaskEvaluation.claimed_by_validator.is_not(None),
        )
        .group_by(TaskEvaluation.run_id, TaskEvaluation.claimed_by_validator)
        .all()
    )
    judge_by_pair: dict[tuple[str, str], float] = {
        (run_id, hotkey): float(judge_cost or 0.0)
        for run_id, hotkey, judge_cost in judge_rows
    }

    rows: list[dict[str, Any]] = []
    for run_id, hotkey, tasks_claimed, oracle_cost in oracle_rows:
        oracle = float(oracle_cost or 0.0)
        judge = judge_by_pair.get((run_id, hotkey), 0.0)
        rows.append({
            "run_id": run_id,
            "hotkey": hotkey,
            "tasks_claimed": int(tasks_claimed or 0),
            "oracle_cost_usd": oracle,
            "judge_cost_usd": judge,
            "total_cost_usd": oracle + judge,
        })
    return rows


def _format_validator_cost_metrics(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    lines: list[str] = []
    metric_defs = [
        (
            "eirel_owner_validator_run_oracle_cost_usd",
            "Validator-paid oracle/reconciler spend (USD) per run.",
            "oracle_cost_usd",
        ),
        (
            "eirel_owner_validator_run_judge_cost_usd",
            "Validator-paid eiretes-judge spend (USD) per run.",
            "judge_cost_usd",
        ),
        (
            "eirel_owner_validator_run_total_cost_usd",
            "Validator-paid total (oracle + judge) spend (USD) per run.",
            "total_cost_usd",
        ),
        (
            "eirel_owner_validator_run_tasks_claimed",
            "Tasks claimed by this validator in the run.",
            "tasks_claimed",
        ),
    ]
    for metric_name, help_text, row_key in metric_defs:
        lines.append(f"# HELP {metric_name} {help_text}")
        lines.append(f"# TYPE {metric_name} gauge")
        for row in rows:
            value = row[row_key]
            formatted = f"{value:.6f}" if isinstance(value, float) else str(value)
            lines.append(
                f'{metric_name}{{run_id="{row["run_id"]}",'
                f'hotkey="{row["hotkey"]}"}} {formatted}'
            )
    lines.append("")
    return "\n".join(lines)
