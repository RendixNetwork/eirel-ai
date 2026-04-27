from __future__ import annotations

"""Task-level evaluation management.

After the pairwise redesign, each TaskEvaluation row represents a single task
in a run. A validator claims the row, fans out to every registered miner for
that task, calls the OpenAI baseline, and submits per-miner pairwise judgments
in one batch. Aggregation rolls per-miner TaskMinerResult rows into
MinerEvaluationSummary via pairwise win rate.
"""

import logging
import math
from datetime import timedelta
from typing import Any, TYPE_CHECKING

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from shared.common.models import (
    EpochTargetSnapshot,
    MinerEvaluationSummary,
    TaskEvaluation,
    TaskMinerResult,
    utcnow,
)
from eirel.groups import ensure_active_family_id

from control_plane.owner_api._helpers import _strip_sensitive_task_metadata

if TYPE_CHECKING:
    from control_plane.owner_api.managed import ManagedOwnerServices

logger = logging.getLogger(__name__)


def _compress_scores(
    effective_scores: dict[str, float],
    *,
    power: float = 0.65,
    outlier_z_threshold: float = 2.0,
) -> dict[str, float]:
    """Power-law compression + outlier dampening to resist cartel gaming."""
    if not effective_scores:
        return effective_scores
    values = [v for v in effective_scores.values() if v > 0]
    if not values:
        return effective_scores

    sorted_vals = sorted(values)
    median = sorted_vals[len(sorted_vals) // 2]
    if len(values) >= 3:
        mean_val = sum(values) / len(values)
        stddev = (sum((v - mean_val) ** 2 for v in values) / len(values)) ** 0.5
        cap = median + outlier_z_threshold * stddev if stddev > 0 else float("inf")
    else:
        cap = float("inf")

    compressed: dict[str, float] = {}
    for hk, score in effective_scores.items():
        if score <= 0:
            compressed[hk] = 0.0
            continue
        capped = min(score, cap)
        compressed[hk] = math.pow(capped, power)
    return compressed


class EvaluationTaskManager:
    """Handles claim/submit/status for task-level distributed evaluation."""

    def __init__(self, owner: ManagedOwnerServices) -> None:
        self._owner = owner

    @property
    def settings(self):
        return self._owner.settings

    # ------------------------------------------------------------------
    # Task initialization (one row per task, no miner fan-out at seed time)
    # ------------------------------------------------------------------

    def initialize_evaluation_tasks(
        self,
        session: Session,
        *,
        run_id: str,
        family_id: str,
        snapshot: EpochTargetSnapshot,
    ) -> int:
        """Create one TaskEvaluation row per task in the run bundle.

        Miner list is stored on the EpochTargetSnapshot (`members_json`) and
        attached to the claim response at claim time, so the validator knows
        which miners to fan out to without the task row carrying it.

        Returns the number of tasks created.
        """
        bundle = self._owner.run_evaluation_bundle(
            session, run_id=run_id, family_id=family_id,
        )
        if not isinstance(bundle, dict):
            logger.warning("no evaluation bundle for run=%s family=%s", run_id, family_id)
            return 0
        tasks = bundle.get("tasks", [])
        if not tasks:
            logger.warning("empty task list in evaluation bundle for run=%s family=%s", run_id, family_id)
            return 0

        # Optional E2E knob: cap tasks per run for smoke tests.
        import os
        task_limit_raw = os.getenv("EIREL_EVAL_TASK_LIMIT", "")
        if task_limit_raw.strip():
            try:
                task_limit = int(task_limit_raw)
                if task_limit > 0 and task_limit < len(tasks):
                    tasks = tasks[:task_limit]
                    logger.info(
                        "EIREL_EVAL_TASK_LIMIT=%d — truncating %s bundle to %d tasks",
                        task_limit, family_id, len(tasks),
                    )
            except ValueError:
                logger.warning("EIREL_EVAL_TASK_LIMIT=%r is not an integer; ignoring", task_limit_raw)

        members = snapshot.members_json or []
        created = 0
        for task_index, task in enumerate(tasks):
            task_id = task.get("task_id", "")
            if not task_id:
                continue
            existing = session.execute(
                select(TaskEvaluation).where(
                    TaskEvaluation.run_id == run_id,
                    TaskEvaluation.family_id == family_id,
                    TaskEvaluation.task_id == task_id,
                ).limit(1)
            ).scalar_one_or_none()
            if existing is not None:
                continue
            session.add(TaskEvaluation(
                run_id=run_id,
                family_id=family_id,
                task_id=task_id,
                task_index=task_index,
                status="pending",
            ))
            created += 1

        # Seed one MinerEvaluationSummary per miner so aggregation has a
        # row to update as task results land.
        for member in members:
            miner_hotkey = member.get("hotkey", "")
            if not miner_hotkey:
                continue
            existing_summary = session.execute(
                select(MinerEvaluationSummary).where(
                    MinerEvaluationSummary.run_id == run_id,
                    MinerEvaluationSummary.family_id == family_id,
                    MinerEvaluationSummary.miner_hotkey == miner_hotkey,
                ).limit(1)
            ).scalar_one_or_none()
            if existing_summary is None:
                session.add(MinerEvaluationSummary(
                    run_id=run_id,
                    family_id=family_id,
                    miner_hotkey=miner_hotkey,
                    total_tasks=len(tasks),
                    status="pending",
                ))

        session.flush()
        logger.info(
            "initialized %d task evaluations for run=%s family=%s miners=%d",
            created, run_id, family_id, len(members),
        )
        return created

    # ------------------------------------------------------------------
    # Claim tasks (task-level)
    # ------------------------------------------------------------------

    def claim_tasks(
        self,
        session: Session,
        *,
        run_id: str,
        family_id: str,
        validator_hotkey: str,
        batch_size: int = 1,
    ) -> list[TaskEvaluation]:
        """Claim up to *batch_size* pending TaskEvaluations.

        Each claim is a task — validator handles all miners for that task.
        Default batch size is 1: a task already fans out to N miners + OpenAI
        baseline + N judges in parallel internally, so claim/submit overhead is
        a tiny fraction of per-task work. Larger batches just lengthen lease
        hold time, concentrate failure blast radius, and skew validator
        consensus by letting one validator monopolize a chunk of the run.
        Operators with high control-plane RTT can bump this via env.
        """
        now = utcnow()
        family_id = ensure_active_family_id(family_id)

        snapshot = session.execute(
            select(EpochTargetSnapshot)
            .where(EpochTargetSnapshot.run_id == run_id)
            .where(EpochTargetSnapshot.family_id == family_id)
        ).scalar_one_or_none()
        if snapshot is None or snapshot.status != "open":
            return []

        # Release expired claims (crashed validators)
        session.execute(
            update(TaskEvaluation)
            .where(TaskEvaluation.run_id == run_id)
            .where(TaskEvaluation.family_id == family_id)
            .where(TaskEvaluation.status == "claimed")
            .where(TaskEvaluation.claim_expires_at <= now)
            .values(
                status="pending",
                claimed_by_validator=None,
                claimed_at=None,
                claim_expires_at=None,
                updated_at=now,
            )
        )
        session.flush()

        candidates = list(
            session.execute(
                select(TaskEvaluation)
                .where(TaskEvaluation.run_id == run_id)
                .where(TaskEvaluation.family_id == family_id)
                .where(TaskEvaluation.status == "pending")
                .order_by(TaskEvaluation.task_index)
                .limit(batch_size)
                .with_for_update(skip_locked=True)
            ).scalars()
        )

        timeout = timedelta(seconds=self.settings.task_claim_timeout_seconds)
        for task in candidates:
            task.status = "claimed"
            task.claimed_by_validator = validator_hotkey
            task.claimed_at = now
            task.claim_expires_at = now + timeout
            task.claim_attempt_count += 1
            task.updated_at = now

        session.flush()
        return candidates

    # ------------------------------------------------------------------
    # Submit task result (per-task batch of per-miner agreement judgments)
    # ------------------------------------------------------------------

    def submit_task_result(
        self,
        session: Session,
        *,
        task_evaluation_id: str,
        validator_hotkey: str,
        baseline_response: dict[str, Any] | None,
        miner_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Record all per-miner agreement results for a claimed task.

        `miner_results` is a list of dicts with keys:
          miner_hotkey, miner_response, miner_citations, judge_output,
          agreement_score, verdict ("matches"|"partially_matches"|
          "contradicts"|"not_applicable"|"error"|"latency_violation"),
          miner_latency_seconds (miner wall-clock), latency_seconds (judge
          wall-clock — kept under the legacy key for back-compat).

        Citations are preserved on the row for dashboard display but do
        not participate in scoring. Returns status info (accepted/rejected
        + family completion flag).
        """
        now = utcnow()
        task = session.get(TaskEvaluation, task_evaluation_id)
        if task is None:
            return {
                "status": "rejected_not_found",
                "family_evaluation_complete": False,
                "remaining_task_count": -1,
            }

        if task.status == "evaluated" and task.claimed_by_validator == validator_hotkey:
            remaining = self._remaining_tasks(session, task)
            return {
                "status": "accepted",
                "family_evaluation_complete": remaining == 0,
                "remaining_task_count": remaining,
            }

        if task.claimed_by_validator != validator_hotkey:
            return {
                "status": "rejected_wrong_validator",
                "family_evaluation_complete": False,
                "remaining_task_count": -1,
            }

        if task.status != "claimed":
            return {
                "status": "rejected_not_claimed",
                "family_evaluation_complete": False,
                "remaining_task_count": -1,
            }

        task.baseline_response_json = baseline_response
        task.status = "evaluated"
        task.evaluated_at = now
        task.updated_at = now

        from shared.core.evaluation_models import VERDICT_SCORES

        valid_verdicts = set(VERDICT_SCORES)  # matches, partially_matches, not_applicable, contradicts, error
        for entry in miner_results:
            miner_hotkey = str(entry.get("miner_hotkey") or "").strip()
            if not miner_hotkey:
                continue
            verdict = str(entry.get("verdict") or "error")
            if verdict not in valid_verdicts:
                verdict = "error"
            # Score the row from the verdict if the validator didn't pre-fill
            # agreement_score, so the DB always has a consistent scalar.
            score = entry.get("agreement_score")
            if score is None:
                score = VERDICT_SCORES.get(verdict, 0.0)
            citations = entry.get("miner_citations") or []
            if not isinstance(citations, list):
                citations = []
            session.add(TaskMinerResult(
                task_evaluation_id=task.id,
                run_id=task.run_id,
                family_id=task.family_id,
                task_id=task.task_id,
                miner_hotkey=miner_hotkey,
                miner_response_json=entry.get("miner_response") or {},
                miner_citations_json=citations,
                judge_output_json=entry.get("judge_output"),
                agreement_verdict=verdict,
                agreement_score=float(score),
                miner_latency_seconds=float(entry.get("miner_latency_seconds") or 0.0),
                latency_seconds=float(entry.get("latency_seconds") or 0.0),
            ))
        session.flush()

        remaining = self._remaining_tasks(session, task)
        family_complete = remaining == 0
        if family_complete:
            self._on_family_evaluation_complete(
                session,
                run_id=task.run_id,
                family_id=task.family_id,
            )

        return {
            "status": "accepted",
            "family_evaluation_complete": family_complete,
            "remaining_task_count": remaining,
        }

    # ------------------------------------------------------------------
    # Submit baseline failure (task returned to pending for retry)
    # ------------------------------------------------------------------

    def mark_baseline_failed(
        self,
        session: Session,
        *,
        task_evaluation_id: str,
        validator_hotkey: str,
    ) -> dict[str, Any]:
        """Mark a claimed task as baseline_failed and return it to pending.

        Called by the validator when the OpenAI baseline call fails — the
        task should be available for another validator to try.
        """
        task = session.get(TaskEvaluation, task_evaluation_id)
        if task is None:
            return {"status": "rejected_not_found"}
        if task.claimed_by_validator != validator_hotkey:
            return {"status": "rejected_wrong_validator"}
        if task.status != "claimed":
            return {"status": "rejected_not_claimed"}
        now = utcnow()
        task.status = "pending"
        task.claimed_by_validator = None
        task.claimed_at = None
        task.claim_expires_at = None
        task.updated_at = now
        session.flush()
        return {"status": "released"}

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def evaluation_status(
        self,
        session: Session,
        *,
        run_id: str,
        family_id: str,
    ) -> dict[str, Any]:
        """Return evaluation progress for a family in a run."""
        family_id = ensure_active_family_id(family_id)

        rows = session.execute(
            select(
                TaskEvaluation.status,
                func.count(TaskEvaluation.id),
            )
            .where(TaskEvaluation.run_id == run_id)
            .where(TaskEvaluation.family_id == family_id)
            .group_by(TaskEvaluation.status)
        ).fetchall()

        status_counts: dict[str, int] = {"pending": 0, "claimed": 0, "evaluated": 0}
        for status, count in rows:
            status_counts[status] = status_counts.get(status, 0) + count

        # Per-miner count of judgments landed so far
        miner_rows = session.execute(
            select(
                TaskMinerResult.miner_hotkey,
                func.count(TaskMinerResult.id),
            )
            .where(TaskMinerResult.run_id == run_id)
            .where(TaskMinerResult.family_id == family_id)
            .group_by(TaskMinerResult.miner_hotkey)
        ).fetchall()
        miners = [
            {"miner_hotkey": hk, "judgments_received": count}
            for hk, count in sorted(miner_rows)
        ]

        total = sum(status_counts.values())
        return {
            "run_id": run_id,
            "family_id": family_id,
            "total_tasks": total,
            "pending_tasks": status_counts["pending"],
            "claimed_tasks": status_counts["claimed"],
            "evaluated_tasks": status_counts["evaluated"],
            "miners": miners,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _remaining_tasks(self, session: Session, task: TaskEvaluation) -> int:
        """Count unevaluated TaskEvaluations in the same run/family."""
        remaining = session.execute(
            select(func.count(TaskEvaluation.id))
            .where(TaskEvaluation.run_id == task.run_id)
            .where(TaskEvaluation.family_id == task.family_id)
            .where(TaskEvaluation.status != "evaluated")
        ).scalar_one()
        return int(remaining or 0)

    def finalize_run_family(
        self,
        session: Session,
        *,
        run_id: str,
        family_id: str,
    ) -> None:
        self._on_family_evaluation_complete(
            session, run_id=run_id, family_id=family_id,
        )

    # ------------------------------------------------------------------
    # Per-miner scoring + family aggregation
    # ------------------------------------------------------------------

    def _aggregate_miner_from_results(
        self,
        session: Session,
        *,
        run_id: str,
        family_id: str,
        miner_hotkey: str,
    ) -> None:
        """Compute MinerEvaluationSummary from TaskMinerResult rows.

        Primary score is the mean of per-task ``agreement_score`` values
        across non-error rows. Miners with error_rate > 30% are capped
        at 0.5 (see MinerRollup).
        """
        from control_plane.owner_api.evaluation.general_chat_scoring import (
            aggregate_miner_score,
        )

        summary = session.execute(
            select(MinerEvaluationSummary).where(
                MinerEvaluationSummary.run_id == run_id,
                MinerEvaluationSummary.family_id == family_id,
                MinerEvaluationSummary.miner_hotkey == miner_hotkey,
            ).limit(1)
        ).scalar_one_or_none()
        if summary is None:
            return

        results = list(session.execute(
            select(TaskMinerResult)
            .where(TaskMinerResult.run_id == run_id)
            .where(TaskMinerResult.family_id == family_id)
            .where(TaskMinerResult.miner_hotkey == miner_hotkey)
        ).scalars())

        rollup = aggregate_miner_score(results)
        summary.completed_tasks = rollup.completed
        summary.failed_tasks = rollup.errored
        # The mean agreement is the authoritative capability signal.
        summary.family_capability_score = rollup.mean_agreement
        # robustness_score / anti_gaming_score columns are legacy — retained
        # on the summary row for backward compat but set to None since we
        # no longer have dimension breakdowns or trace-gate metrics.
        summary.robustness_score = None
        summary.anti_gaming_score = None
        summary.official_family_score = rollup.final_score
        summary.protocol_gate_passed = rollup.reliable
        summary.rollout_metadata_json = rollup.to_metadata()
        summary.status = "scored"
        summary.updated_at = utcnow()
        session.flush()

    def _on_family_evaluation_complete(
        self,
        session: Session,
        *,
        run_id: str,
        family_id: str,
    ) -> None:
        """Aggregate all per-miner results into summaries + snapshot.

        Runs after the final task for a family has been evaluated. Iterates
        every miner, rolls up their TaskMinerResult rows, and rebuilds the
        AggregateFamilyScoreSnapshot + DeploymentScoreRecord entries.
        """
        from shared.common.models import (
            AggregateFamilyScoreSnapshot,
            DeploymentScoreRecord,
            EpochTargetSnapshot,
            ManagedDeployment,
        )
        from sqlalchemy import delete

        now = utcnow()

        # Recompute summary rows for every miner that produced results
        miner_hotkeys = [
            hk for (hk,) in session.execute(
                select(TaskMinerResult.miner_hotkey.distinct())
                .where(TaskMinerResult.run_id == run_id)
                .where(TaskMinerResult.family_id == family_id)
            ).fetchall()
        ]
        for miner_hotkey in miner_hotkeys:
            self._aggregate_miner_from_results(
                session, run_id=run_id, family_id=family_id, miner_hotkey=miner_hotkey,
            )

        summaries = list(session.execute(
            select(MinerEvaluationSummary)
            .where(MinerEvaluationSummary.run_id == run_id)
            .where(MinerEvaluationSummary.family_id == family_id)
            .where(MinerEvaluationSummary.status == "scored")
        ).scalars())

        if not summaries:
            logger.warning("no scored summaries for run=%s family=%s", run_id, family_id)
            return

        miner_scores: dict[str, float] = {
            s.miner_hotkey: s.official_family_score or 0.0 for s in summaries
        }

        effective_scores = _compress_scores(dict(miner_scores))
        total = sum(max(0.0, v) for v in effective_scores.values())
        if total > 0:
            normalized_weights = {
                hk: max(0.0, v) / total for hk, v in sorted(effective_scores.items())
            }
        else:
            normalized_weights = {hk: 0.0 for hk in sorted(effective_scores)}

        contributing_validators = {
            v for (v,) in session.execute(
                select(TaskEvaluation.claimed_by_validator.distinct())
                .where(TaskEvaluation.run_id == run_id)
                .where(TaskEvaluation.family_id == family_id)
                .where(TaskEvaluation.status == "evaluated")
                .where(TaskEvaluation.claimed_by_validator.isnot(None))
            ).fetchall()
        }
        validator_weights = {v_hotkey: 1 for v_hotkey in contributing_validators}

        miner_score_breakdowns = {
            s.miner_hotkey: {
                "family_capability_score": s.family_capability_score or 0.0,
                "robustness_score": s.robustness_score or 0.0,
                "anti_gaming_score": s.anti_gaming_score or 0.0,
                "official_family_score": s.official_family_score or 0.0,
            }
            for s in summaries
        }

        snapshot = session.execute(
            select(EpochTargetSnapshot).where(
                EpochTargetSnapshot.run_id == run_id,
                EpochTargetSnapshot.family_id == family_id,
            ).limit(1)
        ).scalar_one_or_none()

        snapshot_json = {
            "run_id": run_id,
            "family_id": family_id,
            "evaluation_plane": "agreement_against_openai_baseline",
            "miner_scores": miner_scores,
            "normalized_weights": normalized_weights,
            "rubric_version": snapshot.rubric_version if snapshot else "agreement_general_chat_v1",
            "miner_score_breakdowns": miner_score_breakdowns,
            "task_count": summaries[0].total_tasks if summaries else 0,
            "judge_model": snapshot.judge_model if snapshot else "local-rubric-judge",
            "evaluation_timestamp": now.isoformat() + "Z",
        }

        session.execute(
            delete(AggregateFamilyScoreSnapshot)
            .where(AggregateFamilyScoreSnapshot.run_id == run_id)
            .where(AggregateFamilyScoreSnapshot.family_id == family_id)
        )
        session.flush()

        aggregate = AggregateFamilyScoreSnapshot(
            run_id=run_id,
            family_id=family_id,
            snapshot_json=snapshot_json,
            validator_count=len(contributing_validators),
            validator_hotkeys_json=sorted(contributing_validators),
            validator_weights_json=validator_weights,
            consensus_method="agreement_against_openai_baseline",
            status="aggregated",
            activated_at=now,
        )
        session.add(aggregate)
        session.flush()

        session.execute(
            delete(DeploymentScoreRecord)
            .where(DeploymentScoreRecord.run_id == run_id)
            .where(DeploymentScoreRecord.family_id == family_id)
        )
        session.flush()

        for s in summaries:
            deployment = session.execute(
                select(ManagedDeployment)
                .where(ManagedDeployment.miner_hotkey == s.miner_hotkey)
                .where(ManagedDeployment.family_id == family_id)
                .where(ManagedDeployment.status != "retired")
                .order_by(ManagedDeployment.created_at.desc())
                .limit(1)
            ).scalar_one_or_none()
            if deployment is None:
                continue

            score_record = DeploymentScoreRecord(
                run_id=run_id,
                family_id=family_id,
                deployment_id=deployment.id,
                submission_id=deployment.submission_id,
                miner_hotkey=s.miner_hotkey,
                deployment_revision=deployment.deployment_revision,
                raw_score=s.official_family_score or 0.0,
                normalized_score=normalized_weights.get(s.miner_hotkey, 0.0),
                is_eligible=deployment.health_status == "healthy",
                metadata_json={
                    "mean_agreement": s.family_capability_score,
                    "final_score": s.official_family_score,
                    "reliable": s.protocol_gate_passed,
                    "completed_tasks": s.completed_tasks,
                    "failed_tasks": s.failed_tasks,
                    "total_tasks": s.total_tasks,
                    "evaluation_method": "agreement_against_openai_baseline",
                },
            )
            self._owner.scoring.populate_cost_columns(
                score_record,
                deployment.id,
                self._owner.settings.run_budget_usd,
            )
            session.add(score_record)

        if snapshot:
            snapshot.status = "scored"
            snapshot.scored_at = now

        session.flush()

        try:
            self._owner.rebalance_family(session, family_id=family_id)
        except Exception:
            logger.exception(
                "rebalance_family failed after pairwise evaluation: run=%s family=%s",
                run_id, family_id,
            )

        logger.info(
            "family evaluation finalized: run=%s family=%s miners=%d validators=%d",
            run_id, family_id, len(summaries), len(contributing_validators),
        )

    # ------------------------------------------------------------------
    # Build claim response payloads (include miner list for fan-out)
    # ------------------------------------------------------------------

    def build_claim_items(
        self,
        session: Session,
        *,
        claimed_tasks: list[TaskEvaluation],
        run_id: str,
        family_id: str,
    ) -> list[dict[str, Any]]:
        """Package claimed tasks with task payload + miner fan-out list."""
        snapshot = session.execute(
            select(EpochTargetSnapshot).where(
                EpochTargetSnapshot.run_id == run_id,
                EpochTargetSnapshot.family_id == family_id,
            ).limit(1)
        ).scalar_one_or_none()
        if snapshot is None:
            return []

        miners = []
        for member in (snapshot.members_json or []):
            hk = member.get("hotkey", "")
            if not hk:
                continue
            metadata = member.get("metadata", {}) or {}
            miners.append({
                "hotkey": hk,
                "endpoint": metadata.get("validator_endpoint") or member.get("endpoint", ""),
                "auth_headers": metadata.get("auth_headers", {}),
            })

        bundle = self._owner.run_evaluation_bundle(
            session, run_id=run_id, family_id=family_id,
        )
        tasks_by_id: dict[str, dict[str, Any]] = {}
        if isinstance(bundle, dict):
            for task_def in bundle.get("tasks", []):
                tid = task_def.get("task_id", "")
                if tid:
                    tasks_by_id[tid] = task_def

        judge_config = None
        if isinstance(bundle, dict):
            judge_config = bundle.get("judge_config")

        items = []
        for task in claimed_tasks:
            task_payload = tasks_by_id.get(task.task_id, {})
            if isinstance(task_payload, dict):
                task_payload = _strip_sensitive_task_metadata(task_payload)
            items.append({
                "task_evaluation_id": task.id,
                "run_id": task.run_id,
                "family_id": task.family_id,
                "task_id": task.task_id,
                "task_index": task.task_index,
                "task_payload": task_payload,
                "miners": miners,
                "claim_expires_at": task.claim_expires_at.isoformat() if task.claim_expires_at else "",
                "judge_config": judge_config,
                "rubric_version": snapshot.rubric_version,
                "benchmark_version": snapshot.benchmark_version,
            })
        return items
