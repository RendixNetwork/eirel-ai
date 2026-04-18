from __future__ import annotations

"""Distributed evaluation task management.

Manages the lifecycle of per-miner per-task evaluation assignments.
Validators claim tasks, evaluate them (invoke miner + judge), and
submit results.  Expired claims are automatically released for
reassignment.
"""

import logging
import math
from datetime import timedelta
from typing import Any, TYPE_CHECKING

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from shared.common.models import (
    EpochTargetSnapshot,
    EvaluationRun,
    ManagedDeployment,
    MinerEvaluationSummary,
    MinerEvaluationTask,
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
    """Fix 11: compress scores to reduce cartel advantage.

    1. Power-law compression (score^0.65) reduces marginal benefit of inflation.
    2. Outlier dampening caps scores that are >2 stddev above median.
    """
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
    """Handles claim/submit/status for distributed evaluation tasks."""

    def __init__(self, owner: ManagedOwnerServices) -> None:
        self._owner = owner

    @property
    def settings(self):
        return self._owner.settings

    # ------------------------------------------------------------------
    # Server-side judge proxy (C4)
    # ------------------------------------------------------------------

    def run_judge_for_assignment(
        self,
        session: Session,
        *,
        task_assignment_id: str,
        miner_response: dict[str, Any],
    ) -> dict[str, Any]:
        """Run the judge server-side for a given task assignment.

        Loads the full task definition (with ``expected_output`` rubric) from
        the stored run bundle, builds the judge excerpt, calls the eiretes
        judge sidecar via :class:`JudgeServiceClient`, and returns the
        ``{task_score, judge_output}`` pair without ever exposing the rubric
        to the calling validator.

        Raises ``LookupError`` if the assignment or bundle is missing, and
        any exception from the judge client if the sidecar is unreachable.
        """
        task = session.get(MinerEvaluationTask, task_assignment_id)
        if task is None:
            raise LookupError(f"task_assignment {task_assignment_id!r} not found")
        bundle = self._owner.run_evaluation_bundle(
            session, run_id=task.run_id, family_id=task.family_id,
        )
        if not isinstance(bundle, dict):
            raise LookupError(
                f"no evaluation bundle for run={task.run_id!r} family={task.family_id!r}"
            )
        task_def: dict[str, Any] | None = None
        for candidate in bundle.get("tasks", []) or []:
            if isinstance(candidate, dict) and candidate.get("task_id") == task.task_id:
                task_def = candidate
                break
        if task_def is None:
            raise LookupError(
                f"task_id={task.task_id!r} not found in bundle for run={task.run_id!r}"
            )

        # Build the judge excerpt from the FULL (unstripped) server-side
        # task definition. The validator supplies the raw miner response;
        # the rubric stays inside the owner process.
        from shared.benchmark._judge import build_judge_excerpt
        from shared.core.evaluation_models import BenchmarkTaskRun
        from shared.core.judge_client import JudgeServiceClient

        task_run = BenchmarkTaskRun(
            task_id=task.task_id,
            family_id=task.family_id,
            prompt=str(task_def.get("prompt") or ""),
            expected_output=dict(task_def.get("expected_output") or {}),
            response=dict(miner_response.get("response") or {}),
            status=str(miner_response.get("status") or "completed"),
            error=miner_response.get("error"),
            metadata=dict(miner_response.get("metadata") or {}),
        )
        excerpt = build_judge_excerpt(family_id=task.family_id, run=task_run)

        judge_client = JudgeServiceClient()
        try:
            judge_result = judge_client.judge(
                family_id=task.family_id,
                prompt=task_run.prompt,
                response_excerpt=excerpt,
            )
        finally:
            judge_client.close()
        quality_score = float(getattr(judge_result, "score", 0.0) or 0.0)

        # Layered post-judge scoring: if the miner voluntarily reports
        # response_text + trace + conversation_id in its response dict,
        # run the full general_chat pipeline — trace integrity gate,
        # body-overlap check, honeytoken short-circuit, latency axis,
        # and economic penalty on gate failure. Otherwise, fall through
        # to the legacy judge-direct behavior.
        final_score = quality_score
        conversation_score_payload: dict[str, Any] | None = None
        if task.family_id == "general_chat":
            final_score, conversation_score_payload = self._layered_general_chat_score(
                session=session,
                task=task,
                task_def=task_def,
                raw_miner_response=miner_response,
                quality_score=quality_score,
            )

        result: dict[str, Any] = {
            "task_score": final_score,
            "judge_output": judge_result.model_dump(mode="json"),
        }
        if conversation_score_payload is not None:
            result["conversation_score"] = conversation_score_payload
        return result

    def _layered_general_chat_score(
        self,
        *,
        session: Session,
        task: MinerEvaluationTask,
        task_def: dict[str, Any],
        raw_miner_response: dict[str, Any],
        quality_score: float,
    ) -> tuple[float, dict[str, Any] | None]:
        """Apply the full general_chat scoring pipeline on top of the judge.

        Returns ``(final_score, conversation_score_payload)`` where
        ``conversation_score_payload`` is the serialized ``ConversationScore``
        when the layered pipeline ran, or ``None`` when the miner did not
        supply the optional trace + response_text fields (legacy path).

        Miners opt in by including in their ``response`` dict:
          * ``response_text: str`` — the raw assistant reply
          * ``trace: dict`` — a ``ConversationTrace`` payload
          * optional ``conversation_id: str`` (echoed back in metadata)

        When either field is missing, the function returns the judge
        score unchanged so legacy miners keep working.
        """
        response_dict = raw_miner_response.get("response") or {}
        if not isinstance(response_dict, dict):
            return quality_score, None

        response_text = response_dict.get("response_text")
        trace_payload = response_dict.get("trace")
        if not isinstance(response_text, str) or not isinstance(trace_payload, dict):
            return quality_score, None

        # Rehydrate the trace — tolerate malformed trace by falling back.
        from shared.core.evaluation_models import ConversationTrace, ConversationTurn
        from control_plane.owner_api.evaluation.general_chat_scoring import (
            _PreComputedQualityJudge,
            budget_for_mode,
            score_general_chat_conversation,
        )
        try:
            trace = ConversationTrace.model_validate(trace_payload)
        except Exception as exc:
            logger.warning(
                "rehydrate trace failed for task=%s: %s", task.task_id, exc,
            )
            return quality_score, None

        # Derive the mode budget from the task definition, fall back to
        # instant if the task omits the mode field.
        budget = budget_for_mode(task_def.get("mode"))

        # Build conversation_history from the task prompt. General-chat
        # rubrics are mostly single-turn in production, but we carry the
        # shape so future multi-turn tasks drop in without refactoring.
        conversation_history: list[ConversationTurn] = [
            ConversationTurn(role="user", content=str(task_def.get("prompt") or "")),
        ]

        # Fetch active honeytokens and penalty settings for the run.
        active_honeytokens = self._active_honeytokens_for_run(session, task.run_id)
        penalty_usd = float(
            getattr(self.settings, "trace_gate_penalty_usd", 0.0)
        )

        # The quality score came from the eiretes judge sidecar — we
        # adapt it as a pre-computed score so score_general_chat_conversation
        # doesn't make a second judge call.
        adapter_judge = _PreComputedQualityJudge(quality_score)

        import asyncio

        try:
            conversation_score = asyncio.run(
                score_general_chat_conversation(
                    conversation_history=conversation_history,
                    response=response_text,
                    trace=trace,
                    budget=budget,
                    judge_client=adapter_judge,
                    trace_gate_penalty_usd=penalty_usd,
                    active_honeytokens=active_honeytokens,
                )
            )
        except RuntimeError as exc:
            # We're already inside an event loop (FastAPI). Schedule on a
            # fresh loop via asyncio.new_event_loop to avoid nesting.
            if "asyncio.run() cannot be called" not in str(exc):
                raise
            loop = asyncio.new_event_loop()
            try:
                conversation_score = loop.run_until_complete(
                    score_general_chat_conversation(
                        conversation_history=conversation_history,
                        response=response_text,
                        trace=trace,
                        budget=budget,
                        judge_client=adapter_judge,
                        trace_gate_penalty_usd=penalty_usd,
                        active_honeytokens=active_honeytokens,
                    )
                )
            finally:
                loop.close()

        # If the gate failed and a penalty is configured, debit the run
        # budget against the miner's deployment. Swallow errors — a
        # failed charge must not break the scoring pipeline.
        if conversation_score.trace_gate_penalty_usd > 0.0:
            deployment_id = self._deployment_id_for_miner(
                session, task.miner_hotkey, task.family_id,
            )
            if deployment_id is not None:
                self._owner.scoring.charge_trace_gate_penalty(
                    deployment_id,
                    amount_usd=conversation_score.trace_gate_penalty_usd,
                    reason="trace_gate_fail",
                )

        return conversation_score.total, conversation_score.model_dump(mode="json")

    def _active_honeytokens_for_run(
        self, session: Session, run_id: str
    ) -> list[str]:
        run = session.get(EvaluationRun, run_id)
        if run is None or not isinstance(run.metadata_json, dict):
            return []
        honeytokens = run.metadata_json.get("honeytokens")
        if not isinstance(honeytokens, list):
            return []
        return [str(url) for url in honeytokens if isinstance(url, str)]

    def _deployment_id_for_miner(
        self, session: Session, miner_hotkey: str, family_id: str
    ) -> str | None:
        deployment = session.execute(
            select(ManagedDeployment)
            .where(ManagedDeployment.miner_hotkey == miner_hotkey)
            .where(ManagedDeployment.family_id == family_id)
            .where(ManagedDeployment.status != "retired")
            .order_by(ManagedDeployment.created_at.desc())
            .limit(1)
        ).scalar_one_or_none()
        return deployment.id if deployment is not None else None

    # ------------------------------------------------------------------
    # Task initialization
    # ------------------------------------------------------------------

    def initialize_evaluation_tasks(
        self,
        session: Session,
        *,
        run_id: str,
        family_id: str,
        snapshot: EpochTargetSnapshot,
    ) -> int:
        """Create MinerEvaluationTask rows for every member × task combination.

        Called once when the EpochTargetSnapshot is first created.
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

        # Optional E2E knob: cap the number of tasks created per miner so
        # test runs don't take 20+ minutes. Unset in production.
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
        for member in members:
            miner_hotkey = member.get("hotkey", "")
            if not miner_hotkey:
                continue
            for task_index, task in enumerate(tasks):
                task_id = task.get("task_id", "")
                if not task_id:
                    continue
                existing = session.execute(
                    select(MinerEvaluationTask).where(
                        MinerEvaluationTask.run_id == run_id,
                        MinerEvaluationTask.family_id == family_id,
                        MinerEvaluationTask.miner_hotkey == miner_hotkey,
                        MinerEvaluationTask.task_id == task_id,
                    ).limit(1)
                ).scalar_one_or_none()
                if existing is not None:
                    continue
                session.add(MinerEvaluationTask(
                    run_id=run_id,
                    family_id=family_id,
                    miner_hotkey=miner_hotkey,
                    task_id=task_id,
                    task_index=task_index,
                    status="pending",
                ))
                created += 1

        # Also create MinerEvaluationSummary rows (one per miner)
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
            "initialized %d evaluation tasks for run=%s family=%s miners=%d",
            created, run_id, family_id, len(members),
        )
        return created

    # ------------------------------------------------------------------
    # Claim tasks
    # ------------------------------------------------------------------

    def claim_tasks(
        self,
        session: Session,
        *,
        run_id: str,
        family_id: str,
        validator_hotkey: str,
        batch_size: int = 5,
    ) -> list[MinerEvaluationTask]:
        """Claim up to *batch_size* pending tasks for the validator.

        Tasks are distributed across validators as a simple work queue.
        Multiple validators can work on different tasks for the same miner
        simultaneously — there is no per-miner-per-validator limit.  The
        ``FOR UPDATE SKIP LOCKED`` pattern ensures concurrent claim requests
        from different validators get different tasks without blocking.

        Expired claims (from crashed validators) are released first so
        those tasks become available for reassignment.
        """
        now = utcnow()
        family_id = ensure_active_family_id(family_id)

        # Gate: only release tasks when all deployments are ready
        snapshot = session.execute(
            select(EpochTargetSnapshot)
            .where(EpochTargetSnapshot.run_id == run_id)
            .where(EpochTargetSnapshot.family_id == family_id)
        ).scalar_one_or_none()
        if snapshot is None or snapshot.status != "open":
            return []

        # Step 1: Release expired claims (crashed validators)
        session.execute(
            update(MinerEvaluationTask)
            .where(MinerEvaluationTask.run_id == run_id)
            .where(MinerEvaluationTask.family_id == family_id)
            .where(MinerEvaluationTask.status == "claimed")
            .where(MinerEvaluationTask.claim_expires_at <= now)
            .values(
                status="pending",
                claimed_by_validator=None,
                claimed_at=None,
                claim_expires_at=None,
                updated_at=now,
            )
        )
        session.flush()

        # Step 2: Grab next available pending tasks.
        # ORDER BY task_index distributes tasks in evaluation-bundle order,
        # which naturally interleaves miners when multiple are present.
        # SKIP LOCKED ensures concurrent validators get different rows.
        candidates = list(
            session.execute(
                select(MinerEvaluationTask)
                .where(MinerEvaluationTask.run_id == run_id)
                .where(MinerEvaluationTask.family_id == family_id)
                .where(MinerEvaluationTask.status == "pending")
                .order_by(MinerEvaluationTask.task_index, MinerEvaluationTask.miner_hotkey)
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
    # Submit task result
    # ------------------------------------------------------------------

    def submit_task_result(
        self,
        session: Session,
        *,
        task_assignment_id: str,
        validator_hotkey: str,
        miner_response: dict[str, Any],
        judge_output: dict[str, Any] | None,
        task_score: float | None,
        task_status: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Record the result of an evaluated task.

        Returns a dict with status info (accepted/rejected, completion flags).
        """
        now = utcnow()
        task = session.get(MinerEvaluationTask, task_assignment_id)
        if task is None:
            return {"status": "rejected_not_found", "miner_evaluation_complete": False, "family_evaluation_complete": False, "remaining_task_count": -1}

        # Idempotent: already evaluated by this validator
        if task.status == "evaluated" and task.claimed_by_validator == validator_hotkey:
            remaining = self._remaining_tasks(session, task)
            return {
                "status": "accepted",
                "miner_evaluation_complete": remaining["miner_remaining"] == 0,
                "family_evaluation_complete": remaining["family_remaining"] == 0,
                "remaining_task_count": remaining["family_remaining"],
            }

        # Reject if not claimed by this validator
        if task.claimed_by_validator != validator_hotkey:
            return {"status": "rejected_wrong_validator", "miner_evaluation_complete": False, "family_evaluation_complete": False, "remaining_task_count": -1}

        # Reject if not in claimed state (expired or already evaluated by someone else)
        if task.status != "claimed":
            return {"status": "rejected_not_claimed", "miner_evaluation_complete": False, "family_evaluation_complete": False, "remaining_task_count": -1}

        # The validator submits ``task_score`` = the raw LLM judge quality
        # score. Apply the server-side anti-gaming pipeline (trace integrity
        # gate, honeytoken detection, latency axis, economic penalty) to
        # derive the final stored score. This keeps honeytoken URLs and
        # gating heuristics hidden from validators while still letting
        # validators carry the LLM judge cost.
        quality_score = float(task_score or 0.0)
        final_score = quality_score
        conversation_score_payload: dict[str, Any] | None = None
        if task.family_id == "general_chat":
            bundle = self._owner.run_evaluation_bundle(
                session, run_id=task.run_id, family_id=task.family_id,
            )
            task_def: dict[str, Any] | None = None
            if isinstance(bundle, dict):
                for candidate in bundle.get("tasks", []) or []:
                    if isinstance(candidate, dict) and candidate.get("task_id") == task.task_id:
                        task_def = candidate
                        break
            if task_def is not None:
                final_score, conversation_score_payload = self._layered_general_chat_score(
                    session=session,
                    task=task,
                    task_def=task_def,
                    raw_miner_response=miner_response,
                    quality_score=quality_score,
                )

        enriched_metadata = dict(metadata or {})
        enriched_metadata["quality_score"] = quality_score
        if conversation_score_payload is not None:
            enriched_metadata["conversation_score"] = conversation_score_payload

        # Accept the result
        task.status = "evaluated"
        task.miner_response_json = miner_response
        task.judge_output_json = judge_output
        task.task_score = final_score
        task.task_status = task_status
        task.result_metadata_json = enriched_metadata
        task.evaluated_at = now
        task.updated_at = now
        session.flush()

        remaining = self._remaining_tasks(session, task)
        miner_complete = remaining["miner_remaining"] == 0
        family_complete = remaining["family_remaining"] == 0

        if miner_complete:
            self._on_miner_evaluation_complete(
                session,
                run_id=task.run_id,
                family_id=task.family_id,
                miner_hotkey=task.miner_hotkey,
            )

        if family_complete:
            self._on_family_evaluation_complete(
                session,
                run_id=task.run_id,
                family_id=task.family_id,
            )

        return {
            "status": "accepted",
            "miner_evaluation_complete": miner_complete,
            "family_evaluation_complete": family_complete,
            "remaining_task_count": remaining["family_remaining"],
        }

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
        """Return evaluation progress for a family in a run (single query)."""
        family_id = ensure_active_family_id(family_id)

        # Single GROUP BY query for all counts
        rows = session.execute(
            select(
                MinerEvaluationTask.miner_hotkey,
                MinerEvaluationTask.status,
                func.count(MinerEvaluationTask.id),
            )
            .where(MinerEvaluationTask.run_id == run_id)
            .where(MinerEvaluationTask.family_id == family_id)
            .group_by(MinerEvaluationTask.miner_hotkey, MinerEvaluationTask.status)
        ).fetchall()

        # Aggregate family-level counts
        status_counts: dict[str, int] = {"pending": 0, "claimed": 0, "evaluated": 0}
        # Per-miner breakdown: {hotkey: {status: count}}
        miner_map: dict[str, dict[str, int]] = {}
        for hotkey, status, count in rows:
            status_counts[status] = status_counts.get(status, 0) + count
            if hotkey not in miner_map:
                miner_map[hotkey] = {"pending": 0, "claimed": 0, "evaluated": 0}
            miner_map[hotkey][status] = count

        total = sum(status_counts.values())
        miners = []
        for hotkey in sorted(miner_map):
            m = miner_map[hotkey]
            miner_total = m["pending"] + m["claimed"] + m["evaluated"]
            miners.append({
                "miner_hotkey": hotkey,
                "total": miner_total,
                "evaluated": m["evaluated"],
                "claimed": m["claimed"],
                "pending": m["pending"],
            })

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

    def _remaining_tasks(
        self,
        session: Session,
        task: MinerEvaluationTask,
    ) -> dict[str, int]:
        """Count unevaluated tasks for this miner and for the whole family in one query."""
        rows = session.execute(
            select(
                MinerEvaluationTask.miner_hotkey,
                func.count(MinerEvaluationTask.id),
            )
            .where(MinerEvaluationTask.run_id == task.run_id)
            .where(MinerEvaluationTask.family_id == task.family_id)
            .where(MinerEvaluationTask.status != "evaluated")
            .group_by(MinerEvaluationTask.miner_hotkey)
        ).fetchall()
        family_remaining = sum(count for _, count in rows)
        miner_remaining = next(
            (count for hotkey, count in rows if hotkey == task.miner_hotkey), 0
        )
        return {"miner_remaining": miner_remaining, "family_remaining": family_remaining}

    def _on_miner_evaluation_complete(
        self,
        session: Session,
        *,
        run_id: str,
        family_id: str,
        miner_hotkey: str,
    ) -> None:
        """Called when all tasks for a miner are evaluated. Computes the full score."""
        from shared.benchmark._orchestration import compute_miner_score_from_results

        summary = session.execute(
            select(MinerEvaluationSummary).where(
                MinerEvaluationSummary.run_id == run_id,
                MinerEvaluationSummary.family_id == family_id,
                MinerEvaluationSummary.miner_hotkey == miner_hotkey,
            ).limit(1)
        ).scalar_one_or_none()
        if summary is None:
            return

        evaluated_tasks = list(session.execute(
            select(MinerEvaluationTask)
            .where(MinerEvaluationTask.run_id == run_id)
            .where(MinerEvaluationTask.family_id == family_id)
            .where(MinerEvaluationTask.miner_hotkey == miner_hotkey)
            .where(MinerEvaluationTask.status == "evaluated")
            .order_by(MinerEvaluationTask.task_index)
        ).scalars())

        completed = sum(1 for t in evaluated_tasks if t.task_status == "completed")
        failed = sum(1 for t in evaluated_tasks if t.task_status != "completed")

        summary.completed_tasks = completed
        summary.failed_tasks = failed
        summary.status = "scoring"
        summary.updated_at = utcnow()
        session.flush()

        # Collect judge outputs aligned by index; miner responses are
        # consumed below to synthesize ConversationScore payloads.
        judge_outputs = [t.judge_output_json or {} for t in evaluated_tasks]

        # Get the snapshot's benchmark_version
        snapshot = session.execute(
            select(EpochTargetSnapshot).where(
                EpochTargetSnapshot.run_id == run_id,
                EpochTargetSnapshot.family_id == family_id,
            ).limit(1)
        ).scalar_one_or_none()
        benchmark_version = snapshot.benchmark_version if snapshot else "family_benchmark_v2"

        # Translate each task into a ConversationScore-shaped dict.  The
        # family scorer requires fully-populated ConversationScore payloads;
        # raw miner_response_json lacks the quality/latency/cost/trace_gate
        # fields and fails Pydantic validation.  For miners that opt into
        # the layered path, task.task_score already carries the layered
        # conversation_score.total — the synthesized wrapper below reuses
        # it as ``quality`` so the dimensions stay consistent.
        #
        # The task's ``mode`` ("instant" or "thinking") is read from the
        # bundle per task; it drives the 60/40 instant/thinking blend in
        # ``aggregate_miner_score``.  Hardcoding "instant" here sends every
        # thinking task into the instant bucket and leaves ``thinking_mean``
        # permanently at 0, which silently caps raw_score at 0.6x the real
        # average.
        from shared.scoring.families._judge_to_conversation_score import (
            build_conversation_score_from_judge,
        )
        bundle = self._owner.run_evaluation_bundle(
            session, run_id=run_id, family_id=family_id,
        )
        task_mode_by_id: dict[str, str] = {}
        if isinstance(bundle, dict):
            for task_def in bundle.get("tasks", []):
                tid = task_def.get("task_id")
                raw_mode = task_def.get("mode")
                if isinstance(tid, str) and raw_mode in ("instant", "thinking"):
                    task_mode_by_id[tid] = raw_mode
        conversation_payloads = [
            build_conversation_score_from_judge(
                task_score=float(t.task_score or 0.0),
                judge_output=t.judge_output_json or {},
                miner_response=t.miner_response_json or {},
                mode=task_mode_by_id.get(t.task_id, "instant"),
            ).model_dump(mode="json")
            for t in evaluated_tasks
        ]

        try:
            rollout_metadata = compute_miner_score_from_results(
                family_id=family_id,
                benchmark_version=benchmark_version,
                miner_hotkey=miner_hotkey,
                task_results=conversation_payloads,
                judge_outputs=judge_outputs,
            )
            evaluation_breakdown = rollout_metadata.get("evaluation_breakdown", {})
            summary.family_capability_score = float(evaluation_breakdown.get("family_capability_score", 0.0) or 0.0)
            summary.robustness_score = float(evaluation_breakdown.get("robustness_score", 0.0) or 0.0)
            summary.anti_gaming_score = float(evaluation_breakdown.get("anti_gaming_score", 0.0) or 0.0)
            summary.official_family_score = float(evaluation_breakdown.get("official_family_score", 0.0) or 0.0)
            # ``compute_miner_score_from_results`` emits the nested form
            # ``{"protocol_gate": {"passed": bool, "reason": str}}``; older
            # callers wrote a flat ``protocol_gate_passed`` key.  Accept
            # both so either shape produces a valid gate flag.
            gate_flat = rollout_metadata.get("protocol_gate_passed")
            gate_nested = (rollout_metadata.get("protocol_gate") or {}).get("passed")
            summary.protocol_gate_passed = bool(
                gate_flat if gate_flat is not None else gate_nested
            )
            summary.rollout_metadata_json = rollout_metadata
            summary.status = "scored"
        except Exception as exc:
            logger.exception(
                "failed to compute miner score: run=%s family=%s miner=%s",
                run_id, family_id, miner_hotkey[:16],
            )
            # Fallback: simple average from task scores
            scores = [t.task_score for t in evaluated_tasks if t.task_score is not None]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            summary.family_capability_score = avg_score
            summary.official_family_score = avg_score
            summary.rollout_metadata_json = {"error": str(exc), "fallback": "average"}
            summary.status = "scored"

        summary.updated_at = utcnow()
        session.flush()

        logger.info(
            "miner evaluation complete: run=%s family=%s miner=%s completed=%d failed=%d "
            "capability=%.4f robustness=%.4f anti_gaming=%.4f official=%.4f gate=%s",
            run_id, family_id, miner_hotkey[:16],
            completed, failed,
            summary.family_capability_score or 0.0,
            summary.robustness_score or 0.0,
            summary.anti_gaming_score or 0.0,
            summary.official_family_score or 0.0,
            summary.protocol_gate_passed,
        )

    def _on_family_evaluation_complete(
        self,
        session: Session,
        *,
        run_id: str,
        family_id: str,
    ) -> None:
        """Called when all tasks for all miners in a family are evaluated.

        Builds the AggregateFamilyScoreSnapshot and DeploymentScoreRecord
        entries from the individual MinerEvaluationSummary scores.
        """
        from shared.common.models import (
            AggregateFamilyScoreSnapshot,
            DeploymentScoreRecord,
            EpochTargetSnapshot,
            ManagedDeployment,
            ManagedMinerSubmission,
        )
        from sqlalchemy import delete

        now = utcnow()
        summaries = list(session.execute(
            select(MinerEvaluationSummary)
            .where(MinerEvaluationSummary.run_id == run_id)
            .where(MinerEvaluationSummary.family_id == family_id)
            .where(MinerEvaluationSummary.status == "scored")
        ).scalars())

        if not summaries:
            logger.warning("no scored summaries for run=%s family=%s", run_id, family_id)
            return

        # Build miner_scores and normalized_weights
        miner_scores: dict[str, float] = {}
        for s in summaries:
            miner_scores[s.miner_hotkey] = s.official_family_score or 0.0

        # Apply protocol gate: gate_passed → full score, else partial/zero
        # Fix 12: raise partial credit threshold from 0.50 → 0.75 with quadratic discount
        _PARTIAL_CREDIT_THRESHOLD = 0.75
        effective_scores: dict[str, float] = {}
        for s in summaries:
            gate_passed = s.protocol_gate_passed
            official = s.official_family_score or 0.0
            rollout = s.rollout_metadata_json or {}
            contract_pass_rate = float(rollout.get("protocol_contract_pass_rate", 0.0) or 0.0)
            if gate_passed:
                effective_scores[s.miner_hotkey] = official
            elif contract_pass_rate >= _PARTIAL_CREDIT_THRESHOLD:
                _discount = (
                    (contract_pass_rate - _PARTIAL_CREDIT_THRESHOLD)
                    / (1.0 - _PARTIAL_CREDIT_THRESHOLD)
                ) ** 2
                effective_scores[s.miner_hotkey] = official * _discount * 0.5
            else:
                effective_scores[s.miner_hotkey] = 0.0

        # Fix 11: power-law score compression + outlier dampening to resist cartel gaming
        effective_scores = _compress_scores(effective_scores)
        total = sum(max(0.0, v) for v in effective_scores.values())
        if total > 0:
            normalized_weights = {hk: max(0.0, v) / total for hk, v in sorted(effective_scores.items())}
        else:
            normalized_weights = {hk: 0.0 for hk in sorted(effective_scores)}

        # Collect validator hotkeys that contributed
        contributing_validators = set()
        for task in session.execute(
            select(MinerEvaluationTask.claimed_by_validator)
            .where(MinerEvaluationTask.run_id == run_id)
            .where(MinerEvaluationTask.family_id == family_id)
            .where(MinerEvaluationTask.status == "evaluated")
            .where(MinerEvaluationTask.claimed_by_validator.isnot(None))
            .distinct()
        ).fetchall():
            contributing_validators.add(task[0])

        validator_weights = {v_hotkey: 1 for v_hotkey in contributing_validators}

        # Build score breakdowns
        miner_score_breakdowns = {}
        for s in summaries:
            miner_score_breakdowns[s.miner_hotkey] = {
                "family_capability_score": s.family_capability_score or 0.0,
                "robustness_score": s.robustness_score or 0.0,
                "anti_gaming_score": s.anti_gaming_score or 0.0,
                "official_family_score": s.official_family_score or 0.0,
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
            "evaluation_plane": "family_protocol",
            "miner_scores": miner_scores,
            "normalized_weights": normalized_weights,
            "query_volume_share": 0.0,
            "rubric_version": snapshot.rubric_version if snapshot else "family_rubric_v2",
            "miner_query_volume_shares": {},
            "miner_score_breakdowns": miner_score_breakdowns,
            "miner_robustness_scores": {s.miner_hotkey: s.robustness_score or 0.0 for s in summaries},
            "miner_anti_gaming_flags": {},
            "task_count": summaries[0].total_tasks if summaries else 0,
            "judge_model": snapshot.judge_model if snapshot else "local-rubric-judge",
            "evaluation_timestamp": now.isoformat() + "Z",
        }

        # Delete existing aggregate for this run+family (idempotent)
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
            consensus_method="distributed_task_evaluation",
            status="aggregated",
            activated_at=now,
        )
        session.add(aggregate)
        session.flush()

        # Delete old DeploymentScoreRecords and create new ones
        session.execute(
            delete(DeploymentScoreRecord)
            .where(DeploymentScoreRecord.run_id == run_id)
            .where(DeploymentScoreRecord.family_id == family_id)
        )
        session.flush()

        for s in summaries:
            # Find the deployment for this miner
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

            submission = session.execute(
                select(ManagedMinerSubmission)
                .where(ManagedMinerSubmission.id == deployment.submission_id)
                .limit(1)
            ).scalar_one_or_none()

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
                    "family_capability_score": s.family_capability_score,
                    "robustness_score": s.robustness_score,
                    "anti_gaming_score": s.anti_gaming_score,
                    "official_family_score": s.official_family_score,
                    "protocol_gate_passed": s.protocol_gate_passed,
                    "completed_tasks": s.completed_tasks,
                    "failed_tasks": s.failed_tasks,
                    "total_tasks": s.total_tasks,
                    "evaluation_method": "distributed_task_evaluation",
                },
            )
            self._owner.scoring.populate_cost_columns(
                score_record,
                deployment.id,
                self._owner.settings.run_budget_usd,
            )
            session.add(score_record)

        # Mark snapshot as scored
        if snapshot:
            snapshot.status = "scored"
            snapshot.scored_at = now

        session.flush()

        # Trigger rebalance
        try:
            self._owner.rebalance_family(session, family_id=family_id)
        except Exception:
            logger.exception("rebalance_family failed after distributed evaluation: run=%s family=%s", run_id, family_id)

        logger.info(
            "family evaluation finalized: run=%s family=%s miners=%d validators=%d",
            run_id, family_id, len(summaries), len(contributing_validators),
        )

    # ------------------------------------------------------------------
    # Build claim response payloads
    # ------------------------------------------------------------------

    def build_claim_items(
        self,
        session: Session,
        *,
        claimed_tasks: list[MinerEvaluationTask],
        run_id: str,
        family_id: str,
    ) -> list[dict[str, Any]]:
        """Build response payloads for claimed tasks with task definitions and miner endpoints."""
        snapshot = session.execute(
            select(EpochTargetSnapshot).where(
                EpochTargetSnapshot.run_id == run_id,
                EpochTargetSnapshot.family_id == family_id,
            ).limit(1)
        ).scalar_one_or_none()
        if snapshot is None:
            return []

        # Build lookup: miner_hotkey -> member dict
        member_by_hotkey: dict[str, dict[str, Any]] = {}
        for member in (snapshot.members_json or []):
            hk = member.get("hotkey", "")
            if hk:
                member_by_hotkey[hk] = member

        # Load evaluation bundle for task definitions
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
            member = member_by_hotkey.get(task.miner_hotkey, {})
            task_payload = tasks_by_id.get(task.task_id, {})
            # Anti-gaming (C2): strip hidden_fixture/visibility/seed_id/topic
            # from the per-task metadata before it reaches the validator.
            # C4 will additionally strip expected_output + judge_rubric.
            if isinstance(task_payload, dict):
                task_payload = _strip_sensitive_task_metadata(task_payload)
            metadata = member.get("metadata", {})
            items.append({
                "task_assignment_id": task.id,
                "run_id": task.run_id,
                "family_id": task.family_id,
                "miner_hotkey": task.miner_hotkey,
                "task_id": task.task_id,
                "task_index": task.task_index,
                "task_payload": task_payload,
                "miner_endpoint": metadata.get("validator_endpoint", member.get("endpoint", "")),
                "miner_auth_headers": metadata.get("auth_headers", {}),
                "claim_expires_at": task.claim_expires_at.isoformat() if task.claim_expires_at else "",
                "judge_config": judge_config,
                "rubric_version": snapshot.rubric_version,
                "benchmark_version": snapshot.benchmark_version,
            })
        return items
