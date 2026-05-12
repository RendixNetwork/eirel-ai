from __future__ import annotations

"""Submission lifecycle, deployment payloads, and candidate registry management.

Extracted from ``ManagedOwnerServices`` (Item 15) to reduce the size of
the god-object.  Each public method here has a thin delegation wrapper
in ``ManagedOwnerServices`` for backward compatibility.
"""

import hashlib
import logging
from datetime import datetime
from typing import Any, TYPE_CHECKING

logger = logging.getLogger(__name__)

from sqlalchemy import select
from sqlalchemy.orm import Session

from shared.common.models import (
    DeploymentHealthEvent,
    DeploymentScoreRecord,
    ManagedDeployment,
    ManagedMinerSubmission,
    SubmissionArtifact,
)
from shared.common.manifest import extract_manifest_from_archive
from control_plane.owner_api._constants import PRODUCTION_FAMILIES
from control_plane.owner_api._helpers import (
    _evaluation_policy_payload,
    _score_record_selection_score,
    family_for_manifest,
    is_supported_family,
    utcnow,
)
from control_plane.owner_api.operations._helpers import latency_score_from_ms
from shared.contracts.models import MinerRegistryEntry

if TYPE_CHECKING:
    from control_plane.owner_api.managed import ManagedOwnerServices


class SubmissionManager:
    """Handles submission lifecycle, deployment payloads, and candidate registry."""

    def __init__(self, owner: ManagedOwnerServices) -> None:
        self._owner = owner

    @property
    def db(self):
        return self._owner.db

    @property
    def settings(self):
        return self._owner.settings

    def record_health_event(
        self,
        session: Session,
        *,
        deployment: ManagedDeployment,
        status: str,
        reason: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        session.add(
            DeploymentHealthEvent(
                deployment_id=deployment.id,
                family_id=deployment.family_id,
                status=status,
                reason=reason,
                details_json=details or {},
            )
        )

    def _snapshot_member_from_deployment(
        self,
        *,
        base_url: str,
        run_id: str,
        deployment: ManagedDeployment,
        quality_score: float,
    ) -> dict[str, Any]:
        del base_url
        validator_endpoint = (
            f"{self.settings.owner_api_internal_url.rstrip('/')}"
            f"/v1/internal/runs/{run_id}/deployments/{deployment.id}"
        )
        return MinerRegistryEntry(
            hotkey=deployment.miner_hotkey,
            family_id=deployment.family_id,
            endpoint=validator_endpoint,
            latency_score=latency_score_from_ms(deployment.latency_ms_p50),
            quality_score=quality_score,
            metadata={
                "deployment_id": deployment.id,
                "submission_id": deployment.submission_id,
                "deployment_revision": deployment.deployment_revision,
                "health_status": deployment.health_status,
                "consumer_endpoint": deployment.endpoint,
                "validator_endpoint": validator_endpoint,
                "validator_health_path": self._owner.validator_health_path(
                    run_id=run_id,
                    deployment_id=deployment.id,
                ),
                "validator_infer_path": self._owner.validator_infer_path(
                    run_id=run_id,
                    deployment_id=deployment.id,
                ),
                "benchmark_endpoint": validator_endpoint,
                "auth_headers": (
                    {"Authorization": f"Bearer {self.settings.internal_service_token}"}
                    if self.settings.internal_service_token
                    else {}
                ),
            },
        ).model_dump(mode="json")

    def latest_submission_for_hotkey(
        self, session: Session, *, miner_hotkey: str
    ) -> ManagedMinerSubmission | None:
        return session.execute(
            select(ManagedMinerSubmission)
            .where(ManagedMinerSubmission.miner_hotkey == miner_hotkey)
            .order_by(ManagedMinerSubmission.submission_seq.desc())
            .limit(1)
            .with_for_update()
        ).scalar_one_or_none()

    def _require_registered_hotkey(self, session: Session, miner_hotkey: str) -> None:
        # Presence in registered_neurons = currently registered on the subnet.
        # Deregistered hotkeys are removed from the table by the metagraph sync.
        from shared.common.models import RegisteredNeuron
        if session.get(RegisteredNeuron, miner_hotkey) is None:
            raise ValueError(
                f"hotkey {miner_hotkey[:16]}... is not registered on the metagraph"
            )

    def create_submission(
        self,
        session: Session,
        *,
        miner_hotkey: str,
        submission_block: int,
        archive_bytes: bytes,
        base_url: str,
        extrinsic_hash: str | None = None,
        block_hash: str | None = None,
    ) -> tuple[ManagedMinerSubmission, ManagedDeployment]:
        self._require_registered_hotkey(session, miner_hotkey)

        manifest = extract_manifest_from_archive(archive_bytes)
        family_id = family_for_manifest(manifest)
        if not is_supported_family(family_id):
            raise ValueError(f"family {family_id} is deferred in managed v1")
        if manifest.inference.requires_subnet_provider_proxy and (
            not self.settings.provider_proxy_url or not self.settings.provider_proxy_token
        ):
            raise ValueError("provider proxy is required but not configured on the owner side")

        # Fee verification: if treasury address is configured, require proof-of-payment.
        fee_verifier = getattr(self._owner, "_fee_verifier", None)
        if fee_verifier is not None:
            if not extrinsic_hash:
                raise ValueError(
                    "submission fee payment required: provide extrinsic_hash of "
                    f"{self.settings.submission_fee_tao} TAO transfer to treasury"
                )
            existing = session.execute(
                select(ManagedMinerSubmission)
                .where(ManagedMinerSubmission.extrinsic_hash == extrinsic_hash)
                .limit(1)
            ).scalar_one_or_none()
            if existing is not None:
                raise ValueError("extrinsic_hash already used for a prior submission")
            result = fee_verifier.verify_payment(extrinsic_hash, miner_hotkey, block_hash=block_hash)
            if not result.valid:
                raise ValueError(f"fee verification failed: {result.reason}")

        requested_cpu_millis, requested_memory_bytes = self._owner.normalize_manifest_resources(manifest)
        evaluation_policy = _evaluation_policy_payload(family_id)

        previous = self.latest_submission_for_hotkey(session, miner_hotkey=miner_hotkey)
        submission_seq = 1 if previous is None else previous.submission_seq + 1

        # Immediately retire previous deployment on resubmit (cross-family aware).
        retired_deployment_id: str | None = None
        retired_family_id: str | None = None
        if previous is not None:
            old_deployment = self._owner.get_deployment_for_submission(session, previous.id)
            if old_deployment is not None and old_deployment.status != "retired":
                retired_deployment_id = old_deployment.id
                retired_family_id = old_deployment.family_id
                old_deployment.is_active = False
                old_deployment.active_set_rank = None
                old_deployment.status = "retired"
                old_deployment.pending_runtime_stop = True
                old_deployment.health_status = "retired"
                old_deployment.retired_at = utcnow()
                old_deployment.updated_at = utcnow()
                old_deployment.health_details_json = {
                    **old_deployment.health_details_json,
                    "stop_reason": "resubmit",
                }
                previous.status = "retired"
                previous.updated_at = utcnow()
                self._owner.record_health_event(
                    session,
                    deployment=old_deployment,
                    status="retired",
                    reason="resubmit",
                    details={"replacement_family_id": family_id},
                )

        target_run = self._owner.submission_target_run(session)
        archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
        artifact = SubmissionArtifact(
            archive_bytes=archive_bytes,
            sha256=archive_sha256,
            size_bytes=len(archive_bytes),
            manifest_json=manifest.model_dump(mode="json"),
        )
        session.add(artifact)
        session.flush()

        submission = ManagedMinerSubmission(
            miner_hotkey=miner_hotkey,
            submission_seq=submission_seq,
            family_id=family_id,
            status="received",
            artifact_id=artifact.id,
            manifest_json=manifest.model_dump(mode="json"),
            archive_sha256=archive_sha256,
            submission_block=submission_block,
            introduced_run_id=target_run.id,
            extrinsic_hash=extrinsic_hash,
        )
        session.add(submission)
        session.flush()

        revision = hashlib.sha256(
            f"{archive_sha256}:{miner_hotkey}:{submission_seq}".encode()
        ).hexdigest()
        deployment = ManagedDeployment(
            submission_id=submission.id,
            miner_hotkey=miner_hotkey,
            family_id=family_id,
            deployment_revision=revision,
            image_ref=f"managed://{family_id}/{revision}",
            endpoint="",
            status="queued",
            health_status="queued",
            health_details_json={
                "build": "queued",
                "deploy": "queued",
                "manifest_agent": manifest.agent.name,
            },
            requested_cpu_millis=requested_cpu_millis,
            requested_memory_bytes=requested_memory_bytes,
            placement_status="queued",
            latency_ms_p50=0,  # unmeasured until runtime metrics land
            benchmark_version=str(evaluation_policy["benchmark_version"]),
            rubric_version=str(evaluation_policy["rubric_version"]),
            judge_model=self._owner.judge_model,
        )
        session.add(deployment)
        session.flush()
        deployment.endpoint = f"{base_url.rstrip('/')}/runtime/{deployment.id}"
        session.commit()
        if retired_family_id:
            self.rebalance_family(session, family_id=retired_family_id)
        session.refresh(submission)
        session.refresh(deployment)
        logger.info(
            "submission created: hotkey=%s family=%s submission_id=%s deployment_id=%s",
            miner_hotkey[:16], family_id, submission.id, deployment.id,
        )
        return submission, deployment

    def rebalance_family(self, session: Session, *, family_id: str) -> None:
        if self._owner.is_rollout_frozen(session, family_id=family_id):
            return
        deployments = list(
            session.execute(
                select(ManagedDeployment)
                .where(ManagedDeployment.family_id == family_id)
                .where(ManagedDeployment.status.notin_(("retired", "queued")))
            ).scalars()
        )
        score_rows = list(
            session.execute(
                select(DeploymentScoreRecord).where(DeploymentScoreRecord.family_id == family_id)
            ).scalars()
        )
        latest_scores: dict[str, tuple[float, datetime]] = {}
        for row in score_rows:
            current = latest_scores.get(row.deployment_id)
            candidate = (_score_record_selection_score(row), row.created_at)
            if current is None or candidate[1] > current[1]:
                latest_scores[row.deployment_id] = candidate

        eligible_deployment_ids = {
            row.deployment_id
            for row in score_rows
            if row.is_eligible
        }
        candidate_ranked = sorted(
            [
                deployment
                for deployment in deployments
                if deployment.health_status == "healthy"
                and deployment.status != "retired"
                and deployment.id in eligible_deployment_ids
            ],
            key=lambda item: (
                latest_scores.get(item.id, (0.0, utcnow()))[0],
                item.created_at,
            ),
            reverse=True,
        )
        ranked: list[ManagedDeployment] = []
        seen_hotkeys: set[str] = set()
        for deployment in candidate_ranked:
            if deployment.miner_hotkey in seen_hotkeys:
                continue
            seen_hotkeys.add(deployment.miner_hotkey)
            ranked.append(deployment)
        active_ids = {item.id for item in ranked[: self._owner.top_k_per_group]}
        for index, deployment in enumerate(ranked, start=1):
            deployment.is_active = deployment.id in active_ids
            deployment.active_set_rank = index if deployment.is_active else None
            deployment.status = "active" if deployment.is_active else "standby_cold"
            deployment.updated_at = utcnow()
        for deployment in deployments:
            if deployment.id not in active_ids and deployment not in ranked:
                deployment.is_active = False
                deployment.active_set_rank = None
                if deployment.health_status == "healthy" and deployment.id in eligible_deployment_ids:
                    deployment.status = "standby_cold"
                deployment.updated_at = utcnow()
        session.commit()

    def submission_payload(
        self,
        submission: ManagedMinerSubmission,
        deployment: ManagedDeployment | None,
        *,
        latest_scorecard_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "id": submission.id,
            "miner_hotkey": submission.miner_hotkey,
            "submission_seq": submission.submission_seq,
            "family_id": submission.family_id,
            "introduced_run_id": submission.introduced_run_id,
            "status": submission.status,
            "submission_block": submission.submission_block,
            "manifest": submission.manifest_json,
            "archive_sha256": submission.archive_sha256,
            "created_at": submission.created_at.isoformat(),
            "updated_at": submission.updated_at.isoformat(),
            "deployment": self.deployment_payload(deployment) if deployment else None,
            "latest_scorecard_summary": latest_scorecard_summary,
        }

    def deployment_payload(self, deployment: ManagedDeployment | None) -> dict[str, Any] | None:
        if deployment is None:
            return None
        return {
            "id": deployment.id,
            "submission_id": deployment.submission_id,
            "miner_hotkey": deployment.miner_hotkey,
            "family_id": deployment.family_id,
            "deployment_revision": deployment.deployment_revision,
            "image_ref": deployment.image_ref,
            "endpoint": deployment.endpoint,
            "status": deployment.status,
            "health_status": deployment.health_status,
            "health_details": deployment.health_details_json,
            "requested_cpu_millis": deployment.requested_cpu_millis,
            "requested_memory_bytes": deployment.requested_memory_bytes,
            "placement_status": deployment.placement_status,
            "assigned_node_name": deployment.assigned_node_name,
            "assigned_cpu_millis": deployment.assigned_cpu_millis,
            "assigned_memory_bytes": deployment.assigned_memory_bytes,
            "placement_error": deployment.placement_error_text,
            "latency_ms_p50": deployment.latency_ms_p50,
            "is_active": deployment.is_active,
            "is_eligible": deployment.status in {"eligible", "active", "standby_cold", "draining"},
            "is_draining": deployment.status == "draining",
            "active_set_rank": deployment.active_set_rank,
            "benchmark_version": deployment.benchmark_version,
            "rubric_version": deployment.rubric_version,
            "judge_model": deployment.judge_model,
            "created_at": deployment.created_at.isoformat(),
            "updated_at": deployment.updated_at.isoformat(),
        }

    def get_deployment_for_submission(
        self, session: Session, submission_id: str
    ) -> ManagedDeployment | None:
        return session.execute(
            select(ManagedDeployment)
            .where(ManagedDeployment.submission_id == submission_id)
            .order_by(ManagedDeployment.created_at.desc())
            .limit(1)
        ).scalar_one_or_none()

    def latest_submission_score_record(
        self,
        session: Session,
        *,
        submission_id: str,
    ) -> DeploymentScoreRecord | None:
        return session.execute(
            select(DeploymentScoreRecord)
            .where(DeploymentScoreRecord.submission_id == submission_id)
            .order_by(DeploymentScoreRecord.created_at.desc())
            .limit(1)
        ).scalar_one_or_none()

    def _submission_task_runs_and_judges(
        self,
        session: Session,
        *,
        submission: ManagedMinerSubmission,
        deployment: ManagedDeployment | None,
        score_record: DeploymentScoreRecord | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        from shared.common.models import TaskMinerResult

        if score_record is None:
            return [], []
        miner_hotkey = (
            deployment.miner_hotkey
            if deployment is not None
            else submission.miner_hotkey
        )
        tasks = list(
            session.execute(
                select(TaskMinerResult)
                .where(TaskMinerResult.run_id == score_record.run_id)
                .where(TaskMinerResult.family_id == score_record.family_id)
                .where(TaskMinerResult.miner_hotkey == miner_hotkey)
            ).scalars()
        )
        task_runs = [
            {
                "task_id": task.task_id,
                "status": "completed" if task.agreement_verdict != "error" else "failed",
                "score": task.agreement_score,
                "error": None,
            }
            for task in tasks
        ]
        judge_outputs = [
            dict(task.judge_output_json or {})
            for task in tasks
            if task.judge_output_json
        ]
        return task_runs, judge_outputs

    def _classify_submission_failures(
        self,
        *,
        task_runs: list[dict[str, Any]],
    ) -> dict[str, int]:
        infra_failed = 0
        task_failed = 0
        infra_tokens = (
            "connection refused",
            "timed out",
            "timeout",
            "temporary failure in name resolution",
            "name or service not known",
            "connection reset",
            "bad gateway",
            "service unavailable",
            "dns",
            "nodename nor servname",
        )
        for run in task_runs:
            if str(run.get("status") or "").lower() == "completed":
                continue
            error_text = str(run.get("error") or "").lower()
            if any(token in error_text for token in infra_tokens):
                infra_failed += 1
            else:
                task_failed += 1
        return {
            "infra_failed_results": infra_failed,
            "task_failed_results": task_failed,
        }

    def submission_progress_payload(
        self,
        session: Session,
        *,
        submission: ManagedMinerSubmission,
    ) -> dict[str, Any]:
        deployment = self.get_deployment_for_submission(session, submission.id)
        score_record = self.latest_submission_score_record(
            session,
            submission_id=submission.id,
        )
        task_runs, judge_outputs = self._submission_task_runs_and_judges(
            session,
            submission=submission,
            deployment=deployment,
            score_record=score_record,
        )
        judge_counts_by_task: dict[str, int] = {}
        for output in judge_outputs:
            task_id = str(output.get("task_id") or "").strip()
            if task_id:
                judge_counts_by_task[task_id] = judge_counts_by_task.get(task_id, 0) + 1
        tasks = [
            {
                "task_id": str(run.get("task_id") or ""),
                "status": str(run.get("status") or "unknown"),
                "error": run.get("error"),
                "score_bearing": True,
                "judge_backed_result_count": judge_counts_by_task.get(
                    str(run.get("task_id") or ""),
                    0,
                ),
            }
            for run in task_runs
        ]
        failure_counts = self._classify_submission_failures(task_runs=task_runs)
        planned = len(tasks) if tasks else 0
        completed = sum(1 for item in tasks if item["status"] == "completed")
        failed = sum(1 for item in tasks if item["status"] == "failed")
        running = max(0, planned - completed - failed)
        return {
            "submission_id": submission.id,
            "submission_status": submission.status,
            "family_id": submission.family_id,
            "deployment": self.deployment_payload(deployment),
            "task_events": tasks,
            "tasks": tasks,
            "validator_reports": [],
            "failure_classification_counts": failure_counts,
            "capability_counts": {
                submission.family_id: {
                    "planned": planned,
                    "running": running,
                    "completed": completed,
                    "failed": failed,
                }
            },
        }

    def submission_canonical_payload(
        self,
        session: Session,
        *,
        submission: ManagedMinerSubmission,
    ) -> dict[str, Any]:
        deployment = self.get_deployment_for_submission(session, submission.id)
        score_record = self.latest_submission_score_record(
            session,
            submission_id=submission.id,
        )
        if score_record is None:
            pending_summary = {
                "family_id": submission.family_id,
                "deployment_id": deployment.id if deployment is not None else None,
                "deployment_health_status": (
                    deployment.health_status if deployment is not None else None
                ),
            }
            if deployment is not None and deployment.health_status in {"unhealthy", "retired"}:
                return {
                    "submission_id": submission.id,
                    "status": "failed",
                    "validator_count": 0,
                    "overall_score": None,
                    "aggregation_method": None,
                    "summary": pending_summary,
                }
            return {
                "submission_id": submission.id,
                "status": "pending",
                "validator_count": 0,
                "overall_score": None,
                "aggregation_method": None,
                "summary": pending_summary,
            }
        task_runs, _judge_outputs = self._submission_task_runs_and_judges(
            session,
            submission=submission,
            deployment=deployment,
            score_record=score_record,
        )
        failure_counts = self._classify_submission_failures(task_runs=task_runs)
        aggregate = self._owner.aggregate_snapshot_for_family(
            session,
            run_id=score_record.run_id,
            family_id=score_record.family_id,
        )
        has_failed_tasks = any(
            str(item.get("status") or "").lower() == "failed"
            for item in task_runs
        )
        status = "failed" if has_failed_tasks or failure_counts["task_failed_results"] > 0 else "finalized"
        metadata = dict(score_record.metadata_json or {})
        return {
            "submission_id": submission.id,
            "status": status,
            "validator_count": (
                int(aggregate.validator_count)
                if aggregate is not None
                else 0
            ),
            "overall_score": float(score_record.raw_score),
            "aggregation_method": (
                aggregate.consensus_method if aggregate is not None else None
            ),
            "summary": {
                "run_id": score_record.run_id,
                "family_id": score_record.family_id,
                "deployment_id": score_record.deployment_id,
                "deployment_revision": score_record.deployment_revision,
                "official_family_score": float(
                    metadata.get("official_family_score", score_record.raw_score) or 0.0
                ),
                "qualifies_for_incentives": bool(
                    metadata.get("qualifies_for_incentives", True)
                ),
                "failure_classification_counts": failure_counts,
                "family_diagnostics": dict(metadata.get("family_diagnostics", {}) or {}),
                "failure_taxonomy": dict(metadata.get("failure_taxonomy", {}) or {}),
            },
        }

    def get_candidate_registry(self, session: Session) -> dict[str, list[MinerRegistryEntry]]:
        deployments = list(
            session.execute(
                select(ManagedDeployment)
                .where(ManagedDeployment.health_status == "healthy")
                .where(ManagedDeployment.status != "retired")
                .where(ManagedDeployment.family_id.in_(PRODUCTION_FAMILIES))
            ).scalars()
        )
        deployments.sort(
            key=lambda item: (
                item.family_id,
                0 if item.is_active else 1,
                item.active_set_rank if item.active_set_rank is not None else 999_999,
                item.created_at,
                item.id,
            )
        )
        grouped: dict[str, list[MinerRegistryEntry]] = {family_id: [] for family_id in PRODUCTION_FAMILIES}
        for deployment in deployments:
            grouped.setdefault(deployment.family_id, []).append(
                MinerRegistryEntry(
                    hotkey=deployment.miner_hotkey,
                    family_id=deployment.family_id,
                    endpoint=self._candidate_runtime_endpoint(deployment_id=deployment.id),
                    latency_score=latency_score_from_ms(deployment.latency_ms_p50),
                    quality_score=self._owner.latest_quality_score(
                        session, deployment_id=deployment.id
                    ),
                    metadata={
                        "deployment_id": deployment.id,
                        "submission_id": deployment.submission_id,
                        "deployment_revision": deployment.deployment_revision,
                        "health_status": deployment.health_status,
                    },
                )
            )
        return grouped

    def _candidate_runtime_endpoint(self, *, deployment_id: str) -> str:
        return f"{self.settings.owner_api_internal_url.rstrip('/')}/runtime/{deployment_id}"
