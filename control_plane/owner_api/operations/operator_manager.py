from __future__ import annotations

"""Operator-facing status, health, and deployment management.

Extracted from ``ManagedOwnerServices`` to reduce the size of the
god-object.  Each public method here has a thin delegation wrapper in
``ManagedOwnerServices`` for backward compatibility.
"""

from typing import Any, TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.orm import Session

from shared.common.models import (
    AggregateFamilyScoreSnapshot,
    DeploymentHealthEvent,
    DeploymentScoreRecord,
    FamilyRolloutState,
    ManagedDeployment,
    ManagedMinerSubmission,
    MetagraphSyncSnapshot,
    ServingRelease,
    WorkflowEpisodeRecord,
)
from shared.workflow_specs import workflow_corpus_report
from control_plane.owner_api._constants import (
    ABV_SERVING_SELECTION_REASON,
    PRODUCTION_FAMILIES,
    WORKFLOW_COMPOSITION_SELECTION_REASON,
)
from control_plane.owner_api._helpers import utcnow

if TYPE_CHECKING:
    from control_plane.owner_api.managed import ManagedOwnerServices


class OperatorManager:
    """Handles operator-facing status summaries, health events, and manual deployment controls."""

    def __init__(self, owner: ManagedOwnerServices) -> None:
        self._owner = owner

    @property
    def db(self):
        return self._owner.db

    @property
    def settings(self):
        return self._owner.settings

    def latest_metagraph_sync_status(self, session: Session) -> dict[str, Any]:
        latest = session.execute(
            select(MetagraphSyncSnapshot)
            .order_by(MetagraphSyncSnapshot.created_at.desc())
            .limit(1)
        ).scalar_one_or_none()
        if latest is None:
            return {
                "status": "never_synced",
                "network": self.settings.bittensor_network,
                "netuid": self.settings.bittensor_netuid,
            }
        return {
            "status": latest.status,
            "network": latest.network,
            "netuid": latest.netuid,
            "validator_count": latest.validator_count,
            "miner_count": latest.miner_count,
            "error": latest.error_text,
            "created_at": latest.created_at.isoformat(),
        }

    def list_health_events(
        self,
        session: Session,
        *,
        deployment_id: str,
    ) -> list[dict[str, Any]]:
        return [
            {
                "status": item.status,
                "reason": item.reason,
                "details": item.details_json,
                "created_at": item.created_at.isoformat(),
            }
            for item in session.execute(
                select(DeploymentHealthEvent)
                .where(DeploymentHealthEvent.deployment_id == deployment_id)
                .order_by(DeploymentHealthEvent.created_at.desc())
            ).scalars()
        ]

    def manual_promote_deployment(self, session: Session, *, deployment_id: str) -> ManagedDeployment:
        deployment = session.get(ManagedDeployment, deployment_id)
        if deployment is None:
            raise ValueError("deployment not found")
        if deployment.status == "retired":
            raise ValueError("deployment is retired")
        for candidate in session.execute(
            select(ManagedDeployment).where(ManagedDeployment.family_id == deployment.family_id)
        ).scalars():
            if candidate.id == deployment.id:
                candidate.is_active = True
                candidate.active_set_rank = 1
                candidate.status = "active"
                candidate.health_details_json = {
                    key: value
                    for key, value in candidate.health_details_json.items()
                    if key != "drain_requested_at"
                }
            elif candidate.is_active and candidate.miner_hotkey == deployment.miner_hotkey:
                candidate.is_active = False
                candidate.active_set_rank = None
                candidate.status = "standby_cold"
        deployment.updated_at = utcnow()
        session.commit()
        session.refresh(deployment)
        return deployment

    def manual_drain_deployment(self, session: Session, *, deployment_id: str) -> ManagedDeployment:
        deployment = session.get(ManagedDeployment, deployment_id)
        if deployment is None:
            raise ValueError("deployment not found")
        deployment.is_active = False
        deployment.active_set_rank = None
        deployment.status = "draining"
        deployment.health_details_json = {
            **deployment.health_details_json,
            "operator_drain": True,
            "drain_requested_at": utcnow().isoformat(),
        }
        deployment.updated_at = utcnow()
        session.commit()
        session.refresh(deployment)
        return deployment

    def _workflow_runtime_health_summary(
        self,
        session: Session,
        *,
        workflow_episode_records: list[WorkflowEpisodeRecord] | None = None,
    ) -> dict[str, Any]:
        records = workflow_episode_records
        if records is None:
            records = list(session.execute(select(WorkflowEpisodeRecord)).scalars())
        now = utcnow()
        lifecycle_counts = {
            "queued": 0,
            "executing": 0,
            "deferred": 0,
            "completed": 0,
            "retry_wait": 0,
            "failed": 0,
            "dead_lettered": 0,
        }
        stale_episode_count = 0
        retryable_episode_count = 0
        retry_wait_episode_count = 0
        task_backed_count = 0
        internal_episode_count = 0
        oldest_stale_episode_age_seconds = 0.0
        oldest_retry_wait_age_seconds = 0.0
        for item in records:
            if self._owner._workflow_episode_task_execution_payload(item) is not None:
                task_backed_count += 1
            else:
                internal_episode_count += 1
            if item.queue_state == "completed":
                lifecycle_counts["completed"] += 1
            elif item.queue_state == "dead_lettered" or item.status == "dead_lettered":
                lifecycle_counts["dead_lettered"] += 1
            elif item.queue_state == "failed" or item.status == "failed":
                lifecycle_counts["failed"] += 1
            elif item.queue_state == "deferred":
                lifecycle_counts["deferred"] += 1
            elif item.queue_state in {"leased", "executing"}:
                lifecycle_counts["executing"] += 1
            elif self._owner._workflow_episode_is_retry_wait(item, now=now):
                lifecycle_counts["retry_wait"] += 1
                retry_wait_episode_count += 1
                oldest_retry_wait_age_seconds = max(
                    oldest_retry_wait_age_seconds,
                    max(0.0, (now - item.updated_at).total_seconds()),
                )
            else:
                lifecycle_counts["queued"] += 1
            if self._owner._workflow_episode_is_stale(item, now=now):
                stale_episode_count += 1
                if item.lease_expires_at is not None:
                    oldest_stale_episode_age_seconds = max(
                        oldest_stale_episode_age_seconds,
                        max(0.0, (now - item.lease_expires_at).total_seconds()),
                    )
            if self._owner._workflow_episode_is_retryable(item, now=now) and int(item.attempt_count or 0) > 0:
                retryable_episode_count += 1
        return {
            "lifecycle_counts": lifecycle_counts,
            "retry_backlog_count": retry_wait_episode_count,
            "retryable_episode_count": retryable_episode_count,
            "retry_wait_episode_count": retry_wait_episode_count,
            "dead_lettered_episode_count": lifecycle_counts["dead_lettered"],
            "stale_episode_count": stale_episode_count,
            "oldest_stale_episode_age_seconds": oldest_stale_episode_age_seconds,
            "oldest_retry_wait_age_seconds": oldest_retry_wait_age_seconds,
            "task_backed_episode_count": task_backed_count,
            "internal_episode_count": internal_episode_count,
            "active_leased_count": sum(
                1
                for item in records
                if item.queue_state in {"leased", "executing"}
            ),
            "ok": stale_episode_count == 0,
        }

    def operator_summary(self, session: Session) -> dict[str, Any]:
        submissions = list(session.execute(select(ManagedMinerSubmission)).scalars())
        deployments = list(session.execute(select(ManagedDeployment)).scalars())
        aggregate_snapshots = list(session.execute(select(AggregateFamilyScoreSnapshot)).scalars())
        workflow_episode_records = list(session.execute(select(WorkflowEpisodeRecord)).scalars())
        runs = self._owner.list_runs(session)
        current_run = self._owner.current_run(session)
        latest_completed_run = self._owner.latest_completed_run(session)
        serving_releases = list(session.execute(select(ServingRelease)).scalars())
        current_serving_release = self._owner.latest_published_release(session)
        current_serving_fleet = self._owner.current_serving_fleet(session)
        workflow_composition_registry = self._owner.workflow_composition_registry(session)
        chain_publication_readiness = self._owner.chain_publication_readiness(session)
        candidate_registry = self._owner.get_candidate_registry(session)
        runtime_capacity = self._owner.runtime_capacity_summary(session)
        latest_scored = session.execute(
            select(DeploymentScoreRecord)
            .order_by(DeploymentScoreRecord.created_at.desc())
            .limit(1)
        ).scalar_one_or_none()
        healthy_candidate_families = {
            family_id
            for family_id, entries in candidate_registry.items()
            if entries
        }
        readiness = {
            "ok": True,
            "ready": len(healthy_candidate_families) == len(PRODUCTION_FAMILIES),
            "active_family_count": len(healthy_candidate_families),
        }
        corpus_report = workflow_corpus_report(corpus_root=self.settings.workflow_corpus_root_path)
        workflow_readiness = {
            "ok": bool(corpus_report.valid),
            "ready": bool(corpus_report.valid and len(self._owner.list_workflow_specs()) >= 4),
            "workflow_spec_count": len(self._owner.list_workflow_specs()),
            "workflow_episode_count": len(workflow_episode_records),
            "corpus_root_path": self.settings.workflow_corpus_root_path,
            "manifest_loaded": corpus_report.manifest is not None,
            "corpus_version": corpus_report.corpus_version,
            "corpus_manifest_digest": corpus_report.manifest_digest,
            "public_slice_count": corpus_report.public_slice_count,
            "hidden_slice_count": corpus_report.hidden_slice_count,
            "baseline_count": corpus_report.baseline_count,
            "errors": list(corpus_report.errors),
        }
        latest_workflow_episode = (
            max(workflow_episode_records, key=lambda item: (item.created_at, item.episode_id))
            if workflow_episode_records
            else None
        )
        runtime_health = self._workflow_runtime_health_summary(
            session,
            workflow_episode_records=workflow_episode_records,
        )
        latest_dead_lettered_episode = (
            max(
                (item for item in workflow_episode_records if self._owner._workflow_episode_is_dead_lettered(item)),
                key=lambda item: (
                    item.dead_lettered_at or item.updated_at,
                    item.episode_id,
                ),
            )
            if any(self._owner._workflow_episode_is_dead_lettered(item) for item in workflow_episode_records)
            else None
        )
        incidents_preview = self._owner.workflow_incidents(session, limit=10)
        summary = {
            "service": "owner",
            "managed_execution": True,
            "submission_status_counts": {
                status: sum(1 for item in submissions if item.status == status)
                for status in sorted({item.status for item in submissions} | {"healthy"})
            },
            "deployment_health_counts": {
                status: sum(1 for item in deployments if item.health_status == status)
                for status in sorted({item.health_status for item in deployments} | {"healthy"})
            },
            "active_deployment_count": sum(1 for item in deployments if item.is_active),
            "serving_release_count": len(serving_releases),
            "published_serving_family_count": len(current_serving_fleet),
            "aggregate_run_snapshot_count": len(aggregate_snapshots),
            "run_count": len(runs),
            "runtime_capacity": runtime_capacity,
            "current_run": (
                {
                    "run_id": current_run.id,
                    "sequence": current_run.sequence,
                    "status": current_run.status,
                    "started_at": current_run.started_at.isoformat(),
                    "ends_at": current_run.ends_at.isoformat(),
                }
                if current_run is not None
                else None
            ),
            "latest_completed_run": (
                {
                    "run_id": latest_completed_run.id,
                    "sequence": latest_completed_run.sequence,
                    "status": latest_completed_run.status,
                    "started_at": latest_completed_run.started_at.isoformat(),
                    "ends_at": latest_completed_run.ends_at.isoformat(),
                    "closed_at": (
                        latest_completed_run.closed_at.isoformat()
                        if latest_completed_run.closed_at
                        else None
                    ),
                }
                if latest_completed_run is not None
                else None
            ),
            "metagraph_sync": self.latest_metagraph_sync_status(session),
            "rollout_freeze_families": [
                state.family_id
                for state in session.execute(select(FamilyRolloutState)).scalars()
                if state.rollout_frozen
            ],
            "scoring_readiness": readiness,
            "workflow_evaluation_readiness": workflow_readiness,
            "workflow_episode_lifecycle": runtime_health["lifecycle_counts"],
            "workflow_runtime_health": {
                key: value
                for key, value in runtime_health.items()
                if key != "lifecycle_counts"
            },
            "workflow_runtime_incidents_preview": incidents_preview,
            "chain_publication": {
                "readiness": chain_publication_readiness,
                "chain_publish_ready": bool(chain_publication_readiness.get("ready")),
                "chain_publish_blockers": list(chain_publication_readiness.get("blockers") or []),
                "authoritative_v3_ready": bool(chain_publication_readiness.get("ready")),
            },
            "runtime_links": {
                "workflow_incidents_url": "/v1/operators/workflow-incidents",
                "dead_lettered_incidents_url": "/v1/operators/workflow-incidents?incident_state=dead_lettered",
                "stale_incidents_url": "/v1/operators/workflow-incidents?incident_state=stale",
                "runtime_remediation_url": "/v1/operators/runtime-remediation",
                "runtime_remediation_policy_url": "/v1/operators/runtime-remediation/policy",
                "runtime_remediation_suppressions_url": "/v1/operators/runtime-remediation/suppressions",
                "execution_worker_runtime_status_url": "/v1/operators/runtime-status",
                "workflow_composition_registry_url": "/v1/workflow-composition/registry",
            },
            "abv_serving_mode": ABV_SERVING_SELECTION_REASON,
            "workflow_composition_mode": WORKFLOW_COMPOSITION_SELECTION_REASON,
            "authority_flags": {
                "auto_remediation_policy_enabled": bool(self.settings.workflow_runtime_auto_remediation_enabled),
                "chain_publish_readiness_enforced": bool(self.settings.chain_publish_readiness_enforced),
                "incentive_authority": "family_protocol",
                "family_serving_authority": "family_protocol",
                "workflow_observability_mode": "owner_observability_only",
                "scoring_authority": "family_protocol",
                "runtime_routing_authority": WORKFLOW_COMPOSITION_SELECTION_REASON,
                "chain_publication_authority": "v3_family_aggregates",
            },
            "current_serving_release": self._owner.serving_release_payload(current_serving_release),
            "workflow_composition_registry": workflow_composition_registry,
            "current_serving_fleet": [
                self._owner.serving_deployment_payload(item) for item in current_serving_fleet
            ],
            "latest_scored_submission": None,
            "latest_workflow_episode": (
                self._owner.workflow_episode_payload(latest_workflow_episode)
                if latest_workflow_episode is not None
                else None
            ),
            "latest_dead_lettered_episode": (
                self._owner.workflow_episode_payload(latest_dead_lettered_episode)
                if latest_dead_lettered_episode is not None
                else None
            ),
        }
        if latest_scored is not None:
            summary["latest_scored_submission"] = {
                "submission_id": latest_scored.submission_id,
                "run_id": latest_scored.run_id,
                "family_id": latest_scored.family_id,
                "deployment_revision": latest_scored.deployment_revision,
            }
        return summary
