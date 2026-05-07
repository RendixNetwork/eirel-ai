from __future__ import annotations

"""Runtime node inventory, placement, and deployment lifecycle management.

Extracted from ``ManagedOwnerServices`` (Item 15) to reduce the size of
the god-object.  Each public method here has a thin delegation wrapper
in ``ManagedOwnerServices`` for backward compatibility.
"""

import logging
from datetime import datetime
from typing import Any, TYPE_CHECKING

logger = logging.getLogger(__name__)

from sqlalchemy import select
from sqlalchemy.orm import Session

from shared.common.models import (
    FamilyRolloutState,
    ManagedDeployment,
    ManagedMinerSubmission,
    RuntimeNodeSnapshot,
    ServingDeployment,
    SubmissionArtifact,
)
from shared.common.manifest import SubmissionManifest
from eirel.groups import ensure_family_id
from control_plane.owner_api._constants import PLACEMENT_RESERVED_STATUSES, PRODUCTION_FAMILIES
from control_plane.owner_api._helpers import utcnow
from infra.miner_runtime.runtime_manager import RuntimeManagerError
from infra.miner_runtime._k8s_helpers import DeploymentStatus, DeploymentStatusCode

if TYPE_CHECKING:
    from control_plane.owner_api.managed import ManagedOwnerServices


_VALID_DEPLOYMENT_TRANSITIONS: dict[str, set[str]] = {
    "queued": {"received", "retired"},
    "received": {"building", "pending_capacity", "retired"},
    "pending_capacity": {"building", "retired"},
    "building": {"deployed_for_eval", "build_failed", "retired"},
    "build_failed": {"building", "retired"},
    "deployed_for_eval": {"eligible", "active", "unhealthy", "retired", "draining", "standby_cold", "building"},
    "eligible": {"active", "unhealthy", "retired", "draining", "standby_cold", "building"},
    "active": {"unhealthy", "retired", "draining", "standby_cold", "eligible", "building"},
    "unhealthy": {"building", "retired", "deployed_for_eval"},
    "draining": {"standby_cold", "retired", "active"},
    "standby_cold": {"active", "retired", "building", "draining"},
    "retired": set(),  # Terminal state -- no transitions allowed
}


class DeploymentManager:
    """Handles runtime node inventory, placement scheduling, and deployment lifecycle."""

    def __init__(self, owner: ManagedOwnerServices) -> None:
        self._owner = owner

    @staticmethod
    def _check_transition(deployment_id: str, current: str, target: str) -> None:
        valid = _VALID_DEPLOYMENT_TRANSITIONS.get(current)
        if valid is not None and target not in valid:
            logger.error(
                "invalid deployment transition blocked: id=%s %s -> %s (allowed: %s)",
                deployment_id, current, target, valid,
            )
            raise ValueError(
                f"invalid deployment state transition: {current} -> {target} "
                f"for deployment {deployment_id}"
            )

    @property
    def db(self):
        return self._owner.db

    @property
    def settings(self):
        return self._owner.settings

    def validator_endpoint_base(self, *, base_url: str, run_id: str, deployment_id: str) -> str:
        return (
            f"{base_url.rstrip('/')}/v1/validator/runs/{run_id}/deployments/{deployment_id}"
        )

    def validator_health_path(self, *, run_id: str, deployment_id: str) -> str:
        return f"/v1/validator/runs/{run_id}/deployments/{deployment_id}/healthz"

    def validator_infer_path(self, *, run_id: str, deployment_id: str) -> str:
        return f"/v1/validator/runs/{run_id}/deployments/{deployment_id}/v1/agent/infer"

    def rollout_state(
        self,
        session: Session,
        *,
        family_id: str,
    ) -> FamilyRolloutState:
        family_id = ensure_family_id(family_id)
        state = session.get(FamilyRolloutState, family_id)
        if state is None:
            state = FamilyRolloutState(family_id=family_id, rollout_frozen=False)
            session.add(state)
            session.flush()
        return state

    def is_rollout_frozen(
        self,
        session: Session,
        *,
        family_id: str,
    ) -> bool:
        return self.rollout_state(session, family_id=family_id).rollout_frozen

    def set_rollout_freeze(
        self,
        session: Session,
        *,
        family_id: str,
        rollout_frozen: bool,
        reason: str | None = None,
    ) -> FamilyRolloutState:
        state = self.rollout_state(session, family_id=family_id)
        state.rollout_frozen = rollout_frozen
        state.freeze_reason = reason
        state.updated_at = utcnow()
        session.commit()
        session.refresh(state)
        return state

    async def refresh_runtime_node_inventory(self) -> list[RuntimeNodeSnapshot]:
        discovered = await self._owner.runtime_manager.list_runtime_nodes()
        now = utcnow()
        with self.db.sessionmaker() as session:
            seen: set[str] = set()
            for node in discovered:
                seen.add(node.node_name)
                verified, error_text = self._verify_runtime_node(node)
                snapshot = session.get(RuntimeNodeSnapshot, node.node_name)
                if snapshot is None:
                    snapshot = RuntimeNodeSnapshot(node_name=node.node_name)
                    session.add(snapshot)
                snapshot.pool_name = "miner"
                snapshot.labels_json = dict(node.labels)
                snapshot.ready = node.ready
                snapshot.schedulable = node.schedulable
                snapshot.verified = verified
                snapshot.allocatable_cpu_millis = node.allocatable_cpu_millis
                snapshot.allocatable_memory_bytes = node.allocatable_memory_bytes
                snapshot.allocatable_pod_count = node.allocatable_pod_count
                snapshot.derived_pod_capacity = max(
                    0,
                    node.allocatable_pod_count - self._owner.runtime_pod_headroom,
                )
                snapshot.verification_error_text = error_text
                snapshot.metadata_json = dict(node.metadata)
                snapshot.last_verified_at = now
                snapshot.updated_at = now
            for snapshot in session.execute(select(RuntimeNodeSnapshot)).scalars():
                if snapshot.node_name in seen:
                    continue
                snapshot.ready = False
                snapshot.schedulable = False
                snapshot.verified = False
                snapshot.verification_error_text = "node_missing_from_cluster"
                snapshot.last_verified_at = now
                snapshot.updated_at = now
            self._release_invalid_placements(session)
            session.commit()
            return list(
                session.execute(
                    select(RuntimeNodeSnapshot).order_by(RuntimeNodeSnapshot.node_name.asc())
                ).scalars()
            )

    def _verify_runtime_node(self, node: Any) -> tuple[bool, str | None]:
        labels = getattr(node, "labels", {}) or {}
        required = {
            self.settings.owner_runtime_pool_label_key: self.settings.owner_runtime_pool_label_value,
            self.settings.owner_runtime_class_label_key: self.settings.owner_runtime_class_label_value,
        }
        for key, value in required.items():
            if str(labels.get(key)) != value:
                return False, f"missing_required_label:{key}"
        if not getattr(node, "ready", False):
            return False, "node_not_ready"
        if not getattr(node, "schedulable", False):
            return False, "node_unschedulable"
        if int(getattr(node, "allocatable_cpu_millis", 0) or 0) <= 0:
            return False, "allocatable_cpu_unavailable"
        if int(getattr(node, "allocatable_memory_bytes", 0) or 0) <= 0:
            return False, "allocatable_memory_unavailable"
        return True, None

    def _release_invalid_placements(self, session: Session) -> None:
        valid_nodes = {
            item.node_name
            for item in session.execute(
                select(RuntimeNodeSnapshot).where(RuntimeNodeSnapshot.verified.is_(True))
            ).scalars()
        }
        for deployment in session.execute(select(ManagedDeployment)).scalars():
            if deployment.assigned_node_name and deployment.assigned_node_name not in valid_nodes:
                self._release_managed_placement(deployment, error="assigned_node_unavailable")
        for serving in session.execute(select(ServingDeployment)).scalars():
            if serving.assigned_node_name and serving.assigned_node_name not in valid_nodes:
                self._release_serving_placement(serving, error="assigned_node_unavailable")

    def _release_managed_placement(
        self,
        deployment: ManagedDeployment,
        *,
        error: str | None = None,
    ) -> None:
        deployment.placement_status = "pending_capacity" if error else "released"
        deployment.assigned_node_name = None
        deployment.assigned_cpu_millis = 0
        deployment.assigned_memory_bytes = 0
        deployment.placement_error_text = error
        deployment.updated_at = utcnow()

    def _release_serving_placement(
        self,
        serving: ServingDeployment,
        *,
        error: str | None = None,
    ) -> None:
        serving.placement_status = "pending_capacity" if error else "released"
        serving.assigned_node_name = None
        serving.assigned_cpu_millis = 0
        serving.assigned_memory_bytes = 0
        serving.placement_error_text = error
        serving.updated_at = utcnow()

    def _runtime_node_usage(self, session: Session) -> dict[str, dict[str, int]]:
        usage: dict[str, dict[str, int]] = {}
        for node in session.execute(select(RuntimeNodeSnapshot)).scalars():
            usage[node.node_name] = {
                "reserved_cpu_millis": 0,
                "reserved_memory_bytes": 0,
                "reserved_pods": 0,
                "serving_count": 0,
            }
        for deployment in session.execute(select(ManagedDeployment)).scalars():
            if deployment.assigned_node_name not in usage:
                continue
            if deployment.placement_status not in PLACEMENT_RESERVED_STATUSES:
                continue
            stats = usage[deployment.assigned_node_name]
            stats["reserved_cpu_millis"] += int(deployment.assigned_cpu_millis or 0)
            stats["reserved_memory_bytes"] += int(deployment.assigned_memory_bytes or 0)
            stats["reserved_pods"] += 1
        for serving in session.execute(select(ServingDeployment)).scalars():
            if serving.assigned_node_name not in usage:
                continue
            if serving.retired_at is not None or serving.placement_status not in PLACEMENT_RESERVED_STATUSES:
                continue
            stats = usage[serving.assigned_node_name]
            stats["reserved_cpu_millis"] += int(serving.assigned_cpu_millis or 0)
            stats["reserved_memory_bytes"] += int(serving.assigned_memory_bytes or 0)
            stats["reserved_pods"] += 1
            stats["serving_count"] += 1
        return usage

    def _remaining_capacity(
        self,
        node: RuntimeNodeSnapshot,
        usage: dict[str, dict[str, int]],
    ) -> tuple[int, int, int]:
        stats = usage.get(node.node_name, {})
        remaining_cpu = max(
            0,
            int(node.allocatable_cpu_millis or 0)
            - self._owner.runtime_cpu_headroom_millis
            - int(stats.get("reserved_cpu_millis", 0)),
        )
        remaining_memory = max(
            0,
            int(node.allocatable_memory_bytes or 0)
            - self._owner.runtime_memory_headroom_bytes
            - int(stats.get("reserved_memory_bytes", 0)),
        )
        remaining_pods = max(
            0,
            int(node.allocatable_pod_count or 0)
            - self._owner.runtime_pod_headroom
            - int(stats.get("reserved_pods", 0)),
        )
        return remaining_cpu, remaining_memory, remaining_pods

    def _select_runtime_node(
        self,
        session: Session,
        *,
        requested_cpu_millis: int,
        requested_memory_bytes: int,
        placement_kind: str,
    ) -> tuple[RuntimeNodeSnapshot | None, str | None]:
        nodes = list(
            session.execute(
                select(RuntimeNodeSnapshot)
                .where(RuntimeNodeSnapshot.verified.is_(True))
                .order_by(RuntimeNodeSnapshot.node_name.asc())
            ).scalars()
        )
        if not nodes:
            return None, "no_verified_runtime_nodes"
        usage = self._runtime_node_usage(session)
        fit: list[tuple[Any, RuntimeNodeSnapshot]] = []
        for node in nodes:
            remaining_cpu, remaining_memory, remaining_pods = self._remaining_capacity(node, usage)
            if (
                remaining_cpu < requested_cpu_millis
                or remaining_memory < requested_memory_bytes
                or remaining_pods < 1
            ):
                continue
            stats = usage.get(node.node_name, {})
            if placement_kind == "serving":
                utilization = (
                    stats.get("reserved_cpu_millis", 0) / max(1, int(node.allocatable_cpu_millis))
                )
                fit.append(((int(stats.get("serving_count", 0)), utilization, node.node_name), node))
            else:
                fit.append(
                    (
                        (
                            remaining_cpu - requested_cpu_millis,
                            remaining_memory - requested_memory_bytes,
                            node.node_name,
                        ),
                        node,
                    )
                )
        if not fit:
            return None, "insufficient_runtime_capacity"
        fit.sort(key=lambda item: item[0])
        return fit[0][1], None

    _ALREADY_RUNNING_STATUSES = {"deployed_for_eval", "eligible", "active"}

    async def ensure_deployment_runtime(
        self,
        *,
        deployment_id: str,
    ) -> ManagedDeployment:
        await self.refresh_runtime_node_inventory()
        with self.db.sessionmaker() as session:
            deployment = session.get(ManagedDeployment, deployment_id)
            if deployment is None:
                raise ValueError("deployment not found")
            if deployment.status == "retired":
                session.refresh(deployment)
                return deployment
            # If already in a running state with a live container, skip rebuild
            if deployment.status in self._ALREADY_RUNNING_STATUSES:
                handle = self._owner.runtime_manager.runtime_handle(deployment_id)
                if handle is not None and handle.state == "healthy":
                    session.refresh(deployment)
                    return deployment
            submission = session.get(ManagedMinerSubmission, deployment.submission_id)
            if submission is None:
                raise ValueError("submission not found")
            artifact = session.get(SubmissionArtifact, submission.artifact_id)
            if artifact is None:
                raise ValueError("artifact not found")
            manifest = SubmissionManifest.model_validate(submission.manifest_json)
            requested_cpu_millis, requested_memory_bytes = self._owner.normalize_manifest_resources(manifest)
            deployment.requested_cpu_millis = requested_cpu_millis
            deployment.requested_memory_bytes = requested_memory_bytes
            if deployment.assigned_node_name:
                node = session.get(RuntimeNodeSnapshot, deployment.assigned_node_name)
                if node is None or not node.verified:
                    self._release_managed_placement(
                        deployment,
                        error="assigned_node_unavailable",
                    )
            if not deployment.assigned_node_name:
                selected_node, placement_error = self._select_runtime_node(
                    session,
                    requested_cpu_millis=requested_cpu_millis,
                    requested_memory_bytes=requested_memory_bytes,
                    placement_kind="candidate",
                )
                if selected_node is None:
                    deployment.status = "pending_capacity"
                    deployment.health_status = "pending_capacity"
                    deployment.placement_status = "pending_capacity"
                    deployment.placement_error_text = placement_error
                    deployment.health_details_json = {
                        **deployment.health_details_json,
                        "build": "pending_capacity",
                        "deploy": "pending_capacity",
                        "placement_error": placement_error,
                    }
                    deployment.updated_at = utcnow()
                    submission.status = "pending_capacity"
                    submission.updated_at = utcnow()
                    session.commit()
                    session.refresh(deployment)
                    return deployment
                deployment.assigned_node_name = selected_node.node_name
                deployment.assigned_cpu_millis = requested_cpu_millis
                deployment.assigned_memory_bytes = requested_memory_bytes
                deployment.placement_status = "assigned"
                deployment.placement_error_text = None
            self._check_transition(deployment_id, deployment.status, "building")
            submission.status = "building"
            deployment.status = "building"
            deployment.health_status = "starting"
            deployment.health_details_json = {
                **deployment.health_details_json,
                "build": "running",
                "deploy": "pending",
                "assigned_node_name": deployment.assigned_node_name,
            }
            deployment.updated_at = utcnow()
            submission.updated_at = utcnow()
            session.commit()
        try:
            handle = await self._owner.runtime_manager.ensure_runtime(
                deployment_id=deployment_id,
                submission_id=deployment_id,
                archive_sha256=submission.archive_sha256,
                archive_bytes=artifact.archive_bytes,
                manifest=manifest,
                owner_api_url=self.settings.owner_api_internal_url,
                internal_service_token=self.settings.internal_service_token,
                provider_proxy_url=self.settings.provider_proxy_url,
                provider_proxy_token=self.settings.provider_proxy_token,
                assigned_node_name=deployment.assigned_node_name,
                requested_cpu_millis=deployment.requested_cpu_millis,
                requested_memory_bytes=deployment.requested_memory_bytes,
                run_budget_usd=self.settings.run_budget_usd,
            )
        except Exception as exc:
            with self.db.sessionmaker() as session:
                deployment = session.get(ManagedDeployment, deployment_id)
                submission = session.get(ManagedMinerSubmission, deployment.submission_id) if deployment else None
                if deployment is not None:
                    deployment.status = "build_failed"
                    deployment.health_status = "unhealthy"
                    deployment.health_details_json = {
                        **deployment.health_details_json,
                        "build": "failed",
                        "deploy": "failed",
                        "reason": str(exc),
                    }
                    deployment.placement_status = "assigned" if deployment.assigned_node_name else "pending_capacity"
                    deployment.updated_at = utcnow()
                    self._owner.record_health_event(
                        session,
                        deployment=deployment,
                        status="unhealthy",
                        reason=str(exc),
                        details={"build_failed": True},
                    )
                if submission is not None:
                    submission.status = "build_failed"
                    submission.updated_at = utcnow()
                session.commit()
            raise
        with self.db.sessionmaker() as session:
            deployment = session.get(ManagedDeployment, deployment_id)
            submission = session.get(ManagedMinerSubmission, deployment.submission_id) if deployment else None
            if deployment is None:
                raise ValueError("deployment not found after runtime start")
            self._check_transition(deployment_id, deployment.status, "deployed_for_eval")
            deployment.status = "deployed_for_eval"
            deployment.health_status = "healthy"
            deployment.health_details_json = {
                **deployment.health_details_json,
                "build": "succeeded",
                "deploy": "succeeded",
                "container_name": handle.container_name,
                "host_port": handle.host_port,
            }
            deployment.placement_status = "running"
            deployment.placement_error_text = None
            deployment.updated_at = utcnow()
            if submission is not None:
                submission.status = "deployed_for_eval"
                submission.updated_at = utcnow()
            self._owner.record_health_event(
                session,
                deployment=deployment,
                status="healthy",
                details={"runtime_started": True},
            )
            session.commit()
            session.refresh(deployment)
            logger.info(
                "deployment runtime started: id=%s node=%s",
                deployment_id, deployment.assigned_node_name,
            )
            return deployment

    async def schedule_queued_deployments(self, deployment_ids: list[str]) -> None:
        for deployment_id in deployment_ids:
            try:
                await self.ensure_deployment_runtime(deployment_id=deployment_id)
            except Exception:
                logger.exception(
                    "schedule_queued: failed to start runtime for deployment %s",
                    deployment_id,
                )

    async def ensure_serving_runtime(
        self,
        *,
        serving_deployment_id: str,
    ) -> ServingDeployment:
        await self.refresh_runtime_node_inventory()
        with self.db.sessionmaker() as session:
            serving = session.get(ServingDeployment, serving_deployment_id)
            if serving is None:
                raise ValueError("serving deployment not found")
            source = session.get(ManagedDeployment, serving.source_deployment_id)
            if source is None:
                raise ValueError("source deployment not found")
            submission = session.get(ManagedMinerSubmission, serving.source_submission_id)
            if submission is None:
                raise ValueError("submission not found")
            artifact = session.get(SubmissionArtifact, submission.artifact_id)
            if artifact is None:
                raise ValueError("artifact not found")
            manifest = SubmissionManifest.model_validate(submission.manifest_json)
            requested_cpu_millis, requested_memory_bytes = self._owner.normalize_manifest_resources(manifest)
            serving.requested_cpu_millis = requested_cpu_millis
            serving.requested_memory_bytes = requested_memory_bytes
            if serving.assigned_node_name:
                node = session.get(RuntimeNodeSnapshot, serving.assigned_node_name)
                if node is None or not node.verified:
                    self._release_serving_placement(serving, error="assigned_node_unavailable")
            if not serving.assigned_node_name:
                selected_node, placement_error = self._select_runtime_node(
                    session,
                    requested_cpu_millis=requested_cpu_millis,
                    requested_memory_bytes=requested_memory_bytes,
                    placement_kind="serving",
                )
                if selected_node is None:
                    serving.status = "pending"
                    serving.health_status = "pending_capacity"
                    serving.placement_status = "pending_capacity"
                    serving.placement_error_text = placement_error
                    serving.health_details_json = {
                        **serving.health_details_json,
                        "deploy": "pending_capacity",
                        "placement_error": placement_error,
                    }
                    serving.updated_at = utcnow()
                    session.commit()
                    session.refresh(serving)
                    return serving
                serving.assigned_node_name = selected_node.node_name
                serving.assigned_cpu_millis = requested_cpu_millis
                serving.assigned_memory_bytes = requested_memory_bytes
                serving.placement_status = "assigned"
                serving.placement_error_text = None
            serving.status = "deploying"
            serving.health_status = "starting"
            serving.health_details_json = {
                **serving.health_details_json,
                "build": "succeeded",
                "deploy": "running",
                "source_deployment_id": source.id,
                "assigned_node_name": serving.assigned_node_name,
            }
            serving.updated_at = utcnow()
            session.commit()
        try:
            handle = await self._owner.runtime_manager.ensure_runtime(
                deployment_id=serving_deployment_id,
                submission_id=serving_deployment_id,
                archive_sha256=submission.archive_sha256,
                archive_bytes=artifact.archive_bytes,
                manifest=manifest,
                owner_api_url=self.settings.owner_api_internal_url,
                internal_service_token=self.settings.internal_service_token,
                provider_proxy_url=self.settings.provider_proxy_url,
                provider_proxy_token=self.settings.provider_proxy_token,
                assigned_node_name=serving.assigned_node_name,
                requested_cpu_millis=serving.requested_cpu_millis,
                requested_memory_bytes=serving.requested_memory_bytes,
                run_budget_usd=self.settings.run_budget_usd,
            )
        except Exception as exc:
            with self.db.sessionmaker() as session:
                serving = session.get(ServingDeployment, serving_deployment_id)
                if serving is not None:
                    serving.status = "failed"
                    serving.health_status = "unhealthy"
                    serving.health_details_json = {
                        **serving.health_details_json,
                        "deploy": "failed",
                        "reason": str(exc),
                    }
                    serving.placement_status = "assigned" if serving.assigned_node_name else "pending_capacity"
                    serving.updated_at = utcnow()
                    session.commit()
            raise
        with self.db.sessionmaker() as session:
            serving = session.get(ServingDeployment, serving_deployment_id)
            if serving is None:
                raise ValueError("serving deployment not found after runtime start")
            serving.status = "healthy"
            serving.health_status = "healthy"
            serving.health_details_json = {
                **serving.health_details_json,
                "deploy": "succeeded",
                "container_name": handle.container_name,
                "host_port": handle.host_port,
            }
            serving.placement_status = "running"
            serving.placement_error_text = None
            serving.updated_at = utcnow()
            session.commit()
            session.refresh(serving)
            return serving

    async def stop_deployment_runtime(
        self,
        *,
        deployment_id: str,
        reason: str,
        retire: bool = False,
    ) -> None:
        logger.info("stopping deployment runtime: id=%s reason=%s retire=%s", deployment_id, reason, retire)
        handle = self._owner.runtime_manager.runtime_handle(deployment_id)
        if handle is not None:
            try:
                await self._owner.runtime_manager.stop_runtime(deployment_id, reason=reason, soft=False)
            except (RuntimeManagerError, RuntimeError, OSError, TimeoutError) as exc:
                logger.warning("runtime stop failed for %s: %s", deployment_id, exc)
                return
        with self.db.sessionmaker() as session:
            deployment = session.get(ManagedDeployment, deployment_id)
            if deployment is None:
                return
            self._release_managed_placement(deployment)
            deployment.is_active = False
            deployment.active_set_rank = None
            deployment.health_status = "unhealthy" if not retire else "retired"
            deployment.status = "retired" if retire else "unhealthy"
            deployment.health_details_json = {
                **deployment.health_details_json,
                "stop_reason": reason,
            }
            if retire:
                deployment.retired_at = utcnow()
            deployment.pending_runtime_stop = False
            deployment.updated_at = utcnow()
            self._owner.record_health_event(
                session,
                deployment=deployment,
                status=deployment.health_status,
                reason=reason,
                details={"retire": retire},
            )
            submission = session.get(ManagedMinerSubmission, deployment.submission_id)
            if submission is not None and retire:
                submission.status = "retired"
                submission.updated_at = utcnow()
            session.commit()

    async def recover_or_demote_deployment(
        self,
        *,
        deployment_id: str,
        reason: str,
    ) -> None:
        with self.db.sessionmaker() as session:
            deployment = session.get(ManagedDeployment, deployment_id)
            if deployment is None or deployment.status == "retired":
                return
            submission_id = deployment.submission_id

        backend = getattr(self._owner.runtime_manager, "backend", None)
        if backend is None:
            return
        status = await backend.deployment_status(submission_id)

        if status.code == DeploymentStatusCode.READY:
            return

        if status.code == DeploymentStatusCode.PENDING_STARTING:
            return

        if status.code == DeploymentStatusCode.PENDING_UNSCHEDULABLE:
            with self.db.sessionmaker() as session:
                deployment = session.get(ManagedDeployment, deployment_id)
                if deployment is not None:
                    deployment.placement_status = "pending_capacity"
                    deployment.health_status = "unhealthy"
                    deployment.updated_at = utcnow()
                    self._owner.record_health_event(
                        session,
                        deployment=deployment,
                        status="unhealthy",
                        reason=reason,
                        details={"k8s_status": status.code},
                    )
                    session.commit()
            return

        if status.code in (DeploymentStatusCode.CRASHLOOP, DeploymentStatusCode.MISSING):
            return await self._attempt_rebuild_or_demote(
                deployment_id=deployment_id, reason=reason, status=status,
            )

        if status.code == DeploymentStatusCode.UNKNOWN:
            return await self._legacy_recover_or_demote(
                deployment_id=deployment_id, reason=reason,
            )

    async def _attempt_rebuild_or_demote(
        self,
        *,
        deployment_id: str,
        reason: str,
        status: DeploymentStatus,
    ) -> None:
        with self.db.sessionmaker() as session:
            deployment = session.get(ManagedDeployment, deployment_id)
            if deployment is None or deployment.status == "retired":
                return
            restart_attempts = int(deployment.health_details_json.get("restart_attempts", 0))
            if restart_attempts >= max(0, self.settings.owner_runtime_restart_budget):
                should_demote = True
            else:
                should_demote = False
                deployment.health_details_json = {
                    **deployment.health_details_json,
                    "restart_attempts": restart_attempts + 1,
                    "last_restart_reason": reason,
                }
                deployment.updated_at = utcnow()
                self._owner.record_health_event(
                    session,
                    deployment=deployment,
                    status="restarting",
                    reason=reason,
                    details={"restart_attempt": restart_attempts + 1},
                )
                session.commit()
        if should_demote:
            await self.stop_deployment_runtime(
                deployment_id=deployment_id,
                reason=f"restart_budget_exhausted:{reason}",
                retire=False,
            )
            return
        try:
            await self._owner.runtime_manager.stop_runtime(
                deployment_id,
                reason=f"restart:{reason}",
                soft=False,
            )
        except (RuntimeError, TimeoutError, ConnectionError, OSError):
            logger.warning("failed to stop runtime before restart: id=%s", deployment_id, exc_info=True)
        try:
            await self.ensure_deployment_runtime(deployment_id=deployment_id)
        except Exception:
            with self.db.sessionmaker() as session:
                deployment = session.get(ManagedDeployment, deployment_id)
                if deployment is not None:
                    restart_attempts = int(
                        deployment.health_details_json.get("restart_attempts", 0)
                    )
                    if restart_attempts >= max(
                        1, self.settings.owner_runtime_restart_budget
                    ):
                        session.commit()
                        await self.stop_deployment_runtime(
                            deployment_id=deployment_id,
                            reason=f"restart_failed:{reason}",
                            retire=False,
                        )
                        return
            raise

    async def _legacy_recover_or_demote(
        self,
        *,
        deployment_id: str,
        reason: str,
    ) -> None:
        await self._attempt_rebuild_or_demote(
            deployment_id=deployment_id, reason=reason,
            status=DeploymentStatus(
                code=DeploymentStatusCode.UNKNOWN,
                ready_replicas=0, desired_replicas=0,
                message="legacy path", last_pod_phase=None,
            ),
        )

    async def reconcile_family_deployments(self, *, family_id: str) -> None:
        with self.db.sessionmaker() as session:
            deployments = list(
                session.execute(
                    select(ManagedDeployment)
                    .where(ManagedDeployment.family_id == family_id)
                    .where(ManagedDeployment.status != "retired")
                ).scalars()
            )
            active_unhealthy_ids = [
                item.id
                for item in deployments
                if item.is_active and item.health_status != "healthy"
            ]
        for deployment_id in active_unhealthy_ids:
            try:
                await self.recover_or_demote_deployment(
                    deployment_id=deployment_id,
                    reason="active_unhealthy_reconcile",
                )
            except Exception:
                logger.exception("failed to recover deployment: id=%s", deployment_id)
                continue
        with self.db.sessionmaker() as session:
            self._owner.rebalance_family(session, family_id=family_id)
            deployments = list(
                session.execute(
                    select(ManagedDeployment)
                    .where(ManagedDeployment.family_id == family_id)
                    .where(ManagedDeployment.status != "retired")
                ).scalars()
            )
            pinned_ids = self._owner.open_run_pinned_deployment_ids(session, family_id=family_id)
            bootstrapping_ids = {
                item.id
                for item in deployments
                if item.status in {"building", "received", "deploying", "pending_capacity"}
                or item.health_status == "starting"
            }
            to_keep = (
                {item.id for item in deployments if item.is_active}
                | pinned_ids
                | bootstrapping_ids
                | {
                    item.id
                    for item in deployments
                    if item.status in {"deployed_for_eval", "eligible"}
                }
            )
            to_start = [item.id for item in deployments if item.id in to_keep]
        for deployment_id in to_start:
            try:
                await self.ensure_deployment_runtime(deployment_id=deployment_id)
            except Exception:
                logger.exception("failed to start deployment runtime: id=%s", deployment_id)
                continue
        with self.db.sessionmaker() as session:
            self._owner.rebalance_family(session, family_id=family_id)
            deployments = list(
                session.execute(
                    select(ManagedDeployment)
                    .where(ManagedDeployment.family_id == family_id)
                    .where(ManagedDeployment.status != "retired")
                ).scalars()
            )
            pinned_ids = self._owner.open_run_pinned_deployment_ids(session, family_id=family_id)
            bootstrapping_ids = {
                item.id
                for item in deployments
                if item.status in {"building", "received", "deploying", "pending_capacity"}
                or item.health_status == "starting"
            }
            desired_keep = (
                {item.id for item in deployments if item.is_active}
                | pinned_ids
                | bootstrapping_ids
                | {
                    item.id
                    for item in deployments
                    if item.status == "deployed_for_eval"
                }
            )
            now = utcnow()
            for deployment in deployments:
                if deployment.id not in desired_keep:
                    if deployment.health_status == "healthy":
                        drain_requested_at = deployment.health_details_json.get("drain_requested_at")
                        if drain_requested_at is None:
                            deployment.health_details_json = {
                                **deployment.health_details_json,
                                "drain_requested_at": now.isoformat(),
                            }
                            deployment.status = "draining"
                            desired_keep.add(deployment.id)
                        else:
                            try:
                                drain_started = datetime.fromisoformat(drain_requested_at)
                            except ValueError:
                                drain_started = now
                            if (now - drain_started).total_seconds() < self._owner.soft_termination_grace_seconds:
                                deployment.status = "draining"
                                desired_keep.add(deployment.id)
                            else:
                                deployment.status = "standby_cold"
                    else:
                        deployment.status = "standby_cold"
                    deployment.active_set_rank = None
                    deployment.updated_at = utcnow()
                else:
                    if deployment.status == "draining":
                        deployment.health_details_json = {
                            key: value
                            for key, value in deployment.health_details_json.items()
                            if key != "drain_requested_at"
                        }
                        deployment.status = "active" if deployment.is_active else deployment.status
            session.commit()
        await self.reconcile_runtime_pool(family_id=family_id)

    def _candidate_runtime_ids_to_keep(
        self,
        session: Session,
        *,
        family_id: str | None = None,
    ) -> set[str]:
        statement = select(ManagedDeployment).where(ManagedDeployment.status != "retired")
        if family_id is not None:
            statement = statement.where(ManagedDeployment.family_id == family_id)
        deployments = list(session.execute(statement).scalars())
        pinned_ids = self._owner.open_run_pinned_deployment_ids(session, family_id=family_id)
        bootstrapping_ids = {
            item.id
            for item in deployments
            if item.status in {"building", "received", "deploying", "pending_capacity"}
            or item.health_status == "starting"
        }
        return (
            {item.id for item in deployments if item.is_active}
            | pinned_ids
            | bootstrapping_ids
            | {
                item.id
                for item in deployments
                if item.status in {"deployed_for_eval", "eligible", "draining"}
            }
        )

    def _serving_runtime_ids_to_keep(
        self,
        session: Session,
        *,
        family_id: str | None = None,
    ) -> set[str]:
        statement = select(ServingDeployment).where(ServingDeployment.retired_at.is_(None))
        if family_id is not None:
            statement = statement.where(ServingDeployment.family_id == family_id)
        keep_ids: set[str] = set()
        now = utcnow()
        for serving in session.execute(statement).scalars():
            if serving.status in {"pending", "deploying", "healthy"} or serving.placement_status == "pending_capacity":
                keep_ids.add(serving.id)
                continue
            if serving.status == "draining":
                drain_started = serving.drain_requested_at or now
                if (now - drain_started).total_seconds() < self._owner.soft_termination_grace_seconds:
                    keep_ids.add(serving.id)
                else:
                    self._release_serving_placement(serving)
                    serving.status = "retired"
                    serving.health_status = "retired"
                    serving.retired_at = now
                    serving.updated_at = now
        return keep_ids

    async def reconcile_runtime_pool(
        self,
        *,
        family_id: str | None = None,
    ) -> None:
        await self.refresh_runtime_node_inventory()
        with self.db.sessionmaker() as session:
            candidate_keep = self._candidate_runtime_ids_to_keep(session, family_id=family_id)
            serving_keep = self._serving_runtime_ids_to_keep(session, family_id=family_id)
            candidate_to_start = list(candidate_keep)
            serving_to_start = [
                item.id
                for item in session.execute(
                    select(ServingDeployment).where(ServingDeployment.retired_at.is_(None))
                ).scalars()
                if (family_id is None or item.family_id == family_id)
                and item.id in serving_keep
                and item.status in {"pending", "deploying", "healthy"}
            ]
            session.commit()
        for deployment_id in candidate_to_start:
            try:
                await self.ensure_deployment_runtime(deployment_id=deployment_id)
            except Exception:
                logger.exception("reconcile: failed to start candidate runtime: id=%s", deployment_id)
                continue
        for serving_deployment_id in serving_to_start:
            try:
                await self.ensure_serving_runtime(serving_deployment_id=serving_deployment_id)
            except Exception:
                logger.exception("reconcile: failed to start serving runtime: id=%s", serving_deployment_id)
                continue
        with self.db.sessionmaker() as session:
            desired_keep = (
                self._candidate_runtime_ids_to_keep(session, family_id=family_id)
                | self._serving_runtime_ids_to_keep(session, family_id=family_id)
            )
            for deployment in session.execute(select(ManagedDeployment)).scalars():
                if (family_id is None or deployment.family_id == family_id) and deployment.id not in desired_keep:
                    if deployment.placement_status in PLACEMENT_RESERVED_STATUSES:
                        self._release_managed_placement(deployment)
            for serving in session.execute(select(ServingDeployment)).scalars():
                if (family_id is None or serving.family_id == family_id) and serving.id not in desired_keep:
                    if serving.placement_status in PLACEMENT_RESERVED_STATUSES:
                        self._release_serving_placement(serving)
            session.commit()
        await self._owner.runtime_manager.reconcile_active_deployments(desired_keep)

    async def reconcile_all_active_deployments(self) -> None:
        for family_id in PRODUCTION_FAMILIES:
            await self.reconcile_family_deployments(family_id=family_id)

    async def proactive_health_check(self) -> None:
        """Check Docker container state for all deployments that should be
        running.  If a container has stopped, trigger recovery or demotion."""
        backend = getattr(self._owner.runtime_manager, "backend", None)
        if backend is None or not hasattr(backend, "is_container_running"):
            return  # Not supported by this runtime manager

        with self.db.sessionmaker() as session:
            target_ids = set()
            for family_id in PRODUCTION_FAMILIES:
                target_ids |= self._candidate_runtime_ids_to_keep(session, family_id=family_id)

        for dep_id in target_ids:
            handle = self._owner.runtime_manager.runtime_handle(dep_id)
            if handle is None:
                continue
            try:
                is_running = await backend.is_container_running(dep_id)
                if is_running is None:
                    continue  # Backend doesn't support this check
                if not is_running:
                    logger.warning(
                        "proactive health: container for deployment %s stopped, triggering recovery",
                        dep_id,
                    )
                    await self.recover_or_demote_deployment(
                        deployment_id=dep_id,
                        reason="proactive_health_container_stopped",
                    )
            except Exception:
                logger.exception("proactive health: check failed for deployment %s", dep_id)

    async def ensure_current_run_and_reconcile(self) -> Any:
        """Ensure a current run exists, and if a run transition occurred,
        stop containers for retired deployments and reconcile all families.

        If the run is still scheduled (not open), skip reconciliation --
        there are no deployments to manage yet.
        """
        with self.db.sessionmaker() as session:
            before_run = self._owner.runs.current_run(session)
            before_id = before_run.id if before_run else None
            current = self._owner.runs.ensure_current_run(session)
            session.commit()

        # Drain queued deployments promoted by start_queued_deployments during
        # the sync run-open path. We schedule these explicitly rather than
        # relying on reconcile_all_active_deployments so tests and edge cases
        # (e.g. repeat calls that don't trigger a run transition) still spin
        # up queued pods.
        queued_ids = list(self._owner.runs._pending_queued_deployment_ids)
        self._owner.runs._pending_queued_deployment_ids = []

        # If run is not open yet (scheduled for the future), nothing to do.
        if current.status != "open":
            return current

        if queued_ids:
            logger.info(
                "run-open %s: scheduling %d queued deployments",
                current.id, len(queued_ids),
            )
            await self.schedule_queued_deployments(queued_ids)

        if current.id != before_id:
            # Run transition occurred -- stop retired containers immediately
            retired_ids = list(self._owner.runs._last_closed_run_retired_ids)
            self._owner.runs._last_closed_run_retired_ids = []
            if retired_ids:
                logger.info(
                    "run transition %s -> %s: stopping %d retired deployments",
                    before_id, current.id, len(retired_ids),
                )
            for dep_id in retired_ids:
                try:
                    await self.stop_deployment_runtime(
                        deployment_id=dep_id,
                        reason="run_closed",
                        retire=True,
                    )
                except Exception:
                    logger.exception("failed to stop retired deployment: %s", dep_id)
            # Start carry-over containers + clean up remaining orphans
            await self.reconcile_all_active_deployments()

        return current
