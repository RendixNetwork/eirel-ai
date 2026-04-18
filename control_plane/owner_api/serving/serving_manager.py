from __future__ import annotations

"""Serving release lifecycle, fleet management, and candidate selection.

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
    DeploymentScoreRecord,
    ManagedDeployment,
    RuntimeNodeSnapshot,
    RunFamilyResult,
    ServingDeployment,
    ServingRelease,
)
from shared.contracts.models import MinerRegistryEntry
from shared.contracts.specialist_contracts import contract_for_family, contracts_payload, SPECIALIST_CONTRACTS_VERSION
from control_plane.owner_api._constants import (
    ABV_SERVING_SELECTION_REASON,
    PRODUCTION_FAMILIES,
    WORKFLOW_COMPOSITION_SELECTION_REASON,
)
from control_plane.owner_api._helpers import _score_record_selection_score, utcnow

if TYPE_CHECKING:
    from control_plane.owner_api.managed import ManagedOwnerServices


class ServingManager:
    """Handles serving release lifecycle, fleet management, and candidate selection."""

    def __init__(self, owner: ManagedOwnerServices) -> None:
        self._owner = owner

    @property
    def db(self):
        return self._owner.db

    @property
    def settings(self):
        return self._owner.settings

    def serving_release_payload(self, release: ServingRelease | None) -> dict[str, Any] | None:
        if release is None:
            return None
        return {
            "id": release.id,
            "trigger_type": release.trigger_type,
            "status": release.status,
            "scheduled_for": release.scheduled_for.isoformat() if release.scheduled_for else None,
            "published_at": release.published_at.isoformat() if release.published_at else None,
            "cancelled_at": release.cancelled_at.isoformat() if release.cancelled_at else None,
            "metadata": release.metadata_json,
            "created_at": release.created_at.isoformat(),
            "updated_at": release.updated_at.isoformat(),
        }

    def serving_deployment_payload(
        self, serving: ServingDeployment | None
    ) -> dict[str, Any] | None:
        if serving is None:
            return None
        return {
            "id": serving.id,
            "release_id": serving.release_id,
            "family_id": serving.family_id,
            "source_deployment_id": serving.source_deployment_id,
            "source_submission_id": serving.source_submission_id,
            "miner_hotkey": serving.miner_hotkey,
            "source_deployment_revision": serving.source_deployment_revision,
            "endpoint": serving.endpoint,
            "status": serving.status,
            "health_status": serving.health_status,
            "health_details": serving.health_details_json,
            "serving_selection_reason": serving.health_details_json.get("serving_selection_reason"),
            "serving_selection_score": serving.health_details_json.get("serving_selection_score"),
            "run_winner_deployment_id": serving.health_details_json.get("run_winner_deployment_id"),
            "requested_cpu_millis": serving.requested_cpu_millis,
            "requested_memory_bytes": serving.requested_memory_bytes,
            "placement_status": serving.placement_status,
            "assigned_node_name": serving.assigned_node_name,
            "assigned_cpu_millis": serving.assigned_cpu_millis,
            "assigned_memory_bytes": serving.assigned_memory_bytes,
            "placement_error": serving.placement_error_text,
            "published_at": serving.published_at.isoformat() if serving.published_at else None,
            "drain_requested_at": (
                serving.drain_requested_at.isoformat() if serving.drain_requested_at else None
            ),
            "retired_at": serving.retired_at.isoformat() if serving.retired_at else None,
            "created_at": serving.created_at.isoformat(),
            "updated_at": serving.updated_at.isoformat(),
        }

    def runtime_node_payload(self, node: RuntimeNodeSnapshot) -> dict[str, Any]:
        return {
            "node_name": node.node_name,
            "pool_name": node.pool_name,
            "labels": node.labels_json,
            "ready": node.ready,
            "schedulable": node.schedulable,
            "verified": node.verified,
            "allocatable_cpu_millis": node.allocatable_cpu_millis,
            "allocatable_memory_bytes": node.allocatable_memory_bytes,
            "allocatable_pod_count": node.allocatable_pod_count,
            "derived_pod_capacity": node.derived_pod_capacity,
            "verification_error": node.verification_error_text,
            "metadata": node.metadata_json,
            "last_verified_at": node.last_verified_at.isoformat(),
            "updated_at": node.updated_at.isoformat(),
        }

    def list_runtime_nodes(self, session: Session) -> list[dict[str, Any]]:
        return [
            self.runtime_node_payload(item)
            for item in session.execute(
                select(RuntimeNodeSnapshot).order_by(RuntimeNodeSnapshot.node_name.asc())
            ).scalars()
        ]

    def runtime_capacity_summary(self, session: Session) -> dict[str, Any]:
        nodes = list(session.execute(select(RuntimeNodeSnapshot)).scalars())
        usage = self._owner._runtime_node_usage(session)
        pending_candidate_count = session.execute(
            select(ManagedDeployment).where(ManagedDeployment.placement_status == "pending_capacity")
        ).scalars()
        pending_serving_count = session.execute(
            select(ServingDeployment).where(ServingDeployment.placement_status == "pending_capacity")
        ).scalars()
        verified_nodes = [item for item in nodes if item.verified]
        total_allocatable_cpu = sum(item.allocatable_cpu_millis for item in verified_nodes)
        total_allocatable_memory = sum(item.allocatable_memory_bytes for item in verified_nodes)
        total_reserved_cpu = sum(stats["reserved_cpu_millis"] for stats in usage.values())
        total_reserved_memory = sum(stats["reserved_memory_bytes"] for stats in usage.values())
        node_entries: list[dict[str, Any]] = []
        for node in verified_nodes:
            remaining_cpu, remaining_memory, remaining_pods = self._owner._remaining_capacity(node, usage)
            stats = usage.get(node.node_name, {})
            node_entries.append(
                {
                    **self.runtime_node_payload(node),
                    "reserved_cpu_millis": stats.get("reserved_cpu_millis", 0),
                    "reserved_memory_bytes": stats.get("reserved_memory_bytes", 0),
                    "reserved_pods": stats.get("reserved_pods", 0),
                    "serving_count": stats.get("serving_count", 0),
                    "free_cpu_millis": remaining_cpu,
                    "free_memory_bytes": remaining_memory,
                    "free_pod_count": remaining_pods,
                }
            )
        return {
            "verified_node_count": len(verified_nodes),
            "pending_candidate_count": len(list(pending_candidate_count)),
            "pending_serving_count": len(list(pending_serving_count)),
            "total_allocatable_cpu_millis": total_allocatable_cpu,
            "total_allocatable_memory_bytes": total_allocatable_memory,
            "total_reserved_cpu_millis": total_reserved_cpu,
            "total_reserved_memory_bytes": total_reserved_memory,
            "nodes": node_entries,
        }

    def latest_published_release(self, session: Session) -> ServingRelease | None:
        return session.execute(
            select(ServingRelease)
            .where(ServingRelease.status == "published")
            .order_by(ServingRelease.published_at.desc(), ServingRelease.created_at.desc())
            .limit(1)
        ).scalar_one_or_none()

    def serving_release_by_id(self, session: Session, *, release_id: str) -> ServingRelease | None:
        return session.get(ServingRelease, release_id)

    def list_release_deployments(
        self, session: Session, *, release_id: str
    ) -> list[ServingDeployment]:
        return list(
            session.execute(
                select(ServingDeployment)
                .where(ServingDeployment.release_id == release_id)
                .order_by(ServingDeployment.family_id.asc(), ServingDeployment.created_at.asc())
            ).scalars()
        )

    def current_serving_fleet(self, session: Session) -> list[ServingDeployment]:
        release = self.latest_published_release(session)
        if release is None:
            return []
        return [
            item
            for item in self.list_release_deployments(session, release_id=release.id)
            if item.status == "healthy" and item.health_status == "healthy" and item.retired_at is None
        ]

    def _serving_runtime_endpoint(self, *, serving_deployment_id: str) -> str:
        return (
            f"{self.settings.owner_api_internal_url.rstrip('/')}"
            f"/runtime/serving/{serving_deployment_id}"
        )

    def get_serving_registry(self, session: Session) -> dict[str, list[MinerRegistryEntry]]:
        grouped: dict[str, list[MinerRegistryEntry]] = {family_id: [] for family_id in PRODUCTION_FAMILIES}
        for serving in self.current_serving_fleet(session):
            grouped.setdefault(serving.family_id, []).append(
                MinerRegistryEntry(
                    hotkey=serving.miner_hotkey,
                    family_id=serving.family_id,
                    endpoint=self._serving_runtime_endpoint(serving_deployment_id=serving.id),
                    latency_score=1.0,
                    quality_score=self._owner.latest_quality_score(
                        session, deployment_id=serving.source_deployment_id
                    ),
                    metadata={
                        "serving_deployment_id": serving.id,
                        "deployment_id": serving.source_deployment_id,
                        "submission_id": serving.source_submission_id,
                        "deployment_revision": serving.source_deployment_revision,
                        "source_candidate_deployment_id": serving.source_deployment_id,
                        "serving_release_id": serving.release_id,
                        "health_status": serving.health_status,
                        "response_owner": "control_plane",
                        "family_contract": contract_for_family(serving.family_id),
                        "serving_selection_reason": serving.health_details_json.get("serving_selection_reason"),
                        "serving_selection_score": serving.health_details_json.get("serving_selection_score"),
                        "run_winner_deployment_id": serving.health_details_json.get("run_winner_deployment_id"),
                    },
                )
            )
        return grouped

    def workflow_composition_registry(self, session: Session) -> dict[str, dict[str, Any]]:
        serving_release = self.latest_published_release(session)
        serving_by_family = {
            serving.family_id: serving
            for serving in self.current_serving_fleet(session)
        }
        registry: dict[str, dict[str, Any]] = {}
        for spec in self._owner.list_workflow_specs():
            selected_node_map: dict[str, dict[str, Any]] = {}
            for node in spec.nodes:
                serving = serving_by_family.get(node.family_id)
                if serving is None:
                    continue
                selected_node_map[node.node_id] = {
                    "node_id": node.node_id,
                    "role_id": node.role_id,
                    "family_id": node.family_id,
                    "miner_hotkey": serving.miner_hotkey,
                    "endpoint": self._serving_runtime_endpoint(serving_deployment_id=serving.id),
                    "deployment_id": serving.source_deployment_id,
                    "submission_id": serving.source_submission_id,
                    "serving_deployment_id": serving.id,
                }
            registry[spec.workflow_spec_id] = {
                "workflow_spec_id": spec.workflow_spec_id,
                "workflow_version": spec.workflow_version,
                "workflow_class": spec.workflow_class,
                "selection_reason": WORKFLOW_COMPOSITION_SELECTION_REASON,
                "source_serving_release_id": serving_release.id if serving_release is not None else None,
                "selected_node_map": selected_node_map,
            }
        return registry

    def _best_eligible_candidate_per_group(
        self,
        session: Session,
        *,
        candidate_overrides: dict[str, str] | None = None,
    ) -> dict[str, ManagedDeployment]:
        overrides = candidate_overrides or {}
        score_rows = list(session.execute(select(DeploymentScoreRecord)).scalars())
        latest_scores: dict[str, tuple[float, datetime]] = {}
        eligible_ids: set[str] = set()
        for row in score_rows:
            candidate = (_score_record_selection_score(row), row.created_at)
            current = latest_scores.get(row.deployment_id)
            if current is None or candidate[1] > current[1]:
                latest_scores[row.deployment_id] = candidate
            if row.is_eligible:
                eligible_ids.add(row.deployment_id)
        selected: dict[str, ManagedDeployment] = {}
        for family_id, deployment_id in overrides.items():
            deployment = session.get(ManagedDeployment, deployment_id)
            if deployment is None:
                raise ValueError(f"override deployment not found for family {family_id}")
            if deployment.family_id != family_id:
                raise ValueError(f"override deployment family mismatch for family {family_id}")
            if deployment.id not in eligible_ids or deployment.health_status != "healthy":
                raise ValueError(f"override deployment is not eligible for family {family_id}")
            selected[family_id] = deployment
        for family_id in PRODUCTION_FAMILIES:
            if family_id in selected:
                continue
            candidates = [
                deployment
                for deployment in session.execute(
                    select(ManagedDeployment)
                    .where(ManagedDeployment.family_id == family_id)
                    .where(ManagedDeployment.status != "retired")
                ).scalars()
                if deployment.id in eligible_ids and deployment.health_status == "healthy"
            ]
            if not candidates:
                continue
            candidates.sort(
                key=lambda item: (
                    latest_scores.get(item.id, (0.0, utcnow()))[0],
                    item.created_at,
                ),
                reverse=True,
            )
            selected[family_id] = candidates[0]
        return selected

    def _winner_candidate_per_family(
        self,
        session: Session,
        *,
        run_id: str,
        candidate_overrides: dict[str, str] | None = None,
    ) -> dict[str, ManagedDeployment]:
        overrides = candidate_overrides or {}
        selected: dict[str, ManagedDeployment] = {}
        for family_id, deployment_id in overrides.items():
            deployment = session.get(ManagedDeployment, deployment_id)
            if deployment is None:
                raise ValueError(f"override deployment not found for family {family_id}")
            if deployment.family_id != family_id:
                raise ValueError(f"override deployment family mismatch for family {family_id}")
            if deployment.status == "retired" or deployment.health_status != "healthy":
                raise ValueError(f"override deployment is not healthy for family {family_id}")
            selected[family_id] = deployment
        results = list(
            session.execute(
                select(RunFamilyResult).where(RunFamilyResult.run_id == run_id)
            ).scalars()
        )
        winner_deployment_ids = {
            item.family_id: item.winner_deployment_id
            for item in results
            if item.has_winner and item.winner_deployment_id
        }
        for family_id in PRODUCTION_FAMILIES:
            if family_id in selected:
                continue
            deployment_id = winner_deployment_ids.get(family_id)
            if not deployment_id:
                continue
            deployment = session.get(ManagedDeployment, deployment_id)
            if deployment is None or deployment.status == "retired" or deployment.health_status != "healthy":
                continue
            selected[family_id] = deployment
        return selected

    def _abv_run_winner_deployment_id(
        self,
        session: Session,
        *,
        run_id: str | None,
        family_id: str,
    ) -> str | None:
        if not run_id:
            return None
        winner = session.execute(
            select(RunFamilyResult)
            .where(RunFamilyResult.run_id == run_id)
            .where(RunFamilyResult.family_id == family_id)
            .limit(1)
        ).scalar_one_or_none()
        return winner.winner_deployment_id if winner is not None else None

    def _abv_serving_candidate_payload(
        self,
        session: Session,
        *,
        row: DeploymentScoreRecord,
    ) -> dict[str, Any]:
        deployment = session.get(ManagedDeployment, row.deployment_id)
        metadata = dict(row.metadata_json or {})
        reward_metrics = self._owner._scorecard_reward_metrics(session, row=row)
        rejection_reasons: list[str] = []
        if deployment is None or deployment.status == "retired":
            rejection_reasons.append("deployment_missing_or_retired")
        elif deployment.health_status != "healthy":
            rejection_reasons.append("deployment_unhealthy")
        if metadata.get("promotion_gate_passed") is not True:
            rejection_reasons.append("promotion_gate_failed")
        if metadata.get("consistency_gate_passed") is not True:
            rejection_reasons.append("consistency_gate_failed")
        if metadata.get("qualifies_for_incentives") is not True:
            rejection_reasons.append("incentives_not_qualified")
        return {
            "deployment": deployment,
            "deployment_id": row.deployment_id,
            "submission_id": row.submission_id,
            "miner_hotkey": row.miner_hotkey,
            "official_family_score": float(
                metadata.get("official_family_score", row.raw_score) or row.raw_score
            ),
            "reliability_score": float(reward_metrics["reliability_score"] or 0.0),
            "serving_selection_score": float(
                reward_metrics["serving_selection_score"] or 0.0
            ),
            "evaluation_path": metadata.get("evaluation_path"),
            "evaluation_plane": metadata.get("evaluation_plane"),
            "promotion_gate_passed": bool(metadata.get("promotion_gate_passed", False)),
            "consistency_gate_passed": bool(metadata.get("consistency_gate_passed", False)),
            "qualifies_for_incentives": bool(metadata.get("qualifies_for_incentives", False)),
            "workflow_product_diagnostics": dict(
                reward_metrics.get("workflow_product_diagnostics") or {}
            ),
            "rejection_reasons": rejection_reasons,
            "eligible": not rejection_reasons,
        }

    def _abv_candidate_selection_details(
        self,
        session: Session,
        *,
        family_id: str,
        run_id: str | None,
        current_source_map: dict[str, str],
        candidate_overrides: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        run_winner_deployment_id = self._abv_run_winner_deployment_id(
            session,
            run_id=run_id,
            family_id=family_id,
        )
        override_id = (candidate_overrides or {}).get(family_id)
        if override_id:
            deployment = session.get(ManagedDeployment, override_id)
            if deployment is None:
                raise ValueError(f"override deployment not found for family {family_id}")
            if deployment.family_id != family_id:
                raise ValueError(f"override deployment family mismatch for family {family_id}")
            if deployment.status == "retired" or deployment.health_status != "healthy":
                raise ValueError(f"override deployment is not healthy for family {family_id}")
            return {
                "selected": deployment,
                "selection_reason": "manual_override",
                "serving_selection_score": None,
                "run_winner_deployment_id": run_winner_deployment_id,
                "shortlist": [
                    {
                        "deployment_id": deployment.id,
                        "submission_id": deployment.submission_id,
                        "miner_hotkey": deployment.miner_hotkey,
                        "eligible": True,
                        "manual_override": True,
                        "rejection_reasons": [],
                    }
                ],
                "rejection_reasons": {},
            }

        rows = self._owner._ordered_run_score_records(session, run_id=run_id, family_id=family_id) if run_id else []
        shortlist = [
            self._abv_serving_candidate_payload(session, row=row)
            for row in rows
        ]
        shortlist.sort(
            key=lambda item: (
                float(item.get("serving_selection_score") or 0.0),
                float(item.get("reliability_score") or 0.0),
                float(item.get("official_family_score") or 0.0),
            ),
            reverse=True,
        )
        selected_payload = next((item for item in shortlist if item["eligible"]), None)
        selection_reason = ABV_SERVING_SELECTION_REASON if selected_payload is not None else None
        selected = (
            selected_payload["deployment"]
            if selected_payload is not None
            else None
        )
        if selected is None:
            current_source_id = current_source_map.get(family_id)
            if current_source_id:
                current_source = session.get(ManagedDeployment, current_source_id)
                if (
                    current_source is not None
                    and current_source.status != "retired"
                    and current_source.health_status == "healthy"
                ):
                    selected = current_source
                    selection_reason = "retained_previous_serving"
        rejection_reasons = {
            str(item["deployment_id"]): list(item.get("rejection_reasons", []))
            for item in shortlist
            if item.get("rejection_reasons")
        }
        return {
            "selected": selected,
            "selection_reason": selection_reason,
            "serving_selection_score": (
                float(selected_payload.get("serving_selection_score") or 0.0)
                if selected_payload is not None
                else None
            ),
            "run_winner_deployment_id": run_winner_deployment_id,
            "shortlist": [
                {
                    key: value
                    for key, value in item.items()
                    if key != "deployment"
                }
                for item in shortlist
            ],
            "rejection_reasons": rejection_reasons,
        }

    def _release_candidate_selection_details(
        self,
        session: Session,
        *,
        run_id: str | None,
        candidate_overrides: dict[str, str] | None = None,
    ) -> tuple[dict[str, ManagedDeployment], dict[str, dict[str, Any]]]:
        selected: dict[str, ManagedDeployment] = {}
        details: dict[str, dict[str, Any]] = {}
        current_source_map = self._current_serving_source_map(session)
        for family_id in PRODUCTION_FAMILIES:
            family_details = self._abv_candidate_selection_details(
                session,
                family_id=family_id,
                run_id=run_id,
                current_source_map=current_source_map,
                candidate_overrides=candidate_overrides,
            )
            if family_details["selected"] is not None:
                selected[family_id] = family_details["selected"]
            details[family_id] = family_details
        return selected, details

    def _release_candidate_selection(
        self,
        session: Session,
        *,
        run_id: str | None,
        candidate_overrides: dict[str, str] | None = None,
    ) -> dict[str, ManagedDeployment]:
        selected, _details = self._release_candidate_selection_details(
            session,
            run_id=run_id,
            candidate_overrides=candidate_overrides,
        )
        return selected

    def _current_serving_source_map(self, session: Session) -> dict[str, str]:
        return {
            item.family_id: item.source_deployment_id
            for item in self.current_serving_fleet(session)
        }

    def _release_due(self, session: Session) -> bool:
        current = self.latest_published_release(session)
        if current is None or current.published_at is None:
            return True
        return (utcnow() - current.published_at).days >= self._owner.serving_release_interval_days

    async def publish_serving_release(
        self,
        *,
        base_url: str,
        trigger_type: str,
        candidate_overrides: dict[str, str] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        with self.db.sessionmaker() as session:
            completed_run = self._owner.latest_completed_run(session)
            selected, selection_details = self._release_candidate_selection_details(
                session,
                run_id=completed_run.id if completed_run is not None else None,
                candidate_overrides=candidate_overrides,
            )
            if not selected:
                raise ValueError("no serving candidates are available")
            current_source_map = self._current_serving_source_map(session)
            selected_source_map = {family_id: item.id for family_id, item in selected.items()}
            if not force and selected_source_map == current_source_map:
                current = self.latest_published_release(session)
                return {
                    "status": "noop",
                    "release": self.serving_release_payload(current),
                    "deployments": [
                        self.serving_deployment_payload(item)
                        for item in self.current_serving_fleet(session)
                    ],
                }
            release = ServingRelease(
                trigger_type=trigger_type,
                status="pending",
                scheduled_for=utcnow(),
                metadata_json={
                    "selected_source_deployments": selected_source_map,
                    "force": force,
                    "run_id": completed_run.id if completed_run is not None else None,
                    "response_owner": "control_plane",
                    "family_contracts_version": SPECIALIST_CONTRACTS_VERSION,
                    "family_contracts": contracts_payload(list(selected)),
                    "rollback_source_deployments": current_source_map,
                    "serving_selection_mode_by_family": {
                        family_id: ABV_SERVING_SELECTION_REASON
                        for family_id in selected
                    },
                    "selection_reason_by_family": {
                        family_id: details.get("selection_reason")
                        for family_id, details in selection_details.items()
                        if details.get("selection_reason") is not None
                    },
                    "candidate_shortlists_by_family": {
                        family_id: details.get("shortlist", [])
                        for family_id, details in selection_details.items()
                    },
                    "rejection_reasons_by_family": {
                        family_id: details.get("rejection_reasons", {})
                        for family_id, details in selection_details.items()
                        if details.get("rejection_reasons")
                    },
                    "run_winner_deployment_id_by_family": {
                        family_id: details.get("run_winner_deployment_id")
                        for family_id, details in selection_details.items()
                        if details.get("run_winner_deployment_id") is not None
                    },
                    "serving_selected_deployment_id_by_family": selected_source_map,
                },
            )
            session.add(release)
            session.flush()
            created_serving_ids: list[str] = []
            for family_id, candidate in selected.items():
                details = selection_details.get(family_id, {})
                serving = ServingDeployment(
                    release_id=release.id,
                    family_id=family_id,
                    source_deployment_id=candidate.id,
                    source_submission_id=candidate.submission_id,
                    miner_hotkey=candidate.miner_hotkey,
                    source_deployment_revision=candidate.deployment_revision,
                    endpoint=f"{base_url.rstrip('/')}/runtime/serving/{candidate.family_id}/{release.id}",
                    status="pending",
                    health_status="starting",
                    health_details_json={
                        "source_deployment_id": candidate.id,
                        "serving_selection_reason": details.get("selection_reason"),
                        "serving_selection_score": details.get("serving_selection_score"),
                        "run_winner_deployment_id": details.get("run_winner_deployment_id"),
                    },
                    requested_cpu_millis=candidate.requested_cpu_millis,
                    requested_memory_bytes=candidate.requested_memory_bytes,
                    placement_status="pending",
                )
                session.add(serving)
                session.flush()
                serving.endpoint = f"{base_url.rstrip('/')}/runtime/serving/{serving.id}"
                created_serving_ids.append(serving.id)
            release.status = "deploying"
            release.updated_at = utcnow()
            session.commit()

        failures: list[dict[str, Any]] = []
        for serving_id in created_serving_ids:
            try:
                serving = await self._owner.ensure_serving_runtime(serving_deployment_id=serving_id)
                if serving.health_status != "healthy":
                    failures.append(
                        {
                            "serving_deployment_id": serving_id,
                            "error": serving.placement_error_text
                            or f"serving_runtime_not_ready:{serving.health_status}",
                        }
                    )
            except Exception as exc:
                failures.append({"serving_deployment_id": serving_id, "error": str(exc)})

        if failures:
            with self.db.sessionmaker() as session:
                release = session.execute(
                    select(ServingRelease).where(ServingRelease.id == release.id)
                ).scalar_one()
                release.status = "failed"
                release.metadata_json = {
                    **release.metadata_json,
                    "failures": failures,
                }
                release.updated_at = utcnow()
                session.commit()
            await self._owner.reconcile_runtime_pool()
            return {
                "status": "failed",
                "release": self.serving_release_payload(release),
                "failures": failures,
            }

        with self.db.sessionmaker() as session:
            current = self.latest_published_release(session)
            release = session.execute(
                select(ServingRelease).where(ServingRelease.id == release.id)
            ).scalar_one()
            new_fleet = self.list_release_deployments(session, release_id=release.id)
            release.status = "published"
            release.published_at = utcnow()
            release.updated_at = utcnow()
            for item in new_fleet:
                item.status = "healthy"
                item.health_status = "healthy"
                item.published_at = release.published_at
                item.updated_at = utcnow()
            if current is not None:
                previous_fleet = self.list_release_deployments(session, release_id=current.id)
                current.status = "superseded"
                current.updated_at = utcnow()
                for item in previous_fleet:
                    item.status = "draining"
                    item.drain_requested_at = utcnow()
                    item.updated_at = utcnow()
            session.commit()
            response = {
                "status": "published",
                "release": self.serving_release_payload(release),
                "deployments": [self.serving_deployment_payload(item) for item in new_fleet],
            }
        await self._owner.reconcile_runtime_pool()
        return response

    async def publish_due_serving_release(self, *, base_url: str) -> dict[str, Any] | None:
        with self.db.sessionmaker() as session:
            if not self._release_due(session):
                return None
        return await self.publish_serving_release(
            base_url=base_url,
            trigger_type="scheduled",
            force=False,
        )

    def cancel_serving_release(self, session: Session, *, release_id: str) -> ServingRelease:
        release = session.get(ServingRelease, release_id)
        if release is None:
            raise ValueError("serving release not found")
        if release.status not in {"pending", "deploying"}:
            raise ValueError("only pending or deploying releases can be cancelled")
        release.status = "cancelled"
        release.cancelled_at = utcnow()
        release.updated_at = utcnow()
        for serving in self.list_release_deployments(session, release_id=release_id):
            if serving.status not in {"retired", "failed"}:
                serving.status = "retired"
                serving.health_status = "retired"
                serving.retired_at = utcnow()
                serving.updated_at = utcnow()
        session.commit()
        session.refresh(release)
        return release

    def manual_drain_serving_deployment(
        self, session: Session, *, serving_deployment_id: str
    ) -> ServingDeployment:
        serving = session.get(ServingDeployment, serving_deployment_id)
        if serving is None:
            raise ValueError("serving deployment not found")
        serving.status = "draining"
        serving.placement_status = "draining"
        serving.drain_requested_at = utcnow()
        serving.updated_at = utcnow()
        session.commit()
        session.refresh(serving)
        return serving

    def manual_retire_serving_deployment(
        self, session: Session, *, serving_deployment_id: str
    ) -> ServingDeployment:
        serving = session.get(ServingDeployment, serving_deployment_id)
        if serving is None:
            raise ValueError("serving deployment not found")
        self._owner._release_serving_placement(serving)
        serving.status = "retired"
        serving.health_status = "retired"
        serving.retired_at = utcnow()
        serving.updated_at = utcnow()
        session.commit()
        session.refresh(serving)
        return serving
