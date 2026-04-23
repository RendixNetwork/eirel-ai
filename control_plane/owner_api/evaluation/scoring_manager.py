from __future__ import annotations

"""Scoring, scorecard, and aggregate snapshot management.

Extracted from ``ManagedOwnerServices`` (Item 15) to reduce the size of
the god-object.  Each public method here has a thin delegation wrapper
in ``ManagedOwnerServices`` for backward compatibility.
"""

import logging
from typing import Any, TYPE_CHECKING

import httpx

logger = logging.getLogger(__name__)

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from shared.common.models import (
    AggregateFamilyScoreSnapshot,
    DeploymentScoreRecord,
    EpochTargetSnapshot,
    EvaluationRun,
    ManagedDeployment,
    ManagedMinerSubmission,
    RunFamilyResult,
    WorkflowEpisodeRecord,
)
from eirel.groups import ensure_family_id
from shared.contracts.models import FamilyScoreSnapshot

if TYPE_CHECKING:
    from control_plane.owner_api.managed import ManagedOwnerServices


from control_plane.owner_api._constants import (
    FAMILY_SERVING_FAMILY_WEIGHT,
    FAMILY_SERVING_RELIABILITY_WEIGHT,
    PRODUCTION_FAMILIES,
)
from control_plane.owner_api._helpers import (
    _metadata_float,
    _metadata_int,
    _score_record_official_family_score,
    _score_record_selection_score,
    fixed_family_weight,
    utcnow,
)


class ScoringManager:
    """Handles scoring records, scorecards, and aggregate score snapshots."""

    def __init__(self, owner: ManagedOwnerServices) -> None:
        self._owner = owner

    @property
    def db(self):
        return self._owner.db

    @property
    def settings(self):
        return self._owner.settings

    # ------------------------------------------------------------------
    # Score record ordering helpers
    # ------------------------------------------------------------------

    def _top_run_deployment_ids(
        self,
        session: Session,
        *,
        run_id: str,
        family_id: str,
        limit: int,
    ) -> list[str]:
        rows = self._ordered_run_score_records(
            session,
            run_id=run_id,
            family_id=family_id,
        )
        top_ids: list[str] = []
        seen_hotkeys: set[str] = set()
        for row in rows:
            if row.miner_hotkey in seen_hotkeys:
                continue
            seen_hotkeys.add(row.miner_hotkey)
            top_ids.append(row.deployment_id)
            if len(top_ids) >= limit:
                break
        return top_ids

    def _ordered_run_score_records(
        self,
        session: Session,
        *,
        run_id: str,
        family_id: str,
    ) -> list[DeploymentScoreRecord]:
        rows = list(
            session.execute(
                select(DeploymentScoreRecord)
                .where(DeploymentScoreRecord.run_id == run_id)
                .where(DeploymentScoreRecord.family_id == family_id)
            ).scalars()
        )

        submission_ids = {row.submission_id for row in rows if row.submission_id}
        deployment_ids = {row.deployment_id for row in rows if row.deployment_id}
        submissions_by_id: dict[str, ManagedMinerSubmission] = {}
        deployments_by_id: dict[str, ManagedDeployment] = {}
        if submission_ids:
            for sub in session.execute(
                select(ManagedMinerSubmission).where(ManagedMinerSubmission.id.in_(submission_ids))
            ).scalars():
                submissions_by_id[sub.id] = sub
        if deployment_ids:
            for dep in session.execute(
                select(ManagedDeployment).where(ManagedDeployment.id.in_(deployment_ids))
            ).scalars():
                deployments_by_id[dep.id] = dep

        def submission_sort_key(row: DeploymentScoreRecord) -> tuple:
            submission = submissions_by_id.get(row.submission_id)
            deployment = deployments_by_id.get(row.deployment_id)
            submission_created = submission.created_at if submission is not None else row.created_at
            deployment_created = deployment.created_at if deployment is not None else row.created_at
            return (submission_created, deployment_created, row.deployment_id)

        return sorted(
            rows,
            key=lambda row: (-_score_record_selection_score(row), *submission_sort_key(row)),
        )

    # ------------------------------------------------------------------
    # Runtime contract mode lookup
    # ------------------------------------------------------------------

    def _runtime_contract_modes_for_miner(
        self,
        session: Session,
        *,
        family_id: str,
        miner_hotkey: str,
        run_id: str | None = None,
    ) -> dict[str, int]:
        statement = select(WorkflowEpisodeRecord).order_by(WorkflowEpisodeRecord.created_at.desc())
        if run_id is not None:
            statement = statement.where(WorkflowEpisodeRecord.run_id == run_id)
        modes: dict[str, int] = {}
        for record in session.execute(statement).scalars():
            result_json = dict(record.result_json or {})
            for trace in list(result_json.get("node_traces", []) or []):
                if not isinstance(trace, dict):
                    continue
                if trace.get("family_id") != family_id or trace.get("miner_hotkey") != miner_hotkey:
                    continue
                mode = str(
                    trace.get("runtime_contract_mode")
                    or (trace.get("metadata") or {}).get("runtime_contract_mode")
                    or ""
                )
                if not mode:
                    continue
                modes[mode] = modes.get(mode, 0) + 1
        return modes

    # ------------------------------------------------------------------
    # Scorecard building
    # ------------------------------------------------------------------

    def _scorecard_reward_metrics(
        self,
        session: Session,
        *,
        row: DeploymentScoreRecord,
    ) -> dict[str, float | None]:
        del session
        metadata = dict(row.metadata_json or {})
        evaluation_breakdown = dict(metadata.get("evaluation_breakdown", {}) or {})
        official_family_score = float(
            metadata.get("official_family_score", row.raw_score) or row.raw_score
        )
        reliability_score = max(
            0.0,
            min(
                1.0,
                (
                    float(
                        evaluation_breakdown.get(
                            "robustness_score", metadata.get("robustness_score", 1.0)
                        )
                        or 0.0
                    )
                    + float(
                        evaluation_breakdown.get(
                            "anti_gaming_score", metadata.get("anti_gaming_score", 1.0)
                        )
                        or 0.0
                    )
                )
                / 2.0,
            ),
        )
        serving_selection_score = max(
            0.0,
            min(
                1.0,
                FAMILY_SERVING_FAMILY_WEIGHT * official_family_score
                + FAMILY_SERVING_RELIABILITY_WEIGHT * reliability_score,
            ),
        )
        return {
            "reliability_score": reliability_score,
            "serving_selection_score": serving_selection_score,
            "workflow_product_diagnostics": {
                "workflow_episode_count": _metadata_int(metadata.get("anchored_episode_count")),
                "public_episode_count": _metadata_int(metadata.get("public_episode_count")),
                "hidden_episode_count": _metadata_int(metadata.get("hidden_episode_count")),
                "replay_episode_count": _metadata_int(metadata.get("replay_episode_count")),
                "workflow_spec_ids": list(metadata.get("workflow_spec_ids", []) or []),
                "corpus_version": (
                    metadata.get("corpus_version")
                    or metadata.get("family_diagnostics", {}).get("corpus_version")
                ),
                "corpus_manifest_digest": (
                    metadata.get("corpus_manifest_digest")
                    or metadata.get("family_diagnostics", {}).get("corpus_manifest_digest")
                ),
                "workflow_slice_id": metadata.get("workflow_slice_id"),
                "workflow_composition_source": metadata.get("workflow_composition_source"),
                "workflow_composition_revision": metadata.get("workflow_composition_revision"),
                "workflow_composition_registry_url": metadata.get(
                    "workflow_composition_registry_url"
                ),
                "workflow_composition_reason": metadata.get("workflow_composition_reason"),
            },
        }

    def _scorecard_payload(
        self,
        session: Session,
        *,
        row: DeploymentScoreRecord,
        rank: int | None = None,
        is_run_winner: bool = False,
    ) -> dict[str, Any]:
        metadata = dict(row.metadata_json or {})
        reward_metrics = self._scorecard_reward_metrics(session, row=row)
        workflow_product_diagnostics = dict(
            reward_metrics.get("workflow_product_diagnostics") or {}
        )
        runtime_contract_modes = dict(
            metadata.get("runtime_contract_modes")
            or metadata.get("family_diagnostics", {}).get("runtime_contract_modes", {})
            or self._runtime_contract_modes_for_miner(
                session,
                family_id=row.family_id,
                miner_hotkey=row.miner_hotkey,
                run_id=row.run_id,
            )
        )
        return {
            "run_id": row.run_id,
            "family_id": row.family_id,
            "deployment_id": row.deployment_id,
            "submission_id": row.submission_id,
            "miner_hotkey": row.miner_hotkey,
            "created_at": row.created_at.isoformat(),
            "raw_score": float(row.raw_score),
            "normalized_score": float(row.normalized_score),
            "family_capability_score": float(
                metadata.get("family_capability_score", row.raw_score) or row.raw_score
            ),
            "robustness_score": float(metadata.get("robustness_score", 1.0) or 0.0),
            "anti_gaming_score": float(metadata.get("anti_gaming_score", 1.0) or 0.0),
            "official_family_score": float(
                metadata.get("official_family_score", row.raw_score) or row.raw_score
            ),
            "evaluation_path": metadata.get("evaluation_path"),
            "evaluation_plane": metadata.get("evaluation_plane"),
            "protocol_suite_id": metadata.get("protocol_suite_id"),
            "protocol_fixture_ids": list(metadata.get("protocol_fixture_ids", []) or []),
            "protocol_gate_passed": bool(metadata.get("protocol_gate_passed", False)),
            "protocol_gate_failures": list(metadata.get("protocol_gate_failures", []) or []),
            "protocol_contract_pass_rate": _metadata_float(
                metadata.get("protocol_contract_pass_rate")
            ),
            "fixture_episode_count": _metadata_int(metadata.get("fixture_episode_count")),
            "hidden_fixture_count": _metadata_int(metadata.get("hidden_fixture_count")),
            "reliability_score": reward_metrics["reliability_score"],
            "serving_selection_score": reward_metrics["serving_selection_score"],
            "runtime_contract_modes": runtime_contract_modes,
            "official_scoring_version": metadata.get("official_scoring_version"),
            "scoring_policy_version": metadata.get("scoring_policy_version"),
            "benchmark_version": metadata.get("benchmark_version"),
            "rubric_version": metadata.get("rubric_version"),
            "agent_gpa_score": metadata.get("agent_gpa_score"),
            "deer_score": metadata.get("deer_score"),
            "ledger_truth_score": metadata.get("ledger_truth_score"),
            "agent_gpa_breakdown": dict(metadata.get("agent_gpa_breakdown", {}) or {}),
            "deer_breakdown": dict(metadata.get("deer_breakdown", {}) or {}),
            "ledger_truth_breakdown": dict(metadata.get("ledger_truth_breakdown", {}) or {}),
            "paper_deer_dimensions": dict(metadata.get("paper_deer_dimensions", {}) or {}),
            "paper_deer_subdimensions": dict(metadata.get("paper_deer_subdimensions", {}) or {}),
            "localized_failure_count": metadata.get("localized_failure_count"),
            "primary_failure_stage": metadata.get("primary_failure_stage"),
            "localized_failure_labels": list(
                metadata.get("localized_failure_labels", []) or []
            ),
            "judge_score": metadata.get("judge_score"),
            "evaluation_breakdown": dict(metadata.get("evaluation_breakdown", {}) or {}),
            "family_diagnostics": dict(metadata.get("family_diagnostics", {}) or {}),
            "analyst_variant_scores": dict(
                metadata.get("analyst_variant_scores", {}) or {}
            ),
            "analyst_variant_diagnostics": dict(
                metadata.get("analyst_variant_diagnostics", {}) or {}
            ),
            "analyst_track_scores": dict(metadata.get("analyst_track_scores", {}) or {}),
            "analyst_track_gate_status": dict(
                metadata.get("analyst_track_gate_status", {}) or {}
            ),
            "analyst_protocol_status": dict(
                metadata.get("analyst_protocol_status", {}) or {}
            ),
            "analyst_truth_metrics": dict(metadata.get("analyst_truth_metrics", {}) or {}),
            "analyst_trace_metrics": dict(metadata.get("analyst_trace_metrics", {}) or {}),
            "analyst_anti_gaming_metrics": dict(
                metadata.get("analyst_anti_gaming_metrics", {}) or {}
            ),
            "analyst_failure_taxonomy": dict(
                metadata.get("analyst_failure_taxonomy", {}) or {}
            ),
            "workflow_product_diagnostics": {
                **workflow_product_diagnostics,
                "workflow_episode_count": (
                    workflow_product_diagnostics.get("workflow_episode_count")
                    if workflow_product_diagnostics.get("workflow_episode_count") is not None
                    else _metadata_int(metadata.get("anchored_episode_count"))
                ),
            },
            "failure_taxonomy": dict(metadata.get("failure_taxonomy", {}) or {}),
            "artifact_evaluation_summary": dict(
                metadata.get("artifact_evaluation_summary", {}) or {}
            ),
            "calibration_policy_version": metadata.get("calibration_policy_version"),
            "promotion_gate_status": metadata.get("promotion_gate_status"),
            "promotion_gate_failures": list(
                metadata.get("promotion_gate_failures", []) or []
            ),
            "promotion_gate_metrics": dict(
                metadata.get("promotion_gate_metrics", {}) or {}
            ),
            "analyst_variant_gate_status": dict(
                metadata.get("analyst_variant_gate_status", {}) or {}
            ),
            "consistency_policy_version": metadata.get("consistency_policy_version"),
            "consistency_gate_status": metadata.get("consistency_gate_status"),
            "consistency_gate_failures": list(
                metadata.get("consistency_gate_failures", []) or []
            ),
            "recent_gate_history": list(metadata.get("recent_gate_history", []) or []),
            "is_eligible": bool(row.is_eligible),
            "qualifies_for_incentives": bool(
                metadata.get("qualifies_for_incentives", False)
            ),
            "rank": rank,
            "reward_rank": None,
            "serving_rank": None,
            "is_run_winner": is_run_winner,
            "is_serving_selected": False,
            "serving_selection_reason": None,
            "run_winner_deployment_id": None,
            "serving_selected_deployment_id": None,
        }

    # ------------------------------------------------------------------
    # Public scorecard query methods
    # ------------------------------------------------------------------

    def latest_scorecard_summary(
        self,
        session: Session,
        *,
        deployment_id: str | None = None,
        submission_id: str | None = None,
    ) -> dict[str, Any] | None:
        statement = select(DeploymentScoreRecord).order_by(
            DeploymentScoreRecord.created_at.desc()
        )
        if deployment_id is not None:
            statement = statement.where(
                DeploymentScoreRecord.deployment_id == deployment_id
            )
        if submission_id is not None:
            statement = statement.where(
                DeploymentScoreRecord.submission_id == submission_id
            )
        row = session.execute(statement.limit(1)).scalar_one_or_none()
        if row is None:
            return None
        return self._scorecard_payload(session, row=row)

    def submission_scorecards(
        self,
        session: Session,
        *,
        submission_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        rows = list(
            session.execute(
                select(DeploymentScoreRecord)
                .where(DeploymentScoreRecord.submission_id == submission_id)
                .order_by(DeploymentScoreRecord.created_at.desc())
                .limit(max(1, min(limit, 100)))
            ).scalars()
        )
        winner_keys = {
            (item.family_id, item.run_id, item.winner_submission_id)
            for item in session.execute(select(RunFamilyResult)).scalars()
            if item.winner_submission_id
        }
        return [
            self._scorecard_payload(
                session,
                row=row,
                is_run_winner=(row.family_id, row.run_id, row.submission_id) in winner_keys,
            )
            for row in rows
        ]

    def family_scorecards(
        self,
        session: Session,
        *,
        family_id: str,
        run_id: str,
    ) -> list[dict[str, Any]]:
        rows = self._ordered_run_score_records(session, run_id=run_id, family_id=family_id)
        winner = session.execute(
            select(RunFamilyResult)
            .where(RunFamilyResult.run_id == run_id)
            .where(RunFamilyResult.family_id == family_id)
            .limit(1)
        ).scalar_one_or_none()
        winner_submission_id = winner.winner_submission_id if winner is not None else None
        payloads = [
            self._scorecard_payload(
                session,
                row=row,
                rank=index,
                is_run_winner=(
                    winner_submission_id is not None
                    and row.submission_id == winner_submission_id
                ),
            )
            for index, row in enumerate(rows, start=1)
        ]
        current_release = self._owner.latest_published_release(session)
        current_release_meta = (
            dict(current_release.metadata_json or {}) if current_release is not None else {}
        )
        current_selected_by_family = dict(
            current_release_meta.get("serving_selected_deployment_id_by_family", {})
            or current_release_meta.get("selected_source_deployments", {})
            or {}
        )
        selection_reason_by_family = dict(
            current_release_meta.get("selection_reason_by_family", {}) or {}
        )
        run_winner_by_family = dict(
            current_release_meta.get("run_winner_deployment_id_by_family", {}) or {}
        )
        reward_ranks = {
            item["deployment_id"]: index
            for index, item in enumerate(
                sorted(
                    payloads,
                    key=lambda payload: (
                        float(payload.get("official_family_score") or 0.0),
                        float(payload.get("reliability_score") or 0.0),
                    ),
                    reverse=True,
                ),
                start=1,
            )
        }
        serving_ranks = {
            item["deployment_id"]: index
            for index, item in enumerate(
                sorted(
                    payloads,
                    key=lambda payload: (
                        float(payload.get("serving_selection_score") or 0.0),
                        float(payload.get("reliability_score") or 0.0),
                        float(payload.get("official_family_score") or 0.0),
                    ),
                    reverse=True,
                ),
                start=1,
            )
        }
        for payload in payloads:
            payload["reward_rank"] = reward_ranks.get(payload["deployment_id"])
            payload["serving_rank"] = serving_ranks.get(payload["deployment_id"])
            payload["run_winner_deployment_id"] = run_winner_by_family.get(
                family_id
            ) or (winner.winner_deployment_id if winner is not None else None)
            payload["serving_selected_deployment_id"] = current_selected_by_family.get(
                family_id
            )
            payload["is_serving_selected"] = (
                payload["serving_selected_deployment_id"] == payload["deployment_id"]
            )
            payload["serving_selection_reason"] = selection_reason_by_family.get(family_id)
        return payloads

    # ------------------------------------------------------------------
    # Family score summaries
    # ------------------------------------------------------------------

    def family_score_summary(
        self,
        session: Session,
        *,
        run_id: str,
        family_id: str,
    ) -> dict[str, Any]:
        rows = self._ordered_run_score_records(session, run_id=run_id, family_id=family_id)
        winner = session.execute(
            select(RunFamilyResult)
            .where(RunFamilyResult.run_id == run_id)
            .where(RunFamilyResult.family_id == family_id)
            .limit(1)
        ).scalar_one_or_none()
        top = rows[0] if rows else None
        top_meta = dict(top.metadata_json or {}) if top is not None else {}
        winner_meta = dict(winner.metadata_json or {}) if winner is not None else {}
        current_release = self._owner.latest_published_release(session)
        current_release_meta = (
            dict(current_release.metadata_json or {}) if current_release is not None else {}
        )
        top_runtime_contract_modes = (
            dict(
                top_meta.get("runtime_contract_modes")
                or top_meta.get("family_diagnostics", {}).get("runtime_contract_modes", {})
                or {}
            )
            if top is not None
            else {}
        )
        if not top_runtime_contract_modes and top is not None:
            top_runtime_contract_modes = self._runtime_contract_modes_for_miner(
                session,
                family_id=family_id,
                miner_hotkey=top.miner_hotkey,
                run_id=run_id,
            )
        return {
            "official_scoring_version": top_meta.get("official_scoring_version"),
            "scoring_policy_version": top_meta.get("scoring_policy_version"),
            "evaluation_path": top_meta.get("evaluation_path"),
            "top_runtime_contract_modes": top_runtime_contract_modes,
            "scored_miner_count": len(rows),
            "eligible_scored_miner_count": sum(1 for row in rows if row.is_eligible),
            "top_official_family_score": (
                float(
                    top_meta.get("official_family_score", top.raw_score) or top.raw_score
                )
                if top is not None
                else None
            ),
            "top_reliability_score": (
                max(
                    float(
                        self._scorecard_reward_metrics(session, row=row)["reliability_score"]
                        or 0.0
                    )
                    for row in rows
                )
                if rows
                else None
            ),
            "top_serving_selection_score": (
                max(
                    float(
                        self._scorecard_reward_metrics(session, row=row)[
                            "serving_selection_score"
                        ]
                        or 0.0
                    )
                    for row in rows
                )
                if rows
                else None
            ),
            "workflow_product_diagnostics": {
                "top_workflow_episode_count": (
                    max(
                        int(
                            (
                                self._scorecard_reward_metrics(session, row=row)
                                .get("workflow_product_diagnostics", {})
                                .get("workflow_episode_count")
                            )
                            or 0
                        )
                        for row in rows
                    )
                    if rows
                    else None
                ),
                "top_hidden_episode_count": (
                    max(
                        int(
                            (
                                self._scorecard_reward_metrics(session, row=row)
                                .get("workflow_product_diagnostics", {})
                                .get("hidden_episode_count")
                            )
                            or 0
                        )
                        for row in rows
                    )
                    if rows
                    else None
                ),
                "top_replay_episode_count": (
                    max(
                        int(
                            (
                                self._scorecard_reward_metrics(session, row=row)
                                .get("workflow_product_diagnostics", {})
                                .get("replay_episode_count")
                            )
                            or 0
                        )
                        for row in rows
                    )
                    if rows
                    else None
                ),
                "corpus_version": (
                    top_meta.get("corpus_version")
                    or top_meta.get("family_diagnostics", {}).get("corpus_version")
                ),
                "corpus_manifest_digest": (
                    top_meta.get("corpus_manifest_digest")
                    or top_meta.get("family_diagnostics", {}).get("corpus_manifest_digest")
                ),
                "workflow_composition_source": top_meta.get("workflow_composition_source"),
                "workflow_composition_revision": top_meta.get(
                    "workflow_composition_revision"
                ),
            },
            "analyst_variant_scores": (
                dict(top_meta.get("analyst_variant_scores", {}) or {})
                if family_id == "analyst"
                else {}
            ),
            "analyst_variant_gate_status": (
                dict(top_meta.get("analyst_variant_gate_status", {}) or {})
                if family_id == "analyst"
                else {}
            ),
            "analyst_track_scores": (
                dict(top_meta.get("analyst_track_scores", {}) or {})
                if family_id == "analyst"
                else {}
            ),
            "analyst_track_gate_status": (
                dict(top_meta.get("analyst_track_gate_status", {}) or {})
                if family_id == "analyst"
                else {}
            ),
            "analyst_protocol_status": (
                dict(top_meta.get("analyst_protocol_status", {}) or {})
                if family_id == "analyst"
                else {}
            ),
            "analyst_truth_metrics": (
                dict(top_meta.get("analyst_truth_metrics", {}) or {})
                if family_id == "analyst"
                else {}
            ),
            "analyst_trace_metrics": (
                dict(top_meta.get("analyst_trace_metrics", {}) or {})
                if family_id == "analyst"
                else {}
            ),
            "analyst_anti_gaming_metrics": (
                dict(top_meta.get("analyst_anti_gaming_metrics", {}) or {})
                if family_id == "analyst"
                else {}
            ),
            "analyst_failure_taxonomy": (
                dict(top_meta.get("analyst_failure_taxonomy", {}) or {})
                if family_id == "analyst"
                else {}
            ),
            "winner_submission_id": winner.winner_submission_id if winner is not None else None,
            "run_winner_deployment_id": (
                winner.winner_deployment_id if winner is not None else None
            ),
            "serving_selected_deployment_id": dict(
                current_release_meta.get(
                    "serving_selected_deployment_id_by_family", {}
                )
                or current_release_meta.get("selected_source_deployments", {})
                or {}
            ).get(family_id),
            "serving_selection_reason": dict(
                current_release_meta.get("selection_reason_by_family", {}) or {}
            ).get(family_id),
            "gate_passed_candidate_count": int(
                winner_meta.get("gate_passed_candidate_count", 0) or 0
            ),
            "consistency_passed_candidate_count": int(
                winner_meta.get("consistency_passed_candidate_count", 0) or 0
            ),
            "top_candidate_gate_status": winner_meta.get("top_candidate_gate_status"),
            "top_candidate_consistency_status": winner_meta.get(
                "top_candidate_consistency_status"
            ),
            "winner_gate_status": winner_meta.get("winner_gate_status"),
            "winner_gate_failures": list(winner_meta.get("winner_gate_failures", []) or []),
            "winner_consistency_status": winner_meta.get("winner_consistency_status"),
        }

    def run_score_summaries(
        self,
        session: Session,
        *,
        run_id: str,
    ) -> dict[str, dict[str, Any]]:
        return {
            family_id: self.family_score_summary(session, run_id=run_id, family_id=family_id)
            for family_id in PRODUCTION_FAMILIES
        }

    def run_family_score_summaries(
        self,
        session: Session,
        *,
        run_id: str,
    ) -> dict[str, dict[str, Any]]:
        return self.run_score_summaries(session, run_id=run_id)

    # ------------------------------------------------------------------
    # Previous winner lookup
    # ------------------------------------------------------------------

    def _previous_winner_result(
        self,
        session: Session,
        *,
        before_sequence: int,
        family_id: str,
    ) -> RunFamilyResult | None:
        candidates = list(
            session.execute(
                select(RunFamilyResult).where(
                    RunFamilyResult.family_id == family_id,
                    RunFamilyResult.has_winner.is_(True),
                )
            ).scalars()
        )
        if not candidates:
            return None
        eligible: list[tuple[int, RunFamilyResult]] = []
        for item in candidates:
            run = session.get(EvaluationRun, item.run_id)
            if run is None or run.sequence >= before_sequence:
                continue
            eligible.append((run.sequence, item))
        if not eligible:
            return None
        eligible.sort(key=lambda pair: pair[0], reverse=True)
        return eligible[0][1]

    # ------------------------------------------------------------------
    # Latest score record helpers
    # ------------------------------------------------------------------

    def _latest_score_record_for_family_miner(
        self,
        session: Session,
        *,
        family_id: str,
        miner_hotkey: str,
        run_ids: set[str] | None = None,
    ) -> DeploymentScoreRecord | None:
        family_id = ensure_family_id(family_id)
        statement = (
            select(DeploymentScoreRecord)
            .where(DeploymentScoreRecord.family_id == family_id)
            .where(DeploymentScoreRecord.miner_hotkey == miner_hotkey)
            .order_by(DeploymentScoreRecord.created_at.desc())
        )
        if run_ids:
            row = session.execute(
                statement.where(DeploymentScoreRecord.run_id.in_(set(run_ids))).limit(1)
            ).scalar_one_or_none()
            if row is not None:
                return row
        return session.execute(statement.limit(1)).scalar_one_or_none()

    def latest_quality_score(self, session: Session, *, deployment_id: str) -> float:
        latest = session.execute(
            select(DeploymentScoreRecord)
            .where(DeploymentScoreRecord.deployment_id == deployment_id)
            .order_by(DeploymentScoreRecord.created_at.desc())
            .limit(1)
        ).scalar_one_or_none()
        if latest is None:
            return 0.0
        return _score_record_official_family_score(latest)

    # ------------------------------------------------------------------
    # Aggregate snapshot helpers
    # ------------------------------------------------------------------

    def aggregate_snapshot_for_family(
        self,
        session: Session,
        *,
        run_id: str,
        family_id: str,
    ) -> AggregateFamilyScoreSnapshot | None:
        return session.execute(
            select(AggregateFamilyScoreSnapshot).where(
                AggregateFamilyScoreSnapshot.run_id == run_id,
                AggregateFamilyScoreSnapshot.family_id == family_id,
            )
        ).scalar_one_or_none()

    # ------------------------------------------------------------------
    # Aggregate status and chain publication
    # ------------------------------------------------------------------

    def aggregate_status_payload(
        self,
        session: Session,
        *,
        run_id: str,
        family_id: str,
    ) -> dict[str, Any]:
        from shared.common.models import TaskEvaluation

        family_id = ensure_family_id(family_id)
        contributing = list(
            session.execute(
                select(TaskEvaluation.claimed_by_validator)
                .where(TaskEvaluation.run_id == run_id)
                .where(TaskEvaluation.family_id == family_id)
                .where(TaskEvaluation.status == "evaluated")
                .where(TaskEvaluation.claimed_by_validator.isnot(None))
                .distinct()
            ).scalars()
        )
        aggregate = self.aggregate_snapshot_for_family(
            session,
            run_id=run_id,
            family_id=family_id,
        )
        summary = self.family_score_summary(session, run_id=run_id, family_id=family_id)
        return {
            "run_id": run_id,
            "family_id": family_id,
            "validator_submission_count": len(contributing),
            "required_quorum": 1,
            "quorum_reached": aggregate is not None and aggregate.status == "aggregated",
            "validators": contributing,
            "late_submission_count": 0,
            "status": aggregate.status if aggregate is not None else "pending",
            "aggregate_snapshot": aggregate.snapshot_json if aggregate is not None else None,
            **summary,
        }

    def chain_publication_readiness(
        self,
        session: Session,
        *,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        target_run = (
            session.get(EvaluationRun, run_id)
            if run_id is not None
            else self._owner.latest_completed_run(session)
        )
        blockers: list[dict[str, Any]] = []
        if target_run is None:
            blockers.append({"type": "run", "reason": "no_completed_run"})
            return {
                "ready": False,
                "run_id": None,
                "blockers": blockers,
            }
        for family_id in PRODUCTION_FAMILIES:
            aggregate = self.aggregate_snapshot_for_family(
                session,
                run_id=target_run.id,
                family_id=family_id,
            )
            if aggregate is None or aggregate.status != "aggregated":
                blockers.append(
                    {
                        "type": "aggregate",
                        "family_id": family_id,
                        "reason": "missing_authoritative_aggregate",
                    }
                )
                continue
            snapshot = FamilyScoreSnapshot.model_validate(aggregate.snapshot_json)
            if snapshot.evaluation_plane != "family_protocol":
                blockers.append(
                    {
                        "type": "aggregate",
                        "family_id": family_id,
                        "reason": "aggregate_not_family_protocol",
                        "evaluation_plane": snapshot.evaluation_plane,
                    }
                )
        return {
            "ready": not blockers,
            "run_id": target_run.id,
            "blockers": blockers,
        }

    def build_chain_weight_inputs(
        self,
        session: Session,
        *,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        readiness = self.chain_publication_readiness(session, run_id=run_id)
        target_run_id = readiness.get("run_id")
        if target_run_id is None:
            raise ValueError("chain publication has no completed run")
        families: list[dict[str, Any]] = []
        for family_id in PRODUCTION_FAMILIES:
            aggregate = self.aggregate_snapshot_for_family(
                session,
                run_id=target_run_id,
                family_id=family_id,
            )
            if aggregate is None or not isinstance(aggregate.snapshot_json, dict):
                raise ValueError(f"missing aggregate snapshot for {family_id}")
            snapshot = FamilyScoreSnapshot.model_validate(aggregate.snapshot_json)
            family_weight = fixed_family_weight(snapshot.family_id)
            built = {
                "run_id": snapshot.run_id,
                "family_id": snapshot.family_id,
                "evaluation_plane": snapshot.evaluation_plane,
                "family_weight": family_weight,
                "weights": snapshot.normalized_weights,
                "scaled_weights": {
                    hotkey: weight * family_weight
                    for hotkey, weight in snapshot.normalized_weights.items()
                },
                "query_volume_share": float(snapshot.query_volume_share or 0.0),
                "rubric_version": snapshot.rubric_version,
                "allocation_mode": "fixed_family_weights_v1",
            }
            families.append(
                {
                    "family_id": family_id,
                    "snapshot": snapshot.model_dump(mode="json"),
                    "evaluation_plane": built["evaluation_plane"],
                    "family_weight": built["family_weight"],
                    "allocation_mode": built["allocation_mode"],
                    "weights": built["weights"],
                    "scaled_weights": built["scaled_weights"],
                    "rubric_version": built["rubric_version"],
                }
            )
        return {
            "run_id": target_run_id,
            "allocation_mode": "fixed_family_weights_v1",
            "families": families,
            "readiness": readiness,
        }

    # ------------------------------------------------------------------
    # Cost accounting helpers
    # ------------------------------------------------------------------

    def fetch_deployment_cost(self, deployment_id: str) -> dict[str, Any]:
        proxy_url = self.settings.provider_proxy_url
        if not proxy_url:
            logger.warning("provider_proxy_url not configured; cost data unavailable")
            return {}
        # Key must match the ``X-Eirel-Job-Id`` the miner SDK sends, which
        # is set by ``infra.miner_runtime.runtime_manager`` via
        # ``EIREL_PROVIDER_PROXY_JOB_ID=miner-<deployment_id>``.  Prefixing
        # with ``miner-`` here keeps both sides aligned.
        url = f"{proxy_url.rstrip('/')}/v1/jobs/miner-{deployment_id}/cost"
        headers: dict[str, str] = {}
        token = getattr(self.settings, "provider_proxy_token", None)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            resp = httpx.get(url, headers=headers, timeout=10.0)
            resp.raise_for_status()
            return dict(resp.json())
        except Exception as exc:
            logger.warning("failed to fetch cost for deployment %s: %s", deployment_id, exc)
            return {}

    def charge_trace_gate_penalty(
        self,
        deployment_id: str,
        *,
        amount_usd: float,
        reason: str = "trace_gate_fail",
    ) -> bool:
        """Debit a USD penalty against a deployment's run budget.

        Called once per conversation that fails the trace integrity gate.
        Returns True on success, False on any failure (logged — we don't
        want penalty charging to break the scoring pipeline). The penalty
        lands unconditionally, even if it overshoots ``max_usd_budget`` —
        that's the whole point of the economic gate.
        """
        if amount_usd <= 0.0:
            return False
        proxy_url = self.settings.provider_proxy_url
        if not proxy_url:
            logger.warning(
                "provider_proxy_url not configured; cannot charge trace-gate penalty",
            )
            return False
        url = f"{proxy_url.rstrip('/')}/v1/jobs/miner-{deployment_id}/charge_penalty"
        headers: dict[str, str] = {}
        token = getattr(self.settings, "provider_proxy_token", None)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            resp = httpx.post(
                url,
                json={"reason": reason, "amount_usd": float(amount_usd)},
                headers=headers,
                timeout=10.0,
            )
            resp.raise_for_status()
            logger.info(
                "charged trace-gate penalty: deployment=%s amount_usd=%.4f reason=%s",
                deployment_id, amount_usd, reason,
            )
            return True
        except Exception as exc:
            logger.warning(
                "failed to charge trace-gate penalty for deployment %s: %s",
                deployment_id, exc,
            )
            return False

    def populate_cost_columns(
        self,
        record: DeploymentScoreRecord,
        deployment_id: str,
        run_budget_usd: float,
    ) -> None:
        cost_data = self.fetch_deployment_cost(deployment_id)
        record.run_budget_usd = run_budget_usd
        record.run_cost_usd_used = float(cost_data.get("cost_usd_used", 0.0))
        record.llm_cost_usd = float(cost_data.get("llm_cost_usd", 0.0))
        record.tool_cost_usd = float(cost_data.get("tool_cost_usd", 0.0))
        record.cost_rejection_count = int(cost_data.get("cost_rejections", 0))

