from __future__ import annotations

"""Run lifecycle management, evaluation bundle construction, and artifact persistence.

Extracted from ``ManagedOwnerServices`` to reduce the size of the god-object.
Each public method here has a thin delegation wrapper in ``ManagedOwnerServices``
for backward compatibility.
"""

import hashlib
import json
import logging
from datetime import timedelta
from pathlib import Path
from typing import Any, TYPE_CHECKING

logger = logging.getLogger(__name__)

from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from shared.common.artifacts import ArtifactStoreError
from shared.common.models import (
    AggregateFamilyScoreSnapshot,
    EpochTargetSnapshot,
    EvaluationRun,
    ManagedArtifact,
    ManagedDeployment,
    ManagedMinerSubmission,
    TaskEvaluation,
    RunFamilyResult,
)
from shared.contracts.models import BenchmarkRunRecord
from shared.contracts.specialist_contracts import contract_for_family
from shared.core.evaluation_models import FamilyEvaluationBundle
from eirel.groups import ensure_family_id

if TYPE_CHECKING:
    from control_plane.owner_api.managed import ManagedOwnerServices


from control_plane.owner_api._constants import PRODUCTION_FAMILIES, WINNER_DOMINANCE_MARGIN
from control_plane.owner_api._helpers import (
    _default_allowed_tool_policy_for_bundle,
    _evaluation_policy_payload,
    _live_research_retrieval_environment_payload,
    _load_owner_evaluation_bundle_seed,
    _score_record_official_family_score,
    _score_record_selection_score,
    utcnow as _utcnow_fn,
)
from control_plane.owner_api.dataset_loader import (
    EvalPoolConfigError,
    EvalPoolFetchError,
    LoaderResult,
    load_owner_evaluation_bundle,
)


def _utcnow():
    return _utcnow_fn()


def _winner_dominance_margin() -> float:
    return WINNER_DOMINANCE_MARGIN


class RunManager:
    """Handles run lifecycle, evaluation bundle construction, and artifact persistence."""

    def __init__(self, owner: ManagedOwnerServices) -> None:
        self._owner = owner
        self._last_closed_run_retired_ids: list[str] = []
        self._pending_queued_deployment_ids: list[str] = []

    @property
    def db(self):
        return self._owner.db

    @property
    def settings(self):
        return self._owner.settings

    # ------------------------------------------------------------------
    # Run queries
    # ------------------------------------------------------------------

    def current_run(self, session: Session) -> EvaluationRun | None:
        return session.execute(
            select(EvaluationRun)
            .where(EvaluationRun.status == "open")
            .order_by(EvaluationRun.sequence.desc())
            .limit(1)
        ).scalar_one_or_none()

    def latest_completed_run(self, session: Session) -> EvaluationRun | None:
        return session.execute(
            select(EvaluationRun)
            .where(EvaluationRun.status == "completed")
            .order_by(EvaluationRun.sequence.desc())
            .limit(1)
        ).scalar_one_or_none()

    def next_scheduled_run(self, session: Session) -> EvaluationRun | None:
        return session.execute(
            select(EvaluationRun)
            .where(EvaluationRun.status == "scheduled")
            .order_by(EvaluationRun.sequence.asc())
            .limit(1)
        ).scalar_one_or_none()

    def list_runs(self, session: Session) -> list[EvaluationRun]:
        return list(
            session.execute(
                select(EvaluationRun).order_by(EvaluationRun.sequence.desc())
            ).scalars()
        )

    # ------------------------------------------------------------------
    # Run creation
    # ------------------------------------------------------------------

    def create_run(
        self,
        session: Session,
        *,
        sequence: int,
        started_at: Any = None,
        status: str = "open",
    ) -> EvaluationRun:
        started = started_at or _utcnow()
        run = EvaluationRun(
            id=f"run-{sequence}",
            sequence=sequence,
            status=status,
            benchmark_version=self._owner.benchmark_version,
            rubric_version=self._owner.rubric_version,
            judge_model=self._owner.judge_model,
            min_scores_json=self._owner.run_min_scores(),
            started_at=started,
            ends_at=started + timedelta(days=self._owner.run_duration_days),
            metadata_json={},
        )
        session.add(run)
        session.flush()
        if status == "open":
            self._initialize_run_benchmarks(session, run=run)
            self.start_queued_deployments(session, run=run)
        return run

    def _initialize_run_benchmarks(self, session: Session, *, run: EvaluationRun) -> None:
        for family_id in PRODUCTION_FAMILIES:
            self.ensure_run_evaluation_bundle(session, run=run, family_id=family_id)

    def start_queued_deployments(self, session: Session, *, run: EvaluationRun) -> list[str]:
        submissions = list(
            session.execute(
                select(ManagedMinerSubmission)
                .where(ManagedMinerSubmission.introduced_run_id == run.id)
            ).scalars()
        )
        now = _utcnow()
        deployment_ids: list[str] = []
        touched_families: set[str] = set()
        for submission in submissions:
            deployment = self._owner.get_deployment_for_submission(session, submission.id)
            if deployment is None or deployment.placement_status != "queued":
                continue
            deployment.status = "received"
            deployment.health_status = "starting"
            deployment.placement_status = "pending"
            deployment.health_details_json = {
                **deployment.health_details_json,
                "build": "pending",
                "deploy": "pending",
            }
            deployment.updated_at = now
            deployment_ids.append(deployment.id)
            touched_families.add(deployment.family_id)
        session.flush()
        for family_id in touched_families:
            self._owner.rebalance_family(session, family_id=family_id)
        if deployment_ids:
            self._pending_queued_deployment_ids.extend(deployment_ids)
        logger.info(
            "started %d queued deployments for run %s",
            len(deployment_ids), run.id,
        )
        return deployment_ids

    # ------------------------------------------------------------------
    # Evaluation bundle construction & persistence
    # ------------------------------------------------------------------

    def _load_evaluation_bundle_seed(
        self,
        *,
        family_id: str,
        run_id: str,
    ) -> FamilyEvaluationBundle:
        """Resolve the evaluation bundle for a run from R2.

        Convention-based: the bundle URI is derived as
        ``s3://${EIREL_EVAL_POOL_BUCKET}/${family_id}/pool-run-${run_id}.json``.
        ``ObjectStore`` is R2-aware via ``EIREL_R2_*`` env vars.

        A failed fetch raises — runs must never open against stale
        or absent data.
        """
        if self._owner.object_store is None:
            raise RuntimeError(
                "evaluation bundle requires an ObjectStore; "
                "owner-api was started without one"
            )
        result = load_owner_evaluation_bundle(
            family_id=family_id,
            run_id=run_id,
            object_store=self._owner.object_store,
            cache_dir=getattr(self.settings, "owner_dataset_cache_dir", None),
        )
        logger.info(
            "loaded evaluation bundle family=%s run_id=%s uri=%s cache_hit=%s bytes=%d",
            family_id, run_id, result.bundle_uri, result.cache_hit, result.bytes_fetched,
        )
        return result.bundle

    def _build_run_evaluation_bundle(
        self,
        *,
        run: EvaluationRun,
        family_id: str,
    ) -> FamilyEvaluationBundle:
        family_id = ensure_family_id(family_id)
        policy = _evaluation_policy_payload(family_id)
        seed = self._load_evaluation_bundle_seed(
            family_id=family_id,
            run_id=str(run.sequence),
        )
        retrieval_environment = (
            _live_research_retrieval_environment_payload(self.settings)
            if any(str(task.execution_mode or "").strip() == "live_web" for task in seed.tasks)
            else (dict(seed.retrieval_environment or {}) if isinstance(seed.retrieval_environment, dict) else None)
        )
        tasks = []
        for task in seed.tasks:
            task_inputs = dict(task.inputs or {})
            if retrieval_environment is not None and str(task.execution_mode or "").strip() == "live_web":
                task_inputs["retrieval_environment"] = dict(retrieval_environment)
            task_metadata = {
                **dict(task.metadata or {}),
                "owner_frozen": True,
                "run_id": run.id,
                "benchmark_version": policy["benchmark_version"],
                "rubric_version": policy["rubric_version"],
                "evaluation_track": (task.metadata or {}).get("evaluation_track"),
            }
            tasks.append(
                task.model_copy(
                    update={
                        "inputs": task_inputs,
                        "metadata": task_metadata,
                    }
                )
            )
        prepared_seed = seed.model_copy(
            update={
                "tasks": tasks,
                "retrieval_environment": retrieval_environment,
                "allowed_tool_policy": _default_allowed_tool_policy_for_bundle(seed),
                "policy_version": str(policy["scoring_policy_version"]),
            }
        )
        bundle = prepared_seed.model_copy(
            update={
                "run_id": run.id,
                "family_id": family_id,
                "benchmark_version": str(policy["benchmark_version"]),
                "rubric_version": str(policy["rubric_version"]),
            }
        )
        return FamilyEvaluationBundle.model_validate(bundle.model_dump(mode="json"))

    def ensure_run_evaluation_bundle(
        self,
        session: Session,
        *,
        run: EvaluationRun,
        family_id: str,
    ) -> dict[str, Any]:
        family_id = ensure_family_id(family_id)
        metadata = dict(run.metadata_json or {})
        evaluation_bundles = dict(metadata.get("evaluation_bundles", {}) or {})
        bundle_payload = evaluation_bundles.get(family_id)
        bundle_metadata = dict(bundle_payload.get("metadata", {}) or {}) if isinstance(bundle_payload, dict) else {}
        if not isinstance(bundle_payload, dict) or not isinstance(bundle_metadata.get("dataset_generator"), dict):
            bundle_payload = self._build_run_evaluation_bundle(
                run=run,
                family_id=family_id,
            ).model_dump(mode="json")
            evaluation_bundles[family_id] = bundle_payload
            metadata["evaluation_bundles"] = evaluation_bundles
            # Index any RAG corpora into the rag-tool-service.
            # Best-effort: log on failure, let the run continue. Fired
            # only on the first-build (cache-miss) branch so we don't
            # re-index every claim.
            corpora_payload = bundle_payload.get("corpora") or []
            if corpora_payload:
                from control_plane.owner_api.evaluation.corpus_indexer import (
                    index_bundle_corpora,
                )
                indexed = index_bundle_corpora(corpora_payload)
                logger.info(
                    "rag indexing summary family=%s run=%s indexed=%d corpora",
                    family_id, run.id, len(indexed),
                )
        tasks = [item for item in bundle_payload.get("tasks", []) if isinstance(item, dict)]
        summary_payload = {
            family_id: {
                "kind": bundle_payload.get("kind"),
                "task_count": len(tasks),
                "benchmark_version": bundle_payload.get("benchmark_version", run.benchmark_version),
                "rubric_version": bundle_payload.get("rubric_version", run.rubric_version),
                "categories": sorted(
                    {
                        str((item.get("metadata") or {}).get("category") or item.get("category"))
                        for item in tasks
                        if (item.get("metadata") or {}).get("category") is not None
                        or item.get("category") is not None
                    }
                ),
                "difficulty_levels": sorted(
                    {
                        str((item.get("metadata") or {}).get("difficulty") or item.get("difficulty"))
                        for item in tasks
                        if (item.get("metadata") or {}).get("difficulty") is not None
                        or item.get("difficulty") is not None
                    }
                ),
                "task_families": sorted(
                    {
                        str((item.get("metadata") or {}).get("task_family") or (item.get("expected_output") or {}).get("task_family"))
                        for item in tasks
                        if (item.get("metadata") or {}).get("task_family") is not None
                        or (item.get("expected_output") or {}).get("task_family") is not None
                    }
                ),
                "evaluation_tracks": sorted(
                    {
                        str(item.get("evaluation_track") or (item.get("metadata") or {}).get("evaluation_track"))
                        for item in tasks
                        if item.get("evaluation_track") is not None
                        or (item.get("metadata") or {}).get("evaluation_track") is not None
                    }
                ),
            },
        }
        metadata["evaluation_bundle_summaries"] = {
            **dict(metadata.get("evaluation_bundle_summaries", {}) or {}),
            **summary_payload,
        }
        target_artifact = self._ensure_run_json_artifact(
            session,
            run=run,
            family_id=family_id,
            artifact_kind="evaluation_bundle",
            payload=bundle_payload,
            metadata_json={
                "run_id": run.id,
                "benchmark_version": bundle_payload.get("benchmark_version", run.benchmark_version),
                "target_kind": bundle_payload.get("kind"),
            },
        )
        metadata["evaluation_bundle_artifacts"] = {
            **dict(metadata.get("evaluation_bundle_artifacts", {}) or {}),
            family_id: target_artifact,
        }
        if isinstance(bundle_payload.get("retrieval_environment"), dict):
            metadata["family_retrieval_environments"] = {
                **dict(metadata.get("family_retrieval_environments", {}) or {}),
                family_id: dict(bundle_payload.get("retrieval_environment") or {}),
            }
        run.metadata_json = metadata
        run.updated_at = _utcnow()
        session.commit()
        session.refresh(run)
        return bundle_payload

    def run_evaluation_bundle(
        self,
        session: Session,
        *,
        run_id: str,
        family_id: str,
    ) -> dict[str, Any] | None:
        family_id = ensure_family_id(family_id)
        run = session.get(EvaluationRun, run_id)
        if run is None:
            return None
        return self.ensure_run_evaluation_bundle(session, run=run, family_id=family_id)

    def run_evaluation_bundle_artifact(
        self,
        session: Session,
        *,
        run_id: str,
        family_id: str,
    ) -> dict[str, Any] | None:
        family_id = ensure_family_id(family_id)
        run = session.get(EvaluationRun, run_id)
        if run is None:
            return None
        self.ensure_run_evaluation_bundle(session, run=run, family_id=family_id)
        return self._artifact_ref_for_kind(
            run=run,
            family_id=family_id,
            artifact_kind="evaluation_bundle",
        )

    def run_live_research_retrieval_environment(
        self,
        session: Session,
        *,
        run_id: str,
    ) -> dict[str, Any] | None:
        run = session.get(EvaluationRun, run_id)
        if run is None:
            return None
        bundle = self.ensure_run_evaluation_bundle(session, run=run, family_id="analyst")
        retrieval_environment = bundle.get("retrieval_environment")
        return dict(retrieval_environment or {}) if isinstance(retrieval_environment, dict) else None

    # ------------------------------------------------------------------
    # Artifact helpers
    # ------------------------------------------------------------------

    def _ensure_run_json_artifact(
        self,
        session: Session,
        *,
        run: EvaluationRun,
        family_id: str,
        artifact_kind: str,
        payload: dict[str, Any],
        metadata_json: dict[str, Any],
    ) -> dict[str, Any]:
        payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(payload_bytes).hexdigest()
        known_ref = self._artifact_ref_for_kind(run=run, family_id=family_id, artifact_kind=artifact_kind)
        if isinstance(known_ref, dict) and known_ref.get("sha256") == digest:
            artifact = session.get(ManagedArtifact, known_ref.get("artifact_id")) if known_ref.get("artifact_id") else None
            if artifact is not None:
                return known_ref
        artifact = session.execute(
            select(ManagedArtifact)
            .where(ManagedArtifact.family_id == family_id)
            .where(ManagedArtifact.artifact_kind == artifact_kind)
            .where(ManagedArtifact.retained_for_run_id == run.id)
            .where(ManagedArtifact.sha256 == digest)
            .limit(1)
        ).scalar_one_or_none()
        if artifact is None:
            storage_key = f"{family_id}/{artifact_kind}/{run.id}/{digest}.json"
            try:
                stored = self._owner.artifact_store.put_bytes(storage_key=storage_key, content=payload_bytes)
            except ArtifactStoreError as exc:
                raise RuntimeError(f"failed to persist {artifact_kind} artifact: {exc}") from exc
            artifact = ManagedArtifact(
                deployment_id=None,
                submission_id=None,
                family_id=family_id,
                artifact_kind=artifact_kind,
                storage_key=stored.storage_key,
                storage_uri=stored.storage_uri,
                mime_type="application/json",
                sha256=stored.sha256,
                size_bytes=stored.size_bytes,
                metadata_json=metadata_json,
                retained_for_run_id=run.id,
            )
            session.add(artifact)
            session.flush()
        return self._artifact_ref_payload(artifact)

    def _artifact_ref_payload(self, artifact: ManagedArtifact) -> dict[str, Any]:
        return {
            "artifact_id": artifact.id,
            "sha256": artifact.sha256,
            "storage_uri": artifact.storage_uri,
            "size_bytes": artifact.size_bytes,
            "mime_type": artifact.mime_type,
            "download_path": f"/v1/internal/artifacts/{artifact.id}",
        }

    def _artifact_ref_for_kind(self, *, run: EvaluationRun, family_id: str, artifact_kind: str) -> dict[str, Any] | None:
        metadata = dict(run.metadata_json or {})
        ref_map_key = {
            "evaluation_bundle": "evaluation_bundle_artifacts",
            "calibration_report": "family_calibration_artifacts",
        }.get(artifact_kind)
        if ref_map_key is None:
            return None
        refs = dict(metadata.get(ref_map_key, {}) or {})
        ref = refs.get(family_id)
        return ref if isinstance(ref, dict) else None

    def _persist_retrieval_ledger_artifacts(
        self,
        session: Session,
        *,
        run_id: str,
        validator_hotkey: str,
        benchmark_run: BenchmarkRunRecord | None,
    ) -> dict[str, dict[str, Any]]:
        if benchmark_run is None:
            return {}
        artifacts_by_miner: dict[str, dict[str, Any]] = {}
        for miner_hotkey, responses in (benchmark_run.miner_responses or {}).items():
            task_artifacts: dict[str, Any] = {}
            for response in responses if isinstance(responses, list) else []:
                if not isinstance(response, dict):
                    continue
                task_id = str(response.get("task_id") or "").strip()
                metadata = response.get("metadata")
                if not task_id or not isinstance(metadata, dict):
                    continue
                ledger = metadata.get("retrieval_ledger")
                if not isinstance(ledger, dict) or not ledger:
                    continue
                payload_bytes = json.dumps(ledger, sort_keys=True, separators=(",", ":")).encode()
                digest = hashlib.sha256(payload_bytes).hexdigest()
                artifact = session.execute(
                    select(ManagedArtifact)
                    .where(ManagedArtifact.family_id == "analyst")
                    .where(ManagedArtifact.artifact_kind == "retrieval_ledger")
                    .where(ManagedArtifact.retained_for_run_id == run_id)
                    .where(ManagedArtifact.sha256 == digest)
                    .limit(1)
                ).scalar_one_or_none()
                if artifact is None:
                    storage_key = f"analyst/retrieval_ledger/{run_id}/{validator_hotkey}/{miner_hotkey}/{task_id}/{digest}.json"
                    try:
                        stored = self._owner.artifact_store.put_bytes(storage_key=storage_key, content=payload_bytes)
                    except ArtifactStoreError as exc:
                        raise RuntimeError(f"failed to persist retrieval ledger artifact: {exc}") from exc
                    artifact = ManagedArtifact(
                        deployment_id=None,
                        submission_id=None,
                        family_id="analyst",
                        artifact_kind="retrieval_ledger",
                        storage_key=stored.storage_key,
                        storage_uri=stored.storage_uri,
                        mime_type="application/json",
                        sha256=stored.sha256,
                        size_bytes=stored.size_bytes,
                        metadata_json={
                            "run_id": run_id,
                            "task_id": task_id,
                            "miner_hotkey": miner_hotkey,
                            "validator_hotkey": validator_hotkey,
                            "retrieval_ledger_id": ledger.get("retrieval_ledger_id"),
                        },
                        retained_for_run_id=run_id,
                    )
                    session.add(artifact)
                    session.flush()
                task_artifacts[task_id] = self._artifact_ref_payload(artifact)
            if task_artifacts:
                artifacts_by_miner[str(miner_hotkey)] = task_artifacts
        return artifacts_by_miner

    # ------------------------------------------------------------------
    # Run closure
    # ------------------------------------------------------------------

    def close_run(self, session: Session, *, run_id: str) -> EvaluationRun:
        run = session.get(EvaluationRun, run_id)
        if run is None:
            raise ValueError("run not found")
        if run.status == "completed":
            return run

        # Expire any remaining pending/claimed evaluation tasks for this run
        now = _utcnow()
        session.execute(
            update(TaskEvaluation)
            .where(TaskEvaluation.run_id == run_id)
            .where(TaskEvaluation.status.in_(("pending", "claimed")))
            .values(status="expired", updated_at=now)
        )
        session.flush()

        for family_id in PRODUCTION_FAMILIES:
            self._owner.evaluation_tasks.finalize_run_family(
                session, run_id=run_id, family_id=family_id,
            )

        session.execute(delete(RunFamilyResult).where(RunFamilyResult.run_id == run_id))
        for family_id in PRODUCTION_FAMILIES:
            aggregate = self._owner.aggregate_snapshot_for_family(session, run_id=run_id, family_id=family_id)
            min_score = float(run.min_scores_json.get(family_id, 0.0))
            top_deployment_ids = self._owner._top_run_deployment_ids(
                session,
                run_id=run_id,
                family_id=family_id,
                limit=self._owner.run_top_carryover_per_family,
            )
            winner_deployment_id: str | None = None
            winner_submission_id: str | None = None
            winner_hotkey: str | None = None
            best_raw_score = 0.0
            has_winner = False
            metadata: dict[str, Any] = {}
            if aggregate is not None and isinstance(aggregate.snapshot_json, dict):
                ranked_records = self._owner._ordered_run_score_records(
                    session,
                    run_id=run_id,
                    family_id=family_id,
                )
                metadata["validator_count"] = aggregate.validator_count
                metadata["consensus_method"] = aggregate.consensus_method
                metadata["margin_threshold"] = _winner_dominance_margin()
                if ranked_records:
                    best_raw_score = float(ranked_records[0].raw_score)
                    best_official_family_score = max(
                        _score_record_official_family_score(row)
                        for row in ranked_records
                    )
                    metadata["best_official_family_score"] = best_official_family_score
                    previous_winner = self._owner._previous_winner_result(
                        session,
                        before_sequence=run.sequence,
                        family_id=family_id,
                    )
                    selected_record = None
                    if previous_winner is None:
                        selected_record = ranked_records[0]
                        metadata["winner_selection"] = "first_run_best_score"
                    else:
                        baseline_record = next(
                            (row for row in ranked_records if row.deployment_id == previous_winner.winner_deployment_id),
                            None,
                        )
                        metadata["previous_winner_run_id"] = previous_winner.run_id
                        metadata["previous_winner_deployment_id"] = previous_winner.winner_deployment_id
                        metadata["previous_winner_submission_id"] = previous_winner.winner_submission_id
                        metadata["previous_winner_hotkey"] = previous_winner.winner_hotkey
                        if baseline_record is None:
                            selected_record = ranked_records[0]
                            metadata["winner_selection"] = "fallback_first_run_logic"
                            metadata["fallback_reason"] = "previous_winner_missing_from_current_run"
                        else:
                            baseline_score = float(baseline_record.raw_score)
                            baseline_official_family_score = _score_record_official_family_score(
                                baseline_record
                            )
                            baseline_selection_score = _score_record_selection_score(baseline_record)
                            metadata["previous_winner_baseline_score"] = baseline_score
                            metadata["previous_winner_official_family_score"] = baseline_official_family_score
                            challengers = [
                                row
                                for row in ranked_records
                                if row.deployment_id != baseline_record.deployment_id
                                and _score_record_selection_score(row)
                                >= baseline_selection_score + _winner_dominance_margin()
                            ]
                            metadata["dominance_threshold_met"] = bool(challengers)
                            if challengers:
                                selected_record = challengers[0]
                                metadata["winner_selection"] = "challenger_replaced_previous"
                            else:
                                selected_record = baseline_record
                                metadata["winner_selection"] = "previous_winner_retained"
                    if selected_record is not None:
                        selected_meta = dict(selected_record.metadata_json or {})
                        winner_deployment_id = selected_record.deployment_id
                        winner_submission_id = selected_record.submission_id
                        winner_hotkey = selected_record.miner_hotkey
                        has_winner = True
                        metadata["winner_raw_score"] = float(selected_record.raw_score)
                        metadata["winner_official_family_score"] = _score_record_official_family_score(
                            selected_record
                        )
                        metadata["promotion_recommendation"] = metadata.get("promotion_recommendation", "promote")
                        metadata["rollback_candidate_deployment_id"] = previous_winner.winner_deployment_id if previous_winner is not None else None
                        metadata["rollback_candidate_submission_id"] = previous_winner.winner_submission_id if previous_winner is not None else None
                        metadata["rollback_candidate_hotkey"] = previous_winner.winner_hotkey if previous_winner is not None else None
                        metadata["qualifies_for_incentives"] = (
                            float(selected_record.raw_score) >= min_score
                        )
                try:
                    metadata["family_contract"] = contract_for_family(family_id)
                except KeyError:
                    pass
            session.add(
                RunFamilyResult(
                    run_id=run_id,
                    family_id=family_id,
                    winner_deployment_id=winner_deployment_id if has_winner else None,
                    winner_submission_id=winner_submission_id if has_winner else None,
                    winner_hotkey=winner_hotkey if has_winner else None,
                    best_raw_score=float(best_raw_score),
                    min_score=min_score,
                    has_winner=has_winner,
                    top_deployment_ids_json=top_deployment_ids,
                    metadata_json=metadata,
                )
            )
        # Retire non-top deployments that were evaluated in this run but
        # did NOT finish in the top-N.  Submissions introduced DURING this
        # run are excluded — they haven't been evaluated yet and will
        # participate in the next run (N+1 model).
        keep_ids: set[str] = set()
        for family_id in PRODUCTION_FAMILIES:
            result = session.execute(
                select(RunFamilyResult)
                .where(RunFamilyResult.run_id == run_id)
                .where(RunFamilyResult.family_id == family_id)
            ).scalar_one_or_none()
            if result is not None:
                keep_ids.update(result.top_deployment_ids_json or [])

        # New submissions introduced during this run → keep their deployments
        for sub in session.execute(
            select(ManagedMinerSubmission)
            .where(ManagedMinerSubmission.introduced_run_id == run_id)
        ).scalars():
            dep = self._owner.get_deployment_for_submission(session, sub.id)
            if dep is not None and dep.status != "retired":
                keep_ids.add(dep.id)

        retired_deployment_ids: list[str] = []
        if keep_ids:
            evaluated_snapshots = list(session.execute(
                select(EpochTargetSnapshot)
                .where(EpochTargetSnapshot.run_id == run_id)
            ).scalars())
            for snap in evaluated_snapshots:
                for member in (snap.members_json or []):
                    dep_id = str((member.get("metadata") or {}).get("deployment_id", ""))
                    if dep_id and dep_id not in keep_ids:
                        dep = session.get(ManagedDeployment, dep_id)
                        if dep is not None and dep.status not in ("retired", "draining"):
                            dep.status = "retired"
                            dep.health_status = "retired"
                            dep.retired_at = now
                            dep.updated_at = now
                            retired_deployment_ids.append(dep_id)
                            logger.info(
                                "retiring non-top deployment: run=%s family=%s deployment=%s miner=%s",
                                run_id, snap.family_id, dep_id, dep.miner_hotkey,
                            )

        # Also retire any non-top deployments introduced in this run that
        # were not captured via EpochTargetSnapshot members above.
        all_run_deployments = list(
            session.execute(
                select(ManagedDeployment)
                .join(
                    ManagedMinerSubmission,
                    ManagedDeployment.submission_id == ManagedMinerSubmission.id,
                )
                .where(ManagedMinerSubmission.introduced_run_id == run_id)
                .where(ManagedDeployment.status.notin_(("retired", "queued")))
            ).scalars()
        )
        for dep in all_run_deployments:
            if dep.id not in keep_ids and dep.id not in {d for d in retired_deployment_ids}:
                dep.status = "retired"
                dep.health_status = "retired"
                dep.retired_at = now
                dep.updated_at = now
                retired_deployment_ids.append(dep.id)
                logger.info(
                    "retiring non-survivor deployment at run close: run=%s deployment=%s miner=%s",
                    run_id, dep.id, dep.miner_hotkey,
                )

        # Store retired IDs so callers can stop their containers
        self._last_closed_run_retired_ids = retired_deployment_ids

        run.status = "completed"
        run.closed_at = _utcnow()
        run.updated_at = _utcnow()
        session.flush()
        return run

    # ------------------------------------------------------------------
    # Ensure current / target run
    # ------------------------------------------------------------------

    def ensure_current_run(self, session: Session) -> EvaluationRun:
        current = self.current_run(session)
        now = _utcnow()
        if current is None:
            scheduled = self.next_scheduled_run(session)
            if scheduled is not None:
                if scheduled.started_at <= now:
                    # Scheduled run is ready to open
                    scheduled.status = "open"
                    scheduled.updated_at = now
                    self._initialize_run_benchmarks(session, run=scheduled)
                    self.start_queued_deployments(session, run=scheduled)
                    session.commit()
                    session.refresh(scheduled)
                    return scheduled
                # Scheduled run exists but not yet time — return as-is
                return scheduled
            # No run exists at all: create the first one
            first_run_start = self._owner.settings.first_run_start_time
            if first_run_start:
                from datetime import datetime as dt
                try:
                    start_time = dt.fromisoformat(first_run_start).replace(tzinfo=None)
                except ValueError:
                    start_time = now
                if start_time > now:
                    # Schedule the first run for the future
                    current = self.create_run(
                        session, sequence=1, started_at=start_time, status="scheduled",
                    )
                    session.commit()
                    session.refresh(current)
                    return current
                current = self.create_run(session, sequence=1, started_at=start_time)
            else:
                current = self.create_run(session, sequence=1, started_at=now)
            session.commit()
            session.refresh(current)
            return current
        if current.ends_at <= now:
            self.close_run(session, run_id=current.id)
            next_run = self.next_scheduled_run(session)
            if next_run is None or next_run.sequence <= current.sequence:
                next_run = self.create_run(
                    session,
                    sequence=current.sequence + 1,
                    started_at=current.ends_at,
                    status="scheduled",
                )
            next_run.status = "open"
            next_run.started_at = current.ends_at
            next_run.ends_at = next_run.started_at + timedelta(days=self._owner.run_duration_days)
            next_run.updated_at = now
            self._initialize_run_benchmarks(session, run=next_run)
            self.start_queued_deployments(session, run=next_run)
            session.commit()
            session.refresh(next_run)
            return next_run
        return current

    def submission_target_run(self, session: Session) -> EvaluationRun:
        current = self.ensure_current_run(session)
        if current.status != "open":
            # Before first run starts (status='scheduled'), target that
            # run directly. Miners submit before the announced start time.
            return current
        # During any open run, submissions always target the next run.
        next_run = self.next_scheduled_run(session)
        if next_run is None or next_run.sequence <= current.sequence:
            next_run = self.create_run(
                session,
                sequence=current.sequence + 1,
                started_at=current.ends_at,
                status="scheduled",
            )
            session.commit()
            session.refresh(next_run)
        return next_run

    # ------------------------------------------------------------------
    # Epoch target snapshots
    # ------------------------------------------------------------------

    def open_run_pinned_deployment_ids(
        self, session: Session, *, family_id: str | None = None
    ) -> set[str]:
        statement = select(EpochTargetSnapshot).where(EpochTargetSnapshot.status == "open")
        if family_id is not None:
            statement = statement.where(EpochTargetSnapshot.family_id == ensure_family_id(family_id))
        deployment_ids: set[str] = set()
        for snapshot in session.execute(statement).scalars():
            for member in snapshot.members_json:
                deployment_id = str(member.get("metadata", {}).get("deployment_id", "")).strip()
                if deployment_id:
                    deployment_ids.add(deployment_id)
        return deployment_ids

    def resolve_run_member(
        self,
        session: Session,
        *,
        run_id: str,
        deployment_id: str,
    ) -> tuple[EpochTargetSnapshot | None, dict[str, Any] | None]:
        snapshots = session.execute(
            select(EpochTargetSnapshot).where(EpochTargetSnapshot.run_id == run_id)
        ).scalars()
        for snapshot in snapshots:
            for member in snapshot.members_json:
                member_deployment_id = member.get("metadata", {}).get("deployment_id")
                if member_deployment_id == deployment_id:
                    return snapshot, member
        return None, None

    def _candidate_pool_deployments(
        self,
        session: Session,
        *,
        run: EvaluationRun,
        family_id: str,
    ) -> list[ManagedDeployment]:
        previous_run = session.execute(
            select(EvaluationRun)
            .where(EvaluationRun.sequence == run.sequence - 1)
            .limit(1)
        ).scalar_one_or_none()
        deployment_ids: set[str] = set()

        # Carryover top performers from previous run
        if previous_run is not None:
            deployment_ids.update(
                self._owner._top_run_deployment_ids(
                    session,
                    run_id=previous_run.id,
                    family_id=family_id,
                    limit=self._owner.run_top_carryover_per_family,
                )
            )

        # Submissions targeting this run were queued during the previous
        # run period (or before the first run for bootstrap).
        submission_source_run_id = run.id
        for submission in session.execute(
            select(ManagedMinerSubmission)
            .where(ManagedMinerSubmission.family_id == family_id)
            .where(ManagedMinerSubmission.introduced_run_id == submission_source_run_id)
        ).scalars():
            deployment = self._owner.get_deployment_for_submission(session, submission.id)
            if deployment is not None and deployment.status != "retired":
                deployment_ids.add(deployment.id)

        deployments = [
            session.get(ManagedDeployment, deployment_id)
            for deployment_id in deployment_ids
        ]
        return [
            item
            for item in deployments
            if item is not None and item.status != "retired"
        ]

    def freeze_run_targets(
        self,
        session: Session,
        *,
        run_id: str,
        family_id: str,
        base_url: str,
        benchmark_version: str | None = None,
        rubric_version: str | None = None,
        judge_model: str | None = None,
    ) -> EpochTargetSnapshot:
        family_id = ensure_family_id(family_id)
        run = session.get(EvaluationRun, run_id)
        if run is None:
            raise ValueError("run not found")
        existing = session.execute(
            select(EpochTargetSnapshot).where(
                EpochTargetSnapshot.run_id == run_id,
                EpochTargetSnapshot.family_id == family_id,
            )
        ).scalar_one_or_none()
        if existing is not None:
            return existing
        deployments = self._candidate_pool_deployments(session, run=run, family_id=family_id)
        deployments.sort(key=lambda item: (item.created_at, item.id))
        members = [
            self._owner._snapshot_member_from_deployment(
                base_url=base_url,
                run_id=run_id,
                deployment=deployment,
                quality_score=self._owner.latest_quality_score(session, deployment_id=deployment.id),
            )
            for deployment in deployments
        ]
        validator_stakes = {}
        snapshot = EpochTargetSnapshot(
            run_id=run_id,
            family_id=family_id,
            benchmark_version=benchmark_version or self._owner.benchmark_version,
            rubric_version=rubric_version or self._owner.rubric_version,
            judge_model=judge_model or self._owner.judge_model,
            status="pending_deployments",
            frozen_validator_stakes_json=validator_stakes,
            members_json=members,
        )
        try:
            session.add(snapshot)
            session.flush()
        except IntegrityError:
            session.rollback()
            existing = session.execute(
                select(EpochTargetSnapshot)
                .where(EpochTargetSnapshot.run_id == run_id)
                .where(EpochTargetSnapshot.family_id == family_id)
            ).scalar_one()
            return existing

        # Initialize distributed evaluation tasks for every member × task
        self._owner.evaluation_tasks.initialize_evaluation_tasks(
            session,
            run_id=run_id,
            family_id=family_id,
            snapshot=snapshot,
        )

        session.commit()
        session.refresh(snapshot)
        return snapshot

    def check_and_open_pending_snapshots(self, session: Session) -> None:
        pending = list(session.execute(
            select(EpochTargetSnapshot)
            .where(EpochTargetSnapshot.status == "pending_deployments")
        ).scalars())
        now = _utcnow()
        timeout_minutes = getattr(
            self._owner.settings, 'snapshot_readiness_timeout_minutes', 15,
        )
        max_wait = timedelta(minutes=timeout_minutes)

        for snapshot in pending:
            timed_out = (now - snapshot.created_at) >= max_wait
            all_ready = True
            has_healthy = False
            for member in (snapshot.members_json or []):
                deployment_id = member.get('metadata', {}).get('deployment_id')
                if deployment_id is None:
                    continue
                deployment = session.get(ManagedDeployment, deployment_id)
                if deployment is None:
                    continue
                if deployment.status in ('build_failed', 'retired'):
                    continue
                if deployment.health_status == 'healthy':
                    has_healthy = True
                else:
                    all_ready = False

            if all_ready and has_healthy:
                snapshot.status = 'open'
                snapshot.updated_at = now
                logger.info(
                    'snapshot %s/%s opened — all deployments ready',
                    snapshot.run_id, snapshot.family_id,
                )
            elif timed_out and has_healthy:
                snapshot.status = 'open'
                snapshot.updated_at = now
                logger.warning(
                    'snapshot %s/%s opened after timeout — some deployments not ready',
                    snapshot.run_id, snapshot.family_id,
                )
        session.commit()
