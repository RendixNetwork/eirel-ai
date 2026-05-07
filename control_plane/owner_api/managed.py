"""Managed owner services facade.

This module is the composition root for the owner API's domain managers.
Method implementations live in the domain manager modules; this facade
initializes them and delegates attribute lookups via ``__getattr__``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from shared.common.config import Settings
from shared.common.artifacts import ArtifactStore
from shared.common.database import Database
from shared.common.manifest import SubmissionManifest
from shared.common.object_store import ObjectStore
from eirel.groups import ensure_family_id
from control_plane.owner_api.deployment import DeploymentManager, ManagedDeploymentRuntimeManager
from control_plane.owner_api.evaluation import (
    EvaluationTaskManager,
    RunManager,
    ScoringManager,
)
from control_plane.owner_api.serving import LeaseManager, ServingManager, WeightManager
from control_plane.owner_api.workflows import MinerRegistry, TaskOrchestrator, WorkflowManager
from control_plane.owner_api.operations import OperatorManager, SubmissionManager

# ── Re-exports for backward compatibility ────────────────────────────────
# Constants
from control_plane.owner_api._constants import (  # noqa: F401
    ABV_FAMILIES,
    ABV_SERVING_SELECTION_REASON,
    DEFAULT_FIXED_FAMILY_WEIGHTS,
    FAMILY_SERVING_FAMILY_WEIGHT,
    FAMILY_SERVING_RELIABILITY_WEIGHT,
    LAUNCH_FIXED_FAMILY_WEIGHTS,
    PLACEMENT_RESERVED_STATUSES,
    PRODUCTION_FAMILIES,
    WINNER_DOMINANCE_MARGIN,
    WORKFLOW_COMPOSITION_SELECTION_REASON,
    WORKFLOW_EPISODE_DEFAULT_MAX_ATTEMPTS,
    WORKFLOW_EPISODE_RETRY_BASE_SECONDS,
    WORKFLOW_EPISODE_RETRY_MAX_SECONDS,
    WORKFLOW_RUNTIME_POLICY_STATE_KEY,
    WORKFLOW_RUNTIME_REMEDIATION_AUDIT_LIMIT,
    WORKFLOW_RUNTIME_SUPPRESSION_TARGET_KINDS,
)

# Helpers
from control_plane.owner_api._helpers import (  # noqa: F401
    _active_families,
    _default_allowed_tool_policy_for_bundle,
    _evaluation_policy_payload,
    _live_research_retrieval_environment_payload,
    _load_owner_evaluation_bundle_seed,
    _metadata_float,
    _metadata_int,
    _score_record_official_family_score,
    _score_record_selection_score,
    _stamp_specialist_tasks,
    _strip_sensitive_bundle_metadata,
    compute_query_volume_shares,
    family_for_manifest,
    fixed_family_weight,
    is_abv_family,
    is_supported_family,
    score_bearing_family_ids,
    utcnow,
)


# ── Facade ───────────────────────────────────────────────────────────────

# Manager lookup table — maps method names to manager attribute names.
# Built lazily on first __getattr__ call to avoid import-time overhead.
_MANAGER_METHOD_MAP: dict[str, str] | None = None


def _build_manager_method_map() -> dict[str, str]:
    """Build mapping of method_name -> manager_attr_name by inspecting manager classes."""
    mapping: dict[str, str] = {}
    manager_classes = {
        "deployments": DeploymentManager,
        "runs": RunManager,
        "scoring": ScoringManager,
        "serving": ServingManager,
        "submissions": SubmissionManager,
        "evaluation_tasks": EvaluationTaskManager,
        "workflows": WorkflowManager,
        "operators": OperatorManager,
        "weights": WeightManager,
        "leases": LeaseManager,
        "tasks": TaskOrchestrator,
        "miners": MinerRegistry,
    }
    for attr_name, cls in manager_classes.items():
        for name in dir(cls):
            if name.startswith("__") and name.endswith("__"):
                continue
            if name in ("db", "settings"):
                continue
            if callable(getattr(cls, name, None)) or isinstance(getattr(cls, name, None), property):
                mapping[name] = attr_name
    return mapping


@dataclass(slots=True)
class ManagedOwnerServices:
    db: Database
    settings: Settings
    runtime_manager: ManagedDeploymentRuntimeManager
    artifact_store: ArtifactStore
    object_store: ObjectStore | None = None
    top_k_per_group: int = 3
    benchmark_version: str = "family_benchmark_v2"
    rubric_version: str = "family_rubric_v2"
    judge_model: str = "local-rubric-judge"

    # Fee verification (None = not enforced)
    _fee_verifier: Any = None

    # Domain managers
    weights: WeightManager = None  # type: ignore[assignment]
    leases: LeaseManager = None  # type: ignore[assignment]
    tasks: TaskOrchestrator = None  # type: ignore[assignment]
    miners: MinerRegistry = None  # type: ignore[assignment]
    deployments: DeploymentManager = None  # type: ignore[assignment]
    runs: RunManager = None  # type: ignore[assignment]
    scoring: ScoringManager = None  # type: ignore[assignment]
    serving: ServingManager = None  # type: ignore[assignment]
    submissions: SubmissionManager = None  # type: ignore[assignment]
    evaluation_tasks: EvaluationTaskManager = None  # type: ignore[assignment]
    workflows: WorkflowManager = None  # type: ignore[assignment]
    operators: OperatorManager = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.object_store is None:
            self.object_store = ObjectStore.from_settings(self.settings)
        self.weights = WeightManager(self)
        self.leases = LeaseManager(self)
        self.tasks = TaskOrchestrator(self)
        self.miners = MinerRegistry(self)
        self.deployments = DeploymentManager(self)
        self.runs = RunManager(self)
        self.scoring = ScoringManager(self)
        self.serving = ServingManager(self)
        self.submissions = SubmissionManager(self)
        self.evaluation_tasks = EvaluationTaskManager(self)
        self.workflows = WorkflowManager(self)
        self.operators = OperatorManager(self)

    def __getattr__(self, name: str) -> Any:
        global _MANAGER_METHOD_MAP
        if _MANAGER_METHOD_MAP is None:
            _MANAGER_METHOD_MAP = _build_manager_method_map()
        manager_attr = _MANAGER_METHOD_MAP.get(name)
        if manager_attr is not None:
            manager = object.__getattribute__(self, manager_attr)
            return getattr(manager, name)
        raise AttributeError(f"'ManagedOwnerServices' object has no attribute {name!r}")

    # ── Config properties (kept on facade — pure setting lookups) ────────

    @property
    def soft_termination_grace_seconds(self) -> int:
        return max(0, self.settings.soft_termination_grace_seconds)

    @property
    def serving_release_interval_days(self) -> int:
        return max(1, self.settings.serving_release_interval_days)

    @property
    def run_duration_days(self) -> int:
        return max(1, self.settings.run_duration_days)

    @property
    def run_top_carryover_per_family(self) -> int:
        return max(1, self.settings.run_top_carryover_per_family)

    def run_min_scores(self) -> dict[str, float]:
        try:
            payload = json.loads(self.settings.run_min_scores_json or "{}")
        except json.JSONDecodeError:
            payload = {}
        values = {family_id: 0.0 for family_id in PRODUCTION_FAMILIES}
        for raw_key, raw_value in payload.items():
            try:
                family_id = ensure_family_id(str(raw_key))
            except ValueError:
                continue
            values[family_id] = max(0.0, float(raw_value))
        return values

    @property
    def runtime_cpu_headroom_millis(self) -> int:
        return max(0, self.settings.owner_runtime_capacity_cpu_headroom_millis)

    @property
    def runtime_memory_headroom_bytes(self) -> int:
        return max(0, self.settings.owner_runtime_capacity_memory_headroom_mb) * 1024 * 1024

    @property
    def runtime_pod_headroom(self) -> int:
        return max(0, self.settings.owner_runtime_capacity_pod_headroom)

    def check_and_open_pending_snapshots(self) -> None:
        with self.db.sessionmaker() as session:
            self.runs.check_and_open_pending_snapshots(session)

    def normalize_manifest_resources(self, manifest: SubmissionManifest) -> tuple[int, int]:
        cpu_millis = max(1, self.settings.owner_runtime_submission_cpu_millis)
        memory_bytes = max(1, self.settings.owner_runtime_submission_memory_mb * 1024 * 1024)
        return cpu_millis, memory_bytes
