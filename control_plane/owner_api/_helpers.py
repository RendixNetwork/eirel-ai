"""Module-level utility functions for the owner API managed services."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from shared.common.config import Settings
from shared.common.models import DeploymentScoreRecord
from eirel.groups import LAUNCH_FAMILIES, ensure_family_id, is_launch_mode
from shared.core.evaluation_models import FamilyEvaluationBundle
from shared.scoring.policy import SCORING_POLICY_VERSION, scoring_policy_for

from control_plane.owner_api._constants import (
    ABV_FAMILIES,
    DEFAULT_FIXED_FAMILY_WEIGHTS,
    LAUNCH_FIXED_FAMILY_WEIGHTS,
    PRODUCTION_FAMILIES,
)


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def family_for_manifest(manifest: Any) -> str:
    return ensure_family_id(str(manifest.family_id))


def _active_families(settings: Settings | None = None) -> tuple[str, ...]:
    """Return dynamically configured active families.

    Reads ``EIREL_ACTIVE_FAMILIES`` (comma-separated) from settings.
    Falls back to the hardcoded constants if settings are unavailable.
    """
    if settings is not None and settings.active_families:
        return tuple(
            f.strip() for f in settings.active_families.split(",") if f.strip()
        )
    return LAUNCH_FAMILIES if is_launch_mode() else PRODUCTION_FAMILIES


def is_supported_family(family_id: str, settings: Settings | None = None) -> bool:
    try:
        return ensure_family_id(family_id) in _active_families(settings)
    except ValueError:
        return False


def score_bearing_family_ids(settings: Settings | None = None) -> tuple[str, ...]:
    return _active_families(settings)


def _parse_family_weights(settings: Settings | None = None) -> dict[str, float]:
    """Parse ``EIREL_FAMILY_WEIGHTS`` from settings.

    Format: ``"analyst:0.45,builder:0.30,verifier:0.25"``
    Falls back to hardcoded constants if not configured.
    """
    if settings is not None and settings.family_weights:
        weights: dict[str, float] = {}
        for entry in settings.family_weights.split(","):
            entry = entry.strip()
            if ":" in entry:
                fam, val = entry.split(":", 1)
                try:
                    weights[fam.strip()] = float(val.strip())
                except ValueError:
                    pass
        if weights:
            return weights
    return LAUNCH_FIXED_FAMILY_WEIGHTS if is_launch_mode() else DEFAULT_FIXED_FAMILY_WEIGHTS


def fixed_family_weight(family_id: str, settings: Settings | None = None) -> float:
    try:
        weights = _parse_family_weights(settings)
        return weights.get(ensure_family_id(family_id), 0.0)
    except ValueError:
        return 0.0


def compute_query_volume_shares(
    query_counts: dict[str, int],
) -> dict[str, float]:
    """Convert per-family query counts into normalized volume shares."""
    total = sum(query_counts.values())
    if total <= 0:
        return {fid: 0.0 for fid in query_counts}
    return {
        fid: round(count / total, 6)
        for fid, count in query_counts.items()
    }


def is_abv_family(family_id: str) -> bool:
    try:
        return ensure_family_id(family_id) in ABV_FAMILIES
    except ValueError:
        return False


def _evaluation_policy_payload(family_id: str) -> dict[str, Any]:
    family_id = ensure_family_id(family_id)
    policy = scoring_policy_for(family_id)
    return {
        "family_id": family_id,
        "benchmark_version": policy.benchmark_version,
        "rubric_version": policy.rubric_version,
        "official_scoring_version": policy.official_scoring_version,
        "judge_mode": policy.judge_mode,
        "scoring_policy_version": SCORING_POLICY_VERSION,
    }


def _stamp_specialist_tasks(
    tasks: list[dict[str, Any]],
    *,
    family_id: str,
    benchmark_version: str,
    rubric_version: str,
    default_task_family: str,
) -> list[dict[str, Any]]:
    family_id = ensure_family_id(family_id)
    stamped: list[dict[str, Any]] = []
    for item in tasks:
        task = dict(item)
        metadata = dict(task.get("metadata", {}) or {})
        inputs = dict(task.get("inputs", {}) or {})
        risk_tags = [str(tag) for tag in (task.get("risk_tags", []) or []) if str(tag).strip()]
        prompt = str(task.get("prompt") or task.get("question") or "").strip()
        task_family = str(
            task.get("task_family")
            or metadata.get("task_family")
            or default_task_family
        ).strip()
        stamped.append(
            {
                **task,
                "task_id": str(task.get("task_id") or ""),
                "family_id": family_id,
                "prompt": prompt,
                "category": str(task.get("category") or family_id),
                "difficulty": str(task.get("difficulty") or "standard"),
                "benchmark_version": benchmark_version,
                "rubric_version": rubric_version,
                "risk_tags": risk_tags,
                "task_family": task_family,
                "inputs": inputs,
                "metadata": {
                    **metadata,
                    "owner_frozen": True,
        "task_contract_version": "family_benchmark_v2",
                    "task_family": task_family,
                },
            }
        )
    return stamped


# Per-task metadata keys that must not reach validators or miners.
# - ``hidden_fixture`` / ``visibility`` label a task as hidden — leaking
#   either defeats the whole point of hidden fixtures.
# - ``seed_id`` names a seed template the task was generated from. Across
#   multiple runs a miner could cluster tasks by seed_id and learn which
#   templates contribute hidden fixtures.
# - ``topic`` names the entry drawn from the topic pool. Similar cross-run
#   inference attack surface.
_SENSITIVE_TASK_METADATA_KEYS: frozenset[str] = frozenset(
    {"hidden_fixture", "visibility", "seed_id", "topic"}
)


def _strip_sensitive_task_metadata(task_dict: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``task_dict`` with anti-gaming-sensitive fields removed.

    Removes:
    - sensitive keys from ``metadata`` (hidden_fixture/visibility/seed_id/topic)
    - the entire ``expected_output`` field (C4) — it contains the rubric
      (must_cover/must_avoid/judge_rubric/required_structure/...) which
      validators must not receive. Validators invoke the miner with
      ``prompt`` + ``inputs`` and then call the owner-api judge proxy
      (``/v1/internal/tasks/{task_assignment_id}/judge``) to score; the
      rubric never leaves the owner process.
    """
    task_metadata = dict(task_dict.get("metadata") or {})
    for key in _SENSITIVE_TASK_METADATA_KEYS:
        task_metadata.pop(key, None)
    result = dict(task_dict)
    result["metadata"] = task_metadata
    # C4: strip the full rubric payload. Keep a minimal breadcrumb so
    # validators can still branch on execution_mode / task_family without
    # being able to read any grading criterion.
    expected_output = task_dict.get("expected_output")
    if isinstance(expected_output, dict):
        result["expected_output"] = {
            key: expected_output[key]
            for key in ("execution_mode", "task_family")
            if key in expected_output
        }
    else:
        result["expected_output"] = {}
    return result


def _strip_sensitive_bundle_metadata(bundle_dict: dict[str, Any]) -> dict[str, Any]:
    """Remove fields that must not be disclosed to validators or miners.

    This is called at every serving boundary (the family_targets endpoint,
    the task-claim endpoint). The stored bundle under ``run.metadata_json``
    keeps the full set so internal scoring/validation code can still see
    hidden markers; only the outbound payload is stripped.

    Stripped:
      - ``metadata.hidden_fixture_ids`` — the full list of hidden task IDs
      - per-task ``metadata.{hidden_fixture, visibility, seed_id, topic}``
    """
    bundle_dict = dict(bundle_dict)
    meta = dict(bundle_dict.get("metadata") or {})
    meta.pop("hidden_fixture_ids", None)
    bundle_dict["metadata"] = meta

    tasks = bundle_dict.get("tasks")
    if isinstance(tasks, list):
        bundle_dict["tasks"] = [
            _strip_sensitive_task_metadata(task) if isinstance(task, dict) else task
            for task in tasks
        ]
    return bundle_dict


def _enforce_strict_analyst_contract(bundle: FamilyEvaluationBundle) -> None:
    """Loader-side analyst constraint enforcement.

    Enforces the analyst contract at the *load* boundary so a hand-crafted
    bundle (or one whose metadata changed under us) cannot bypass it.
    """
    if bundle.family_id != "analyst":
        return
    live_research_tasks = [task for task in bundle.tasks if str(task.evaluation_track or "") == "live_research"]
    replay_research_tasks = [task for task in bundle.tasks if str(task.evaluation_track or "") == "replay_research"]
    reasoning_tasks = [task for task in bundle.tasks if str(task.evaluation_track or "") == "reasoning"]
    hidden_tasks = [
        task
        for task in bundle.tasks
        if bool(dict(task.metadata or {}).get("hidden_fixture"))
        or str(dict(task.metadata or {}).get("visibility") or "").strip().lower() == "hidden"
    ]
    hidden_live_research_tasks = [task for task in live_research_tasks if task in hidden_tasks]
    hidden_replay_research_tasks = [task for task in replay_research_tasks if task in hidden_tasks]
    hidden_reasoning_tasks = [task for task in reasoning_tasks if task in hidden_tasks]
    if len(bundle.tasks) != 40:
        raise ValueError("analyst owner dataset must contain exactly 40 fixtures")
    if len(live_research_tasks) != 14:
        raise ValueError("analyst owner dataset must contain exactly 14 live_research fixtures")
    if len(replay_research_tasks) != 14:
        raise ValueError("analyst owner dataset must contain exactly 14 replay_research fixtures")
    if len(reasoning_tasks) != 12:
        raise ValueError("analyst owner dataset must contain exactly 12 reasoning fixtures")
    if len(hidden_tasks) < 12:
        raise ValueError("analyst owner dataset must contain at least 12 hidden fixtures")
    if len(hidden_live_research_tasks) < 4:
        raise ValueError("analyst owner dataset must contain at least 4 hidden live_research fixtures")
    if len(hidden_replay_research_tasks) < 4:
        raise ValueError("analyst owner dataset must contain at least 4 hidden replay_research fixtures")
    if len(hidden_reasoning_tasks) < 4:
        raise ValueError("analyst owner dataset must contain at least 4 hidden reasoning fixtures")


def _load_owner_evaluation_bundle_seed(
    *,
    root_path: str,
    family_id: str,
) -> FamilyEvaluationBundle:
    """Filesystem loader.

    Reads ``<root_path>/<family_id>.json`` from disk and re-runs the analyst
    contract. This path is used in dev mode and as a fallback when no
    ``OwnerDatasetBinding`` exists for the run; production with bundles
    registered in the bindings table goes through
    ``load_owner_evaluation_bundle_via_binding``.
    """
    family_id = ensure_family_id(family_id)
    payload = json.loads((Path(root_path) / f"{family_id}.json").read_text(encoding="utf-8"))
    bundle = FamilyEvaluationBundle.model_validate(payload)
    if bundle.family_id != family_id:
        raise ValueError("owner evaluation bundle family_id mismatch")
    _enforce_strict_analyst_contract(bundle)
    return bundle


def _default_allowed_tool_policy_for_bundle(
    bundle: FamilyEvaluationBundle,
) -> dict[str, Any] | None:
    if isinstance(bundle.allowed_tool_policy, dict) and bundle.allowed_tool_policy:
        return dict(bundle.allowed_tool_policy)
    allowed_tools = sorted(
        {
            str(tool)
            for task in bundle.tasks
            for tool in list(task.allowed_tools or [])
            if str(tool).strip()
        }
    )
    if not allowed_tools:
        return None
    return {
        "provider_proxy_required": True,
        "provider_proxy_only": False,
        "allowed_tools": allowed_tools,
    }


def _live_research_retrieval_environment_payload(settings: Settings) -> dict[str, Any]:
    return {
        "mode": "live_web",
        "backend": settings.research_tool_backend,
        "search_provider": "brave" if settings.research_tool_backend == "brave_live_web" else "catalog",
        "base_url": settings.research_tool_service_url,
        "timeout_seconds": settings.research_tool_timeout_seconds,
        "policy_version": settings.research_tool_policy_version,
        "allowed_tool_policy": {
            "provider_proxy_required": True,
            "provider_proxy_only": False,
            "allowed_tools": ["retrieval_search", "browser_open", "browser_find_on_page"],
        },
        "budget_policy": {
            "retrieval_request_soft_limit": settings.research_tool_max_requests,
        },
        "trusted_domain_mode": "prefer_configured_domains",
    }



def _score_record_official_family_score(row: DeploymentScoreRecord) -> float:
    return float(row.metadata_json.get("official_family_score", row.raw_score) or row.raw_score)


def _score_record_selection_score(row: DeploymentScoreRecord) -> float:
    family_score = _score_record_official_family_score(row)
    if 0.0 <= family_score <= 1.0:
        return family_score * 100.0
    return family_score


def _metadata_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _metadata_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None
