from __future__ import annotations

"""Dataset generator compatibility shim.

The legacy analyst 5-domain harvest-driven generator has been retired.
The owner_api run_manager still calls ``build_generated_run_bundle`` as
a convenience wrapper around the dataset loader seed; this module now
simply returns the seed as-is, copying over the run_id / family_id /
policy metadata so downstream code keeps working.
"""

from typing import Any

from shared.core.evaluation_models import FamilyEvaluationBundle


def build_generated_run_bundle(
    *,
    seed: FamilyEvaluationBundle,
    run_id: str,
    family_id: str,
    benchmark_version: str,
    rubric_version: str,
    dataset_source_root: str,
    retrieval_environment: dict[str, Any] | None = None,
    allowed_tool_policy: dict[str, Any] | None = None,
    judge_config: dict[str, Any] | None = None,
    policy_version: str | None = None,
) -> FamilyEvaluationBundle:
    del dataset_source_root
    return seed.model_copy(
        update={
            "run_id": run_id,
            "family_id": family_id,
            "benchmark_version": benchmark_version,
            "rubric_version": rubric_version,
            "retrieval_environment": retrieval_environment,
            "allowed_tool_policy": allowed_tool_policy,
            "judge_config": judge_config,
            "policy_version": policy_version,
        }
    )
