from __future__ import annotations

from typing import Any

from shared.core.evaluation_models import (
    BenchmarkRunContext,
    BenchmarkTask,
    FamilyEvaluationBundle,
)


TOOL_NAME_ALIASES = {
    "retrieval_open_page": "browser_open",
    "retrieval_find_on_page": "browser_find_on_page",
}
NON_CALLABLE_TOOLS = {"provider_proxy"}
DEFAULT_RESEARCH_TOOLS = (
    "retrieval_search",
    "browser_open",
    "browser_find_on_page",
)


def load_family_benchmarks(family_id: str, *, context: BenchmarkRunContext | None = None) -> list[BenchmarkTask]:
    evaluation_bundle = (context.metadata or {}).get("evaluation_bundle") if context is not None else None
    if not isinstance(evaluation_bundle, dict):
        raise ValueError("family requires context.metadata['evaluation_bundle']")
    return _load_evaluation_bundle_tasks(
        evaluation_bundle,
        retrieval_environment=context.retrieval_environment if context is not None else None,
    )


def _normalize_allowed_tools(values: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        name = TOOL_NAME_ALIASES.get(str(item), str(item))
        if name in NON_CALLABLE_TOOLS or name in seen:
            continue
        normalized.append(name)
        seen.add(name)
    return normalized


def _load_evaluation_bundle_tasks(
    payload: dict[str, Any],
    *,
    retrieval_environment: dict[str, Any] | None,
) -> list[BenchmarkTask]:
    bundle = FamilyEvaluationBundle.model_validate(payload)
    effective_retrieval_environment = (
        dict(bundle.retrieval_environment or {})
        if isinstance(bundle.retrieval_environment, dict)
        else dict(retrieval_environment or {})
    )
    tasks: list[BenchmarkTask] = []
    for task in bundle.tasks:
        task_inputs = dict(task.inputs or {})
        task_expected_output = dict(task.expected_output or {})
        task_metadata = dict(task.metadata or {})
        if effective_retrieval_environment and task.execution_mode in {"live_web", "replay_web"}:
            task_inputs = {
                **task_inputs,
                "retrieval_environment": dict(
                    task_inputs.get("retrieval_environment")
                    if isinstance(task_inputs.get("retrieval_environment"), dict)
                    else effective_retrieval_environment
                ),
            }
        tasks.append(
            BenchmarkTask(
                task_id=task.task_id,
                family_id=task.family_id,
                prompt=task.prompt,
                task_mode=task.task_mode,
                execution_mode=task.execution_mode,
                allowed_tools=_normalize_allowed_tools(list(task.allowed_tools or [])),
                retrieval_constraints=dict(task.retrieval_constraints or {}),
                expected_output=task_expected_output,
                inputs=task_inputs,
                metadata={
                    **task_metadata,
                    "benchmark_version": bundle.benchmark_version,
                    "rubric_version": bundle.rubric_version,
                    "category": task.category,
                    "difficulty": task.difficulty,
                    "risk_tags": list(task.risk_tags),
                    "task_contract_version": "family_benchmark_v2",
                    "owner_frozen": True,
                    "evaluation_bundle_kind": bundle.kind,
                },
            )
        )
    return tasks
