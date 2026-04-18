from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from shared.common.exceptions import WorkflowEpisodeAbortedError
from shared.common.security import sha256_hex
from shared.common.miner_client import invoke_miner
from eirel.schemas import AgentInvocationResponse
from shared.benchmark.run import evaluate_multimodal_artifacts, score_media_generation_payload
from shared.contracts.models import NodeTrace, WorkflowEpisode
from shared.contracts.specialist_contracts import FAMILY_REQUIRED_OUTPUT_FIELDS


V3_RUNTIME_METADATA_KEYS = ("checkpoint_events", "runtime_state_patch")
MAX_EPISODE_RESUME_ATTEMPTS = 3

WorkflowNodeUpdateCallback = Callable[[dict[str, Any]], Awaitable[None] | None]
WorkflowAbortCallback = Callable[[str, str, int], Awaitable[str | None] | str | None]


def merge_checkpoint_state(
    current_state: dict[str, object],
    runtime_state_patch: dict[str, object],
) -> dict[str, object]:
    return {
        **dict(current_state or {}),
        **dict(runtime_state_patch or {}),
    }


def response_runtime_list(
    response: AgentInvocationResponse,
    field_name: str,
) -> list[dict[str, object]]:
    value = getattr(response, field_name, None)
    if isinstance(value, list):
        return list(value)
    return []


def response_runtime_patch(response: AgentInvocationResponse) -> dict[str, object]:
    value = response.runtime_state_patch
    if isinstance(value, dict):
        return dict(value)
    return {}


def response_resume_token(response: AgentInvocationResponse) -> str | None:
    value = response.resume_token
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def response_score_hint(
    response: AgentInvocationResponse,
    field_name: str,
) -> float | None:
    value = getattr(response, field_name, None)
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    return None


def normalized_runtime_metadata(response: AgentInvocationResponse) -> dict[str, object]:
    metadata = dict(response.metadata or {})
    metadata["checkpoint_events"] = response_runtime_list(response, "checkpoint_events")
    metadata["runtime_state_patch"] = response_runtime_patch(response)
    resume_token = response_resume_token(response)
    if resume_token is not None:
        metadata["resume_token"] = resume_token
    tool_calls = response_runtime_list(response, "tool_calls")
    if tool_calls:
        metadata["tool_calls"] = tool_calls
    reliability_score = response_score_hint(response, "reliability_score")
    if reliability_score is not None:
        metadata["reliability_score"] = reliability_score
    recovery_score = response_score_hint(response, "recovery_score")
    if recovery_score is not None:
        metadata["recovery_score"] = recovery_score
    return metadata


def completed_output_from_response(response: AgentInvocationResponse) -> dict[str, object]:
    return {
        "output": dict(response.output or {}),
        "artifacts": [artifact.model_dump(mode="json") for artifact in response.artifacts],
        "metadata": normalized_runtime_metadata(response),
        "status": response.status,
        "error": response.error,
    }


def runtime_contract_mode_from_payload(payload: dict[str, object]) -> str:
    if all(key in payload for key in V3_RUNTIME_METADATA_KEYS):
        return "canonical_v3"
    return "invalid_runtime_contract"


def validate_runtime_response_payload(*, payload: dict[str, object]) -> None:
    status = str(payload.get("status") or "")
    missing = [key for key in V3_RUNTIME_METADATA_KEYS if key not in payload]
    if missing:
        raise ValueError(f"workflow episode response missing required runtime metadata: {', '.join(missing)}")
    checkpoint_events = payload.get("checkpoint_events")
    runtime_state_patch = payload.get("runtime_state_patch")
    resume_token = payload.get("resume_token")
    if not isinstance(checkpoint_events, list):
        raise ValueError("workflow episode response checkpoint_events must be a list")
    if not isinstance(runtime_state_patch, dict):
        raise ValueError("workflow episode response runtime_state_patch must be an object")
    if status == "deferred" and not str(resume_token or "").strip():
        raise ValueError("workflow episode deferred response missing resume_token")


def validate_protocol_compliance(
    response: AgentInvocationResponse,
    *,
    family_id: str,
) -> list[str]:
    """Check that a completed response satisfies the family output contract.

    Returns a list of violation strings (empty = compliant).
    Does NOT raise — violations are recorded in NodeTrace for scoring.
    Only applied when status == "completed".
    """
    if response.status != "completed":
        return []
    contract = FAMILY_REQUIRED_OUTPUT_FIELDS.get(family_id)
    if contract is None:
        return []
    violations: list[str] = []
    if contract.get("artifacts_required"):
        if not response.artifacts:
            violations.append(f"{family_id}: artifacts must be non-empty for completed response")
        return violations
    output = dict(response.output or {})
    def _expected_python_type(value: object) -> type | tuple[type, ...] | None:
        if isinstance(value, type):
            return value
        if isinstance(value, str):
            return {
                "str": str,
                "float": float,
                "int": int,
                "bool": bool,
                "dict": dict,
                "list": list,
            }.get(value)
        return None
    for field in contract.get("required", []):
        if field not in output:
            violations.append(f"{family_id}: output missing required field '{field}'")
            continue
        expected_type = _expected_python_type(contract.get("required_types", {}).get(field))
        if expected_type is not None and not isinstance(output[field], expected_type):
            violations.append(
                f"{family_id}: output.{field} must be {expected_type.__name__}, "
                f"got {type(output[field]).__name__}"
            )
    allowed_values = contract.get("allowed_verdict_values")
    if allowed_values and "verdict" in output:
        if output["verdict"] not in allowed_values:
            violations.append(
                f"{family_id}: output.verdict '{output['verdict']}' not in {allowed_values}"
            )
    return violations


def node_quality_score(metadata: dict[str, object], *, status: str) -> float:
    raw = metadata.get("official_family_score")
    if not isinstance(raw, (int, float)):
        raw = metadata.get("overall_score")
    if not isinstance(raw, (int, float)):
        raw = 0.75 if status == "completed" else 0.0
    return max(0.0, min(1.0, float(raw)))


async def _emit_update(
    callback: WorkflowNodeUpdateCallback | None,
    payload: dict[str, Any],
) -> None:
    if callback is None:
        return
    maybe_awaitable = callback(payload)
    if maybe_awaitable is not None:
        await maybe_awaitable


async def execute_workflow_episode_node(
    *,
    episode: WorkflowEpisode,
    node,
    task_prompt: str,
    completed_outputs: dict[str, dict[str, object]],
    replay_mode: bool = False,
    update_callback: WorkflowNodeUpdateCallback | None = None,
    abort_callback: WorkflowAbortCallback | None = None,
    invoke_miner_fn: Callable[..., Awaitable[dict[str, object]]] | None = None,
) -> tuple[dict[str, object], NodeTrace]:
    if not node.endpoint:
        raise ValueError(f"episode node {node.node_id} has no endpoint")
    current_checkpoint_state = dict(node.checkpoint_state or {})
    resume_token = node.resume_token
    checkpoint_events: list[dict[str, object]] = []
    tool_calls: list[dict[str, object]] = []
    total_latency_ms = 0
    total_cost_tao = 0.0
    deferred_rounds = 0
    last_response: AgentInvocationResponse | None = None
    last_payload: dict[str, object] | None = None
    last_error: str | None = None
    runtime_contract_mode = "invalid_runtime_contract"
    while deferred_rounds <= MAX_EPISODE_RESUME_ATTEMPTS:
        if abort_callback is not None:
            abort_reason = abort_callback(episode.episode_id, node.node_id, deferred_rounds)
            if abort_reason is not None and hasattr(abort_reason, "__await__"):
                abort_reason = await abort_reason
            if isinstance(abort_reason, str) and abort_reason.strip():
                raise WorkflowEpisodeAbortedError(abort_reason.strip())
        upstream_outputs = {
            node_id: completed_outputs[node_id]
            for node_id in node.input_node_ids
            if node_id in completed_outputs
        }
        payload = {
            "task_id": f"{episode.episode_id}:{node.node_id}:{deferred_rounds}",
            "session_id": episode.episode_id,
            "primary_goal": task_prompt,
            "subtask": node.episode_input.get("node_description") or node.role_id,
            "family_id": node.family_id,
            "episode_id": episode.episode_id,
            "workflow_spec_id": episode.workflow_spec_id,
            "workflow_version": episode.workflow_version,
            "planner_node_id": node.node_id,
            "role_id": node.role_id,
            "upstream_node_outputs": upstream_outputs,
            "context_bundle": {
                "initial_context": dict(episode.initial_context or {}),
                "workflow_class": episode.workflow_class,
            },
            "checkpoint_state": dict(current_checkpoint_state),
            "resume_token": resume_token,
            "artifact_requirements": dict(node.artifact_requirements or {}),
            "trace_policy": dict(node.trace_policy or {}),
            "inputs": {
                **dict(node.episode_input or {}),
                "workflow_episode_id": episode.episode_id,
                "upstream_node_outputs": upstream_outputs,
                "checkpoint_state": dict(current_checkpoint_state),
                "resume_token": resume_token,
                "artifact_requirements": dict(node.artifact_requirements or {}),
                "context_bundle": {
                    "initial_context": dict(episode.initial_context or {}),
                    "workflow_class": episode.workflow_class,
                },
            },
            "metadata": {
                "planner_node_id": node.node_id,
                "role_id": node.role_id,
                "upstream_node_ids": list(node.input_node_ids),
                "workflow_spec_id": episode.workflow_spec_id,
                "workflow_version": episode.workflow_version,
                "trace_policy": dict(node.trace_policy or {}),
                "replay_mode": replay_mode,
            },
        }
        try:
            raw_response = await (invoke_miner_fn or invoke_miner)(
                endpoint=node.endpoint,
                payload=payload,
                timeout_seconds=30.0,
            )
            runtime_contract_mode = runtime_contract_mode_from_payload(raw_response)
            validate_runtime_response_payload(payload=raw_response)
            response = AgentInvocationResponse.model_validate(raw_response)
            metadata = normalized_runtime_metadata(response)
            total_latency_ms += int(response.latency_ms)
            total_cost_tao += float(response.cost_tao)
            checkpoint_events.extend(list(metadata.get("checkpoint_events", []) or []))
            tool_calls.extend(list(metadata.get("tool_calls", []) or []))
            last_response = response
            last_payload = payload
            if response.status == "deferred":
                deferred_rounds += 1
                resume_token = str(metadata.get("resume_token") or "")
                runtime_state_patch = (
                    metadata.get("runtime_state_patch", {})
                    if isinstance(metadata.get("runtime_state_patch"), dict)
                    else {}
                )
                current_checkpoint_state = merge_checkpoint_state(current_checkpoint_state, runtime_state_patch)
                await _emit_update(
                    update_callback,
                    {
                        "event_type": "deferred",
                        "episode_id": episode.episode_id,
                        "node_id": node.node_id,
                        "role_id": node.role_id,
                        "family_id": node.family_id,
                        "checkpoint_state": dict(current_checkpoint_state),
                        "runtime_state_patch": dict(runtime_state_patch),
                        "resume_token": resume_token,
                        "checkpoint_events": list(checkpoint_events),
                        "tool_calls": list(tool_calls),
                        "retry_count": deferred_rounds,
                        "runtime_contract_mode": runtime_contract_mode,
                    },
                )
                continue
            break
        except Exception as exc:
            last_error = str(exc)
            last_response = None
            last_payload = payload
            break
    if last_response is None:
        trace = NodeTrace(
            episode_id=episode.episode_id,
            node_id=node.node_id,
            role_id=node.role_id,
            family_id=node.family_id,
            miner_hotkey=node.miner_hotkey,
            status="failed",
            input_digest=sha256_hex(json.dumps(last_payload or {}, sort_keys=True, default=str).encode()),
            output_digest=None,
            tool_calls=tool_calls,
            artifact_refs=[],
            latency_ms=total_latency_ms,
            cost_tao=total_cost_tao,
            retry_count=deferred_rounds,
            runtime_state_patch=dict(current_checkpoint_state or {}),
            checkpoint_events=checkpoint_events,
            handoff_payload={},
            metadata={
                "error": last_error or "workflow episode node failed",
                "resume_attempt_count": deferred_rounds,
                "replay_mode": replay_mode,
                "runtime_contract_mode": runtime_contract_mode,
            },
            runtime_contract_mode=runtime_contract_mode,
        )
        await _emit_update(
            update_callback,
            {
                "event_type": "failed",
                "episode_id": episode.episode_id,
                "node_id": node.node_id,
                "role_id": node.role_id,
                "family_id": node.family_id,
                "trace": trace.model_dump(mode="json"),
                "checkpoint_state": dict(current_checkpoint_state),
                "runtime_contract_mode": runtime_contract_mode,
                "error": last_error or "workflow episode node failed",
            },
        )
        return (
            {"output": {}, "artifacts": [], "metadata": dict(trace.metadata), "status": "failed", "error": last_error},
            trace,
        )
    if last_response.status == "deferred":
        error = "workflow episode node exceeded max resume attempts"
        trace = NodeTrace(
            episode_id=episode.episode_id,
            node_id=node.node_id,
            role_id=node.role_id,
            family_id=node.family_id,
            miner_hotkey=node.miner_hotkey,
            status="failed",
            input_digest=sha256_hex(json.dumps(last_payload or {}, sort_keys=True, default=str).encode()),
            output_digest=None,
            tool_calls=tool_calls,
            artifact_refs=[artifact.model_dump(mode="json") for artifact in last_response.artifacts],
            latency_ms=total_latency_ms,
            cost_tao=total_cost_tao,
            retry_count=deferred_rounds,
            runtime_state_patch=merge_checkpoint_state(current_checkpoint_state, response_runtime_patch(last_response)),
            checkpoint_events=checkpoint_events,
            handoff_payload={},
            metadata={
                **normalized_runtime_metadata(last_response),
                "error": error,
                "resume_attempt_count": deferred_rounds,
                "replay_mode": replay_mode,
                "runtime_contract_mode": runtime_contract_mode,
            },
            runtime_contract_mode=runtime_contract_mode,
        )
        await _emit_update(
            update_callback,
            {
                "event_type": "failed",
                "episode_id": episode.episode_id,
                "node_id": node.node_id,
                "role_id": node.role_id,
                "family_id": node.family_id,
                "trace": trace.model_dump(mode="json"),
                "checkpoint_state": dict(trace.runtime_state_patch or {}),
                "runtime_contract_mode": runtime_contract_mode,
                "error": error,
            },
        )
        return (
            {"output": {}, "artifacts": [], "metadata": dict(trace.metadata), "status": "failed", "error": error},
            trace,
        )
    metadata = normalized_runtime_metadata(last_response)
    quality_score = node_quality_score(metadata, status=last_response.status)
    if node.family_id == "media" and last_response.status == "completed":
        workflow_slice_metadata = dict(episode.metadata.get("workflow_slice_metadata", {}) or {})
        task_inputs = {
            **dict(episode.initial_context or {}),
            **workflow_slice_metadata,
        }
        expected_output = {
            **dict(node.artifact_requirements or {}),
            "required_artifact_kind": str(
                workflow_slice_metadata.get("required_artifact_kind")
                or workflow_slice_metadata.get("output_modality")
                or ""
            ).strip()
            or dict(node.artifact_requirements or {}).get("required_artifact_kind"),
        }
        artifact_evaluation_summary = await evaluate_multimodal_artifacts(
            family_id="media",
            payload=last_response.model_dump(mode="json"),
            task_inputs=task_inputs,
            expected_output=expected_output,
            timeout_seconds=30.0,
        )
        media_scores = score_media_generation_payload(
            miner_hotkey=node.miner_hotkey or "unknown-media-miner",
            prompt=task_prompt,
            payload=last_response.model_dump(mode="json"),
            task_inputs=task_inputs,
            expected_output=expected_output,
            artifact_evaluation_summary=artifact_evaluation_summary,
        )
        metadata = {
            **metadata,
            "artifact_evaluation_summary": artifact_evaluation_summary,
            "family_diagnostics": dict(media_scores.get("family_diagnostics", {}) or {}),
            "failure_taxonomy": dict(media_scores.get("failure_taxonomy", {}) or {}),
        }
        quality_score = max(
            0.0,
            min(1.0, float(media_scores.get("overall_score", quality_score) or quality_score)),
        )
    trace = NodeTrace(
        episode_id=episode.episode_id,
        node_id=node.node_id,
        role_id=node.role_id,
        family_id=node.family_id,
        miner_hotkey=node.miner_hotkey,
        status="completed" if last_response.status == "completed" else "failed",
        input_digest=sha256_hex(json.dumps(last_payload or {}, sort_keys=True, default=str).encode()),
        output_digest=sha256_hex(json.dumps(last_response.output, sort_keys=True, default=str).encode()),
        tool_calls=tool_calls,
        artifact_refs=[artifact.model_dump(mode="json") for artifact in last_response.artifacts],
        latency_ms=total_latency_ms,
        cost_tao=total_cost_tao,
        retry_count=deferred_rounds,
        runtime_state_patch=merge_checkpoint_state(
            current_checkpoint_state,
            dict(metadata.get("runtime_state_patch", {}) or {}),
        ),
        checkpoint_events=checkpoint_events,
        handoff_payload={
            "output": dict(last_response.output or {}),
            "artifacts": [artifact.model_dump(mode="json") for artifact in last_response.artifacts],
            "metadata": metadata,
        },
        local_role_score_hint=quality_score,
        reliability_score_hint=response_score_hint(last_response, "reliability_score"),
        recovery_score_hint=response_score_hint(last_response, "recovery_score"),
        metadata={
            **metadata,
            "resume_attempt_count": deferred_rounds,
            "replay_mode": replay_mode,
            "runtime_contract_mode": runtime_contract_mode,
        },
        runtime_contract_mode=runtime_contract_mode,
    )
    protocol_violations = validate_protocol_compliance(last_response, family_id=node.family_id)
    if protocol_violations:
        trace.metadata["protocol_violations"] = protocol_violations
        trace.metadata["protocol_contract_pass"] = False
    else:
        trace.metadata["protocol_contract_pass"] = True
    completed_output = completed_output_from_response(last_response)
    await _emit_update(
        update_callback,
        {
            "event_type": "completed" if last_response.status == "completed" else "failed",
            "episode_id": episode.episode_id,
            "node_id": node.node_id,
            "role_id": node.role_id,
            "family_id": node.family_id,
            "trace": trace.model_dump(mode="json"),
            "completed_output": completed_output,
            "checkpoint_state": dict(trace.runtime_state_patch or {}),
            "runtime_contract_mode": runtime_contract_mode,
        },
    )
    return completed_output, trace


async def execute_episode_nodes(
    *,
    episode: WorkflowEpisode,
    task_prompt: str,
    completed_outputs: dict[str, dict[str, object]] | None = None,
    existing_traces: list[NodeTrace] | None = None,
    update_callback: WorkflowNodeUpdateCallback | None = None,
    abort_callback: WorkflowAbortCallback | None = None,
    invoke_miner_fn: Callable[..., Awaitable[dict[str, object]]] | None = None,
) -> tuple[dict[str, dict[str, object]], list[NodeTrace], dict[str, float]]:
    completed_outputs = {
        str(node_id): dict(value)
        for node_id, value in dict(completed_outputs or {}).items()
    }
    traces_by_node = {
        trace.node_id: trace
        for trace in list(existing_traces or [])
    }
    family_skill_scores: dict[str, float] = {}
    for trace in traces_by_node.values():
        if trace.miner_hotkey:
            family_skill_scores[str(trace.miner_hotkey)] = max(
                family_skill_scores.get(str(trace.miner_hotkey), 0.0),
                max(0.0, min(1.0, float(trace.local_role_score_hint or node_quality_score(dict(trace.metadata or {}), status=trace.status)))),
            )
    for node in episode.nodes:
        if node.node_id in completed_outputs:
            continue
        if abort_callback is not None:
            abort_reason = abort_callback(episode.episode_id, node.node_id, 0)
            if abort_reason is not None and hasattr(abort_reason, "__await__"):
                abort_reason = await abort_reason
            if isinstance(abort_reason, str) and abort_reason.strip():
                raise WorkflowEpisodeAbortedError(abort_reason.strip())
        completed_output, trace = await execute_workflow_episode_node(
            episode=episode,
            node=node,
            task_prompt=task_prompt,
            completed_outputs=completed_outputs,
            update_callback=update_callback,
            abort_callback=abort_callback,
            invoke_miner_fn=invoke_miner_fn,
        )
        completed_outputs[node.node_id] = completed_output
        traces_by_node[node.node_id] = trace
        quality_score = node_quality_score(dict(trace.metadata or {}), status=trace.status)
        if trace.miner_hotkey:
            family_skill_scores[str(trace.miner_hotkey)] = max(
                family_skill_scores.get(str(trace.miner_hotkey), 0.0),
                max(0.0, min(1.0, quality_score)),
            )
    ordered_traces = [traces_by_node[node.node_id] for node in episode.nodes if node.node_id in traces_by_node]
    return completed_outputs, ordered_traces, family_skill_scores
