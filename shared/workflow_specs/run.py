from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any

from shared.contracts.models import (
    ContributionScoreRecord,
    HandoffTrace,
    NodeTrace,
    WorkflowEpisode,
    WorkflowEpisodeResult,
)


LOCAL_ROLE_WEIGHT = 0.30
HANDOFF_WEIGHT = 0.20
MARGINAL_LIFT_WEIGHT = 0.30
RECOVERY_WEIGHT = 0.10
RELIABILITY_WEIGHT = 0.10
def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _digest_payload(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _node_output_score(trace: NodeTrace) -> float:
    if trace.local_role_score_hint is not None:
        return _clamp(float(trace.local_role_score_hint))
    metadata = dict(trace.metadata or {})
    for key in ("overall_score", "official_family_score", "score"):
        raw = metadata.get(key)
        if isinstance(raw, (int, float)):
            return _clamp(float(raw))
    if trace.status != "completed":
        return 0.0
    if trace.output_digest:
        return 0.7
    return 0.5


def _reliability_score(trace: NodeTrace) -> float:
    if trace.reliability_score_hint is not None:
        return _clamp(float(trace.reliability_score_hint))
    penalty = 0.0
    if trace.status != "completed":
        penalty += 0.6
    penalty += min(0.25, trace.retry_count * 0.05)
    if trace.latency_ms > 0:
        penalty += min(0.15, trace.latency_ms / 120_000.0)
    return _clamp(1.0 - penalty)


def _recovery_score(trace: NodeTrace) -> float:
    if trace.recovery_score_hint is not None:
        return _clamp(float(trace.recovery_score_hint))
    metadata = dict(trace.metadata or {})
    if isinstance(metadata.get("repaired_after_feedback"), bool):
        return 1.0 if metadata["repaired_after_feedback"] else 0.25
    if trace.role_id in {"repair", "reaudit", "verification"} and trace.status == "completed":
        return 0.65
    return 0.25 if trace.status == "completed" else 0.0


def _resolved_final_outcome_score(
    *,
    traces: list[NodeTrace],
    provided_score: float | None,
    final_output: dict[str, Any],
) -> float:
    if provided_score is not None:
        return _clamp(float(provided_score))
    successful = [_node_output_score(trace) for trace in traces if trace.status == "completed"]
    if successful:
        return _clamp(sum(successful) / len(successful))
    return 0.75 if final_output else 0.0


def evaluate_workflow_episode(
    *,
    episode: WorkflowEpisode,
    node_traces: list[NodeTrace],
    final_output: dict[str, Any] | None = None,
    final_outcome_score: float | None = None,
) -> WorkflowEpisodeResult:
    final_output = dict(final_output or {})
    trace_by_node = {trace.node_id: trace for trace in node_traces}
    handoff_traces: list[HandoffTrace] = []
    outgoing_scores: dict[str, list[float]] = defaultdict(list)
    successful_nodes: set[str] = set()
    failed_nodes: set[str] = set()

    for trace in node_traces:
        if trace.status == "completed":
            successful_nodes.add(trace.node_id)
        else:
            failed_nodes.add(trace.node_id)

    resolved_final_outcome_score = _resolved_final_outcome_score(
        traces=node_traces,
        provided_score=final_outcome_score,
        final_output=final_output,
    )

    for node in episode.nodes:
        source_trace = trace_by_node.get(node.node_id)
        for upstream_node_id in node.input_node_ids:
            upstream_trace = trace_by_node.get(upstream_node_id)
            if upstream_trace is None:
                usefulness_score = 0.0
                payload_digest = None
                accepted = False
            else:
                local_hint = _node_output_score(upstream_trace)
                accepted = upstream_trace.status == "completed" and source_trace is not None and source_trace.status == "completed"
                usefulness_score = local_hint * (1.0 if accepted else 0.4)
                payload_digest = _digest_payload(upstream_trace.handoff_payload or upstream_trace.metadata or {})
            usefulness_score = _clamp(usefulness_score)
            outgoing_scores[upstream_node_id].append(usefulness_score)
            handoff_traces.append(
                HandoffTrace(
                    episode_id=episode.episode_id,
                    edge_id=f"{upstream_node_id}->{node.node_id}",
                    from_node_id=upstream_node_id,
                    to_node_id=node.node_id,
                    usefulness_score=usefulness_score,
                    payload_digest=payload_digest,
                    accepted=accepted,
                    metadata={"workflow_spec_id": episode.workflow_spec_id},
                )
            )

    contribution_records: list[ContributionScoreRecord] = []
    for trace in node_traces:
        local_role_score = _node_output_score(trace)
        handoff_score = _clamp(
            sum(outgoing_scores.get(trace.node_id, [])) / len(outgoing_scores[trace.node_id])
        ) if outgoing_scores.get(trace.node_id) else (0.7 if trace.status == "completed" and trace.node_id == episode.nodes[-1].node_id else 0.0)
        if trace.counterfactual_final_outcome_score is not None:
            marginal_lift_score = _clamp(resolved_final_outcome_score - float(trace.counterfactual_final_outcome_score))
        else:
            marginal_lift_score = _clamp(min(local_role_score, resolved_final_outcome_score))
        recovery_score = _recovery_score(trace)
        reliability_score = _reliability_score(trace)
        contribution_score = _clamp(
            LOCAL_ROLE_WEIGHT * local_role_score
            + HANDOFF_WEIGHT * handoff_score
            + MARGINAL_LIFT_WEIGHT * marginal_lift_score
            + RECOVERY_WEIGHT * recovery_score
            + RELIABILITY_WEIGHT * reliability_score
        )
        contribution_records.append(
            ContributionScoreRecord(
                episode_id=episode.episode_id,
                node_id=trace.node_id,
                role_id=trace.role_id,
                family_id=trace.family_id,
                miner_hotkey=trace.miner_hotkey,
                local_role_score=local_role_score,
                handoff_score=handoff_score,
                marginal_lift_score=marginal_lift_score,
                recovery_score=recovery_score,
                reliability_score=reliability_score,
                contribution_score=contribution_score,
                credit_assignment_version=str(
                    episode.metadata.get("credit_assignment_version") or "workflow_contribution_v1"
                ),
                metadata={"status": trace.status},
            )
        )

    result_status = "completed" if failed_nodes == set() else "failed"
    lineage = [node.node_id for node in episode.nodes if node.node_id in trace_by_node]
    return WorkflowEpisodeResult(
        episode_id=episode.episode_id,
        workflow_spec_id=episode.workflow_spec_id,
        workflow_version=episode.workflow_version,
        workflow_class=episode.workflow_class,
        status=result_status,
        final_output=final_output,
        final_outcome_score=resolved_final_outcome_score,
        node_traces=node_traces,
        handoff_traces=handoff_traces,
        contribution_records=contribution_records,
        metadata={
            "run_id": episode.run_id,
            "successful_node_ids": sorted(successful_nodes),
            "failed_node_ids": sorted(failed_nodes),
            "lineage": lineage,
            "credit_assignment_version": episode.metadata.get("credit_assignment_version"),
        },
    )

