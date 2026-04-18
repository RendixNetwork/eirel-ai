from __future__ import annotations

from copy import deepcopy
from typing import Any

from shared.contracts.models import WorkflowEpisode, WorkflowEpisodeNode, WorkflowSpec, WorkflowSpecNode


WORKFLOW_SPECS: tuple[WorkflowSpec, ...] = (
    WorkflowSpec(
        workflow_spec_id="general_chat_single_node_v1",
        workflow_version="v1",
        workflow_class="general_chat",
        description="Single-node general_chat conversation — one family, one miner, multi-turn.",
        allowed_tool_policy={
            "mode": "bounded",
            "families": {"general_chat": "standard"},
        },
        episode_budget={
            "max_nodes": 1,
            "max_wall_clock_seconds": 120,
            "max_retry_count": 1,
        },
        expected_deliverables={"final_output": "conversation_response"},
        scoring_hooks={
            "final_outcome": "general_chat_conversation_v1",
            "handoff_edges": [],
        },
        credit_assignment_version="general_chat_v1",
        nodes=[
            WorkflowSpecNode(
                node_id="general_chat_agent",
                role_id="conversation",
                family_id="general_chat",
                description="Answer a multi-turn user conversation, optionally invoking tool services.",
                expected_deliverables={"kind": "conversation_response"},
                scoring_hooks={"local_role": "general_chat_conversation_v1"},
                terminal=True,
            ),
        ],
    ),
)


def list_workflow_specs() -> list[WorkflowSpec]:
    return [spec.model_copy(deep=True) for spec in WORKFLOW_SPECS]


def get_workflow_spec(workflow_spec_id: str) -> WorkflowSpec:
    for spec in WORKFLOW_SPECS:
        if spec.workflow_spec_id == workflow_spec_id:
            return spec.model_copy(deep=True)
    raise KeyError(workflow_spec_id)


def build_workflow_episode(
    *,
    workflow_spec: WorkflowSpec,
    task_prompt: str,
    run_id: str | None = None,
    coalition: dict[str, dict[str, Any]] | None = None,
    initial_context: dict[str, Any] | None = None,
    hidden_slice: bool = False,
    metadata: dict[str, Any] | None = None,
) -> WorkflowEpisode:
    coalition = coalition or {}
    initial_context = initial_context or {}
    metadata = metadata or {}
    nodes: list[WorkflowEpisodeNode] = []
    for spec_node in workflow_spec.nodes:
        coalition_entry = coalition.get(spec_node.node_id) or {}
        nodes.append(
            WorkflowEpisodeNode(
                node_id=spec_node.node_id,
                role_id=spec_node.role_id,
                family_id=spec_node.family_id,
                input_node_ids=list(spec_node.input_node_ids),
                miner_hotkey=coalition_entry.get("miner_hotkey"),
                endpoint=coalition_entry.get("endpoint"),
                episode_input={
                    "task_prompt": task_prompt,
                    "workflow_spec_id": workflow_spec.workflow_spec_id,
                    "workflow_version": workflow_spec.workflow_version,
                    "role_id": spec_node.role_id,
                    "expected_deliverables": deepcopy(spec_node.expected_deliverables),
                    "node_description": spec_node.description,
                },
                checkpoint_state={},
                tool_budget=deepcopy(workflow_spec.episode_budget),
                artifact_requirements=deepcopy(spec_node.expected_deliverables),
                trace_policy={"capture_tool_calls": True, "capture_state_patch": True},
                metadata={
                    "workflow_class": workflow_spec.workflow_class,
                    "scoring_hooks": deepcopy(spec_node.scoring_hooks),
                    **{
                        str(key): deepcopy(value)
                        for key, value in coalition_entry.items()
                        if str(key) not in {"miner_hotkey", "endpoint"}
                    },
                },
            )
        )
    return WorkflowEpisode(
        workflow_spec_id=workflow_spec.workflow_spec_id,
        workflow_version=workflow_spec.workflow_version,
        workflow_class=workflow_spec.workflow_class,
        run_id=run_id,
        task_prompt=task_prompt,
        nodes=nodes,
        initial_context=initial_context,
        hidden_slice=hidden_slice,
        metadata={
            "workflow_spec": workflow_spec.model_dump(mode="json"),
            "credit_assignment_version": workflow_spec.credit_assignment_version,
            **metadata,
        },
    )
