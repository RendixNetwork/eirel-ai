from __future__ import annotations

"""Workflow episode management and task orchestration.

Extracted from ``ManagedOwnerServices`` (Item 15) to reduce the size of
the god-object.  Each public method here has a thin delegation wrapper
in ``ManagedOwnerServices`` for backward compatibility.
"""

from typing import Any, TYPE_CHECKING

from shared.contracts.models import WorkflowSpec
from shared.workflow_specs import (
    get_workflow_spec as _get_workflow_spec,
    list_workflow_specs as _list_workflow_specs,
    workflow_corpus_public_metadata,
)

if TYPE_CHECKING:
    from control_plane.owner_api.managed import ManagedOwnerServices


class TaskOrchestrator:
    """Handles workflow specs, workflow episodes, and DAG dispatch."""

    def __init__(self, owner: ManagedOwnerServices) -> None:
        self._owner = owner

    @property
    def db(self):
        return self._owner.db

    @property
    def settings(self):
        return self._owner.settings

    def list_workflow_specs(self) -> list[WorkflowSpec]:
        return _list_workflow_specs()

    def get_workflow_spec(self, workflow_spec_id: str) -> WorkflowSpec:
        return _get_workflow_spec(workflow_spec_id)

    def workflow_spec_payload(self, workflow_spec: WorkflowSpec) -> dict[str, Any]:
        return workflow_spec.model_dump(mode="json")

    def workflow_corpus_public_payload(self) -> dict[str, Any]:
        return workflow_corpus_public_metadata()
