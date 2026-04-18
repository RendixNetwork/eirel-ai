"""Module-level constants for the owner API managed services."""
from __future__ import annotations

# Post-clean-slate-refactor the subnet runs a single family. The legacy
# multi-family tuples and weight maps are gone — everything is general_chat.
PRODUCTION_FAMILIES = ("general_chat",)
ABV_FAMILIES = ("general_chat",)
DEFAULT_FIXED_FAMILY_WEIGHTS = {"general_chat": 1.0}
LAUNCH_FIXED_FAMILY_WEIGHTS = {"general_chat": 1.0}
FAMILY_SERVING_FAMILY_WEIGHT = 0.85
FAMILY_SERVING_RELIABILITY_WEIGHT = 0.15
ABV_SERVING_SELECTION_REASON = "family_protocol"
WORKFLOW_COMPOSITION_SELECTION_REASON = "derived_family_winners"
WORKFLOW_EPISODE_DEFAULT_MAX_ATTEMPTS = 3
WORKFLOW_EPISODE_RETRY_BASE_SECONDS = 30
WORKFLOW_EPISODE_RETRY_MAX_SECONDS = 300
WORKFLOW_RUNTIME_REMEDIATION_AUDIT_LIMIT = 20
WORKFLOW_RUNTIME_POLICY_STATE_KEY = "runtime_remediation_policy"
CHAIN_PUBLICATION_STATE_KEY = "chain_publication"
WORKFLOW_RUNTIME_SUPPRESSION_TARGET_KINDS = {
    "episode_id",
    "workflow_spec_id",
    "incident_state",
    "task_id",
}
WINNER_DOMINANCE_MARGIN = 5.0
PLACEMENT_RESERVED_STATUSES = {"assigned", "running", "draining"}
