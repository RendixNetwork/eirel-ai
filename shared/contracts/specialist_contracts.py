from __future__ import annotations

from copy import deepcopy
from typing import Any

from eirel.groups import FAMILY_IDS, FamilyId, ensure_family_id


FAMILY_CONTRACTS_VERSION = "family-contracts"
SPECIALIST_CONTRACTS_VERSION = FAMILY_CONTRACTS_VERSION

# ---------------------------------------------------------------------------
# Family protocol: required output fields
#
# The "protocol" is not HTTP — it is the required fields each family agent
# must populate in its AgentInvocationResponse to communicate correctly with
# the control plane.
#
# Runtime fields required by ALL families (top-level on AgentInvocationResponse):
#   checkpoint_events: list[dict]   — may be []
#   runtime_state_patch: dict       — may be {}
#   resume_token: str               — required only when status == "deferred"
#
# Family-specific required fields live in `output` (or `artifacts` for media).
# ---------------------------------------------------------------------------

RUNTIME_REQUIRED_FIELDS: tuple[str, ...] = ("checkpoint_events", "runtime_state_patch")
RUNTIME_DEFERRED_REQUIRED: tuple[str, ...] = ("resume_token",)

# Per-family required and optional output field definitions.
# "required" fields must be present in response.output when status == "completed".
# "artifacts_required" means response.artifacts must be non-empty instead.
FAMILY_REQUIRED_OUTPUT_FIELDS: dict[FamilyId, dict[str, Any]] = {
    "analyst": {
        "required": ["summary"],
        "optional": ["reasoning", "confidence", "citations"],
        "required_types": {"summary": str},
        "optional_types": {"reasoning": str, "confidence": float},
        "artifacts_required": False,
    },
    "browser": {
        "required": ["content"],
        "optional": ["screenshots", "extracted_data", "navigation_trace"],
        "required_types": {"content": str},
        "optional_types": {"screenshots": list, "extracted_data": dict, "navigation_trace": list},
        "artifacts_required": False,
    },
    "builder": {
        "required": ["project_state"],
        "optional": ["code", "language", "tests", "implementation_notes", "commit_log", "test_results"],
        "required_types": {"project_state": dict},
        "optional_types": {"code": str, "language": str, "commit_log": list, "test_results": dict},
        "artifacts_required": False,
    },
    "media": {
        # media agents return results via top-level `artifacts`, not `output`
        "required": [],
        "optional": [],
        "required_types": {},
        "optional_types": {},
        "artifacts_required": True,
    },
    "data": {
        "required": ["result"],
        "optional": ["query", "schema", "row_count"],
        "required_types": {"result": (dict, list, str)},
        "optional_types": {"query": str, "schema": dict, "row_count": int},
        "artifacts_required": False,
    },
    "memory": {
        "required": ["recalled_context"],
        "optional": ["relevance_scores", "source_documents"],
        "required_types": {"recalled_context": (str, list, dict)},
        "optional_types": {"relevance_scores": list, "source_documents": list},
        "artifacts_required": False,
    },
    "planner": {
        "required": ["plan"],
        "optional": ["reasoning", "confidence"],
        "required_types": {"plan": (dict, list)},
        "optional_types": {"reasoning": str, "confidence": float},
        "artifacts_required": False,
    },
    "verifier": {
        "required": ["verdict"],
        "optional": ["issues", "false_positive_probability"],
        "required_types": {"verdict": str},
        "optional_types": {"issues": list, "false_positive_probability": float},
        "allowed_verdict_values": ["pass", "fail", "conditional"],
        "artifacts_required": False,
    },
}

_FAMILY_CONTRACTS: dict[FamilyId, dict[str, Any]] = {
    "analyst": {
        "role": "task understanding, decomposition, research, reasoning, evidence-grounded synthesis, and recommendations",
        "context_contract": "task packet + evidence bundle + workflow state + upstream outputs",
        "latency_budget_ms": 45000,
        "output_fields": FAMILY_REQUIRED_OUTPUT_FIELDS["analyst"],
        "default_tools": [
            {"name": "retrieval_search", "description": "Search the web for information"},
            {"name": "browser_open", "description": "Open a web page and read its content"},
            {"name": "browser_find_on_page", "description": "Find specific text on the current page"},
        ],
        "quality_constraints": [
            "correctness",
            "completeness",
            "evidence use",
            "reasoning quality",
            "calibrated confidence",
            "schema compliance",
        ],
        "failure_semantics": [
            "unsupported_claim",
            "unsupported_citation",
            "unfaithful_synthesis",
            "over_refusal",
            "schema_violation",
        ],
        "trust_policy": {
            "control_plane_may_trust": ["output_shape", "plan skeleton"],
            "control_plane_must_verify": ["claims", "evidence use", "confidence calibration"],
        },
    },
    "browser": {
        "role": "autonomous web navigation, page interaction, content extraction, and web-based data gathering",
        "context_contract": "task packet + target URLs + extraction schema + upstream outputs",
        "latency_budget_ms": 90000,
        "output_fields": FAMILY_REQUIRED_OUTPUT_FIELDS["browser"],
        "default_tools": [
            {"name": "browser_navigate", "description": "Navigate to a URL"},
            {"name": "browser_click", "description": "Click an element on the page"},
            {"name": "browser_extract", "description": "Extract structured data from the page"},
            {"name": "browser_screenshot", "description": "Take a screenshot of the current page"},
        ],
        "quality_constraints": [
            "content accuracy",
            "extraction completeness",
            "navigation efficiency",
            "page interaction correctness",
        ],
        "failure_semantics": ["navigation_error", "extraction_incomplete", "page_timeout", "captcha_blocked", "content_mismatch"],
        "trust_policy": {
            "control_plane_may_trust": ["output shape", "navigation trace"],
            "control_plane_must_verify": ["content accuracy", "extraction completeness"],
        },
    },
    "builder": {
        "role": "Autonomous code generation from specification documents. Builds complete projects including code, tests, documentation, and configuration.",
        "context_contract": "specification document + optional upstream analyst output",
        "latency_budget_ms": 172800000,
        "output_fields": FAMILY_REQUIRED_OUTPUT_FIELDS["builder"],
        "default_tools": [
            {"name": "code_interpreter", "description": "Execute code in a sandboxed environment"},
            {"name": "file_write", "description": "Write content to a file"},
            {"name": "file_read", "description": "Read content from a file"},
            {"name": "run_tests", "description": "Run test suite against implementation"},
            {"name": "shell_execute", "description": "Execute shell commands for setup and tooling"},
            {"name": "web_search", "description": "Search documentation and references"},
        ],
        "quality_constraints": [
            "functional correctness (tests must pass)",
            "specification coverage (all requirements addressed)",
            "code quality (linting, typing, complexity)",
            "autonomous iteration (self-correction on failures)",
            "safety (no exfiltration, no escapes)",
        ],
        "failure_semantics": [
            "timeout_exceeded",
            "no_tests_passing",
            "spec_not_covered",
            "safety_violation",
            "trace_fabrication",
        ],
        "trust_policy": {
            "control_plane_may_trust": ["artifact shape", "checkpoint events"],
            "control_plane_must_verify": ["test results", "spec coverage", "code quality", "safety"],
        },
    },
    "media": {
        "role": "image, audio, video, and multimodal asset generation from structured briefs",
        "context_contract": "task packet + analyst brief + optional reference artifacts",
        "latency_budget_ms": 120000,
        "output_fields": FAMILY_REQUIRED_OUTPUT_FIELDS["media"],
        "default_tools": [
            {"name": "image_generate", "description": "Generate an image from a text prompt"},
            {"name": "audio_synthesize", "description": "Synthesize audio from text or parameters"},
            {"name": "video_render", "description": "Render a video from a structured brief"},
        ],
        "quality_constraints": [
            "artifact delivery",
            "brief adherence",
            "multimodal consistency",
            "safe output contract",
        ],
        "failure_semantics": ["artifact_missing", "brief_miss", "render_failure", "unsafe_output"],
        "trust_policy": {
            "control_plane_may_trust": ["artifact metadata"],
            "control_plane_must_verify": ["artifact validity", "brief adherence", "consistency"],
        },
    },
    "data": {
        "role": "data extraction, transformation, SQL generation, structured data processing, and visualization",
        "context_contract": "task packet + data sources + schema context + upstream outputs",
        "latency_budget_ms": 60000,
        "output_fields": FAMILY_REQUIRED_OUTPUT_FIELDS["data"],
        "default_tools": [
            {"name": "sql_execute", "description": "Execute a SQL query against the data source"},
            {"name": "data_transform", "description": "Apply a transformation to a dataset"},
            {"name": "schema_inspect", "description": "Inspect the schema of a data source"},
        ],
        "quality_constraints": [
            "query correctness",
            "schema adherence",
            "transformation accuracy",
            "result completeness",
        ],
        "failure_semantics": ["query_error", "schema_mismatch", "data_loss", "type_coercion_error", "empty_result"],
        "trust_policy": {
            "control_plane_may_trust": ["output shape", "schema metadata"],
            "control_plane_must_verify": ["query correctness", "data integrity", "result completeness"],
        },
    },
    "memory": {
        "role": "long-term context persistence, user preferences, RAG over history, and session state management",
        "context_contract": "task packet + user context + session history + upstream outputs",
        "latency_budget_ms": 30000,
        "output_fields": FAMILY_REQUIRED_OUTPUT_FIELDS["memory"],
        "default_tools": [
            {"name": "memory_store", "description": "Store a key-value pair in long-term memory"},
            {"name": "memory_retrieve", "description": "Retrieve context by semantic query"},
            {"name": "memory_delete", "description": "Delete a stored memory entry"},
        ],
        "quality_constraints": [
            "recall relevance",
            "context freshness",
            "source attribution",
            "retrieval completeness",
        ],
        "failure_semantics": ["recall_miss", "stale_context", "irrelevant_retrieval", "attribution_error"],
        "trust_policy": {
            "control_plane_may_trust": ["output shape", "source metadata"],
            "control_plane_must_verify": ["recall relevance", "context freshness"],
        },
    },
    "planner": {
        "role": "task decomposition, workflow planning, intent classification, and family routing",
        "context_contract": "task packet + available families + workflow catalog",
        "latency_budget_ms": 15000,
        "output_fields": FAMILY_REQUIRED_OUTPUT_FIELDS["planner"],
        "default_tools": [],
        "quality_constraints": [
            "decomposition accuracy",
            "family assignment correctness",
            "dependency graph validity",
            "minimal node count",
        ],
        "failure_semantics": ["invalid_family", "circular_dependency", "missing_verifier", "over_decomposition"],
        "trust_policy": {
            "control_plane_may_trust": ["plan shape"],
            "control_plane_must_verify": ["family routing", "dependency validity"],
        },
    },
    "verifier": {
        "role": "perception, output auditing, factual cross-checking, issue detection, policy review, and repair guidance",
        "context_contract": "task packet + upstream outputs + evidence bundle + reference artifacts",
        "latency_budget_ms": 45000,
        "output_fields": FAMILY_REQUIRED_OUTPUT_FIELDS["verifier"],
        "default_tools": [
            {"name": "retrieval_search", "description": "Search for evidence to verify claims"},
            {"name": "browser_open", "description": "Open a page to cross-check facts"},
        ],
        "quality_constraints": [
            "issue detection quality",
            "grounded inspection",
            "low false-positive rate",
            "useful repair suggestions",
        ],
        "failure_semantics": ["missed_defect", "false_positive", "ungrounded_issue", "low_confidence_verdict"],
        "trust_policy": {
            "control_plane_may_trust": ["verdict shape"],
            "control_plane_must_verify": ["inspection grounding", "repair usefulness"],
        },
    },
}


def contract_for_family(family_id: FamilyId | str) -> dict[str, Any]:
    normalized = ensure_family_id(str(family_id))
    contract = deepcopy(_FAMILY_CONTRACTS[normalized])

    def _json_safe(value: Any) -> Any:
        if isinstance(value, type):
            return value.__name__
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [_json_safe(item) for item in value]
        return value

    return {
        "family_id": normalized,
        "contract_version": FAMILY_CONTRACTS_VERSION,
        **_json_safe(contract),
    }


def contracts_payload(family_ids: list[FamilyId] | tuple[FamilyId, ...] | set[FamilyId] | None = None) -> dict[str, Any]:
    families = [ensure_family_id(item) for item in (family_ids or FAMILY_IDS)]
    return {
        "contract_version": FAMILY_CONTRACTS_VERSION,
        "contracts": {
            family_id: contract_for_family(family_id)
            for family_id in families
            if family_id in _FAMILY_CONTRACTS
        },
    }


# ---------------------------------------------------------------------------
# Platform tool contracts (Tier 1 — subnet-owned infrastructure)
# ---------------------------------------------------------------------------

PLATFORM_TOOL_CONTRACTS_VERSION = "platform-tools"

PLATFORM_TOOL_CONTRACTS: dict[str, dict[str, Any]] = {
    "code_exec": {
        "name": "code_exec",
        "tier": 1,
        "description": "Execute code in a sandboxed Firecracker microVM",
        "supported_languages": ["python", "javascript", "shell"],
        "limits": {
            "max_cpu": "2 vCPU",
            "max_memory": "512MB",
            "max_runtime": "10 min",
            "network": "none",
            "cold_start": "<200ms",
        },
        "output_fields": {
            "required": ["stdout", "exit_code"],
            "optional": ["stderr", "language"],
        },
    },
    "web_search": {
        "name": "web_search",
        "tier": 1,
        "description": "Search the web for current information",
        "limits": {
            "max_results": 10,
            "timeout": "20s",
        },
        "output_fields": {
            "required": ["query", "results"],
            "optional": ["total"],
        },
    },
    "file_manager": {
        "name": "file_manager",
        "tier": 1,
        "description": "Session-scoped temporary file storage",
        "limits": {
            "max_file_size": "10MB",
            "session_scoped": True,
        },
        "output_fields": {
            "required": ["filename"],
            "optional": ["content", "size", "files"],
        },
    },
    "image_gen": {
        "name": "image_gen",
        "tier": 1,
        "description": "Generate images from text prompts via provider APIs",
        "limits": {
            "max_images_per_request": 4,
            "timeout": "60s",
        },
        "output_fields": {
            "required": ["images"],
            "optional": ["prompt"],
        },
    },
    "memory_recall": {
        "name": "memory_recall",
        "tier": 1,
        "description": "Recall context from conversation history and knowledge retrieval",
        "limits": {
            "max_results": 5,
            "timeout": "10s",
        },
        "output_fields": {
            "required": ["conversation_context", "retrieved_knowledge"],
            "optional": [],
        },
    },
}


def platform_tool_contract(tool_name: str) -> dict[str, Any] | None:
    """Return the contract for a platform tool, or None if unknown."""
    contract = PLATFORM_TOOL_CONTRACTS.get(tool_name)
    if contract is None:
        return None
    return {
        "contract_version": PLATFORM_TOOL_CONTRACTS_VERSION,
        **deepcopy(contract),
    }


def all_platform_tool_contracts() -> dict[str, Any]:
    """Return all platform tool contracts."""
    return {
        "contract_version": PLATFORM_TOOL_CONTRACTS_VERSION,
        "tools": {
            name: platform_tool_contract(name)
            for name in PLATFORM_TOOL_CONTRACTS
        },
    }
