from __future__ import annotations

import os
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_fixture_path(
    *,
    env_name: str,
    container_default: str,
    workspace_default: str,
    repo_relative_default: str,
    expected_kind: str,
) -> str:
    configured = os.getenv(env_name)
    repo_default = (_repo_root() / repo_relative_default).resolve()
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    else:
        candidates.extend(
            [
                Path(container_default),
                Path(workspace_default),
                repo_default,
            ]
        )
    checked_paths: list[str] = []
    for candidate in candidates:
        path = candidate.resolve()
        checked_paths.append(str(path))
        if expected_kind == "file" and path.is_file():
            return str(path)
        if expected_kind == "directory" and path.is_dir():
            return str(path)
    configured_hint = f"{env_name}={configured!r}" if configured is not None else f"{env_name} is unset"
    raise ValueError(
        f"invalid {env_name}: expected an existing {expected_kind}; "
        f"checked {', '.join(checked_paths)} ({configured_hint})"
    )


def _validate_owner_dataset_root(path: str) -> str:
    resolved = Path(path).resolve()
    if "examples" in resolved.parts:
        raise ValueError(
            "invalid EIREL_OWNER_DATASET_ROOT_PATH: official owner dataset roots may not live under examples/**"
        )
    return str(resolved)


_DATASET_SOURCE_TYPES = frozenset({"filesystem", "s3"})


def _validate_dataset_source_type(value: str) -> None:
    if value not in _DATASET_SOURCE_TYPES:
        raise ValueError(
            f"invalid EIREL_OWNER_DATASET_SOURCE_TYPE={value!r}: "
            f"must be one of {sorted(_DATASET_SOURCE_TYPES)}"
        )


def _validate_forge_cross_vendor(
    *,
    generator_provider: str,
    generator_base_url: str,
    judge_base_url: str,
    must_differ: bool,
) -> None:
    if not must_differ:
        return
    if not generator_provider and not generator_base_url:
        return  # Forge not configured yet; no collision to check.
    if generator_base_url and judge_base_url and generator_base_url == judge_base_url:
        raise ValueError(
            "invalid EIREL_DATASET_FORGE_GENERATOR_BASE_URL: must differ from EIREL_JUDGE_BASE_URL "
            "(set EIREL_DATASET_FORGE_JUDGE_PROVIDER_MUST_DIFFER=false to override)"
        )


def _resolve_baremetal_inventory_path() -> str:
    """Resolve the baremetal inventory path.

    Priority:
    1. ``OWNER_BAREMETAL_INVENTORY_PATH`` env var (explicit override)
    2. ``<repo>/eirel-ai/local/inventory.yaml`` (gitignored, operator-specific)
    3. ``/etc/eirel/inventory.yaml`` (system-wide, outside repo)
    4. ``<repo>/eirel-ai/deploy/baremetal/inventory/hosts.yaml`` (in-repo template)
    5. Empty string (no inventory configured)
    """
    explicit = os.getenv("OWNER_BAREMETAL_INVENTORY_PATH")
    if explicit is not None:
        return explicit

    repo_root = _repo_root()
    candidates = [
        repo_root / "eirel-ai" / "local" / "inventory.yaml",
        Path("/etc/eirel/inventory.yaml"),
        repo_root / "eirel-ai" / "deploy" / "baremetal" / "inventory" / "hosts.yaml",
    ]
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return str(resolved)
    return ""


@dataclass(slots=True)
class Settings:
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", "postgresql+psycopg://eirel:eirel@postgres:5432/eirel"
        )
    )
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", ""))
    launch_mode: str = field(
        default_factory=lambda: os.getenv("LAUNCH_MODE", "development")
    )
    use_redis_pool: bool = field(
        default_factory=lambda: _bool_env("USE_REDIS_POOL", False)
    )
    archive_limit_bytes: int = field(
        default_factory=lambda: int(os.getenv("ARCHIVE_LIMIT_BYTES", str(200 * 1024 * 1024)))
    )
    lease_seconds: int = field(default_factory=lambda: int(os.getenv("LEASE_SECONDS", "60")))
    assignment_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("ASSIGNMENT_TIMEOUT_SECONDS", "600"))
    )
    signature_ttl_seconds: int = field(
        default_factory=lambda: int(os.getenv("SIGNATURE_TTL_SECONDS", "60"))
    )
    minimum_validator_stake: int = field(
        default_factory=lambda: int(os.getenv("MINIMUM_VALIDATOR_STAKE", "0"))
    )
    metagraph_sync_interval_seconds: int = field(
        default_factory=lambda: int(os.getenv("METAGRAPH_SYNC_INTERVAL_SECONDS", "60"))
    )
    pool_reconcile_interval_seconds: int = field(
        default_factory=lambda: int(os.getenv("POOL_RECONCILE_INTERVAL_SECONDS", "30"))
    )
    result_aggregation_interval_seconds: int = field(
        default_factory=lambda: int(
            os.getenv("RESULT_AGGREGATION_INTERVAL_SECONDS", "15")
        )
    )
    task_quorum: int = field(
        default_factory=lambda: int(os.getenv("TASK_QUORUM", "3"))
    )
    max_task_attempts: int = field(
        default_factory=lambda: int(os.getenv("MAX_TASK_ATTEMPTS", "3"))
    )
    max_inflight_tasks_per_submission: int = field(
        default_factory=lambda: int(os.getenv("MAX_INFLIGHT_TASKS_PER_SUBMISSION", "2"))
    )
    epoch_id: str = field(
        default_factory=lambda: os.getenv("EIREL_EPOCH_ID", "epoch-public-v2")
    )
    epoch_seed: str = field(
        default_factory=lambda: os.getenv("EIREL_EPOCH_SEED", "public-epoch-seed-v2")
    )
    epoch_seed_commitment: str = field(
        default_factory=lambda: os.getenv("EIREL_EPOCH_SEED_COMMITMENT", "")
    )
    validator_poll_interval_seconds: float = field(
        default_factory=lambda: float(
            os.getenv("EIREL_VALIDATOR_POLL_INTERVAL_SECONDS")
            or os.getenv("VALIDATOR_POLL_INTERVAL_SECONDS")
            or "1"
        )
    )
    validator_max_concurrent_jobs: int = field(
        default_factory=lambda: int(os.getenv("VALIDATOR_MAX_CONCURRENT_JOBS", "2"))
    )
    validator_max_concurrent_builds: int = field(
        default_factory=lambda: int(os.getenv("VALIDATOR_MAX_CONCURRENT_BUILDS", "1"))
    )
    owner_api_url: str = field(
        default_factory=lambda: os.getenv("OWNER_API_URL", "http://127.0.0.1:8000")
    )
    sandbox_service_url: str = field(
        default_factory=lambda: os.getenv(
            "SANDBOX_SERVICE_URL", "http://127.0.0.1:8090"
        )
    )
    sandbox_service_auth_token: str = field(
        default_factory=lambda: os.getenv("SANDBOX_SERVICE_AUTH_TOKEN", "")
    )
    sandbox_poll_interval_seconds: float = field(
        default_factory=lambda: _float_env("SANDBOX_POLL_INTERVAL_SECONDS", 5.0)
    )
    sandbox_job_timeout_seconds: float = field(
        default_factory=lambda: _float_env("SANDBOX_JOB_TIMEOUT_SECONDS", 900.0)
    )
    sandbox_cancellation_timeout_seconds: float = field(
        default_factory=lambda: _float_env(
            "SANDBOX_CANCELLATION_TIMEOUT_SECONDS", 30.0
        )
    )
    provider_proxy_url: str = field(
        default_factory=lambda: os.getenv("EIREL_PROVIDER_PROXY_URL", "")
    )
    provider_proxy_token: str = field(
        default_factory=lambda: os.getenv("EIREL_PROVIDER_PROXY_TOKEN", "")
    )
    provider_max_requests: int = field(
        default_factory=lambda: int(os.getenv("EIREL_PROVIDER_MAX_REQUESTS", "24"))
    )
    provider_max_total_tokens: int = field(
        default_factory=lambda: int(
            os.getenv("EIREL_PROVIDER_MAX_TOTAL_TOKENS", "60000")
        )
    )
    provider_max_wall_clock_seconds: int = field(
        default_factory=lambda: int(
            os.getenv("EIREL_PROVIDER_MAX_WALL_CLOCK_SECONDS", "300")
        )
    )
    provider_request_timeout_seconds: int = field(
        default_factory=lambda: int(
            os.getenv("EIREL_PROVIDER_REQUEST_TIMEOUT_SECONDS", "60")
        )
    )
    # -- general_chat tool services -------------------------------------
    # Six specialized tool services routed through owner-api. Each service
    # has its own URL and HMAC auth token. Agent miners never call these
    # directly — all traffic goes through owner-api trace capture middleware.
    web_search_tool_service_url: str = field(
        default_factory=lambda: os.getenv(
            "EIREL_WEB_SEARCH_TOOL_URL", "http://web-search-tool-service:8085"
        )
    )
    web_search_tool_service_token: str = field(
        default_factory=lambda: os.getenv("EIREL_WEB_SEARCH_TOOL_TOKEN", "")
    )
    web_search_tool_backends: str = field(
        default_factory=lambda: os.getenv("EIREL_WEB_SEARCH_TOOL_BACKENDS", "")
    )
    serper_api_key: str = field(
        default_factory=lambda: os.getenv("EIREL_SERPER_API_KEY", "")
    )
    tavily_api_key: str = field(
        default_factory=lambda: os.getenv("EIREL_TAVILY_API_KEY", "")
    )
    web_search_per_backend_timeout_seconds: float = field(
        default_factory=lambda: _float_env(
            "EIREL_WEB_SEARCH_PER_BACKEND_TIMEOUT_SECONDS", 10.0
        )
    )
    x_tool_service_url: str = field(
        default_factory=lambda: os.getenv(
            "EIREL_X_TOOL_URL", "http://x-tool-service:8086"
        )
    )
    x_tool_service_token: str = field(
        default_factory=lambda: os.getenv("EIREL_X_TOOL_TOKEN", "")
    )
    semantic_scholar_tool_service_url: str = field(
        default_factory=lambda: os.getenv(
            "EIREL_SEMANTIC_SCHOLAR_TOOL_URL",
            "http://semantic-scholar-tool-service:8087",
        )
    )
    semantic_scholar_tool_service_token: str = field(
        default_factory=lambda: os.getenv("EIREL_SEMANTIC_SCHOLAR_TOOL_TOKEN", "")
    )
    semantic_scholar_api_key: str = field(
        default_factory=lambda: os.getenv("EIREL_SEMANTIC_SCHOLAR_API_KEY", "")
    )
    sandbox_tool_service_url: str = field(
        default_factory=lambda: os.getenv(
            "EIREL_SANDBOX_TOOL_URL", "http://sandbox-tool-service:8091"
        )
    )
    sandbox_tool_service_token: str = field(
        default_factory=lambda: os.getenv("EIREL_SANDBOX_TOOL_TOKEN", "")
    )

    # -- general_chat mode budgets --------------------------------------
    # Four modes: instant / thinking × web_search off / on. Budgets are
    # per-task (one conversation = one task). Exceeding a budget triggers
    # a hard-zero for the conversation at scoring time.
    general_chat_instant_latency_seconds: float = field(
        default_factory=lambda: _float_env("EIREL_GC_INSTANT_LATENCY_SECONDS", 15.0)
    )
    general_chat_instant_output_tokens: int = field(
        default_factory=lambda: int(os.getenv("EIREL_GC_INSTANT_OUTPUT_TOKENS", "1024"))
    )
    general_chat_instant_tool_calls: int = field(
        default_factory=lambda: int(os.getenv("EIREL_GC_INSTANT_TOOL_CALLS", "3"))
    )
    general_chat_instant_reasoning_tokens: int = field(
        default_factory=lambda: int(os.getenv("EIREL_GC_INSTANT_REASONING_TOKENS", "0"))
    )
    general_chat_instant_web_search_latency_seconds: float = field(
        default_factory=lambda: _float_env("EIREL_GC_INSTANT_WEB_LATENCY_SECONDS", 20.0)
    )
    general_chat_instant_web_search_extra_calls: int = field(
        default_factory=lambda: int(os.getenv("EIREL_GC_INSTANT_WEB_EXTRA_CALLS", "3"))
    )
    general_chat_thinking_latency_seconds: float = field(
        default_factory=lambda: _float_env("EIREL_GC_THINKING_LATENCY_SECONDS", 60.0)
    )
    general_chat_thinking_output_tokens: int = field(
        default_factory=lambda: int(os.getenv("EIREL_GC_THINKING_OUTPUT_TOKENS", "4096"))
    )
    general_chat_thinking_tool_calls: int = field(
        default_factory=lambda: int(os.getenv("EIREL_GC_THINKING_TOOL_CALLS", "8"))
    )
    general_chat_thinking_reasoning_tokens: int = field(
        default_factory=lambda: int(os.getenv("EIREL_GC_THINKING_REASONING_TOKENS", "16384"))
    )
    general_chat_thinking_web_search_latency_seconds: float = field(
        default_factory=lambda: _float_env("EIREL_GC_THINKING_WEB_LATENCY_SECONDS", 75.0)
    )
    general_chat_thinking_web_search_extra_calls: int = field(
        default_factory=lambda: int(os.getenv("EIREL_GC_THINKING_WEB_EXTRA_CALLS", "5"))
    )

    # Per-API hard caps (enforced at owner-api before dispatching to tool
    # services). Exceeding these returns 429 to the miner.
    general_chat_x_api_calls_per_task: int = field(
        default_factory=lambda: int(os.getenv("EIREL_GC_X_API_CALLS_PER_TASK", "1"))
    )

    # Per-conversation cost budget (used as denominator for the cost
    # dimension in 4D scoring).
    general_chat_instant_cost_budget_usd: float = field(
        default_factory=lambda: _float_env("EIREL_GC_INSTANT_COST_BUDGET_USD", 0.10)
    )
    general_chat_thinking_cost_budget_usd: float = field(
        default_factory=lambda: _float_env("EIREL_GC_THINKING_COST_BUDGET_USD", 0.50)
    )
    owner_dataset_root_path: str = field(
        default_factory=lambda: _resolve_fixture_path(
            env_name="EIREL_OWNER_DATASET_ROOT_PATH",
            container_default="/app/eirel-ai/data/owner_datasets/families",
            workspace_default="/eirel-ai/data/owner_datasets/families",
            repo_relative_default="eirel-ai/data/owner_datasets/families",
            expected_kind="directory",
        )
    )
    judge_base_url: str = field(
        default_factory=lambda: os.getenv("EIREL_JUDGE_BASE_URL", "")
    )
    judge_api_key: str = field(
        default_factory=lambda: os.getenv("EIREL_JUDGE_API_KEY", "")
    )
    judge_timeout_seconds: float = field(
        default_factory=lambda: _float_env("EIREL_JUDGE_TIMEOUT_SECONDS", 30.0)
    )
    ensemble_judge_base_url: str = field(
        default_factory=lambda: os.getenv("EIREL_ENSEMBLE_JUDGE_BASE_URL", "")
    )
    ensemble_judge_api_key: str = field(
        default_factory=lambda: os.getenv("EIREL_ENSEMBLE_JUDGE_API_KEY", "")
    )
    ensemble_judge_timeout_seconds: float = field(
        default_factory=lambda: _float_env("EIREL_ENSEMBLE_JUDGE_TIMEOUT_SECONDS", 30.0)
    )
    ensemble_judge_disagreement_threshold: float = field(
        default_factory=lambda: _float_env("EIREL_ENSEMBLE_JUDGE_DISAGREEMENT_THRESHOLD", 0.20)
    )
    bittensor_network: str = field(
        default_factory=lambda: os.getenv("BITTENSOR_NETWORK", "finney")
    )
    bittensor_netuid: int = field(
        default_factory=lambda: int(os.getenv("BITTENSOR_NETUID", "0"))
    )
    metagraph_snapshot_path: str = field(
        default_factory=lambda: os.getenv("METAGRAPH_SNAPSHOT_PATH", "")
    )
    soft_termination_grace_seconds: int = field(
        default_factory=lambda: int(
            os.getenv("EIREL_SOFT_TERMINATION_GRACE_SECONDS", "600")
        )
    )
    validator_submission_sync_interval_seconds: float = field(
        default_factory=lambda: _float_env("VALIDATOR_SUBMISSION_SYNC_INTERVAL_SECONDS", 15.0)
    )
    validator_env_loop_interval_seconds: float = field(
        default_factory=lambda: _float_env("VALIDATOR_ENV_LOOP_INTERVAL_SECONDS", 2.0)
    )
    owner_docker_binary_path: str = field(
        default_factory=lambda: os.getenv("OWNER_DOCKER_BINARY_PATH", "/usr/bin/docker")
    )
    owner_miner_runtime_image: str = field(
        default_factory=lambda: os.getenv(
            "OWNER_MINER_RUNTIME_IMAGE", "eirel-managed-miner-runtime:local"
        )
    )
    # Path to a local checkout of the eirel SDK repo — the owner-api reads
    # miner manifests from this directory. Set OWNER_SDK_REPO_ROOT in the
    # environment to override; the default assumes a sibling checkout
    # next to the eirel-ai repo.
    owner_sdk_repo_root: str = field(
        default_factory=lambda: os.getenv("OWNER_SDK_REPO_ROOT", "../eirel")
    )
    owner_runtime_work_root: str = field(
        default_factory=lambda: os.getenv(
            "OWNER_RUNTIME_WORK_ROOT", tempfile.gettempdir() + "/eirel-owner-runtimes"
        )
    )
    owner_runtime_endpoint_host: str = field(
        default_factory=lambda: os.getenv("OWNER_RUNTIME_ENDPOINT_HOST", "127.0.0.1")
    )
    owner_runtime_bind_host: str = field(
        default_factory=lambda: os.getenv("OWNER_RUNTIME_BIND_HOST", "127.0.0.1")
    )
    owner_runtime_docker_network: str = field(
        default_factory=lambda: os.getenv("OWNER_RUNTIME_DOCKER_NETWORK", "")
    )
    owner_runtime_health_timeout_seconds: float = field(
        default_factory=lambda: _float_env("OWNER_RUNTIME_HEALTH_TIMEOUT_SECONDS", 60.0)
    )
    owner_runtime_restart_budget: int = field(
        default_factory=lambda: int(os.getenv("OWNER_RUNTIME_RESTART_BUDGET", "2"))
    )
    # Production default is "kubernetes" (miner pods run in k3s alongside
    # owner-api). "docker" and "baremetal" remain supported for local dev
    # and non-k8s deployments but require opt-in.
    owner_runtime_backend: str = field(
        default_factory=lambda: os.getenv("OWNER_RUNTIME_BACKEND", "kubernetes")
    )
    owner_kubectl_binary_path: str = field(
        default_factory=lambda: os.getenv("OWNER_KUBECTL_BINARY_PATH", "/usr/bin/kubectl")
    )
    # owner_runtime_namespace is defined below with EIREL_OWNER_RUNTIME_NAMESPACE;
    # the earlier duplicate declaration that read OWNER_RUNTIME_NAMESPACE was
    # shadowed and therefore dead — removed.
    owner_runtime_service_account: str = field(
        default_factory=lambda: os.getenv("OWNER_RUNTIME_SERVICE_ACCOUNT", "default")
    )
    owner_runtime_init_image: str = field(
        default_factory=lambda: os.getenv("OWNER_RUNTIME_INIT_IMAGE", "curlimages/curl:8.7.1")
    )
    owner_runtime_pool_label_key: str = field(
        default_factory=lambda: os.getenv("OWNER_RUNTIME_POOL_LABEL_KEY", "eirel.dev/runtime-pool")
    )
    owner_runtime_pool_label_value: str = field(
        default_factory=lambda: os.getenv("OWNER_RUNTIME_POOL_LABEL_VALUE", "true")
    )
    owner_runtime_class_label_key: str = field(
        default_factory=lambda: os.getenv("OWNER_RUNTIME_CLASS_LABEL_KEY", "eirel.dev/runtime-class")
    )
    owner_runtime_class_label_value: str = field(
        default_factory=lambda: os.getenv("OWNER_RUNTIME_CLASS_LABEL_VALUE", "miner")
    )
    owner_runtime_capacity_refresh_interval_seconds: float = field(
        default_factory=lambda: _float_env("OWNER_RUNTIME_CAPACITY_REFRESH_INTERVAL_SECONDS", 30.0)
    )
    owner_runtime_reaper_interval_seconds: float = field(
        default_factory=lambda: _float_env("EIREL_OWNER_RUNTIME_REAPER_INTERVAL_SECONDS", 30.0)
    )
    owner_pending_capacity_retry_interval_seconds: float = field(
        default_factory=lambda: _float_env(
            "EIREL_OWNER_PENDING_CAPACITY_RETRY_INTERVAL_SECONDS", 30.0,
        )
    )
    owner_kubeconfig_path: str = field(
        default_factory=lambda: os.getenv("EIREL_OWNER_KUBECONFIG_PATH", "")
    )
    owner_runtime_namespace: str = field(
        default_factory=lambda: os.getenv("EIREL_OWNER_RUNTIME_NAMESPACE", "eirel-miners")
    )
    owner_runtime_system_namespace: str = field(
        default_factory=lambda: os.getenv("EIREL_OWNER_RUNTIME_SYSTEM_NAMESPACE", "eirel-system")
    )
    owner_runtime_control_plane_namespace: str = field(
        default_factory=lambda: os.getenv(
            "EIREL_OWNER_RUNTIME_CONTROL_PLANE_NAMESPACE", "eirel-control-plane")
    )
    owner_runtime_image: str = field(
        default_factory=lambda: os.getenv(
            "EIREL_OWNER_RUNTIME_IMAGE", "registry.eirel.internal/miner-runtime:v1")
    )
    owner_runtime_shared_secret_name: str = field(
        default_factory=lambda: os.getenv(
            "EIREL_OWNER_RUNTIME_SHARED_SECRET_NAME", "eirel-runtime-shared")
    )
    owner_runtime_service_domain: str = field(
        default_factory=lambda: os.getenv(
            "EIREL_OWNER_RUNTIME_SERVICE_DOMAIN", "svc.cluster.local")
    )
    owner_runtime_probe_period_seconds: int = field(
        default_factory=lambda: int(os.getenv(
            "EIREL_OWNER_RUNTIME_PROBE_PERIOD_SECONDS", "5"))
    )
    owner_runtime_submission_cpu_millis: int = field(
        default_factory=lambda: int(os.getenv("EIREL_OWNER_RUNTIME_SUBMISSION_CPU_MILLIS", "2000"))
    )
    owner_runtime_submission_memory_mb: int = field(
        default_factory=lambda: int(os.getenv("EIREL_OWNER_RUNTIME_SUBMISSION_MEMORY_MB", "2048"))
    )
    owner_runtime_capacity_cpu_headroom_millis: int = field(
        default_factory=lambda: int(os.getenv("OWNER_RUNTIME_CAPACITY_CPU_HEADROOM_MILLIS", "500"))
    )
    owner_runtime_capacity_memory_headroom_mb: int = field(
        default_factory=lambda: int(os.getenv("OWNER_RUNTIME_CAPACITY_MEMORY_HEADROOM_MB", "512"))
    )
    owner_runtime_capacity_pod_headroom: int = field(
        default_factory=lambda: int(os.getenv("OWNER_RUNTIME_CAPACITY_POD_HEADROOM", "1"))
    )
    owner_baremetal_inventory_path: str = field(
        default_factory=lambda: _resolve_baremetal_inventory_path()
    )
    owner_baremetal_ssh_user: str = field(
        default_factory=lambda: os.getenv("OWNER_BAREMETAL_SSH_USER", "eirel")
    )
    owner_baremetal_ssh_key_path: str = field(
        default_factory=lambda: os.getenv("OWNER_BAREMETAL_SSH_KEY_PATH", "~/.ssh/eirel_deploy")
    )
    owner_baremetal_ssh_port: int = field(
        default_factory=lambda: int(os.getenv("OWNER_BAREMETAL_SSH_PORT", "22"))
    )
    owner_baremetal_storage_root: str = field(
        default_factory=lambda: os.getenv("OWNER_BAREMETAL_STORAGE_ROOT", "/var/lib/eirel")
    )
    owner_baremetal_provider_proxy_url: str = field(
        default_factory=lambda: os.getenv("OWNER_BAREMETAL_PROVIDER_PROXY_URL", "")
    )
    owner_baremetal_web_search_tool_url: str = field(
        default_factory=lambda: os.getenv("OWNER_BAREMETAL_WEB_SEARCH_TOOL_URL", "")
    )
    orchestrator_url: str = field(
        default_factory=lambda: os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8050")
    )
    orchestrator_timeout_seconds: float = field(
        default_factory=lambda: _float_env("ORCHESTRATOR_TIMEOUT_SECONDS", 120.0)
    )
    platform_tools_enabled: bool = field(
        default_factory=lambda: _bool_env("PLATFORM_TOOLS_ENABLED", True)
    )
    control_plane_internal_url: str = field(
        default_factory=lambda: os.getenv("CONTROL_PLANE_INTERNAL_URL", "http://orchestrator:8050")
    )
    owner_api_internal_url: str = field(
        default_factory=lambda: os.getenv("OWNER_API_INTERNAL_URL", "http://owner-api:8000")
    )
    execution_worker_internal_url: str = field(
        default_factory=lambda: os.getenv(
            "EXECUTION_WORKER_INTERNAL_URL", "http://execution-worker:8006"
        )
    )
    weight_setter_internal_url: str = field(
        default_factory=lambda: os.getenv(
            "WEIGHT_SETTER_INTERNAL_URL", "http://weight-setter:8012"
        )
    )
    workflow_runtime_auto_remediation_enabled: bool = field(
        default_factory=lambda: _bool_env("WORKFLOW_RUNTIME_AUTO_REMEDIATION_ENABLED", False)
    )
    workflow_runtime_auto_remediation_interval_seconds: float = field(
        default_factory=lambda: _float_env("WORKFLOW_RUNTIME_AUTO_REMEDIATION_INTERVAL_SECONDS", 30.0)
    )
    workflow_runtime_auto_remediation_cooldown_seconds: int = field(
        default_factory=lambda: int(os.getenv("WORKFLOW_RUNTIME_AUTO_REMEDIATION_COOLDOWN_SECONDS", "120"))
    )
    workflow_runtime_auto_remediation_max_actions: int = field(
        default_factory=lambda: int(os.getenv("WORKFLOW_RUNTIME_AUTO_REMEDIATION_MAX_ACTIONS", "10"))
    )
    workflow_runtime_auto_remediation_requeue_limit: int = field(
        default_factory=lambda: int(os.getenv("WORKFLOW_RUNTIME_AUTO_REMEDIATION_REQUEUE_LIMIT", "3"))
    )
    workflow_runtime_auto_remediation_escalation_window_seconds: int = field(
        default_factory=lambda: int(
            os.getenv("WORKFLOW_RUNTIME_AUTO_REMEDIATION_ESCALATION_WINDOW_SECONDS", "900")
        )
    )
    workflow_runtime_auto_remediation_worker_failure_backoff_seconds: int = field(
        default_factory=lambda: int(
            os.getenv("WORKFLOW_RUNTIME_AUTO_REMEDIATION_WORKER_FAILURE_BACKOFF_SECONDS", "120")
        )
    )
    chain_publish_readiness_enforced: bool = field(
        default_factory=lambda: _bool_env("CHAIN_PUBLISH_READINESS_ENFORCED", True)
    )
    internal_service_token: str = field(
        default_factory=lambda: os.getenv("EIREL_INTERNAL_SERVICE_TOKEN", "")
    )
    object_storage_backend: str = field(
        default_factory=lambda: os.getenv("OBJECT_STORAGE_BACKEND", "filesystem")
    )
    object_storage_endpoint_url: str = field(
        default_factory=lambda: os.getenv("OBJECT_STORAGE_ENDPOINT_URL", "")
    )
    object_storage_region: str = field(
        default_factory=lambda: os.getenv("OBJECT_STORAGE_REGION", "")
    )
    object_storage_access_key_id: str = field(
        default_factory=lambda: os.getenv("OBJECT_STORAGE_ACCESS_KEY_ID", "")
    )
    object_storage_secret_access_key: str = field(
        default_factory=lambda: os.getenv("OBJECT_STORAGE_SECRET_ACCESS_KEY", "")
    )
    object_storage_prefix: str = field(
        default_factory=lambda: os.getenv("OBJECT_STORAGE_PREFIX", "")
    )
    object_storage_use_ssl: bool = field(
        default_factory=lambda: _bool_env("OBJECT_STORAGE_USE_SSL", True)
    )
    object_storage_addressing_style: str = field(
        default_factory=lambda: os.getenv("OBJECT_STORAGE_ADDRESSING_STYLE", "auto")
    )
    artifact_storage_root: str = field(
        default_factory=lambda: os.getenv(
            "ARTIFACT_STORAGE_ROOT", tempfile.gettempdir() + "/eirel-managed-artifacts"
        )
    )
    artifact_storage_bucket: str = field(
        default_factory=lambda: os.getenv("ARTIFACT_STORAGE_BUCKET", "eirel-managed")
    )
    validator_epoch_quorum: int = field(
        default_factory=lambda: int(os.getenv("VALIDATOR_EPOCH_QUORUM", "1"))
    )
    run_duration_days: int = field(
        default_factory=lambda: int(os.getenv("EIREL_RUN_DURATION_DAYS", "3"))
    )
    run_top_carryover_per_family: int = field(
        default_factory=lambda: int(os.getenv("EIREL_RUN_TOP_CARRYOVER_PER_FAMILY", "3"))
    )
    run_budget_usd: float = field(
        default_factory=lambda: float(os.getenv("EIREL_RUN_BUDGET_USD", "30.0"))
    )
    trace_gate_penalty_usd: float = field(
        default_factory=lambda: float(os.getenv("EIREL_TRACE_GATE_PENALTY_USD", "0.50"))
    )
    honeytoken_count_per_run: int = field(
        default_factory=lambda: int(os.getenv("EIREL_HONEYTOKEN_COUNT_PER_RUN", "8"))
    )
    honeytoken_injection_rate: float = field(
        default_factory=lambda: float(os.getenv("EIREL_HONEYTOKEN_INJECTION_RATE", "0.02"))
    )
    trace_store_backend: str = field(
        default_factory=lambda: os.getenv("EIREL_TRACE_STORE_BACKEND", "memory")
    )
    trace_store_redis_url: str | None = field(
        default_factory=lambda: os.getenv("EIREL_TRACE_STORE_REDIS_URL") or None
    )
    run_min_scores_json: str = field(
        default_factory=lambda: os.getenv("EIREL_RUN_MIN_SCORES_JSON", "")
    )
    submission_fee_tao: float = field(
        default_factory=lambda: float(os.getenv("EIREL_SUBMISSION_FEE_TAO", "0.1"))
    )
    submission_treasury_address: str = field(
        default_factory=lambda: os.getenv("EIREL_SUBMISSION_TREASURY_ADDRESS", "")
    )
    submission_rate_limit_requests: int = field(
        default_factory=lambda: int(os.getenv("EIREL_SUBMISSION_RATE_LIMIT_REQUESTS", "5"))
    )
    submission_rate_limit_window_seconds: int = field(
        default_factory=lambda: int(os.getenv("EIREL_SUBMISSION_RATE_LIMIT_WINDOW_SECONDS", "3600"))
    )
    snapshot_readiness_timeout_minutes: int = field(
        default_factory=lambda: int(os.getenv(
            'EIREL_SNAPSHOT_READINESS_TIMEOUT_MINUTES', '15'))
    )
    # Distributed evaluation settings
    task_claim_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("EIREL_TASK_CLAIM_TIMEOUT_SECONDS", "600"))
    )
    first_run_start_time: str = field(
        default_factory=lambda: os.getenv("EIREL_FIRST_RUN_START_TIME", "")
    )
    spot_check_duplicate_rate: float = field(
        default_factory=lambda: _float_env("EIREL_SPOT_CHECK_DUPLICATE_RATE", 0.05)
    )
    active_families: str = field(
        default_factory=lambda: os.getenv("EIREL_ACTIVE_FAMILIES", "general_chat")
    )
    family_weights: str = field(
        default_factory=lambda: os.getenv("EIREL_FAMILY_WEIGHTS", "general_chat:1.0")
    )
    serving_release_interval_days: int = field(
        default_factory=lambda: int(os.getenv("SERVING_RELEASE_INTERVAL_DAYS", "7"))
    )
    execution_stream_name: str = field(
        default_factory=lambda: os.getenv("EXECUTION_STREAM_NAME", "eirel:tasks")
    )
    execution_stream_group: str = field(
        default_factory=lambda: os.getenv("EXECUTION_STREAM_GROUP", "execution-workers")
    )
    execution_worker_consumer_name: str = field(
        default_factory=lambda: os.getenv("EXECUTION_WORKER_CONSUMER_NAME", "")
    )
    execution_worker_lease_seconds: int = field(
        default_factory=lambda: int(os.getenv("EXECUTION_WORKER_LEASE_SECONDS", "60"))
    )
    execution_worker_idle_reclaim_ms: int = field(
        default_factory=lambda: int(os.getenv("EXECUTION_WORKER_IDLE_RECLAIM_MS", "30000"))
    )
    execution_worker_block_ms: int = field(
        default_factory=lambda: int(os.getenv("EXECUTION_WORKER_BLOCK_MS", "1000"))
    )
    execution_worker_max_retries: int = field(
        default_factory=lambda: int(os.getenv("EXECUTION_WORKER_MAX_RETRIES", "3"))
    )
    execution_worker_poll_interval_seconds: float = field(
        default_factory=lambda: _float_env("EXECUTION_WORKER_POLL_INTERVAL_SECONDS", 0.5)
    )
    consumer_api_keys: str = field(
        default_factory=lambda: os.getenv("CONSUMER_API_KEYS", "")
    )
    consumer_rate_limit_requests: int = field(
        default_factory=lambda: int(os.getenv("CONSUMER_RATE_LIMIT_REQUESTS", "30"))
    )
    consumer_rate_limit_window_seconds: int = field(
        default_factory=lambda: int(os.getenv("CONSUMER_RATE_LIMIT_WINDOW_SECONDS", "60"))
    )
    object_storage_base_url: str = field(
        default_factory=lambda: os.getenv("OBJECT_STORAGE_BASE_URL", "")
    )

    # -- Dataset source (filesystem vs S3) ------------------------------
    owner_dataset_source_type: str = field(
        default_factory=lambda: os.getenv("EIREL_OWNER_DATASET_SOURCE_TYPE", "filesystem")
    )
    owner_dataset_s3_bucket: str = field(
        default_factory=lambda: os.getenv("EIREL_OWNER_DATASET_S3_BUCKET", "")
    )
    owner_dataset_s3_prefix: str = field(
        default_factory=lambda: os.getenv("EIREL_OWNER_DATASET_S3_PREFIX", "datasets/")
    )
    owner_dataset_cache_dir: str = field(
        default_factory=lambda: os.getenv(
            "EIREL_OWNER_DATASET_CACHE_DIR",
            str(Path(tempfile.gettempdir()) / "eirel-dataset-cache"),
        )
    )

    # -- Dataset forge (generator LLM config) ---------------------------
    dataset_forge_generator_provider: str = field(
        default_factory=lambda: os.getenv("EIREL_DATASET_FORGE_GENERATOR_PROVIDER", "")
    )
    dataset_forge_generator_model: str = field(
        default_factory=lambda: os.getenv("EIREL_DATASET_FORGE_GENERATOR_MODEL", "")
    )
    dataset_forge_generator_base_url: str = field(
        default_factory=lambda: os.getenv("EIREL_DATASET_FORGE_GENERATOR_BASE_URL", "")
    )
    dataset_forge_generator_api_key: str = field(
        default_factory=lambda: os.getenv("EIREL_DATASET_FORGE_GENERATOR_API_KEY", "")
    )
    dataset_forge_generator_timeout_seconds: float = field(
        default_factory=lambda: _float_env("EIREL_DATASET_FORGE_GENERATOR_TIMEOUT_SECONDS", 60.0)
    )
    dataset_forge_judge_provider_must_differ: bool = field(
        default_factory=lambda: _bool_env("EIREL_DATASET_FORGE_JUDGE_PROVIDER_MUST_DIFFER", True)
    )
    dataset_forge_owner_secret: str = field(
        default_factory=lambda: os.getenv("EIREL_DATASET_FORGE_OWNER_SECRET", "")
    )
    auto_trigger_dataset_forge: bool = field(
        default_factory=lambda: _bool_env("EIREL_AUTO_TRIGGER_DATASET_FORGE", False)
    )

    # -- Owner identity / signing ---------------------------------------
    # The owner SS58 is derived from the wallet + hotkey names at startup.
    # `owner_hotkey_ss58` is runtime-only (resolved from the wallet files).
    owner_wallet_name: str = field(
        default_factory=lambda: os.getenv("EIREL_OWNER_WALLET_NAME", "")
    )
    owner_hotkey_name: str = field(
        default_factory=lambda: os.getenv("EIREL_OWNER_HOTKEY_NAME", "")
    )
    owner_hotkey_ss58: str = field(default="")

    def __post_init__(self) -> None:
        self.owner_dataset_root_path = _validate_owner_dataset_root(self.owner_dataset_root_path)
        _validate_dataset_source_type(self.owner_dataset_source_type)
        _validate_forge_cross_vendor(
            generator_provider=self.dataset_forge_generator_provider,
            generator_base_url=self.dataset_forge_generator_base_url,
            judge_base_url=self.judge_base_url,
            must_differ=self.dataset_forge_judge_provider_must_differ,
        )
        # H1: in S3 mode the forge must use a non-empty owner secret so the
        # hidden allocation RNG and the task-ID hash (C1) depend on something
        # only the owner knows. In filesystem/dev mode an empty secret is
        # allowed for convenience but logged as a warning by the forge CLI.
        if (
            self.owner_dataset_source_type == "s3"
            and not self.dataset_forge_owner_secret
        ):
            raise ValueError(
                "EIREL_DATASET_FORGE_OWNER_SECRET must be set when "
                "EIREL_OWNER_DATASET_SOURCE_TYPE=s3 — without it the hidden "
                "allocation RNG and task-ID hash are predictable from run_id alone"
            )

        # Interval and timeout validation
        for _field_name in (
            'owner_runtime_capacity_refresh_interval_seconds',
            'owner_runtime_health_timeout_seconds',
            'owner_runtime_reaper_interval_seconds',
            'owner_pending_capacity_retry_interval_seconds',
            'task_claim_timeout_seconds',
        ):
            _val = getattr(self, _field_name, None)
            if _val is not None and float(_val) <= 0:
                raise ValueError(f'{_field_name} must be positive, got {_val}')

        # Resource validation
        if self.owner_runtime_submission_cpu_millis <= 0:
            raise ValueError('owner_runtime_submission_cpu_millis must be positive')
        if self.owner_runtime_submission_memory_mb <= 0:
            raise ValueError('owner_runtime_submission_memory_mb must be positive')


_settings_instance: Settings | None = None
_settings_lock = threading.Lock()


def get_settings() -> Settings:
    global _settings_instance
    if _settings_instance is not None:
        return _settings_instance
    with _settings_lock:
        if _settings_instance is None:
            _settings_instance = Settings()
        return _settings_instance


def reset_settings() -> None:
    """Clear cached settings (useful for tests)."""
    global _settings_instance
    _settings_instance = None
