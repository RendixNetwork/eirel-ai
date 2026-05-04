from __future__ import annotations

import logging

from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy import inspect

_logger = logging.getLogger(__name__)

_MIGRATION_ADVISORY_LOCK_ID = 42


@dataclass(frozen=True, slots=True)
class Migration:
    version: str
    description: str
    apply: Callable[[Engine], None]


def run_migrations(engine: Engine) -> list[str]:
    dialect = engine.dialect.name
    if dialect == "postgresql":
        with engine.begin() as conn:
            conn.execute(text("SELECT pg_advisory_lock(:lock_id)"),
                         {"lock_id": _MIGRATION_ADVISORY_LOCK_ID})
        _logger.info("acquired migration advisory lock")
    try:
        return _run_migrations_unlocked(engine)
    finally:
        if dialect == "postgresql":
            with engine.begin() as conn:
                conn.execute(text("SELECT pg_advisory_unlock(:lock_id)"),
                             {"lock_id": _MIGRATION_ADVISORY_LOCK_ID})
            _logger.info("released migration advisory lock")


def _run_migrations_unlocked(engine: Engine) -> list[str]:
    _ensure_schema_migrations_table(engine)
    applied = _applied_versions(engine)
    executed: list[str] = []
    for migration in MIGRATIONS:
        if migration.version in applied:
            continue
        migration.apply(engine)
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO schema_migrations (version, description)
                    VALUES (:version, :description)
                    """
                ),
                {"version": migration.version, "description": migration.description},
            )
        executed.append(migration.version)
    return executed


def _ensure_schema_migrations_table(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version VARCHAR(64) NOT NULL PRIMARY KEY,
                    description VARCHAR(255) NOT NULL,
                    applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        )


def _applied_versions(engine: Engine) -> set[str]:
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT version FROM schema_migrations")).fetchall()
    return {str(row[0]) for row in rows}


def _migration_family_native_staging_bootstrap(engine: Engine) -> None:
    del engine


def _migration_remove_legacy_workflow_market_schema(engine: Engine) -> None:
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    with engine.begin() as conn:
        if "coalition_score_snapshots" in tables:
            conn.execute(text("DROP TABLE coalition_score_snapshots"))
        episode_columns = {
            column["name"]
            for column in inspector.get_columns("workflow_episode_records")
        } if "workflow_episode_records" in tables else set()
        if "coalition_json" in episode_columns:
            conn.execute(text("ALTER TABLE workflow_episode_records DROP COLUMN coalition_json"))


def _migration_add_distributed_evaluation_schema(engine: Engine) -> None:
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    with engine.begin() as conn:
        if "miner_evaluation_tasks" not in tables:
            conn.execute(text("""
                CREATE TABLE miner_evaluation_tasks (
                    id VARCHAR(36) PRIMARY KEY,
                    epoch_id VARCHAR(128) NOT NULL,
                    family_id VARCHAR(64) NOT NULL,
                    miner_hotkey VARCHAR(128) NOT NULL,
                    task_id VARCHAR(128) NOT NULL,
                    task_index INTEGER NOT NULL,
                    status VARCHAR(32) NOT NULL DEFAULT 'pending',
                    claimed_by_validator VARCHAR(128),
                    claimed_at TIMESTAMP,
                    claim_expires_at TIMESTAMP,
                    claim_attempt_count INTEGER NOT NULL DEFAULT 0,
                    miner_response_json JSON,
                    judge_output_json JSON,
                    task_score FLOAT,
                    task_status VARCHAR(32),
                    result_metadata_json JSON NOT NULL DEFAULT '{}',
                    evaluated_at TIMESTAMP,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(epoch_id, family_id, miner_hotkey, task_id)
                )
            """))
            conn.execute(text(
                "CREATE INDEX idx_met_claimable ON miner_evaluation_tasks (epoch_id, family_id, status, claim_expires_at)"
            ))
            conn.execute(text(
                "CREATE INDEX idx_met_miner ON miner_evaluation_tasks (epoch_id, family_id, miner_hotkey)"
            ))
            conn.execute(text(
                "CREATE INDEX idx_met_epoch_id ON miner_evaluation_tasks (epoch_id)"
            ))
            conn.execute(text(
                "CREATE INDEX idx_met_family_id ON miner_evaluation_tasks (family_id)"
            ))
            conn.execute(text(
                "CREATE INDEX idx_met_status ON miner_evaluation_tasks (status)"
            ))
            conn.execute(text(
                "CREATE INDEX idx_met_claimed_by ON miner_evaluation_tasks (claimed_by_validator)"
            ))
        if "miner_evaluation_summaries" not in tables:
            conn.execute(text("""
                CREATE TABLE miner_evaluation_summaries (
                    id VARCHAR(36) PRIMARY KEY,
                    epoch_id VARCHAR(128) NOT NULL,
                    family_id VARCHAR(64) NOT NULL,
                    miner_hotkey VARCHAR(128) NOT NULL,
                    total_tasks INTEGER NOT NULL,
                    completed_tasks INTEGER NOT NULL DEFAULT 0,
                    failed_tasks INTEGER NOT NULL DEFAULT 0,
                    family_capability_score FLOAT,
                    robustness_score FLOAT,
                    anti_gaming_score FLOAT,
                    official_family_score FLOAT,
                    protocol_gate_passed BOOLEAN,
                    status VARCHAR(32) NOT NULL DEFAULT 'pending',
                    rollout_metadata_json JSON NOT NULL DEFAULT '{}',
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(epoch_id, family_id, miner_hotkey)
                )
            """))
            conn.execute(text(
                "CREATE INDEX idx_mes_epoch_id ON miner_evaluation_summaries (epoch_id)"
            ))
            conn.execute(text(
                "CREATE INDEX idx_mes_family_id ON miner_evaluation_summaries (family_id)"
            ))
            conn.execute(text(
                "CREATE INDEX idx_mes_miner ON miner_evaluation_summaries (miner_hotkey)"
            ))
            conn.execute(text(
                "CREATE INDEX idx_mes_status ON miner_evaluation_summaries (status)"
            ))


def _migration_add_neuron_uid_table(engine: Engine) -> None:
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    if "neuron_uids" not in tables:
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE neuron_uids (
                    hotkey VARCHAR(128) PRIMARY KEY,
                    uid INTEGER NOT NULL,
                    stake BIGINT NOT NULL DEFAULT 0,
                    is_validator BOOLEAN NOT NULL DEFAULT FALSE,
                    is_active BOOLEAN NOT NULL DEFAULT TRUE,
                    last_synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.execute(text("CREATE INDEX idx_neuron_uids_uid ON neuron_uids (uid)"))


def _migration_add_owner_dataset_bindings(engine: Engine) -> None:
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    if "owner_dataset_bindings" in tables:
        return
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE owner_dataset_bindings (
                id VARCHAR(36) PRIMARY KEY,
                family_id VARCHAR(64) NOT NULL,
                run_id VARCHAR(128) NOT NULL,
                bundle_uri VARCHAR(1024) NOT NULL,
                bundle_sha256 VARCHAR(64) NOT NULL,
                generator_version VARCHAR(128) NOT NULL,
                generated_by VARCHAR(128) NOT NULL,
                signature_hex VARCHAR(256) NOT NULL,
                generator_provider VARCHAR(64) NOT NULL DEFAULT '',
                generator_model VARCHAR(128) NOT NULL DEFAULT '',
                status VARCHAR(32) NOT NULL DEFAULT 'pending',
                provenance_json JSON NOT NULL DEFAULT '{}',
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                activated_at TIMESTAMP NULL,
                CONSTRAINT uq_owner_dataset_bindings_family_run UNIQUE (family_id, run_id)
            )
        """))
        conn.execute(text(
            "CREATE INDEX idx_owner_dataset_bindings_family ON owner_dataset_bindings (family_id)"
        ))
        conn.execute(text(
            "CREATE INDEX idx_owner_dataset_bindings_run ON owner_dataset_bindings (run_id)"
        ))
        conn.execute(text(
            "CREATE INDEX idx_owner_dataset_bindings_family_status "
            "ON owner_dataset_bindings (family_id, status)"
        ))


def _migration_add_cost_accounting_columns(engine: Engine) -> None:
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    if "deployment_score_records" not in tables:
        return
    existing = {col["name"] for col in inspector.get_columns("deployment_score_records")}
    new_columns = {
        "run_budget_usd": "FLOAT NOT NULL DEFAULT 30.0",
        "run_cost_usd_used": "FLOAT NOT NULL DEFAULT 0.0",
        "llm_cost_usd": "FLOAT NOT NULL DEFAULT 0.0",
        "tool_cost_usd": "FLOAT NOT NULL DEFAULT 0.0",
        "cost_rejection_count": "INTEGER NOT NULL DEFAULT 0",
    }
    with engine.begin() as conn:
        for col_name, col_def in new_columns.items():
            if col_name not in existing:
                conn.execute(text(
                    f"ALTER TABLE deployment_score_records ADD COLUMN {col_name} {col_def}"
                ))


def _migration_add_pending_runtime_stop(engine: Engine) -> None:
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    if "managed_deployments" not in tables:
        return
    existing = {col["name"] for col in inspector.get_columns("managed_deployments")}
    if "pending_runtime_stop" in existing:
        return
    default_literal = "false" if engine.dialect.name == "postgresql" else "0"
    with engine.begin() as conn:
        conn.execute(text(
            f"ALTER TABLE managed_deployments ADD COLUMN pending_runtime_stop BOOLEAN NOT NULL DEFAULT {default_literal}"
        ))


def _migration_add_snapshot_unique_constraint(engine: Engine) -> None:
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    if "epoch_target_snapshots" not in tables:
        return
    indexes = inspector.get_indexes("epoch_target_snapshots")
    existing_names = {idx["name"] for idx in indexes}
    if "uq_snapshot_run_family" in existing_names:
        return
    if "uq_epoch_target_snapshots_epoch_family" in existing_names:
        return
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE UNIQUE INDEX uq_snapshot_run_family "
            "ON epoch_target_snapshots (epoch_id, family_id)"
        ))


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        version="family_native_staging_bootstrap",
        description="Initialize family-native staging schema baseline",
        apply=_migration_family_native_staging_bootstrap,
    ),
    Migration(
        version="remove_legacy_workflow_market_schema",
        description="Drop legacy workflow-market coalition storage",
        apply=_migration_remove_legacy_workflow_market_schema,
    ),
    Migration(
        version="add_distributed_evaluation_schema",
        description="Add miner_evaluation_tasks and miner_evaluation_summaries tables for distributed task evaluation",
        apply=_migration_add_distributed_evaluation_schema,
    ),
    Migration(
        version="add_neuron_uid_table",
        description="Add neuron_uids table for all hotkey-to-uid mappings from metagraph",
        apply=_migration_add_neuron_uid_table,
    ),
    Migration(
        version="add_owner_dataset_bindings",
        description="Add owner_dataset_bindings table for private dataset pipeline",
        apply=_migration_add_owner_dataset_bindings,
    ),
    Migration(
        version="add_cost_accounting_columns",
        description="Add per-run USD cost accounting columns to deployment_score_records",
        apply=_migration_add_cost_accounting_columns,
    ),
    Migration(
        version="add_pending_runtime_stop",
        description="Add pending_runtime_stop flag to managed_deployments for orphan cleanup",
        apply=_migration_add_pending_runtime_stop,
    ),
    Migration(
        version="add_snapshot_unique_constraint",
        description="Add unique constraint on (epoch_id, family_id) to epoch_target_snapshots",
        apply=_migration_add_snapshot_unique_constraint,
    ),
    Migration(
        version="drop_registered_neurons_is_active",
        description="Drop is_active column from registered_neurons — presence = registered",
        apply=lambda engine: _drop_registered_neurons_is_active(engine),
    ),
    Migration(
        version="refactor_to_task_level_evaluation",
        description=(
            "Replace miner_evaluation_tasks with task_evaluations + "
            "task_miner_results for task-level validator claims and pairwise "
            "judging vs OpenAI baseline"
        ),
        apply=lambda engine: _migration_refactor_to_task_level_evaluation(engine),
    ),
    Migration(
        version="add_miner_latency_to_task_miner_results",
        description=(
            "Add miner_latency_seconds column to task_miner_results so the "
            "leaderboard can show miner response latency separately from "
            "judge latency, and for the latency-violation scoring gate"
        ),
        apply=lambda engine: _migration_add_miner_latency_to_task_miner_results(engine),
    ),
    Migration(
        version="add_miner_first_token_to_task_miner_results",
        description=(
            "Add miner_first_token_seconds column to task_miner_results to "
            "store time-to-first-token from the streaming invocation path; "
            "feeds the mode-agnostic 10s TTFB SLA gate"
        ),
        apply=lambda engine: _migration_add_miner_first_token_to_task_miner_results(engine),
    ),
    Migration(
        version="drop_miner_first_token_seconds",
        description=(
            "Drop miner_first_token_seconds — TTFB metric removed; only "
            "completion-time latency is recorded and gated. Most LLM "
            "providers stream first token <20s anyway, so the gate added "
            "noise without changing miner ranking."
        ),
        apply=lambda engine: _migration_drop_miner_first_token_seconds(engine),
    ),
    Migration(
        version="add_session_mode_web_search",
        description=(
            "Persist per-session user toggles (mode, web_search) on "
            "consumer_sessions so the orchestrator can apply them on every "
            "turn without the client re-asserting them. Owned by the "
            "orchestrator now that consumer-chat-api is a thin facade."
        ),
        apply=lambda engine: _migration_add_session_mode_web_search(engine),
    ),
    Migration(
        version="add_proxy_and_judge_cost_to_task_miner_results",
        description=(
            "Add proxy_cost_usd + judge_cost_usd to task_miner_results"
        ),
        apply=lambda engine: _migration_add_proxy_and_judge_cost(engine),
    ),
    Migration(
        version="drop_anti_gaming_artifacts",
        description=(
            "Strip honeytokens / trace_gate artifacts from JSON metadata "
            "after the anti-gaming subsystem was removed"
        ),
        apply=lambda engine: _migration_drop_anti_gaming_artifacts(engine),
    ),
)


def _migration_add_miner_latency_to_task_miner_results(engine: Engine) -> None:
    inspector = inspect(engine)
    if "task_miner_results" not in set(inspector.get_table_names()):
        return
    cols = {c["name"] for c in inspector.get_columns("task_miner_results")}
    if "miner_latency_seconds" in cols:
        return
    with engine.begin() as conn:
        conn.execute(text(
            "ALTER TABLE task_miner_results "
            "ADD COLUMN miner_latency_seconds FLOAT NOT NULL DEFAULT 0.0"
        ))


def _migration_add_miner_first_token_to_task_miner_results(engine: Engine) -> None:
    inspector = inspect(engine)
    if "task_miner_results" not in set(inspector.get_table_names()):
        return
    cols = {c["name"] for c in inspector.get_columns("task_miner_results")}
    if "miner_first_token_seconds" in cols:
        return
    with engine.begin() as conn:
        conn.execute(text(
            "ALTER TABLE task_miner_results "
            "ADD COLUMN miner_first_token_seconds FLOAT NOT NULL DEFAULT 0.0"
        ))


def _migration_drop_miner_first_token_seconds(engine: Engine) -> None:
    inspector = inspect(engine)
    if "task_miner_results" not in set(inspector.get_table_names()):
        return
    cols = {c["name"] for c in inspector.get_columns("task_miner_results")}
    if "miner_first_token_seconds" not in cols:
        return
    with engine.begin() as conn:
        conn.execute(text(
            "ALTER TABLE task_miner_results DROP COLUMN miner_first_token_seconds"
        ))


def _migration_add_session_mode_web_search(engine: Engine) -> None:
    inspector = inspect(engine)
    if "consumer_sessions" not in set(inspector.get_table_names()):
        return
    cols = {c["name"] for c in inspector.get_columns("consumer_sessions")}
    with engine.begin() as conn:
        if "mode" not in cols:
            conn.execute(text(
                "ALTER TABLE consumer_sessions "
                "ADD COLUMN mode VARCHAR(16) NOT NULL DEFAULT 'instant'"
            ))
        if "web_search" not in cols:
            conn.execute(text(
                "ALTER TABLE consumer_sessions "
                "ADD COLUMN web_search BOOLEAN NOT NULL DEFAULT FALSE"
            ))


def _migration_add_proxy_and_judge_cost(engine: Engine) -> None:
    inspector = inspect(engine)
    if "task_miner_results" not in set(inspector.get_table_names()):
        return
    cols = {c["name"] for c in inspector.get_columns("task_miner_results")}
    with engine.begin() as conn:
        if "proxy_cost_usd" not in cols:
            conn.execute(text(
                "ALTER TABLE task_miner_results "
                "ADD COLUMN proxy_cost_usd FLOAT NOT NULL DEFAULT 0.0"
            ))
        if "judge_cost_usd" not in cols:
            conn.execute(text(
                "ALTER TABLE task_miner_results "
                "ADD COLUMN judge_cost_usd FLOAT NOT NULL DEFAULT 0.0"
            ))


def _migration_drop_anti_gaming_artifacts(engine: Engine) -> None:
    """Strip anti-gaming residue from existing rows.

    The honeytoken / trace-gate code path was removed.  No SQL columns
    were ever added for it, but ``evaluation_runs.metadata_json`` carried
    a ``"honeytokens"`` key with the per-run canary URL list, and
    ``deployment_score_records`` (via Pydantic JSON serialization) may
    have ``trace_gate``, ``trace_gate_penalty_usd``, or ``honeytoken_cited``
    keys embedded in stored conversation-score blobs.  Strip both for
    cleanliness so reads don't surface stale fields.
    """
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    if "evaluation_runs" not in tables:
        return
    cols = {col["name"] for col in inspector.get_columns("evaluation_runs")}
    if "metadata_json" not in cols:
        return
    dialect = engine.dialect.name
    with engine.begin() as conn:
        if dialect == "postgresql":
            # JSONB-aware: strip the "honeytokens" key in place.
            conn.execute(text(
                "UPDATE evaluation_runs "
                "SET metadata_json = metadata_json::jsonb - 'honeytokens' "
                "WHERE metadata_json::jsonb ? 'honeytokens'"
            ))
        else:
            # SQLite / others: rebuild the JSON without the key.  Cheap
            # and correct because run counts are small.
            rows = conn.execute(text(
                "SELECT id, metadata_json FROM evaluation_runs "
                "WHERE metadata_json IS NOT NULL"
            )).fetchall()
            import json as _json
            for row in rows:
                raw = row[1]
                if not raw:
                    continue
                try:
                    parsed = _json.loads(raw) if isinstance(raw, str) else dict(raw)
                except (ValueError, TypeError):
                    continue
                if not isinstance(parsed, dict) or "honeytokens" not in parsed:
                    continue
                parsed.pop("honeytokens", None)
                conn.execute(
                    text(
                        "UPDATE evaluation_runs SET metadata_json = :m "
                        "WHERE id = :id"
                    ),
                    {"m": _json.dumps(parsed), "id": row[0]},
                )


def _drop_registered_neurons_is_active(engine: Engine) -> None:
    inspector = inspect(engine)
    if "registered_neurons" not in set(inspector.get_table_names()):
        return
    if "is_active" not in {c["name"] for c in inspector.get_columns("registered_neurons")}:
        return
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE registered_neurons DROP COLUMN is_active"))


def _migration_refactor_to_task_level_evaluation(engine: Engine) -> None:
    """Replace per-(miner, task) rows with task-level rows + per-miner results.

    Drops `miner_evaluation_tasks` (per-pair claim rows) and creates
    `task_evaluations` (one row per task, validator claims this) plus
    `task_miner_results` (one row per miner per task, stores pairwise judge
    output vs OpenAI baseline). Non-reversible: any in-flight pairs are lost.
    """
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    with engine.begin() as conn:
        if "miner_evaluation_tasks" in tables:
            conn.execute(text("DROP TABLE miner_evaluation_tasks"))
        if "task_evaluations" not in tables:
            conn.execute(text("""
                CREATE TABLE task_evaluations (
                    id VARCHAR(36) PRIMARY KEY,
                    epoch_id VARCHAR(128) NOT NULL,
                    family_id VARCHAR(64) NOT NULL,
                    task_id VARCHAR(128) NOT NULL,
                    task_index INTEGER NOT NULL,
                    status VARCHAR(32) NOT NULL DEFAULT 'pending',
                    claimed_by_validator VARCHAR(128),
                    claimed_at TIMESTAMP,
                    claim_expires_at TIMESTAMP,
                    claim_attempt_count INTEGER NOT NULL DEFAULT 0,
                    baseline_response_json JSON,
                    evaluated_at TIMESTAMP,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(epoch_id, family_id, task_id)
                )
            """))
            conn.execute(text(
                "CREATE INDEX idx_te_claimable ON task_evaluations (epoch_id, family_id, status, claim_expires_at)"
            ))
            conn.execute(text(
                "CREATE INDEX idx_te_epoch_id ON task_evaluations (epoch_id)"
            ))
            conn.execute(text(
                "CREATE INDEX idx_te_status ON task_evaluations (status)"
            ))
        if "task_miner_results" not in tables:
            conn.execute(text("""
                CREATE TABLE task_miner_results (
                    id VARCHAR(36) PRIMARY KEY,
                    task_evaluation_id VARCHAR(36) NOT NULL REFERENCES task_evaluations(id) ON DELETE CASCADE,
                    epoch_id VARCHAR(128) NOT NULL,
                    family_id VARCHAR(64) NOT NULL,
                    task_id VARCHAR(128) NOT NULL,
                    miner_hotkey VARCHAR(128) NOT NULL,
                    miner_response_json JSON NOT NULL,
                    miner_citations_json JSON NOT NULL DEFAULT '[]',
                    judge_output_json JSON,
                    agreement_verdict VARCHAR(32) NOT NULL,
                    agreement_score FLOAT NOT NULL DEFAULT 0.0,
                    latency_seconds FLOAT NOT NULL DEFAULT 0.0,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(task_evaluation_id, miner_hotkey)
                )
            """))
            conn.execute(text(
                "CREATE INDEX idx_tmr_miner ON task_miner_results (epoch_id, family_id, miner_hotkey)"
            ))
            conn.execute(text(
                "CREATE INDEX idx_tmr_task_eval ON task_miner_results (task_evaluation_id)"
            ))
