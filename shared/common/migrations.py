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
)


def _drop_registered_neurons_is_active(engine: Engine) -> None:
    inspector = inspect(engine)
    if "registered_neurons" not in set(inspector.get_table_names()):
        return
    if "is_active" not in {c["name"] for c in inspector.get_columns("registered_neurons")}:
        return
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE registered_neurons DROP COLUMN is_active"))
