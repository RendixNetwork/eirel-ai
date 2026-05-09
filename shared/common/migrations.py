"""Schema migration runner for eirel-ai.

The schema is owned by ``shared.common.models.Base.metadata.create_all``;
migrations are a thin advisory-lock wrapper that records "the schema as
of this revision" in the ``schema_migrations`` table.

Single migration: ``initial_schema``. Pre-launch we collapsed every
prior migration step into ``Base.metadata.create_all`` since no
production DBs need in-place ALTERs. Future migrations append a new
``Migration`` entry to ``MIGRATIONS`` with one-purpose ALTER DDL.
"""
from __future__ import annotations

import logging

from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.engine import Engine

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
            conn.execute(
                text("SELECT pg_advisory_lock(:lock_id)"),
                {"lock_id": _MIGRATION_ADVISORY_LOCK_ID},
            )
        _logger.info("acquired migration advisory lock")
    try:
        return _run_migrations_unlocked(engine)
    finally:
        if dialect == "postgresql":
            with engine.begin() as conn:
                conn.execute(
                    text("SELECT pg_advisory_unlock(:lock_id)"),
                    {"lock_id": _MIGRATION_ADVISORY_LOCK_ID},
                )
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
        rows = conn.execute(
            text("SELECT version FROM schema_migrations")
        ).fetchall()
    return {str(row[0]) for row in rows}


def _migration_initial_schema(engine: Engine) -> None:
    """Marker migration — the schema itself is created by
    ``Base.metadata.create_all`` in ``Database.create_all``.

    Pre-launch we collapsed every prior schema step (validators table,
    submission/deployment plumbing, evaluation runs/tasks, owner dataset
    bindings, consumer-side product tables, MCP catalog, server-attested
    tool-call ledger, etc.) into the ``Base`` SQLAlchemy declaration.
    Fresh DBs come up via ``create_all``; this migration row records
    that fact in ``schema_migrations`` so future ALTER-style migrations
    have a baseline to anchor against.
    """
    del engine


def _migration_multi_metric_scoring(engine: Engine) -> None:
    """Add per-dimension score columns to ``task_miner_results``.

    Each task is now scored along six independent dimensions
    (``pairwise_preference_score`` + 5 outer metrics) plus an aggregate
    ``final_task_score``. ``applied_weights_json`` records the actual
    weights after N/A re-normalization for the task type. New columns
    are nullable so legacy pairwise-only rows coexist.

    Skipped on fresh databases — there ``Base.metadata.create_all``
    runs after migrations and creates the table with the new columns
    already in place. Migrations only do work on pre-existing DBs that
    were bootstrapped before this column set was added.
    """
    from sqlalchemy import inspect, text
    inspector = inspect(engine)
    if "task_miner_results" not in inspector.get_table_names():
        return
    existing_columns = {
        col["name"] for col in inspector.get_columns("task_miner_results")
    }
    columns_to_add = (
        ("pairwise_preference_score", "DOUBLE PRECISION"),
        ("grounded_correctness", "DOUBLE PRECISION"),
        ("retrieval_quality", "DOUBLE PRECISION"),
        ("tool_routing", "DOUBLE PRECISION"),
        ("instruction_safety", "DOUBLE PRECISION"),
        ("latency_cost", "DOUBLE PRECISION"),
        ("computation_correctness", "DOUBLE PRECISION"),
        ("final_task_score", "DOUBLE PRECISION"),
        ("applied_weights_json", "JSON"),
        ("applicable_metrics_json", "JSON"),
        ("task_type", "VARCHAR(64)"),
    )
    with engine.begin() as conn:
        for name, col_type in columns_to_add:
            if name in existing_columns:
                continue
            conn.execute(
                text(
                    f"ALTER TABLE task_miner_results ADD COLUMN {name} {col_type}"
                )
            )


def _migration_eval_feedback_table(engine: Engine) -> None:
    """Create the ``eval_feedback`` table for per-(run, miner, task)
    EvalJudge outcomes.

    Skipped on fresh databases — there ``Base.metadata.create_all``
    runs after migrations and creates the table from the SQLAlchemy
    model declaration. This migration only does work on pre-existing
    DBs that were bootstrapped before the table existed.
    """
    from sqlalchemy import inspect, text
    inspector = inspect(engine)
    if "eval_feedback" in inspector.get_table_names():
        return
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS eval_feedback (
                    id VARCHAR(36) PRIMARY KEY,
                    run_id VARCHAR(64) NOT NULL,
                    miner_hotkey VARCHAR(64) NOT NULL,
                    task_id VARCHAR(64) NOT NULL,
                    outcome VARCHAR(32) NOT NULL,
                    failure_mode VARCHAR(64),
                    guidance TEXT NOT NULL DEFAULT '',
                    prompt_excerpt TEXT NOT NULL DEFAULT '',
                    response_excerpt TEXT NOT NULL DEFAULT '',
                    composite_score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                    knockout_reasons_json JSON NOT NULL,
                    oracle_status VARCHAR(32),
                    created_at TIMESTAMP NOT NULL,
                    CONSTRAINT uq_eval_feedback_run_miner_task
                        UNIQUE (run_id, miner_hotkey, task_id)
                )
                """
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_eval_feedback_miner_run "
                "ON eval_feedback (miner_hotkey, run_id)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_eval_feedback_run_task "
                "ON eval_feedback (run_id, task_id)"
            )
        )


def _migration_validator_cost_tracking(engine: Engine) -> None:
    """Add ``oracle_cost_usd`` to ``task_evaluations`` so the validator-
    cost dashboard can roll up per-validator per-run spend.

    Skipped on fresh databases — there ``Base.metadata.create_all``
    runs after migrations and creates the column from the SQLAlchemy
    model. Only does work on pre-existing DBs.
    """
    from sqlalchemy import inspect, text
    inspector = inspect(engine)
    if "task_evaluations" not in inspector.get_table_names():
        return
    existing_columns = {
        col["name"] for col in inspector.get_columns("task_evaluations")
    }
    if "oracle_cost_usd" in existing_columns:
        return
    with engine.begin() as conn:
        conn.execute(
            text(
                "ALTER TABLE task_evaluations "
                "ADD COLUMN oracle_cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0.0"
            )
        )


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        version="initial_schema",
        description=(
            "Initial schema — owned by Base.metadata.create_all. "
            "Subsequent migrations append ALTER DDL for in-place upgrades."
        ),
        apply=_migration_initial_schema,
    ),
    Migration(
        version="multi_metric_scoring",
        description=(
            "Per-dimension score columns on task_miner_results: "
            "pairwise + grounded + retrieval + tool_routing + safety + "
            "latency_cost + computation_correctness + final_task_score, "
            "plus applied_weights_json / applicable_metrics_json / task_type."
        ),
        apply=_migration_multi_metric_scoring,
    ),
    Migration(
        version="add_eval_feedback_table",
        description=(
            "Create eval_feedback table for per-(run, miner, task) "
            "EvalJudge outcomes. Indexed on (miner_hotkey, run_id) for "
            "the per-miner read path and (run_id, task_id) for "
            "operator dashboard cross-miner drill-in."
        ),
        apply=_migration_eval_feedback_table,
    ),
    Migration(
        version="validator_cost_tracking",
        description=(
            "Add task_evaluations.oracle_cost_usd — validator's "
            "grounding-layer spend (oracle fanout + reconciler) per "
            "task. Combined with task_miner_results.judge_cost_usd in "
            "the validator-costs dashboard endpoint."
        ),
        apply=_migration_validator_cost_tracking,
    ),
)
