from __future__ import annotations

from sqlalchemy import inspect, text

from shared.common.database import Database
from shared.common.migrations import MIGRATIONS, run_migrations


def test_fresh_db_boot_creates_family_native_schema(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'fresh.db'}")
    db.create_all()

    inspector = inspect(db.engine)
    tables = set(inspector.get_table_names())

    assert "aggregate_family_score_snapshots" in tables
    assert "family_rollout_states" in tables
    assert "run_family_results" in tables
    assert "aggregate_group_score_snapshots" not in tables
    assert "group_rollout_states" not in tables
    assert "run_group_results" not in tables

    for table_name in (
        "managed_miner_submissions",
        "managed_deployments",
        "deployment_score_records",
        "epoch_target_snapshots",
        "validator_score_submissions",
        "aggregate_family_score_snapshots",
        "family_rollout_states",
        "run_family_results",
        "managed_artifacts",
        "dag_node_executions",
        "serving_deployments",
        "deployment_health_events",
    ):
        columns = {column["name"] for column in inspector.get_columns(table_name)}
        assert "family_id" in columns
        assert "group_id" not in columns

    dag_columns = {column["name"]: column for column in inspector.get_columns("dag_node_executions")}
    assert dag_columns["family_id"]["nullable"] is True


def test_migrations_are_idempotent_for_fresh_family_schema(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'fresh.db'}")

    first = run_migrations(db.engine)
    second = run_migrations(db.engine)

    assert first == [migration.version for migration in MIGRATIONS]
    assert second == []

    with db.engine.begin() as conn:
        rows = conn.execute(text("SELECT version FROM schema_migrations ORDER BY applied_at")).fetchall()

    assert [row[0] for row in rows] == [migration.version for migration in MIGRATIONS]
