"""Smoke test: the product runtime tables exist after Database.create_all."""
from __future__ import annotations

from sqlalchemy import inspect

from shared.common.database import Database


_PRODUCT_TABLES = {
    "consumer_users",
    "consumer_projects",
    "consumer_conversations",
    "consumer_messages",
    "consumer_preferences",
    "consumer_project_memory",
    "serving_promotions",
}


def test_product_runtime_tables_exist_after_create_all(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'm.db'}")
    db.create_all()
    inspector = inspect(db.engine)
    tables = set(inspector.get_table_names())
    missing = _PRODUCT_TABLES - tables
    assert not missing, f"missing product tables: {sorted(missing)}"


def test_migration_records_initial_schema_version(tmp_path):
    """The collapsed ``initial_schema`` migration is registered and applied.

    Pre-launch we collapsed every prior per-feature migration into the
    ``Base.metadata.create_all`` declaration, so this single marker is
    the only entry in ``schema_migrations`` for a fresh DB.
    """
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'm.db'}")
    db.create_all()
    with db.engine.connect() as conn:
        from sqlalchemy import text
        applied = {
            row[0]
            for row in conn.execute(text("SELECT version FROM schema_migrations"))
        }
    assert "initial_schema" in applied
