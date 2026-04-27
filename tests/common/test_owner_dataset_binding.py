from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError

from shared.common.database import Database
from shared.common.models import OwnerDatasetBinding


def _make_binding(
    *,
    family_id: str = "analyst",
    run_id: str = "run-001",
    status: str = "pending",
    bundle_uri: str = "s3://eirel-owner-private/datasets/bundles/run-001/analyst.json",
) -> OwnerDatasetBinding:
    return OwnerDatasetBinding(
        family_id=family_id,
        run_id=run_id,
        bundle_uri=bundle_uri,
        bundle_sha256="a" * 64,
        generator_version="owner_dataset_v1",
        generated_by="5Fxxx",
        signature_hex="deadbeef",
        generator_provider="openai",
        generator_model="gpt-4o",
        status=status,
        provenance_json={"seed_hash": "1234", "topic_pool_version": "v2"},
    )


def test_migration_creates_owner_dataset_bindings_table(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'bindings.db'}")
    db.create_all()

    inspector = inspect(db.engine)
    assert "owner_dataset_bindings" in inspector.get_table_names()

    columns = {col["name"] for col in inspector.get_columns("owner_dataset_bindings")}
    expected = {
        "id",
        "family_id",
        "run_id",
        "bundle_uri",
        "bundle_sha256",
        "generator_version",
        "generated_by",
        "signature_hex",
        "generator_provider",
        "generator_model",
        "status",
        "provenance_json",
        "created_at",
        "activated_at",
    }
    assert expected.issubset(columns)


def test_binding_insert_and_query_roundtrip(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'bindings.db'}")
    db.create_all()

    with db.sessionmaker() as session:
        session.add(_make_binding())
        session.commit()

    with db.sessionmaker() as session:
        row = session.execute(
            select(OwnerDatasetBinding).where(
                OwnerDatasetBinding.family_id == "analyst",
                OwnerDatasetBinding.run_id == "run-001",
            )
        ).scalar_one()
        assert row.bundle_sha256 == "a" * 64
        assert row.status == "pending"
        assert row.provenance_json["seed_hash"] == "1234"
        assert row.activated_at is None


def test_binding_unique_family_run_constraint(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'bindings.db'}")
    db.create_all()

    with db.sessionmaker() as session:
        session.add(_make_binding())
        session.commit()

    with db.sessionmaker() as session:
        session.add(_make_binding())  # same (analyst, run-001)
        with pytest.raises(IntegrityError):
            session.commit()


def test_binding_allows_multiple_runs_for_family(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'bindings.db'}")
    db.create_all()

    with db.sessionmaker() as session:
        session.add(_make_binding(run_id="run-001"))
        session.add(_make_binding(run_id="run-002"))
        session.add(_make_binding(run_id="run-003", status="active"))
        session.commit()

    with db.sessionmaker() as session:
        rows = session.execute(
            select(OwnerDatasetBinding).where(OwnerDatasetBinding.family_id == "analyst")
        ).scalars().all()
        assert {row.run_id for row in rows} == {"run-001", "run-002", "run-003"}


def test_binding_activation_timestamp(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'bindings.db'}")
    db.create_all()

    with db.sessionmaker() as session:
        session.add(_make_binding(status="pending"))
        session.commit()

    activation = datetime.now(UTC).replace(tzinfo=None)
    with db.sessionmaker() as session:
        row = session.execute(
            select(OwnerDatasetBinding).where(OwnerDatasetBinding.run_id == "run-001")
        ).scalar_one()
        row.status = "active"
        row.activated_at = activation
        session.commit()

    with db.sessionmaker() as session:
        row = session.execute(
            select(OwnerDatasetBinding).where(OwnerDatasetBinding.run_id == "run-001")
        ).scalar_one()
        assert row.status == "active"
        assert row.activated_at == activation
