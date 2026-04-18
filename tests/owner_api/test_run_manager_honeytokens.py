from __future__ import annotations

import pytest

from shared.common.config import Settings, reset_settings
from shared.common.database import Database
from shared.common.artifacts import create_artifact_store
from shared.common.models import EvaluationRun
from shared.core.honeytokens import HONEYTOKEN_MARKER, generate_honeytoken_set
from control_plane.owner_api.managed import ManagedOwnerServices
from control_plane.owner_api.deployment.runtime_manager import ManagedDeploymentRuntimeManager
from tests.owner_api.test_submission_lifecycle_budget import _FakeBackend


@pytest.fixture
def services(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("METAGRAPH_SYNC_INTERVAL_SECONDS", "3600")
    monkeypatch.setenv("RESULT_AGGREGATION_INTERVAL_SECONDS", "3600")
    monkeypatch.setenv("USE_REDIS_POOL", "0")
    monkeypatch.setenv("VALIDATOR_EPOCH_QUORUM", "1")
    monkeypatch.setenv("METAGRAPH_SNAPSHOT_PATH", str(tmp_path / "metagraph.json"))
    monkeypatch.setenv("EIREL_PROVIDER_PROXY_URL", "http://provider-proxy.test")
    monkeypatch.setenv("EIREL_PROVIDER_PROXY_TOKEN", "provider-token")
    monkeypatch.setenv("EIREL_INTERNAL_SERVICE_TOKEN", "internal-token")
    monkeypatch.setenv("EIREL_ACTIVE_FAMILIES", "general_chat")
    monkeypatch.setenv("EIREL_LAUNCH_MODE", "1")
    from tests.conftest import FIXTURES_ROOT
    monkeypatch.setenv(
        "EIREL_OWNER_DATASET_ROOT_PATH",
        str(FIXTURES_ROOT / "owner_datasets" / "families"),
    )
    # _initialize_run_benchmarks would try to load a general_chat.json
    # dataset bundle from the fixtures path, which doesn't exist yet.
    # These tests are about create_run's honeytoken population, not
    # bundle loading — short-circuit the benchmark init.
    from control_plane.owner_api.evaluation import run_manager as rm_mod
    monkeypatch.setattr(
        rm_mod.RunManager,
        "_initialize_run_benchmarks",
        lambda self, session, *, run: None,
    )

    reset_settings()
    settings = Settings()
    db = Database(settings.database_url)
    db.create_all()
    backend = _FakeBackend()
    runtime_mgr = ManagedDeploymentRuntimeManager(backend=backend)
    artifact_store = create_artifact_store(settings)
    svc = ManagedOwnerServices(
        db=db,
        settings=settings,
        runtime_manager=runtime_mgr,
        artifact_store=artifact_store,
    )
    yield svc
    reset_settings()


def test_create_run_populates_honeytokens_on_open_run(services):
    with services.db.sessionmaker() as session:
        run = services.runs.create_run(session, sequence=1, status="open")
        session.commit()
        session.refresh(run)
    assert isinstance(run.metadata_json, dict)
    honeytokens = run.metadata_json.get("honeytokens")
    assert isinstance(honeytokens, list)
    assert len(honeytokens) == services.settings.honeytoken_count_per_run
    for url in honeytokens:
        assert url.startswith("https://")
        assert HONEYTOKEN_MARKER in url


def test_create_run_populates_honeytokens_on_scheduled_run(services):
    with services.db.sessionmaker() as session:
        run = services.runs.create_run(session, sequence=7, status="scheduled")
        session.commit()
        session.refresh(run)
    # Scheduled runs get the honeytoken set populated too, so the tool
    # proxy can load them as soon as miners for this run spin up.
    honeytokens = run.metadata_json.get("honeytokens")
    assert isinstance(honeytokens, list)
    assert len(honeytokens) > 0


def test_create_run_honeytokens_are_deterministic_by_run_id(services):
    # Two create_run calls with the same sequence (different DBs) must
    # produce identical honeytoken sets — deterministic per run_id.
    with services.db.sessionmaker() as session:
        run_a = services.runs.create_run(session, sequence=42, status="scheduled")
        session.commit()
        session.refresh(run_a)
        expected = generate_honeytoken_set(
            run_a.id, count=services.settings.honeytoken_count_per_run,
        )
    assert run_a.metadata_json["honeytokens"] == expected


def test_create_run_honeytokens_vary_per_sequence(services):
    with services.db.sessionmaker() as session:
        run_1 = services.runs.create_run(session, sequence=1, status="scheduled")
        run_2 = services.runs.create_run(session, sequence=2, status="scheduled")
        session.commit()
        session.refresh(run_1)
        session.refresh(run_2)
    set_1 = set(run_1.metadata_json["honeytokens"])
    set_2 = set(run_2.metadata_json["honeytokens"])
    assert set_1 & set_2 == set()  # disjoint


def test_create_run_honeytoken_count_configurable(services, monkeypatch):
    monkeypatch.setattr(services.settings, "honeytoken_count_per_run", 3)
    with services.db.sessionmaker() as session:
        run = services.runs.create_run(session, sequence=99, status="scheduled")
        session.commit()
        session.refresh(run)
    assert len(run.metadata_json["honeytokens"]) == 3


def test_create_run_zero_count_produces_empty_list(services, monkeypatch):
    monkeypatch.setattr(services.settings, "honeytoken_count_per_run", 0)
    with services.db.sessionmaker() as session:
        run = services.runs.create_run(session, sequence=100, status="scheduled")
        session.commit()
        session.refresh(run)
    assert run.metadata_json["honeytokens"] == []
