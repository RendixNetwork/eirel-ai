"""Latency-weighted ServingPicker tests."""
from __future__ import annotations

import random
from collections import Counter
from datetime import datetime
from uuid import uuid4

from shared.common.database import Database
from shared.common.models import (
    ManagedDeployment,
    ManagedMinerSubmission,
    ServingDeployment,
    ServingRelease,
)
from orchestration.orchestrator.serving_picker import ServingPicker


def _make_db(tmp_path) -> Database:
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'spl.db'}")
    db.create_all()
    return db


def _seed_serving_with_latency(
    session,
    *,
    deployment_id: str,
    miner_hotkey: str,
    latency_ms_p50: int,
) -> None:
    submission_id = str(uuid4())
    session.add(ManagedMinerSubmission(
        id=submission_id, miner_hotkey=miner_hotkey, submission_seq=1,
        family_id="general_chat", status="deployed", artifact_id=str(uuid4()),
        manifest_json={"runtime": {"kind": "graph"}},
        archive_sha256="0" * 64, submission_block=0,
    ))
    source_id = str(uuid4())
    session.add(ManagedDeployment(
        id=source_id, submission_id=submission_id, miner_hotkey=miner_hotkey,
        family_id="general_chat", deployment_revision=str(uuid4()),
        image_ref="img:x", endpoint=f"http://eval-{deployment_id}.test:8080",
        status="active", health_status="healthy", placement_status="placed",
        latency_ms_p50=latency_ms_p50,
    ))
    release_id = str(uuid4())
    session.add(ServingRelease(
        id=release_id, trigger_type="t",
        status="published", published_at=datetime.utcnow(),
    ))
    session.add(ServingDeployment(
        id=deployment_id, release_id=release_id, family_id="general_chat",
        source_deployment_id=source_id, source_submission_id=submission_id,
        miner_hotkey=miner_hotkey, source_deployment_revision=str(uuid4()),
        endpoint=f"http://serving-{deployment_id}.test:8080",
        status="healthy", health_status="healthy",
        published_at=datetime.utcnow(),
    ))
    session.commit()


def test_picker_carries_latency_in_candidate(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_serving_with_latency(
            session, deployment_id="d1", miner_hotkey="hk1", latency_ms_p50=300,
        )
    picker = ServingPicker(database=db)
    cand = picker.pick(family_id="general_chat")
    assert cand.deployment_id == "d1"
    assert cand.latency_ms_p50 == 300


def test_picker_weights_inverse_to_latency(tmp_path):
    """Fast replica gets ≈5× the share of a 5×-slower replica."""
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_serving_with_latency(
            session, deployment_id="fast", miner_hotkey="hk-fast",
            latency_ms_p50=200,
        )
        _seed_serving_with_latency(
            session, deployment_id="slow", miner_hotkey="hk-slow",
            latency_ms_p50=1000,
        )
    rng = random.Random(1234)
    picker = ServingPicker(database=db, rng=rng)
    counts: Counter[str] = Counter()
    n = 2000
    for _ in range(n):
        cand = picker.pick(family_id="general_chat")
        counts[cand.deployment_id] += 1
    # Expected ratio: weight_fast / weight_slow = (1/200) / (1/1000) = 5.
    fast_share = counts["fast"] / n
    slow_share = counts["slow"] / n
    # Hard tolerance: ratio between 4 and 6 over 2000 trials.
    ratio = fast_share / slow_share
    assert 3.8 < ratio < 6.2, (
        f"weighting off — fast={fast_share:.3f} slow={slow_share:.3f} "
        f"ratio={ratio:.2f}"
    )


def test_picker_floor_caps_pathologically_fast_replica(tmp_path):
    """Replica reporting 1ms gets clamped to floor — no runaway weight."""
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_serving_with_latency(
            session, deployment_id="micro", miner_hotkey="hk-micro",
            latency_ms_p50=1,
        )
        _seed_serving_with_latency(
            session, deployment_id="normal", miner_hotkey="hk-normal",
            latency_ms_p50=400,
        )
    rng = random.Random(99)
    # Floor 200ms → micro clamps to 200 → ratio is 400/200 = 2.0, not 400.
    picker = ServingPicker(database=db, rng=rng, latency_floor_ms=200)
    counts: Counter[str] = Counter()
    n = 1500
    for _ in range(n):
        counts[picker.pick(family_id="general_chat").deployment_id] += 1
    ratio = counts["micro"] / counts["normal"]
    # With clamping the ratio should be near 2 — not the unclamped 400×.
    assert 1.4 < ratio < 2.8, f"clamp ineffective — ratio={ratio:.2f}"


def test_picker_newcomer_gets_median_weight(tmp_path):
    """Replica with latency_ms_p50=0 (no telemetry) gets median weight.

    Avoids two pathological behaviours: (a) infinite weight from 1/0
    (the code uses None → median fallback), (b) zero weight that
    starves newcomers from ever seeing traffic.
    """
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_serving_with_latency(
            session, deployment_id="seasoned", miner_hotkey="hk-old",
            latency_ms_p50=400,
        )
        _seed_serving_with_latency(
            session, deployment_id="newcomer", miner_hotkey="hk-new",
            latency_ms_p50=0,
        )
    rng = random.Random(42)
    picker = ServingPicker(database=db, rng=rng)
    counts: Counter[str] = Counter()
    n = 1000
    for _ in range(n):
        counts[picker.pick(family_id="general_chat").deployment_id] += 1
    # Median of one known weight is itself → newcomer gets the same weight
    # → ~50/50 split.
    new_share = counts["newcomer"] / n
    assert 0.40 < new_share < 0.60, (
        f"newcomer share off — got {new_share:.3f}"
    )


def test_picker_weighted_disabled_falls_back_to_round_robin(tmp_path):
    """weighted=False reverts to deterministic round-robin cursor behavior."""
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_serving_with_latency(
            session, deployment_id="a", miner_hotkey="hk1",
            latency_ms_p50=100,
        )
        _seed_serving_with_latency(
            session, deployment_id="b", miner_hotkey="hk2",
            latency_ms_p50=10000,  # very slow, would otherwise get ~0 traffic
        )
    picker = ServingPicker(database=db, weighted=False)
    seen: list[str] = []
    for _ in range(4):
        seen.append(picker.pick(family_id="general_chat").deployment_id)
    # Round-robin cycles deterministically; each replica is hit at least once.
    assert set(seen) == {"a", "b"}


def test_picker_single_candidate_always_picks_it_and_returns_weight_one(tmp_path):
    db = _make_db(tmp_path)
    with db.sessionmaker() as session:
        _seed_serving_with_latency(
            session, deployment_id="lonely", miner_hotkey="hk1",
            latency_ms_p50=500,
        )
    picker = ServingPicker(database=db)
    cand = picker.pick(family_id="general_chat")
    assert cand.deployment_id == "lonely"
    assert cand.picker_weight == 1.0
