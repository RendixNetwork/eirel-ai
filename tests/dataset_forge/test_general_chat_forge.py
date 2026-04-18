from __future__ import annotations

from shared.dataset_forge.general_chat_forge import (
    CATEGORY_PERCENTAGES,
    SCRIPTED_FRACTION,
    forge_general_chat_bundle,
    forge_scripted_conversation,
    forge_simulated_user_conversation,
)
import random


def test_forge_general_chat_bundle_is_deterministic_for_seed():
    bundle_a = forge_general_chat_bundle(size=100, rng_seed=42)
    bundle_b = forge_general_chat_bundle(size=100, rng_seed=42)
    assert bundle_a.bundle_id == bundle_b.bundle_id
    assert len(bundle_a.fixtures) == len(bundle_b.fixtures) == 100
    ids_a = [f.conversation_id for f in bundle_a.fixtures]
    ids_b = [f.conversation_id for f in bundle_b.fixtures]
    assert ids_a == ids_b


def test_forge_general_chat_bundle_differs_with_different_seeds():
    bundle_a = forge_general_chat_bundle(size=50, rng_seed=1)
    bundle_b = forge_general_chat_bundle(size=50, rng_seed=2)
    assert bundle_a.bundle_id != bundle_b.bundle_id


def test_forge_general_chat_bundle_respects_category_percentages():
    bundle = forge_general_chat_bundle(size=1000, rng_seed=99)
    counts: dict[str, int] = {}
    for fixture in bundle.fixtures:
        counts[fixture.category] = counts.get(fixture.category, 0) + 1
    for category, pct in CATEGORY_PERCENTAGES.items():
        observed = counts.get(category, 0)
        expected = int(round(1000 * pct))
        # Allow +/- 1 slot of rounding slack on the tiny safety bucket.
        assert abs(observed - expected) <= 2, (
            f"category {category}: expected ≈{expected}, observed {observed}"
        )


def test_forge_general_chat_bundle_respects_scripted_fraction():
    bundle = forge_general_chat_bundle(size=200, rng_seed=7)
    scripted = [f for f in bundle.fixtures if f.fixture_type == "scripted"]
    simulated = [f for f in bundle.fixtures if f.fixture_type == "simulated"]
    assert len(scripted) + len(simulated) == 200
    expected_scripted = int(round(200 * SCRIPTED_FRACTION))
    assert abs(len(scripted) - expected_scripted) <= 2


def test_forge_scripted_conversation_has_user_turn_and_canonical_state():
    rng = random.Random(0)
    fixture = forge_scripted_conversation("factual", "standard", rng)
    assert fixture.fixture_type == "scripted"
    assert fixture.category == "factual"
    assert fixture.difficulty == "standard"
    assert fixture.turns
    assert fixture.turns[0].role == "user"
    assert fixture.canonical_end_state is not None


def test_forge_simulated_user_conversation_locks_persona_and_seed():
    rng = random.Random(0)
    fixture = forge_simulated_user_conversation(
        "social_x", "What is trending today?", rng
    )
    assert fixture.fixture_type == "simulated"
    assert fixture.persona
    assert isinstance(fixture.rng_seed, int)
    assert fixture.canonical_end_state["goal_summary"] == "What is trending today?"


def test_forge_general_chat_bundle_rejects_non_positive_size():
    import pytest

    with pytest.raises(ValueError):
        forge_general_chat_bundle(size=0)
