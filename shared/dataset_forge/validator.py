from __future__ import annotations

"""Forge-side bundle validation (general_chat edition).

Legacy analyst/per-track validation has been removed. The general_chat
forge produces ``GeneralChatBundle`` objects directly, which carry their
own category/type invariants; this module exposes a minimal report type
for back-compat with callers that still reference ``BundleValidationReport``.
"""

from collections import Counter
from dataclasses import dataclass

from shared.dataset_forge.general_chat_forge import (
    CATEGORY_PERCENTAGES,
    GeneralChatBundle,
)


class BundleValidationError(RuntimeError):
    """Raised when a general_chat bundle fails a hard invariant."""


@dataclass(slots=True)
class BundleValidationReport:
    bundle_id: str
    total_fixtures: int
    scripted_count: int
    simulated_count: int
    category_distribution: dict[str, int]
    difficulty_distribution: dict[str, int]


def validate_general_chat_bundle(bundle: GeneralChatBundle) -> BundleValidationReport:
    if not bundle.fixtures:
        raise BundleValidationError("general_chat bundle is empty")

    scripted = 0
    simulated = 0
    categories: Counter[str] = Counter()
    difficulties: Counter[str] = Counter()
    seen_ids: set[str] = set()

    for fixture in bundle.fixtures:
        if fixture.conversation_id in seen_ids:
            raise BundleValidationError(
                f"duplicate conversation_id in bundle: {fixture.conversation_id!r}"
            )
        seen_ids.add(fixture.conversation_id)

        categories[fixture.category] += 1
        difficulties[getattr(fixture, "difficulty", "standard")] += 1
        if fixture.fixture_type == "scripted":
            scripted += 1
        else:
            simulated += 1

    unknown = set(categories) - set(CATEGORY_PERCENTAGES)
    if unknown:
        raise BundleValidationError(
            f"bundle contains unknown categories: {sorted(unknown)}"
        )

    return BundleValidationReport(
        bundle_id=bundle.bundle_id,
        total_fixtures=len(bundle.fixtures),
        scripted_count=scripted,
        simulated_count=simulated,
        category_distribution=dict(categories),
        difficulty_distribution=dict(difficulties),
    )
