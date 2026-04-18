"""Random selection of hidden fixtures per track.

The owner-api loader (``_load_owner_evaluation_bundle_seed``) requires that
the analyst bundle has at least 4 hidden fixtures per track and at least 12
total. Hidden fixtures are tasks the validators score against but which the
miners are not told about. Today the marker is metadata-level: ``metadata``
contains either ``hidden_fixture: true`` or ``visibility: "hidden"``.

In the new pipeline, *which* tasks are hidden is randomized at forge time
(seeded by ``run_id`` for reproducibility), so miners can't pre-classify the
fixture set even if a previous bundle leaks.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from shared.dataset_forge.diversity import CandidatePrompt


# Analyst contract — kept in sync with the loader assertions.
ANALYST_HIDDEN_PER_TRACK = 4


class HiddenAllocationError(RuntimeError):
    pass


@dataclass(slots=True)
class HiddenAllocationResult:
    hidden_seed_ids: set[str]
    hidden_indices_by_track: dict[str, list[int]]


def _seeded_rng(*, run_id: str, owner_secret: str = "") -> random.Random:
    digest = hashlib.sha256(f"hidden:{owner_secret}:{run_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def allocate_hidden(
    candidates_by_track: dict[str, list[CandidatePrompt]],
    *,
    run_id: str,
    per_track: int = ANALYST_HIDDEN_PER_TRACK,
    owner_secret: str = "",
) -> HiddenAllocationResult:
    """Mark ``per_track`` candidates per track as hidden.

    Mutates each selected ``CandidatePrompt.expected_output`` and is callers'
    responsibility to also stamp the metadata when materializing the final
    ``FamilyEvaluationTask``. The result captures both the indices (so callers
    can re-stamp) and the seed ids that were drawn.
    """
    rng = _seeded_rng(run_id=run_id, owner_secret=owner_secret)
    hidden_seed_ids: set[str] = set()
    hidden_indices_by_track: dict[str, list[int]] = {}
    for track, candidates in candidates_by_track.items():
        if len(candidates) < per_track:
            raise HiddenAllocationError(
                f"track {track!r} has only {len(candidates)} candidates; "
                f"need at least {per_track} for hidden allocation"
            )
        indices = list(range(len(candidates)))
        rng.shuffle(indices)
        chosen = sorted(indices[:per_track])
        hidden_indices_by_track[track] = chosen
        for idx in chosen:
            hidden_seed_ids.add(candidates[idx].seed_id)
    return HiddenAllocationResult(
        hidden_seed_ids=hidden_seed_ids,
        hidden_indices_by_track=hidden_indices_by_track,
    )
