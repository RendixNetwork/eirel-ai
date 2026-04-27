"""Outcome-only agreement aggregation for the general_chat family.

After the redesign, miner scoring is the mean of per-task ``agreement_score``
values, derived from the eiretes agreement judge's verdict. Each task maps
to exactly one scalar in [0, 1]:

    matches            → 1.0
    partially_matches  → 0.6
    not_applicable     → 0.7   (open-ended tasks where agreement is N/A)
    contradicts        → 0.0
    error              → not counted

Error rows are excluded from the mean to avoid double-penalizing (a miner
whose pod failed and whose agreement couldn't be assessed). Instead,
``error_rate`` is tracked separately and caps the final score when a miner
fails on too many tasks, matching the old reliability gate.

Citations are NOT evaluated here. They are preserved on ``TaskMinerResult``
(``miner_citations_json``) purely for dashboard readback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from shared.core.evaluation_models import VERDICT_SCORES

if TYPE_CHECKING:
    from shared.common.models import TaskMinerResult


# Miners with error_rate above this threshold have their final score capped
# so a miner that crashes on 50% of tasks can't ride a high mean on the
# half that happened to succeed.
_ERROR_RATE_CAP_THRESHOLD = 0.30
_UNRELIABLE_SCORE_CAP = 0.5


@dataclass(frozen=True)
class MinerRollup:
    """Per-miner aggregation across all TaskMinerResult rows in a run."""

    miner_hotkey: str
    total_judged: int
    matches: int
    partially_matches: int
    not_applicable: int
    contradicts: int
    errored: int
    mean_agreement: float
    verdict_counts: dict[str, int] = field(default_factory=dict)
    final_score: float = 0.0
    reliable: bool = True

    @property
    def completed(self) -> int:
        return self.matches + self.partially_matches + self.not_applicable + self.contradicts

    @property
    def error_rate(self) -> float:
        denom = self.completed + self.errored
        return self.errored / denom if denom else 0.0

    def to_metadata(self) -> dict[str, Any]:
        return {
            "mean_agreement": self.mean_agreement,
            "final_score": self.final_score,
            "reliable": self.reliable,
            "error_rate": self.error_rate,
            "verdict_counts": dict(self.verdict_counts),
        }


def aggregate_miner_score(results: list["TaskMinerResult"]) -> MinerRollup:
    """Roll up agreement results for a single miner into a MinerRollup.

    mean_agreement = sum(agreement_score over non-error rows) / completed.
    Error rows do not contribute to numerator or denominator of the mean,
    but do contribute to error_rate which caps the final score when high.

    An empty-result miner returns an all-zero rollup marked unreliable.
    """
    if not results:
        return MinerRollup(
            miner_hotkey="",
            total_judged=0,
            matches=0,
            partially_matches=0,
            not_applicable=0,
            contradicts=0,
            errored=0,
            mean_agreement=0.0,
            final_score=0.0,
            reliable=False,
        )

    miner_hotkey = results[0].miner_hotkey

    verdict_counts: dict[str, int] = {
        "matches": 0, "partially_matches": 0,
        "not_applicable": 0, "contradicts": 0, "error": 0,
    }
    for r in results:
        v = r.agreement_verdict if r.agreement_verdict in verdict_counts else "error"
        verdict_counts[v] += 1

    completed = (
        verdict_counts["matches"]
        + verdict_counts["partially_matches"]
        + verdict_counts["not_applicable"]
        + verdict_counts["contradicts"]
    )

    # Mean of the scalar agreement scores over non-error rows. Prefer
    # stored agreement_score (already derived from verdict) so consumers
    # that backfill scalars directly still work.
    score_sum = 0.0
    for r in results:
        if r.agreement_verdict == "error":
            continue
        stored = float(r.agreement_score or 0.0)
        if stored <= 0.0 and r.agreement_verdict in VERDICT_SCORES:
            # Fallback: derive from the verdict if the scalar wasn't persisted.
            stored = VERDICT_SCORES[r.agreement_verdict]
        score_sum += stored
    mean_agreement = score_sum / completed if completed else 0.0

    total_judged = len(results)
    error_rate = verdict_counts["error"] / total_judged if total_judged else 0.0
    reliable = error_rate <= _ERROR_RATE_CAP_THRESHOLD
    final_score = mean_agreement if reliable else min(mean_agreement, _UNRELIABLE_SCORE_CAP)

    return MinerRollup(
        miner_hotkey=miner_hotkey,
        total_judged=total_judged,
        matches=verdict_counts["matches"],
        partially_matches=verdict_counts["partially_matches"],
        not_applicable=verdict_counts["not_applicable"],
        contradicts=verdict_counts["contradicts"],
        errored=verdict_counts["error"],
        mean_agreement=mean_agreement,
        verdict_counts=verdict_counts,
        final_score=final_score,
        reliable=reliable,
    )
