"""Outcome-only agreement aggregation for the general_chat family.

Per-task agreement maps the verdict to a scalar in [0, 1]:

    matches            → 1.0
    partially_matches  → 0.6
    not_applicable     → 0.7   (open-ended tasks where agreement is N/A)
    contradicts        → 0.0
    error              → not counted in the denominator

Gate-aware bucketing: a task whose multi-metric composite was zeroed
by a gate (tool_attestation / hallucination_knockout / grounded_gate /
safety_gate / cost_attestation / safety_attestation) is reclassified
OUT of its pass bucket and into ``gate_knockout``. The ``mean_agreement``
treats those rows as contributing 0 to the numerator — so the displayed
verdict counts and the agreement mean reflect the same reality as the
canonical multi-metric score.

Error rows are excluded from the mean's denominator to avoid double-
penalizing (a miner whose pod failed and whose agreement couldn't be
assessed). Instead, ``error_rate`` is tracked separately and caps the
final score when a miner fails on too many tasks.

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

    # Gate-aware verdict bucketing. A pass-bucket verdict whose
    # multi-metric ``final_task_score`` was zeroed by a composite gate
    # is reclassified into ``gate_knockout`` so the displayed bucket
    # counts and the agreement mean reflect the same reality as the
    # canonical score.
    pass_verdicts = {"matches", "partially_matches", "not_applicable"}
    verdict_counts: dict[str, int] = {
        "matches": 0, "partially_matches": 0,
        "not_applicable": 0, "contradicts": 0,
        "gate_knockout": 0, "error": 0,
    }

    def _effective(r: "TaskMinerResult") -> str:
        v = r.agreement_verdict if r.agreement_verdict in verdict_counts else "error"
        if v not in pass_verdicts:
            return v
        final = getattr(r, "final_task_score", None)
        if final is not None and float(final) <= 0.0:
            return "gate_knockout"
        return v

    for r in results:
        verdict_counts[_effective(r)] += 1

    completed = (
        verdict_counts["matches"]
        + verdict_counts["partially_matches"]
        + verdict_counts["not_applicable"]
        + verdict_counts["contradicts"]
        + verdict_counts["gate_knockout"]
    )

    # mean_agreement: sum effective agreement_score over non-error rows.
    # Gate-knocked rows contribute 0 (the gate established that they
    # didn't really pass). All other non-error rows use the stored
    # agreement_score (derived from verdict).
    score_sum = 0.0
    for r in results:
        effective = _effective(r)
        if effective == "error":
            continue
        if effective == "gate_knockout":
            continue  # contributes 0; still counted in ``completed``
        stored = float(r.agreement_score or 0.0)
        if stored <= 0.0 and r.agreement_verdict in VERDICT_SCORES:
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
