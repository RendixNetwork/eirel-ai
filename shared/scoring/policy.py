from __future__ import annotations

"""Scoring policy — general_chat 4-dimension weights."""

from dataclasses import dataclass


SCORING_POLICY_VERSION = "general_chat_scoring_v1"


# Per-conversation weights (cost moved to miner-level multiplicative modifier).
# Trace integrity is applied as a multiplicative gate, not a weighted term.
GENERAL_CHAT_CONV_WEIGHTS: dict[str, float] = {
    "quality": 0.80,
    "latency": 0.20,
}

GENERAL_CHAT_COST_MODIFIER_RANGE: tuple[float, float] = (0.80, 1.00)


# Per-miner mode blending: 0.6 × instant_mean + 0.4 × thinking_mean.
MINER_SCORE_MODE_BLEND: dict[str, float] = {
    "instant": 0.60,
    "thinking": 0.40,
}


@dataclass(frozen=True, slots=True)
class ScoringPolicy:
    family_id: str
    benchmark_version: str
    rubric_version: str
    official_scoring_version: str
    judge_mode: str
    quality_weight: float
    latency_weight: float
    cost_modifier_range: tuple[float, float]


_GENERAL_CHAT_POLICY = ScoringPolicy(
    family_id="general_chat",
    benchmark_version="general_chat_v1",
    rubric_version="general_chat_rubric_v1",
    official_scoring_version="general_chat_scoring_v1",
    judge_mode="ensemble",
    quality_weight=GENERAL_CHAT_CONV_WEIGHTS["quality"],
    latency_weight=GENERAL_CHAT_CONV_WEIGHTS["latency"],
    cost_modifier_range=GENERAL_CHAT_COST_MODIFIER_RANGE,
)


SCORING_POLICY_CATALOG: dict[str, ScoringPolicy] = {
    "general_chat": _GENERAL_CHAT_POLICY,
}


def scoring_policy_for(family_id: str) -> ScoringPolicy:
    policy = SCORING_POLICY_CATALOG.get(family_id)
    if policy is None:
        raise KeyError(f"no scoring policy registered for family_id={family_id!r}")
    return policy
