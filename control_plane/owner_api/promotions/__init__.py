"""Eval-winner → product-mode promotion helpers.

The actual serving rollout (creating ``ServingRelease`` rows and
materializing ``ServingDeployment`` pods) is owned by
``control_plane.owner_api.serving.serving_manager``. This package adds
the audit + dispatch layer on top: at run close, decide which
deployment per family has the highest eval score, record the choice
into :class:`~shared.common.models.ServingPromotion`, and (optionally)
trigger the existing serving-manager publish flow.
"""
from __future__ import annotations

from control_plane.owner_api.promotions.promote_winners import (
    PromotionDecision,
    PromotionResult,
    promote_winners_for_run,
    select_winners_for_run,
)

__all__ = [
    "PromotionDecision",
    "PromotionResult",
    "promote_winners_for_run",
    "select_winners_for_run",
]
