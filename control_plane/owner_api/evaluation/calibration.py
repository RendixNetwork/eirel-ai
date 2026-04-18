from __future__ import annotations

"""Calibration and promotion gate stubs.

The legacy analyst calibration pipeline has been retired; the general_chat
4D scorer does not require a separate calibration pass. This module keeps
the policy-version constants that ``run_manager`` writes into run metadata
so historical rows stay consistent, but the promotion gates are now a
no-op (all families pass by default).
"""

CALIBRATION_POLICY_VERSION = "general_chat_calibration_v1"
CONSISTENCY_POLICY_VERSION = "general_chat_consistency_v1"
CONSISTENCY_WINDOW = 1
CONSISTENCY_REQUIRED_PASS_COUNT = 1


def supports_family_promotion_gates(family_id: str) -> bool:
    """Return whether a family requires the legacy promotion-gate pipeline.

    In the general_chat-only world there are no promotion gates; the
    4D conversation scorer produces the final score directly. Kept as a
    function so future families (``deep_research``, ``coding``) can opt in
    without touching callers.
    """
    del family_id
    return False
