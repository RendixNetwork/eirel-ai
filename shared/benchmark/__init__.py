"""Benchmark execution helpers (general_chat edition).

Single-family surface:

- :func:`score_family_epoch` — single-family entry point (general_chat only).
- :mod:`run` — thin task source plus task-invocation helper.
- :mod:`catalog` — family benchmark catalog lookup.
"""

from shared.benchmark._orchestration import (
    compute_miner_score_from_results,
    score_family_epoch,
)

__all__ = [
    "compute_miner_score_from_results",
    "score_family_epoch",
]
