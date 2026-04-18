"""Benchmark execution helpers (general_chat edition).

The legacy multi-family benchmark harness (analyst/builder/media/verifier)
has been retired. This package now exposes a small surface:

- :func:`score_family_epoch` — single-family entry point (general_chat only).
- :func:`evaluate_multimodal_artifacts` / :func:`score_media_generation_payload`
  — no-op stubs kept for workflow_runtime compatibility.
- :mod:`run` — thin task source plus task-invocation helper.
- :mod:`catalog` — family benchmark catalog lookup.
"""

from shared.benchmark._media import (
    evaluate_multimodal_artifacts,
    score_media_generation_payload,
)
from shared.benchmark._orchestration import (
    compute_miner_score_from_results,
    score_family_epoch,
)

__all__ = [
    "compute_miner_score_from_results",
    "evaluate_multimodal_artifacts",
    "score_family_epoch",
    "score_media_generation_payload",
]
