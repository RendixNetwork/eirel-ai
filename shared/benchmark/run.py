"""Backward-compat re-exports for ``shared.benchmark.run``.

Legacy analyst/builder/media/verifier run helpers are retired. This module
only re-exports what non-test callers still import:

- ``evaluate_multimodal_artifacts`` / ``score_media_generation_payload`` —
  used by :mod:`shared.workflow_runtime.executor`.
"""
from __future__ import annotations

from shared.benchmark._media import (
    evaluate_multimodal_artifacts,
    score_media_generation_payload,
)

__all__ = [
    "evaluate_multimodal_artifacts",
    "score_media_generation_payload",
]
