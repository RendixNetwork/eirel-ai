from __future__ import annotations

"""Multimodal artifact scoring stubs.

The legacy media family is retired. These helpers remain as no-op shims so
``shared/workflow_runtime/executor.py`` and other callers import cleanly.
"""

from typing import Any


async def evaluate_multimodal_artifacts(
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    del args, kwargs
    return {
        "evaluated": False,
        "reason": "media family retired — multimodal scoring not implemented",
    }


def score_media_generation_payload(
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    del args, kwargs
    return {
        "score": 0.0,
        "reason": "media family retired — scoring not implemented",
    }


# Internal alias kept for the legacy ``_invocation`` caller.
async def _evaluate_multimodal_artifacts(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return await evaluate_multimodal_artifacts(*args, **kwargs)
