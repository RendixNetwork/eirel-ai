from __future__ import annotations

"""Minimal judge-excerpt helper.

Legacy multi-family judge excerpts (research, reasoning, builder, media)
are retired. For general_chat the judge excerpt is simply the final
response text the miner produced, truncated to keep the request small.
"""

import json
from typing import Any

from shared.core.evaluation_models import BenchmarkTaskRun


_MAX_EXCERPT_CHARS = 8000


def build_judge_excerpt(*, family_id: str, run: BenchmarkTaskRun) -> str:
    del family_id  # general_chat is the only family
    response = run.response or {}
    if not isinstance(response, dict):
        return str(response)[:_MAX_EXCERPT_CHARS]

    for key in ("content", "answer", "final_answer", "text", "output"):
        value = response.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:_MAX_EXCERPT_CHARS]
        if isinstance(value, dict) and value:
            return json.dumps(value, sort_keys=True)[:_MAX_EXCERPT_CHARS]
    try:
        return json.dumps(response, sort_keys=True)[:_MAX_EXCERPT_CHARS]
    except (TypeError, ValueError):
        return str(response)[:_MAX_EXCERPT_CHARS]
