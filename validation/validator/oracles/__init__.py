"""Validator-side oracle layer.

Three frontier oracles (OpenAI / Gemini / Grok) called in parallel at
task-claim time for items tagged ``oracle_source: three_oracle``. Each
oracle gets the FULL input the miner sees — full prompt, full attached
document, full conversation history. No summarization for oracle calls.

Failures (timeout, 5xx exhausted, vendor blocking, malformed response)
are surfaced as ``OracleGrounding(status="error", ...)`` so the
reconciler can degrade gracefully to 2-oracle (or 0-oracle disputed).

The reconciler that consumes these groundings lives one level up at
``validation/validator/reconciler.py``.
"""

from __future__ import annotations

from validation.validator.oracles.base import (
    OracleClient,
    OracleContext,
    OracleGrounding,
    OracleStatus,
)
from validation.validator.oracles.fanout import OracleFanout
from validation.validator.oracles.gemini_oracle import GeminiOracle
from validation.validator.oracles.grok_oracle import GrokOracle
from validation.validator.oracles.openai_oracle import OpenAIOracle

__all__ = [
    "GeminiOracle",
    "GrokOracle",
    "OpenAIOracle",
    "OracleClient",
    "OracleContext",
    "OracleFanout",
    "OracleGrounding",
    "OracleStatus",
]
