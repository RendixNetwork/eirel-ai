"""Shared oracle request/response shaping.

All three oracle vendors get the same input context and the same
single-call structured-output schema. Vendor differences are limited
to which provider client carries the request (OpenAI/Gemini/Grok).
"""

from __future__ import annotations

import json
from typing import Any

from validation.validator.oracles.base import OracleContext


_SYSTEM_PROMPT = """You are an authoritative ground-truth oracle for an \
agent-evaluation pipeline. Answer the user's question as accurately as \
possible. Be terse and load-bearing — short canonical answer (a name, \
number, year, or single phrase) when one exists; a few sentences when \
the question requires synthesis. Never refuse, hedge, or add filler. \
If a question genuinely has no defensible answer (false premise, not \
yet known, opinion), state that explicitly and concisely.

Return strict JSON: {"answer": "<your answer>"}.
"""


_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def build_oracle_messages(context: OracleContext) -> tuple[str, str]:
    """Build the (system, user) prompt pair for any oracle vendor.

    The user prompt is a JSON envelope so the model sees structured
    fields (prompt / turns / attached_document) clearly delimited.
    """
    user_payload: dict[str, Any] = {"prompt": context.prompt}
    if context.conversation_recent:
        # Multi-turn fixture: include the prior turns verbatim. The
        # oracle should answer the LAST user turn; earlier turns are
        # context.
        user_payload["conversation_recent"] = context.conversation_recent
    if context.attached_document:
        user_payload["attached_document"] = context.attached_document
    if context.category:
        user_payload["category"] = context.category
    user_prompt = json.dumps(user_payload, ensure_ascii=False, sort_keys=True)
    return _SYSTEM_PROMPT, user_prompt


def response_schema() -> dict[str, Any]:
    """JSON-schema for the oracle response. Caller passes this to the
    provider client's ``complete_structured`` ``response_schema`` arg."""
    return dict(_RESPONSE_SCHEMA)


def extract_answer(raw_text: str) -> str:
    """Pull the ``answer`` field out of the oracle's structured JSON.

    Returns the answer string. Raises ``ValueError`` when the response
    is not valid JSON or doesn't carry an ``answer`` field — caller
    converts to ``OracleGrounding(status="error", ...)``.
    """
    try:
        data = json.loads(raw_text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"oracle returned non-JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"oracle JSON is not an object: {type(data).__name__}"
        )
    answer = data.get("answer")
    if not isinstance(answer, str):
        raise ValueError(
            f"oracle JSON missing string 'answer' field: {data!r}"
        )
    return answer.strip()
