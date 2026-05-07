"""JSON-repair retry wrapper tests.

The wrapper sits in front of the provider client (OpenAI / Gemini /
Chutes). On the 90-98% parse-rate band, a 2-retry repair loop turns
flaky models into usable ones. Below 90%, the model isn't ready and
should be swapped; the wrapper is for the recoverable zone.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from validation.validator.providers.json_repair import (
    JsonRepairClient,
    with_json_repair,
)
from validation.validator.providers.types import (
    ProviderError,
    ProviderResponse,
)


pytestmark = pytest.mark.asyncio


class _ScriptedClient:
    """Provider-client stub that returns scripted responses in order."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete_structured(
        self,
        *,
        system: str,
        user: str,
        response_schema: dict[str, Any],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        schema_name: str = "response",
    ) -> ProviderResponse:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "schema": response_schema,
                "schema_name": schema_name,
            }
        )
        if not self._responses:
            raise RuntimeError("scripted client exhausted")
        text = self._responses.pop(0)
        return ProviderResponse(
            text=text, latency_ms=10, usage_usd=0.0, finish_reason="stop",
        )


# -- Happy path ----------------------------------------------------------


async def test_first_call_succeeds_no_retry():
    """Valid JSON on the first call → no retries, no repair prompt."""
    valid = json.dumps({"answer": "ok"})
    inner = _ScriptedClient([valid])
    wrapper = with_json_repair(inner, max_retries=2)
    resp = await wrapper.complete_structured(
        system="s", user="u", response_schema={"type": "object"},
    )
    assert json.loads(resp.text) == {"answer": "ok"}
    assert len(inner.calls) == 1


# -- Recovery zone (90-98%) ----------------------------------------------


async def test_first_call_malformed_second_succeeds():
    """First response is invalid JSON; wrapper retries once and
    succeeds. Caller sees the second response."""
    inner = _ScriptedClient(["not json", json.dumps({"answer": "ok"})])
    wrapper = with_json_repair(inner, max_retries=2)
    resp = await wrapper.complete_structured(
        system="s", user="please answer", response_schema={"type": "object"},
    )
    assert json.loads(resp.text) == {"answer": "ok"}
    assert len(inner.calls) == 2
    # Repair instruction is appended to the user prompt on retry.
    repair_user = inner.calls[1]["user"]
    assert "please answer" in repair_user  # original ask preserved
    assert "malformed JSON" in repair_user
    assert "STRICT JSON" in repair_user


async def test_two_retries_consumed_then_succeeds():
    inner = _ScriptedClient(["malformed", "still bad", json.dumps({"a": 1})])
    wrapper = with_json_repair(inner, max_retries=2)
    resp = await wrapper.complete_structured(
        system="s", user="u", response_schema={"type": "object"},
    )
    assert json.loads(resp.text) == {"a": 1}
    assert len(inner.calls) == 3


# -- Below recovery zone -------------------------------------------------


async def test_all_retries_fail_raises_provider_error():
    """3 attempts (initial + 2 retries) all malformed → ProviderError.
    The caller (judge / reconciler) treats this as a failed call and
    surfaces 'disputed' or equivalent fallback."""
    inner = _ScriptedClient(["bad", "still bad", "no really bad"])
    wrapper = with_json_repair(inner, max_retries=2)
    with pytest.raises(ProviderError, match="JSON parse failed after 3"):
        await wrapper.complete_structured(
            system="s", user="u", response_schema={"type": "object"},
        )
    assert len(inner.calls) == 3


# -- Edge cases ----------------------------------------------------------


async def test_zero_retries_means_one_attempt_total():
    """max_retries=0 → no retries, original call only. Useful when
    the caller wants to know about parse failures immediately
    without paying for repairs."""
    inner = _ScriptedClient(["malformed"])
    wrapper = with_json_repair(inner, max_retries=0)
    with pytest.raises(ProviderError, match="JSON parse failed after 1"):
        await wrapper.complete_structured(
            system="s", user="u", response_schema={"type": "object"},
        )
    assert len(inner.calls) == 1


async def test_repair_prompt_includes_parse_error_message():
    """The repair instruction includes the parse error so the model
    sees what failed. Caps the error text at 200 chars to keep the
    prompt small."""
    inner = _ScriptedClient(["malformed", json.dumps({"x": 1})])
    wrapper = with_json_repair(inner, max_retries=2)
    await wrapper.complete_structured(
        system="s", user="u", response_schema={"type": "object"},
    )
    repair_user = inner.calls[1]["user"]
    # Some recognizable token from the JSONDecodeError message.
    assert any(
        marker in repair_user
        for marker in ("Expecting", "Extra data", "char", "line")
    )


async def test_kwargs_pass_through_to_inner_client():
    """temperature / max_tokens / schema_name forwarded as-is."""
    inner = _ScriptedClient([json.dumps({"a": 1})])
    wrapper = with_json_repair(inner, max_retries=2)
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
    await wrapper.complete_structured(
        system="sys",
        user="usr",
        response_schema=schema,
        temperature=0.7,
        max_tokens=512,
        schema_name="my_schema",
    )
    call = inner.calls[0]
    assert call["system"] == "sys"
    assert call["user"] == "usr"
    assert call["schema"] == schema
    assert call["schema_name"] == "my_schema"


async def test_empty_string_treated_as_malformed():
    """Empty response is also malformed JSON — triggers repair."""
    inner = _ScriptedClient(["", json.dumps({"x": 1})])
    wrapper = with_json_repair(inner, max_retries=2)
    resp = await wrapper.complete_structured(
        system="s", user="u", response_schema={"type": "object"},
    )
    assert json.loads(resp.text) == {"x": 1}
    assert len(inner.calls) == 2
