"""JudgeServiceClient adaptation tests for the outcome-only agreement judge.

Covers:
  * Request serialization to ``/v1/judge/agreement``
  * Response deserialization into shared ``AgreementJudgeOutput``
  * Swap flag pass-through
  * Retry on transient 502/503/504
  * Invalid verdict from server coerced to ``"error"``
"""

from __future__ import annotations

import json

import httpx
import pytest

from shared.core.evaluation_models import VERDICT_SCORES
from shared.core.judge_client import JudgeServiceClient


def _eiretes_response(
    *,
    verdict: str = "matches",
    rationale: str = "claims align",
    swap_applied: bool = False,
) -> dict:
    """Shape of eiretes' AgreementJudgeResult JSON body."""
    return {
        "model": "mock-llm",
        "rubric_name": "agreement_general_chat_v1:test",
        "verdict": verdict,
        "agreement_score": VERDICT_SCORES.get(verdict, 0.0),
        "rationale": rationale,
        "latency_seconds": 0.5,
        "swap_applied": swap_applied,
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        "metadata": {"family_id": "general_chat", "rubric_version": "test"},
    }


def _make_client(handler) -> JudgeServiceClient:
    transport = httpx.MockTransport(handler)
    return JudgeServiceClient(base_url="http://mock", transport=transport)


def test_client_posts_to_agreement_endpoint_and_adapts_response():
    captured: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_eiretes_response(verdict="matches"))

    client = _make_client(_handler)
    result = client.judge_agreement(
        family_id="general_chat",
        prompt="What is X?",
        response_a="miner answer",
        response_b="baseline answer",
        task_mode="instant",
        task_category="factual_web",
        swap=False,
    )
    assert captured["url"].endswith("/v1/judge/agreement")
    body = captured["body"]
    assert body["family_id"] == "general_chat"
    assert body["response_a"] == "miner answer"
    assert body["response_b"] == "baseline answer"
    assert body["task_mode"] == "instant"
    assert body["task_category"] == "factual_web"
    assert body["swap"] is False

    assert result.verdict == "matches"
    assert result.agreement_score == 1.0
    assert result.swap_applied is False
    client.close()


def test_client_passes_swap_flag_through():
    captured: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_eiretes_response(
            verdict="contradicts", swap_applied=True,
        ))

    client = _make_client(_handler)
    result = client.judge_agreement(
        family_id="general_chat",
        prompt="p", response_a="a", response_b="b", swap=True,
    )
    assert captured["body"]["swap"] is True
    assert result.verdict == "contradicts"
    assert result.agreement_score == 0.0
    assert result.swap_applied is True
    client.close()


def test_client_maps_all_verdicts_to_scores():
    for verdict, expected_score in [
        ("matches", 1.0),
        ("partially_matches", 0.6),
        ("not_applicable", 0.7),
        ("contradicts", 0.0),
    ]:
        def _handler(request, _v=verdict):
            return httpx.Response(200, json=_eiretes_response(verdict=_v))
        client = _make_client(_handler)
        result = client.judge_agreement(
            family_id="general_chat",
            prompt="p", response_a="a", response_b="b",
        )
        assert result.verdict == verdict
        assert result.agreement_score == expected_score
        client.close()


def test_client_coerces_unknown_verdict_to_error():
    def _handler(request):
        return httpx.Response(
            200,
            json={"verdict": "banana", "agreement_score": 0.0, "rationale": ""},
        )
    client = _make_client(_handler)
    result = client.judge_agreement(
        family_id="general_chat",
        prompt="p", response_a="a", response_b="b",
    )
    assert result.verdict == "error"
    assert result.agreement_score == 0.0
    client.close()


def test_client_surfaces_http_errors():
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"detail": "bad family"})

    client = _make_client(_handler)
    with pytest.raises(httpx.HTTPStatusError):
        client.judge_agreement(
            family_id="analyst",
            prompt="p", response_a="a", response_b="b",
        )
    client.close()


def test_client_retries_on_transient_status():
    call_count = {"n": 0}

    def _handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        if call_count["n"] < 2:
            return httpx.Response(502, json={"detail": "upstream down"})
        return httpx.Response(200, json=_eiretes_response())

    client = _make_client(_handler)
    result = client.judge_agreement(
        family_id="general_chat",
        prompt="p", response_a="a", response_b="b",
    )
    assert call_count["n"] == 2
    assert result.verdict == "matches"
    client.close()
