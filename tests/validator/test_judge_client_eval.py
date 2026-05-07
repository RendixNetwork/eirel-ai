"""JudgeServiceClient ``judge_eval`` + ``judge_eval_composite`` adaptation.

Covers:
  * Request serialization to ``/v1/judge/eval``
  * Optional prompt vs. turns dispatch
  * Composite endpoint shape
"""
from __future__ import annotations

import json

import httpx

from shared.core.judge_client import JudgeServiceClient


def _make_client(handler) -> JudgeServiceClient:
    transport = httpx.MockTransport(handler)
    return JudgeServiceClient(base_url="http://mock", transport=transport)


def test_judge_eval_posts_full_payload_for_single_turn():
    captured: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "outcome": "correct",
                "failure_mode": None,
                "guidance": "",
            },
        )

    client = _make_client(_handler)
    result = client.judge_eval(
        bundle={
            "question": "What is 2+2?",
            "answers": ["The answer is 4."],
        },
        expected_answer="4",
        must_not_claim=["five", "six"],
        oracle_source="three_oracle",
    )
    assert captured["url"] == "http://mock/v1/judge/eval"
    body = captured["body"]
    assert body["bundle"]["question"] == "What is 2+2?"
    assert body["bundle"]["answers"] == ["The answer is 4."]
    assert body["expected_answer"] == "4"
    assert body["must_not_claim"] == ["five", "six"]
    assert body["oracle_source"] == "three_oracle"
    assert result["outcome"] == "correct"


def test_judge_eval_passes_turns_for_multi_turn_item():
    captured: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={"outcome": "correct", "guidance": ""},
        )

    client = _make_client(_handler)
    turns = [
        {"role": "user", "content": "I work in Python."},
        {"role": "assistant", "content": "Got it."},
        {"role": "user", "content": "What language do I work in?"},
    ]
    client.judge_eval(
        bundle={
            "question": "What language do I work in?",
            "conversation_recent": turns,
            "answers": ["You work in Python."],
        },
        expected_answer="Python",
        oracle_source="deterministic",
    )
    body = captured["body"]
    assert body["bundle"]["conversation_recent"] == turns
    assert body["bundle"]["answers"] == ["You work in Python."]
    assert body["oracle_source"] == "deterministic"


def test_judge_eval_composite_posts_to_composite_endpoint():
    captured: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "composite": 1.0,
                "outcome_score": 1.0,
                "tool_attestation_factor": 1.0,
                "efficiency_factor": 1.0,
                "hallucination_knockout": 1.0,
                "cost_attestation_knockout": 1.0,
                "knockout_reason": None,
            },
        )

    client = _make_client(_handler)
    result = client.judge_eval_composite(
        outcome="correct",
        candidate_response="ok",
        must_not_claim=[],
        required_tool="web_search",
        ledger_tools=["web_search"],
        latency_ms=200,
        cost_usd=0.001,
        latency_budget_ms=5000,
        cost_budget_usd=0.01,
        cost_floor_usd=0.00005,
    )
    assert captured["url"] == "http://mock/v1/judge/eval/composite"
    body = captured["body"]
    assert body["outcome"] == "correct"
    assert body["required_tool"] == "web_search"
    assert body["ledger_tools"] == ["web_search"]
    assert body["latency_budget_ms"] == 5000
    assert body["cost_budget_usd"] == 0.01
    assert body["cost_floor_usd"] == 0.00005
    assert result["composite"] == 1.0
