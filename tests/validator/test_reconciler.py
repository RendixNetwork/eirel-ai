"""Reconciler tests.

Exercises:
  * 3 successful groundings → reconciler call → ReconciledOracle with
    expected_claims = consensus + majority.
  * 2 successful (1 down) → reconciler still runs.
  * <2 successful → skip reconciler call, return disputed.
  * Reconciler error → disputed with template floor preserved.
  * Reconciler returns malformed JSON → disputed.
  * Additive must_not_claim: floor + extras, deduplicated.
  * from_deterministic constructor for non-three_oracle items.
  * Telemetry fields populated (vendor_status, latency, cost).
"""

from __future__ import annotations

import json

import httpx
import pytest

from validation.validator.eval_config import ProviderConfig
from validation.validator.oracles.base import OracleGrounding
from validation.validator.providers.openai_compatible import (
    OpenAICompatibleClient,
)
from validation.validator.reconciler import (
    ReconciledOracle,
    Reconciler,
)


pytestmark = pytest.mark.asyncio


def _cfg() -> ProviderConfig:
    return ProviderConfig(
        base_url="http://chutes.test",
        api_key="tok",
        model="zai-org/GLM-5.1-TEE",
        timeout_seconds=5.0,
        max_tokens=512,
    )


def _ok_response(parsed: dict) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": json.dumps(parsed)}}],
            "usage": {"total_cost_usd": 0.0007},
        },
    )


def _three_groundings() -> list[OracleGrounding]:
    return [
        OracleGrounding(vendor="openai", status="ok", raw_text="Paris is the capital."),
        OracleGrounding(vendor="gemini", status="ok", raw_text="The capital of France is Paris."),
        OracleGrounding(vendor="grok", status="ok", raw_text="Paris."),
    ]


async def test_reconciler_consensus_path():
    captured: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content.decode("utf-8"))
        return _ok_response({
            "consensus_claims": ["Paris is the capital of France"],
            "majority_claims": [],
            "minority_claims": [],
            "must_not_claim_extras": [],
            "oracle_status": "consensus",
            "disagreement_note": None,
        })

    client = OpenAICompatibleClient(_cfg(), transport=httpx.MockTransport(handler))
    rec = Reconciler(client=client)
    result = await rec.reconcile(
        prompt="What is the capital of France?",
        groundings=_three_groundings(),
    )
    await rec.aclose()

    assert result.oracle_status == "consensus"
    assert result.expected_claims == ["Paris is the capital of France"]
    assert result.must_not_claim == []
    assert result.vendor_status == {"openai": "ok", "gemini": "ok", "grok": "ok"}
    assert result.reconciler_cost_usd == pytest.approx(0.0007)

    # User prompt sent to reconciler must include both the user
    # question AND each oracle answer.
    user_msg = next(
        m for m in captured["body"]["messages"] if m["role"] == "user"
    )
    payload = json.loads(user_msg["content"])
    assert payload["user_prompt"] == "What is the capital of France?"
    assert {a["vendor"] for a in payload["oracle_answers"]} == {"openai", "gemini", "grok"}


async def test_reconciler_majority_path():
    """2-of-3 oracles agree on a fact; reconciler emits majority_claim
    + correctly maps it into expected_claims."""

    def handler(req: httpx.Request) -> httpx.Response:
        return _ok_response({
            "consensus_claims": ["Eiffel Tower is in Paris"],
            "majority_claims": [
                {"claim": "Built in 1889", "supporting_oracles": ["openai", "gemini"]},
            ],
            "minority_claims": [
                {"claim": "Tower is 312m tall", "supporting_oracle": "grok"},
            ],
            "must_not_claim_extras": [],
            "oracle_status": "majority",
            "disagreement_note": "grok mentioned tower height, others didn't",
        })

    client = OpenAICompatibleClient(_cfg(), transport=httpx.MockTransport(handler))
    rec = Reconciler(client=client)
    result = await rec.reconcile(
        prompt="Tell me about the Eiffel Tower",
        groundings=_three_groundings(),
    )
    await rec.aclose()

    assert result.oracle_status == "majority"
    # consensus + majority → expected_claims (minority dropped from scoring)
    assert result.expected_claims == [
        "Eiffel Tower is in Paris",
        "Built in 1889",
    ]
    # minority claims preserved for telemetry
    assert len(result.minority_claims) == 1
    assert result.minority_claims[0]["claim"] == "Tower is 312m tall"
    assert result.minority_claims[0]["supporting_oracle"] == "grok"
    assert result.disagreement_note == "grok mentioned tower height, others didn't"


async def test_reconciler_disputed_path():
    def handler(req: httpx.Request) -> httpx.Response:
        return _ok_response({
            "consensus_claims": [],
            "majority_claims": [],
            "minority_claims": [
                {"claim": "Yes, X is true", "supporting_oracle": "openai"},
                {"claim": "Actually X is false", "supporting_oracle": "gemini"},
            ],
            "must_not_claim_extras": [],
            "oracle_status": "disputed",
            "disagreement_note": "openai and gemini disagree on X",
        })

    client = OpenAICompatibleClient(_cfg(), transport=httpx.MockTransport(handler))
    rec = Reconciler(client=client)
    result = await rec.reconcile(prompt="Is X true?", groundings=_three_groundings())
    await rec.aclose()

    assert result.oracle_status == "disputed"
    assert result.expected_claims == []
    assert result.disagreement_note == "openai and gemini disagree on X"


async def test_must_not_claim_is_additive():
    """Reconciler extras + template floor are unioned, dedup preserves
    floor order so the floor ends up first in the final list."""

    def handler(req: httpx.Request) -> httpx.Response:
        return _ok_response({
            "consensus_claims": ["X is the answer"],
            "majority_claims": [],
            "minority_claims": [],
            "must_not_claim_extras": [
                "claim about Y is wrong",
                "do not assert Z",
            ],
            "oracle_status": "consensus",
            "disagreement_note": None,
        })

    client = OpenAICompatibleClient(_cfg(), transport=httpx.MockTransport(handler))
    rec = Reconciler(client=client)
    result = await rec.reconcile(
        prompt="What's the answer?",
        groundings=_three_groundings(),
        must_not_claim_floor=["never claim A", "do not assert Z"],  # 'do not assert Z' overlaps
    )
    await rec.aclose()

    # Floor entries first (preserving order), then non-overlapping extras.
    assert result.must_not_claim == [
        "never claim A",
        "do not assert Z",
        "claim about Y is wrong",
    ]


async def test_reconciler_skipped_when_fewer_than_two_oracles_succeeded():
    """1 ok + 2 errors → skip reconciler call, return disputed with
    template floor."""
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _ok_response({})  # would succeed if called

    client = OpenAICompatibleClient(_cfg(), transport=httpx.MockTransport(handler))
    rec = Reconciler(client=client)
    result = await rec.reconcile(
        prompt="...",
        groundings=[
            OracleGrounding(vendor="openai", status="ok", raw_text="Paris"),
            OracleGrounding(vendor="gemini", status="error", error_msg="boom"),
            OracleGrounding(vendor="grok", status="error", error_msg="boom"),
        ],
        must_not_claim_floor=["never X"],
    )
    await rec.aclose()

    assert calls["n"] == 0  # reconciler call NOT made
    assert result.oracle_status == "disputed"
    assert result.expected_claims == []
    assert result.must_not_claim == ["never X"]
    assert result.vendor_status == {"openai": "ok", "gemini": "error", "grok": "error"}
    assert "1/3 oracles" in (result.disagreement_note or "")


async def test_reconciler_all_oracles_failed_returns_disputed():
    client = OpenAICompatibleClient(_cfg(), transport=httpx.MockTransport(lambda r: _ok_response({})))
    rec = Reconciler(client=client)
    result = await rec.reconcile(
        prompt="...",
        groundings=[
            OracleGrounding(vendor="openai", status="error", error_msg="x"),
            OracleGrounding(vendor="gemini", status="error", error_msg="x"),
            OracleGrounding(vendor="grok", status="error", error_msg="x"),
        ],
    )
    await rec.aclose()

    assert result.oracle_status == "disputed"
    assert result.expected_claims == []


async def test_reconciler_error_falls_back_to_disputed_with_floor():
    """Provider 5xx after retries → reconciler returns disputed with
    expected_claims=[] and must_not_claim=template_floor only."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "always 503"})

    client = OpenAICompatibleClient(
        _cfg(), transport=httpx.MockTransport(handler),
        max_retries=1, backoff_base_seconds=0.001,
    )
    rec = Reconciler(client=client)
    result = await rec.reconcile(
        prompt="...",
        groundings=_three_groundings(),
        must_not_claim_floor=["floor item"],
    )
    await rec.aclose()

    assert result.oracle_status == "disputed"
    assert result.expected_claims == []
    assert result.must_not_claim == ["floor item"]
    assert "reconciler_error" in (result.disagreement_note or "")


async def test_reconciler_malformed_json_falls_back_to_disputed():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "not json"}}]},
        )

    client = OpenAICompatibleClient(_cfg(), transport=httpx.MockTransport(handler))
    rec = Reconciler(client=client)
    result = await rec.reconcile(
        prompt="...",
        groundings=_three_groundings(),
        must_not_claim_floor=["floor"],
    )
    await rec.aclose()

    assert result.oracle_status == "disputed"
    assert result.must_not_claim == ["floor"]
    assert "malformed_json" in (result.disagreement_note or "")


async def test_reconciler_invalid_status_downgrades_to_disputed():
    def handler(req: httpx.Request) -> httpx.Response:
        return _ok_response({
            "consensus_claims": ["X"],
            "majority_claims": [],
            "minority_claims": [],
            "must_not_claim_extras": [],
            "oracle_status": "totally-bogus-status",
            "disagreement_note": None,
        })

    client = OpenAICompatibleClient(_cfg(), transport=httpx.MockTransport(handler))
    rec = Reconciler(client=client)
    result = await rec.reconcile(prompt="...", groundings=_three_groundings())
    await rec.aclose()

    assert result.oracle_status == "disputed"


async def test_from_deterministic_factory():
    """Pool kinds with built-in graders skip the reconciler entirely."""
    rec = ReconciledOracle.from_deterministic(
        answer="Paris",
        must_not_claim_floor=["never claim London"],
    )
    assert rec.oracle_status == "deterministic"
    assert rec.expected_claims == ["Paris"]
    assert rec.must_not_claim == ["never claim London"]
    assert rec.consensus_claims == ["Paris"]
    assert rec.vendor_status == {}


async def test_from_deterministic_with_empty_answer():
    rec = ReconciledOracle.from_deterministic(answer="")
    assert rec.expected_claims == []
    assert rec.consensus_claims == []
    assert rec.oracle_status == "deterministic"


async def test_telemetry_populated_on_success():
    def handler(req: httpx.Request) -> httpx.Response:
        return _ok_response({
            "consensus_claims": ["X"],
            "majority_claims": [],
            "minority_claims": [],
            "must_not_claim_extras": [],
            "oracle_status": "consensus",
            "disagreement_note": None,
        })

    client = OpenAICompatibleClient(_cfg(), transport=httpx.MockTransport(handler))
    rec = Reconciler(client=client)
    result = await rec.reconcile(prompt="...", groundings=_three_groundings())
    await rec.aclose()

    assert result.reconciler_latency_ms >= 0
    assert result.reconciler_cost_usd == pytest.approx(0.0007)
    assert result.vendor_status == {"openai": "ok", "gemini": "ok", "grok": "ok"}
