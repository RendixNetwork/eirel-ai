from __future__ import annotations

from control_plane.owner_api.evaluation.general_chat_scoring import (
    score_general_chat_conversation,
)
from shared.core.evaluation_models import (
    ConversationTrace,
    ConversationTurn,
    ModeBudget,
    TraceEntry,
)


class _NoopJudge:
    async def score_quality(self, *, conversation_history, response, mode):
        return 1.0


def _perfect_budget(mode: str = "instant") -> ModeBudget:
    return ModeBudget(
        mode=mode,
        web_search=True,
        latency_seconds=20.0,
        output_tokens=1024,
        reasoning_tokens=0,
        cost_usd=0.10,
    )


def _trace_containing_url(url: str) -> ConversationTrace:
    return ConversationTrace(
        conversation_id="c",
        entries=[
            TraceEntry(
                tool_name="web_search",
                args={"query": "find reference", "url": url},
                result_digest=url,
                latency_ms=100,
                cost_usd=0.001,
            )
        ],
    )


async def test_gate_zero_when_response_cites_url_not_in_trace():
    trace = _trace_containing_url("https://real.example/a")
    response = (
        "Here is what I found at https://fabricated.example/b — it says the "
        "answer is 42."
    )
    score = await score_general_chat_conversation(
        conversation_history=[ConversationTurn(role="user", content="look it up")],
        response=response,
        trace=trace,
        budget=_perfect_budget(),
        judge_client=_NoopJudge(),
    )
    assert score.trace_gate == 0.0
    assert score.total == 0.0
    assert score.quality == 1.0  # judge still ran; gate just zeroed the total
    assert score.metadata["trace_gate_passed"] is False


async def test_gate_one_when_response_cites_trace_url():
    url = "https://real.example/a"
    trace = _trace_containing_url(url)
    response = f"According to {url}, the answer is 42."
    score = await score_general_chat_conversation(
        conversation_history=[ConversationTurn(role="user", content="look it up")],
        response=response,
        trace=trace,
        budget=_perfect_budget(),
        judge_client=_NoopJudge(),
    )
    assert score.trace_gate == 1.0
    assert score.total > 0.9
    assert score.metadata["trace_gate_passed"] is True


async def test_gate_one_when_response_has_no_citations():
    trace = _trace_containing_url("https://ignored.example/x")
    response = "The answer is 42. (No sources cited.)"
    score = await score_general_chat_conversation(
        conversation_history=[ConversationTurn(role="user", content="what is 6x7?")],
        response=response,
        trace=trace,
        budget=_perfect_budget(),
        judge_client=_NoopJudge(),
    )
    assert score.trace_gate == 1.0
