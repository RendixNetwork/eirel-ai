"""Multi-turn replay tests for ``_invoke_task``.

A fixture with a ``turns`` list is replayed against the miner one user
turn at a time, accumulating the miner's reply as ``assistant`` history
between turns. The final live turn's reply is what becomes the run's
``response`` (and what the judge eventually scores).
"""
from __future__ import annotations

import json

import httpx
import pytest

from shared.benchmark._invocation import _invoke_task
from shared.core.evaluation_models import (
    EvaluationConversationTurn,
    FamilyEvaluationTask,
    MinerBenchmarkTarget,
)


def test_family_evaluation_task_accepts_multi_turn_dict_form():
    """The bundle loader will hand us raw dicts — confirm the schema
    parses both single-turn (``prompt`` only, no ``turns``) and
    multi-turn (``turns`` array, with mixed scripted-or-live turns)."""
    single = FamilyEvaluationTask.model_validate({
        "task_id": "s-1", "family_id": "general_chat",
        "prompt": "hello", "mode": "instant",
    })
    assert single.turns is None

    multi = FamilyEvaluationTask.model_validate({
        "task_id": "m-1", "family_id": "general_chat",
        "prompt": "first user turn",  # legacy mirror for older readers
        "mode": "thinking",
        "turns": [
            {"user": "first user turn"},
            {"user": "second user turn", "assistant": "scripted reply"},
            {"user": "final question"},
        ],
    })
    assert multi.turns is not None
    assert len(multi.turns) == 3
    assert isinstance(multi.turns[0], EvaluationConversationTurn)
    assert multi.turns[1].assistant == "scripted reply"
    assert multi.turns[2].assistant is None  # live final turn


class _Task:
    """Fixture stand-in. ``turns`` is the multi-turn script."""

    def __init__(self, *, turns) -> None:
        self.task_id = "mt-1"
        self.family_id = "general_chat"
        self.prompt = turns[0].get("user") if turns and isinstance(turns[0], dict) else ""
        self.expected_output = {}
        self.inputs = {"mode": "instant", "web_search": False}
        self.metadata = {}
        self.turns = turns


def _miner() -> MinerBenchmarkTarget:
    return MinerBenchmarkTarget(
        hotkey="5X" * 32, endpoint="http://miner.local", stake=0, metadata={},
    )


def _ndjson_body(answer: str) -> bytes:
    chunks = [
        {"event": "delta", "text": answer},
        {
            "event": "done",
            "output": {"answer": answer},
            "citations": [],
            "status": "completed",
            "metadata": {},
        },
    ]
    return ("\n".join(json.dumps(c) for c in chunks) + "\n").encode("utf-8")


class _ScriptedTransport(httpx.AsyncBaseTransport):
    """Records every request body and replies in scripted order."""

    def __init__(self, replies: list[bytes]) -> None:
        self._replies = list(replies)
        self.bodies: list[dict] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        try:
            self.bodies.append(json.loads(request.content or b"{}"))
        except Exception:
            self.bodies.append({})
        body = self._replies.pop(0)
        return httpx.Response(
            200, content=body,
            headers={"content-type": "application/x-ndjson"},
        )


@pytest.fixture
def _patch_client(monkeypatch):
    import shared.benchmark._invocation as mod
    original = mod.httpx.AsyncClient
    holder: dict[str, httpx.AsyncBaseTransport] = {}

    def _install(transport: httpx.AsyncBaseTransport) -> None:
        holder["t"] = transport

    def _patched(*args, **kwargs):
        kwargs["transport"] = holder["t"]
        return original(*args, **kwargs)

    monkeypatch.setattr(mod.httpx, "AsyncClient", _patched)
    monkeypatch.setattr(mod, "_RETRY_BACKOFF_SECONDS", 0.0)
    return _install


async def test_multi_turn_live_replay_accumulates_history(_patch_client):
    """Three live user turns → three miner calls; each call sees the
    history accumulated from previous turns. Final answer is from the
    last turn."""
    transport = _ScriptedTransport([
        _ndjson_body("turn-1 answer"),
        _ndjson_body("turn-2 answer"),
        _ndjson_body("turn-3 answer"),
    ])
    _patch_client(transport)

    task = _Task(turns=[
        {"user": "first question"},
        {"user": "follow up"},
        {"user": "another follow up"},
    ])

    run = await _invoke_task(miner=_miner(), task=task, timeout_seconds=5.0)

    assert run.status == "completed"
    assert run.response.get("output", {}).get("answer") == "turn-3 answer"
    assert run.metadata.get("turn_count") == 3

    # Turn 1: empty history.
    assert transport.bodies[0]["history"] == []
    assert transport.bodies[0]["prompt"] == "first question"

    # Turn 2: history holds turn-1 user + turn-1 assistant reply.
    assert transport.bodies[1]["history"] == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "turn-1 answer"},
    ]
    assert transport.bodies[1]["prompt"] == "follow up"

    # Turn 3: full accumulated history through turn 2.
    assert transport.bodies[2]["history"][-2:] == [
        {"role": "user", "content": "follow up"},
        {"role": "assistant", "content": "turn-2 answer"},
    ]


async def test_scripted_intermediate_turn_skips_miner(_patch_client):
    """Scripted ``assistant`` on a non-final turn injects the canned
    exchange into history without calling the miner."""
    transport = _ScriptedTransport([
        # Only one miner call expected — for the final live turn.
        _ndjson_body("final answer"),
    ])
    _patch_client(transport)

    task = _Task(turns=[
        {"user": "earlier question", "assistant": "earlier scripted reply"},
        {"user": "final question"},
    ])

    run = await _invoke_task(miner=_miner(), task=task, timeout_seconds=5.0)

    assert run.status == "completed"
    assert len(transport.bodies) == 1
    # The single miner call sees the scripted exchange in history.
    assert transport.bodies[0]["prompt"] == "final question"
    assert transport.bodies[0]["history"] == [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier scripted reply"},
    ]
    # Per-turn breakdown reflects scripted turn at index 0.
    breakdown = run.metadata.get("turns") or []
    assert breakdown[0]["scripted"] is True
    assert breakdown[1]["scripted"] is False


async def test_per_turn_max_latency_recorded_for_sla_gate(_patch_client):
    """``max_turn_latency_seconds`` reflects the slowest turn (so the
    validator's per-turn budget gate fires on any over-budget turn,
    not just the average)."""
    transport = _ScriptedTransport([
        _ndjson_body("a"),
        _ndjson_body("b"),
    ])
    _patch_client(transport)

    task = _Task(turns=[{"user": "one"}, {"user": "two"}])

    run = await _invoke_task(miner=_miner(), task=task, timeout_seconds=5.0)

    assert run.metadata.get("turn_count") == 2
    # Both should be tiny; the meaningful invariant is that the sum is
    # at least the max — sanity-checking the relationship rather than
    # specific timing.
    total = run.metadata.get("latency_seconds") or 0.0
    per_turn_max = run.metadata.get("max_turn_latency_seconds") or 0.0
    assert per_turn_max <= total
    assert per_turn_max > 0.0


async def test_failure_at_intermediate_turn_aborts_replay(_patch_client):
    """If turn N fails, we stop and surface the error — turns N+1..
    are never sent."""
    transport = _ScriptedTransport([
        _ndjson_body("turn-1 ok"),
        # Turn 2 will get a 500 (we'll script that via custom transport).
    ])

    class _Transport(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.bodies: list[dict] = []
            self.calls = 0

        async def handle_async_request(self, request):
            self.calls += 1
            try:
                self.bodies.append(json.loads(request.content or b"{}"))
            except Exception:
                self.bodies.append({})
            if self.calls == 1:
                return httpx.Response(
                    200, content=_ndjson_body("turn-1 ok"),
                    headers={"content-type": "application/x-ndjson"},
                )
            return httpx.Response(500, text="boom")

    t = _Transport()
    _patch_client(t)

    task = _Task(turns=[
        {"user": "first"},
        {"user": "second"},
        {"user": "third — must never be sent"},
    ])

    run = await _invoke_task(miner=_miner(), task=task, timeout_seconds=5.0)

    assert run.status == "failed"
    # Only turn 1 (success) + turn 2 (500 → retry → 500). Turn 3 never sent.
    # Confirm by absence of turn-3 prompt in any body.
    sent_prompts = [b.get("prompt") for b in t.bodies]
    assert "third — must never be sent" not in sent_prompts
    assert run.metadata.get("failed_at_turn") == 1
