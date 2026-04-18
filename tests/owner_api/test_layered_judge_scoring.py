from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from shared.common.models import MinerEvaluationTask
from shared.core.evaluation_models import ConversationTrace
from shared.core.honeytokens import generate_honeytoken_set


class _FakeScoringManager:
    def __init__(self) -> None:
        self.charges: list[tuple[str, float, str]] = []
        self.return_value = True

    def charge_trace_gate_penalty(
        self, deployment_id: str, *, amount_usd: float, reason: str = "trace_gate_fail"
    ) -> bool:
        self.charges.append((deployment_id, amount_usd, reason))
        return self.return_value


def _settings(penalty_usd: float = 0.50) -> SimpleNamespace:
    return SimpleNamespace(
        trace_gate_penalty_usd=penalty_usd,
        provider_proxy_url="http://provider-proxy.test",
        provider_proxy_token="internal-token",
    )


def _owner(penalty_usd: float = 0.50) -> SimpleNamespace:
    return SimpleNamespace(
        settings=_settings(penalty_usd),
        scoring=_FakeScoringManager(),
    )


def _task_manager_with_stubs(
    *,
    owner: SimpleNamespace,
    active_honeytokens: list[str] | None = None,
    deployment_id: str | None = "dep-123",
):
    from control_plane.owner_api.evaluation.evaluation_task_manager import (
        EvaluationTaskManager,
    )

    manager = EvaluationTaskManager(owner)
    # Stub the methods that would otherwise hit the DB. We're not testing
    # the DB lookups here — just the scoring layering.
    manager._active_honeytokens_for_run = lambda session, run_id: list(
        active_honeytokens or []
    )
    manager._deployment_id_for_miner = lambda session, miner_hotkey, family_id: deployment_id
    return manager


def _fake_task(task_id: str = "t-1", run_id: str = "run-1") -> MinerEvaluationTask:
    return MinerEvaluationTask(
        run_id=run_id,
        family_id="general_chat",
        miner_hotkey="5HKtest" + "A" * 42,
        task_id=task_id,
        task_index=0,
        status="evaluated",
    )


def _trace_with_url(url: str) -> dict[str, Any]:
    return {
        "conversation_id": "conv-1",
        "entries": [
            {
                "tool_name": "web_search",
                "args": {"url": url, "query": "look up x"},
                "result_digest": url,
                "result_body_excerpt": "quarterly earnings rose on mortgage volume",
                "latency_ms": 200,
                "cost_usd": 0.001,
                "metadata": {"url": url},
            }
        ],
    }


def test_layered_score_falls_back_when_response_text_missing():
    owner = _owner()
    manager = _task_manager_with_stubs(owner=owner)
    task = _fake_task()
    task_def = {"prompt": "look up x", "mode": "instant"}
    # Miner response without response_text / trace — legacy path.
    miner_response = {"response": {"some_other_field": "hello"}}

    score, payload = manager._layered_general_chat_score(
        session=None,
        task=task,
        task_def=task_def,
        raw_miner_response=miner_response,
        quality_score=0.8,
    )
    assert score == 0.8
    assert payload is None
    # No penalty charged on legacy path.
    assert owner.scoring.charges == []


def test_layered_score_falls_back_when_trace_missing():
    owner = _owner()
    manager = _task_manager_with_stubs(owner=owner)
    task = _fake_task()
    miner_response = {
        "response": {
            "response_text": "See https://example.com/a",
            # no trace
        }
    }
    score, payload = manager._layered_general_chat_score(
        session=None,
        task=task,
        task_def={"prompt": "p", "mode": "instant"},
        raw_miner_response=miner_response,
        quality_score=0.9,
    )
    assert score == 0.9
    assert payload is None


def test_layered_score_full_pipeline_happy_path():
    owner = _owner()
    manager = _task_manager_with_stubs(owner=owner)
    task = _fake_task()
    # Miner supplies both fields. URL is in the trace, body excerpt
    # shares content words with the surrounding sentence — gate passes.
    url = "https://example.com/earnings"
    miner_response = {
        "response": {
            "response_text": (
                f"According to {url}, quarterly earnings rose on mortgage "
                f"volume. Full details in the report."
            ),
            "trace": _trace_with_url(url),
        }
    }
    score, payload = manager._layered_general_chat_score(
        session=None,
        task=task,
        task_def={"prompt": "look up earnings", "mode": "instant"},
        raw_miner_response=miner_response,
        quality_score=0.80,
    )
    assert payload is not None
    assert payload["trace_gate"] == 1.0
    # total = trace_gate * (0.80*quality + 0.20*latency)
    #       = 1.0 * (0.80*0.80 + 0.20*~0.99) = ~0.838
    assert 0.80 < score <= 1.0
    assert payload["trace_gate_penalty_usd"] == 0.0
    assert owner.scoring.charges == []


def test_layered_score_gate_fail_zeros_score_and_charges_penalty():
    owner = _owner(penalty_usd=0.50)
    manager = _task_manager_with_stubs(owner=owner, deployment_id="dep-bad")
    task = _fake_task()
    miner_response = {
        "response": {
            "response_text": (
                "I checked https://fabricated.example/fake and it says 42."
            ),
            "trace": _trace_with_url("https://example.com/real"),
        }
    }
    score, payload = manager._layered_general_chat_score(
        session=None,
        task=task,
        task_def={"prompt": "p", "mode": "instant"},
        raw_miner_response=miner_response,
        quality_score=1.0,
    )
    assert score == 0.0
    assert payload["trace_gate"] == 0.0
    assert payload["trace_gate_penalty_usd"] == 0.50
    # Penalty debited via scoring manager.
    assert len(owner.scoring.charges) == 1
    dep_id, amount, reason = owner.scoring.charges[0]
    assert dep_id == "dep-bad"
    assert amount == 0.50
    assert reason == "trace_gate_fail"


def test_layered_score_penalty_skipped_when_deployment_not_found():
    owner = _owner(penalty_usd=0.50)
    manager = _task_manager_with_stubs(owner=owner, deployment_id=None)
    task = _fake_task()
    miner_response = {
        "response": {
            "response_text": "See https://fabricated.example/fake",
            "trace": _trace_with_url("https://example.com/real"),
        }
    }
    score, payload = manager._layered_general_chat_score(
        session=None,
        task=task,
        task_def={"prompt": "p", "mode": "instant"},
        raw_miner_response=miner_response,
        quality_score=1.0,
    )
    assert score == 0.0
    assert payload["trace_gate_penalty_usd"] == 0.50
    # No charge made because deployment_id lookup returned None.
    assert owner.scoring.charges == []


def test_layered_score_honeytoken_citation_zeros_miner():
    active = generate_honeytoken_set("run-layer", count=4)
    owner = _owner(penalty_usd=0.50)
    manager = _task_manager_with_stubs(
        owner=owner,
        active_honeytokens=active,
        deployment_id="dep-ht",
    )
    task = _fake_task()
    # Miner cites a honeytoken URL — this is a confirmed fabrication.
    miner_response = {
        "response": {
            "response_text": f"Reference: {active[0]}",
            "trace": _trace_with_url(active[0]),  # even with matching trace
        }
    }
    score, payload = manager._layered_general_chat_score(
        session=None,
        task=task,
        task_def={"prompt": "p", "mode": "instant"},
        raw_miner_response=miner_response,
        quality_score=1.0,
    )
    assert score == 0.0
    assert payload["metadata"]["honeytoken_cited"] is True
    # Honeytoken citation counts as gate failure → penalty fires.
    assert len(owner.scoring.charges) == 1


def test_layered_score_uses_thinking_budget_when_task_mode_thinking():
    owner = _owner()
    manager = _task_manager_with_stubs(owner=owner)
    task = _fake_task()
    url = "https://example.com/deep"
    miner_response = {
        "response": {
            "response_text": (
                f"According to {url}, the analysis runs across multiple "
                f"dimensions. Full earnings rose on mortgage volume."
            ),
            "trace": _trace_with_url(url),
        }
    }
    _, payload = manager._layered_general_chat_score(
        session=None,
        task=task,
        task_def={"prompt": "deep analysis", "mode": "thinking"},
        raw_miner_response=miner_response,
        quality_score=0.90,
    )
    assert payload is not None
    assert payload["mode"] == "thinking"


def test_layered_score_bad_trace_payload_falls_back():
    owner = _owner()
    manager = _task_manager_with_stubs(owner=owner)
    task = _fake_task()
    # Malformed trace payload — must not crash, just fall through to legacy.
    miner_response = {
        "response": {
            "response_text": "hello",
            "trace": {"entries": "not-a-list"},
        }
    }
    score, payload = manager._layered_general_chat_score(
        session=None,
        task=task,
        task_def={"prompt": "p", "mode": "instant"},
        raw_miner_response=miner_response,
        quality_score=0.7,
    )
    assert score == 0.7
    assert payload is None


def test_layered_score_penalty_disabled_in_settings():
    owner = _owner(penalty_usd=0.0)  # disabled
    manager = _task_manager_with_stubs(owner=owner)
    task = _fake_task()
    miner_response = {
        "response": {
            "response_text": "See https://fabricated.example/x",
            "trace": _trace_with_url("https://example.com/real"),
        }
    }
    score, payload = manager._layered_general_chat_score(
        session=None,
        task=task,
        task_def={"prompt": "p", "mode": "instant"},
        raw_miner_response=miner_response,
        quality_score=1.0,
    )
    assert score == 0.0  # gate still zeros
    assert payload["trace_gate_penalty_usd"] == 0.0
    assert owner.scoring.charges == []
