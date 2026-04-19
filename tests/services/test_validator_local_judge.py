from __future__ import annotations

"""Regression test for the validator-side local-judge refactor.

Before this change, the validator called owner-api's
/v1/families/<id>/tasks/<id>/judge endpoint (the server held the
rubric). After the change, the validator runs its own eiretes-judge
sidecar and submits the quality_score + judge_output to /result.

This test mocks JudgeServiceClient to verify:

1. validator-engine calls JudgeServiceClient.judge() once per task with
   the task's prompt and the response excerpt derived from the miner
   response — NOT the removed owner-api /judge proxy.
2. The value submitted to /result has task_score = JudgeResult.score
   (the raw LLM quality score) and includes judge_output.
3. When the local judge raises, the validator still submits with
   task_score=0 + judge_output=None (graceful fallback).
"""

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _ScriptedTransport(httpx.AsyncBaseTransport):
    """Records requests and returns scripted responses by URL path.

    For ``/tasks/claim``: first call returns ``claim_response``, subsequent
    calls return an empty batch so the engine's ``while True`` loop exits.
    """

    def __init__(
        self,
        *,
        claim_response: dict[str, Any],
        routes: dict[str, tuple[int, dict[str, Any]]] | None = None,
    ) -> None:
        self._claim_response = claim_response
        self._routes = routes or {}
        self._claim_calls = 0
        self.requests: list[tuple[str, str, dict]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        import json
        body: dict[str, Any] = {}
        if request.content:
            try:
                body = json.loads(request.content.decode())
            except Exception:
                body = {"_raw": request.content.decode(errors="ignore")}
        self.requests.append((request.method, request.url.path, body))

        if "/tasks/claim" in request.url.path:
            self._claim_calls += 1
            if self._claim_calls == 1:
                return httpx.Response(200, json=self._claim_response, request=request)
            return httpx.Response(
                200,
                json={"tasks": [], "remaining_task_count": 0},
                request=request,
            )

        for pattern, (status_code, payload) in self._routes.items():
            if pattern in request.url.path:
                return httpx.Response(status_code, json=payload, request=request)
        return httpx.Response(404, json={"detail": "unscripted"}, request=request)


def _task_claim(task_assignment_id: str, *, task_id: str, miner_hotkey: str) -> dict[str, Any]:
    return {
        "task_assignment_id": task_assignment_id,
        "task_id": task_id,
        "task_payload": {
            "prompt": "sample prompt",
            "mode": "instant",
            "allowed_tools": [],
            "expected_output": {},
        },
        "miner_hotkey": miner_hotkey,
        "miner_endpoint": "http://miner.test:9000",
        "miner_auth_headers": {},
    }


def _fake_judge_result(score: float = 0.73) -> MagicMock:
    mock = MagicMock()
    mock.score = score
    mock.model_dump.return_value = {
        "model": "test-judge",
        "rubric_name": "general_chat_rubric_v1",
        "score": score,
        "rationale": "mock",
        "latency_seconds": 0.1,
        "dimension_scores": {},
        "constraint_flags": [],
        "usage": {},
        "metadata": {},
    }
    return mock


@pytest.fixture
def _patched_run(monkeypatch, tmp_path):
    """Wire the validator engine to a mock judge + scripted owner-api.

    Returns (judge_mock, transport_recorder) so the test can assert
    after running ``run_distributed_benchmarks``.
    """
    import validation.validator.engine as eng

    # 1. Point at a test owner URL + provide a fake wallet signer.
    monkeypatch.setenv("OWNER_API_URL", "http://owner.test")
    monkeypatch.setattr(
        eng, "_load_validator_signer",
        lambda: MagicMock(hotkey="5Validator", sign=lambda s: b"sig"),
    )
    monkeypatch.setattr(
        eng, "_signed_headers",
        lambda **kw: {"X-Hotkey": "5Validator", "X-Signature": "sig"},
    )

    # 2. Mock miner invocation — return a deterministic task_run.
    async def _fake_invoke_task(*, miner, task, timeout_seconds):
        from shared.core.evaluation_models import BenchmarkTaskRun
        return BenchmarkTaskRun(
            task_id=task.task_id,
            family_id=task.family_id,
            prompt=task.prompt,
            expected_output={},
            response={"content": "miner reply for " + task.task_id},
            status="completed",
            error=None,
            metadata={"latency_seconds": 0.05},
        )
    monkeypatch.setattr(
        "shared.benchmark._invocation._invoke_task",
        _fake_invoke_task,
    )

    # 3. Replace JudgeServiceClient with a mock. The engine imports it
    #    inside the function body via ``from shared.core.judge_client
    #    import JudgeServiceClient``, so patch at the source module.
    judge_mock = MagicMock()
    judge_mock.judge.return_value = _fake_judge_result(score=0.73)
    monkeypatch.setattr(
        "shared.core.judge_client.JudgeServiceClient",
        lambda *a, **kw: judge_mock,
    )

    return judge_mock


async def _run(transport: _ScriptedTransport, **kwargs):
    """Invoke run_distributed_benchmarks with the scripted owner-api."""
    import validation.validator.engine as eng

    original = httpx.AsyncClient

    def _patched(*args, **kw):
        kw["transport"] = transport
        return original(*args, **kw)

    # Monkey-patch the httpx.AsyncClient used inside engine.py for this run.
    eng_httpx = eng.httpx
    saved = eng_httpx.AsyncClient
    eng_httpx.AsyncClient = _patched
    try:
        return await eng.run_distributed_benchmarks(**kwargs)
    finally:
        eng_httpx.AsyncClient = saved


async def test_local_judge_is_called_and_result_is_submitted(_patched_run):
    """Validator calls judge locally, then submits quality_score to /result."""
    judge_mock = _patched_run

    # Scripted owner-api: first /claim returns one task, then empty,
    # and /result accepts.
    transport = _ScriptedTransport(
        claim_response={
            "tasks": [_task_claim("assign-1", task_id="t-1", miner_hotkey="5Alice")],
            "remaining_task_count": 0,
        },
        routes={
            "/tasks/assign-1/result": (200, {
                "status": "accepted",
                "miner_evaluation_complete": True,
                "family_evaluation_complete": True,
                "remaining_task_count": 0,
            }),
        },
    )

    await _run(
        transport,
        family_id="general_chat",
        run_id="run-test",
        judge_model="test-judge",
        batch_size=5,
        max_parallel=1,
    )

    # 1. The local judge was called exactly once.
    judge_mock.judge.assert_called_once()
    judge_call = judge_mock.judge.call_args
    assert judge_call.kwargs["family_id"] == "general_chat"
    assert judge_call.kwargs["prompt"] == "sample prompt"
    assert "miner reply for t-1" in judge_call.kwargs["response_excerpt"]
    assert judge_call.kwargs["mode"] == "instant"

    # 2. The validator did NOT call the removed /judge endpoint.
    assert not any("/judge" in path for _method, path, _body in transport.requests), (
        f"validator still calls /judge proxy: {transport.requests}"
    )

    # 3. The validator POSTed to /result with the local judge's score.
    result_requests = [
        (method, path, body) for method, path, body in transport.requests
        if "/tasks/assign-1/result" in path
    ]
    assert len(result_requests) == 1
    method, _path, body = result_requests[0]
    assert method == "POST"
    assert body["task_score"] == pytest.approx(0.73)
    assert body["judge_output"] is not None
    assert body["judge_output"]["score"] == pytest.approx(0.73)
    assert body["task_status"] == "completed"


async def test_local_judge_failure_falls_through_with_zero_score(_patched_run):
    """When JudgeServiceClient.judge() raises, submit with task_score=0."""
    judge_mock = _patched_run
    judge_mock.judge.side_effect = RuntimeError("chutes returned 503")

    transport = _ScriptedTransport(
        claim_response={
            "tasks": [_task_claim("assign-2", task_id="t-2", miner_hotkey="5Bob")],
            "remaining_task_count": 0,
        },
        routes={
            "/tasks/assign-2/result": (200, {
                "status": "accepted",
                "miner_evaluation_complete": True,
                "family_evaluation_complete": True,
                "remaining_task_count": 0,
            }),
        },
    )

    await _run(
        transport,
        family_id="general_chat",
        run_id="run-test",
        judge_model="test-judge",
        batch_size=5,
        max_parallel=1,
    )

    result_requests = [
        (method, path, body) for method, path, body in transport.requests
        if "/tasks/assign-2/result" in path
    ]
    assert len(result_requests) == 1
    _method, _path, body = result_requests[0]
    assert body["task_score"] == 0.0
    assert body["judge_output"] is None
