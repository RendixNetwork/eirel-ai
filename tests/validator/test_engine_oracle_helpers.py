"""Engine.py oracle-enrichment integration helpers.

Covers the building blocks ``_enrich_task_oracle``,
``_fetch_ledger_tools``, and ``_build_oracle_layer``. These are the
units the validator's ``_evaluate_task`` and ``_judge_miner`` consume
to wire oracle enrichment + composite scoring; the full closure
chain is exercised end-to-end in production smoke — mocking the
closure's many captured dependencies isn't worth the test friction.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from validation.validator.engine import (
    _build_oracle_layer,
    _enrich_task_oracle,
    _fetch_ledger_tools,
)
from validation.validator.oracles.base import OracleGrounding
from validation.validator.reconciler import ReconciledOracle, Reconciler


pytestmark = pytest.mark.asyncio


def _stub_signer(token: str = "hk-test-validator") -> SimpleNamespace:
    """Minimal signer stub. Returns a fixed Authorization header so the
    test handlers can confirm the request was hotkey-signed."""
    return SimpleNamespace(
        signed_headers=lambda method, path, body_hash: {
            "Authorization": f"Hotkey {token}",
            "X-Eirel-Hotkey": token,
        },
        hotkey=token,
    )


# -- _enrich_task_oracle --------------------------------------------------


def _task(*, oracle_source: str | None = None, **expected_output) -> SimpleNamespace:
    """Build the duck-typed task_obj the validator engine creates."""
    return SimpleNamespace(
        task_id="t-1",
        prompt="What is 2+2?",
        turns=None,
        category="factual",
        expected_output=expected_output,
        oracle_source=oracle_source,
    )


async def test_deterministic_path_wraps_pool_answer():
    """``oracle_source=deterministic`` (or unset) builds a
    ReconciledOracle from the task's pre-baked expected_output —
    no LLM calls, no fanout, no reconciler needed."""
    task = _task(answer="4", must_not_claim=["five"])
    rec = await _enrich_task_oracle(task, fanout=None, reconciler=None)
    assert rec.oracle_status == "deterministic"
    assert rec.expected_claims == ["4"]
    assert rec.must_not_claim == ["five"]


async def test_unset_oracle_source_treated_as_deterministic():
    task = _task(oracle_source=None, answer="4")
    rec = await _enrich_task_oracle(task, fanout=None, reconciler=None)
    assert rec.oracle_status == "deterministic"


async def test_three_oracle_without_layer_falls_back_to_disputed():
    """Operator forgot to configure oracle creds — three_oracle items
    surface as ``disputed`` with the template floor preserved instead
    of crashing the run."""
    task = _task(oracle_source="three_oracle", must_not_claim=["never X"])
    rec = await _enrich_task_oracle(task, fanout=None, reconciler=None)
    assert rec.oracle_status == "disputed"
    assert rec.expected_claims == []
    assert rec.must_not_claim == ["never X"]
    assert rec.disagreement_note == "oracle_layer_not_configured"


async def test_three_oracle_with_mocked_layer_runs_fanout_and_reconciler():
    """Happy path: fanout returns 3 OK groundings, reconciler emits
    consensus claims; expected_claims surface on the result."""

    class _StubFanout:
        async def run(self, ctx):
            return [
                OracleGrounding(vendor="openai", status="ok", raw_text="Paris"),
                OracleGrounding(vendor="gemini", status="ok", raw_text="Paris"),
                OracleGrounding(vendor="grok", status="ok", raw_text="Paris"),
            ]

    class _StubReconciler:
        async def reconcile(self, *, prompt, groundings, must_not_claim_floor):
            return ReconciledOracle(
                expected_claims=["Paris is the capital"],
                must_not_claim=list(must_not_claim_floor),
                oracle_status="consensus",
                consensus_claims=["Paris is the capital"],
            )

    task = _task(oracle_source="three_oracle", must_not_claim=["never London"])
    rec = await _enrich_task_oracle(
        task, fanout=_StubFanout(), reconciler=_StubReconciler(),
    )
    assert rec.oracle_status == "consensus"
    assert rec.expected_claims == ["Paris is the capital"]
    assert rec.must_not_claim == ["never London"]


# -- _fetch_ledger_tools --------------------------------------------------


def _patch_engine_async_client(transport: httpx.MockTransport):
    """Replace ``engine.httpx.AsyncClient`` with one that always uses
    the given mock transport. ``patch.object`` on the class itself
    keeps the constructor's other kwargs (e.g. ``timeout``) intact."""

    real = httpx.AsyncClient

    class _Fake(real):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    from validation.validator import engine as _engine
    return patch.object(_engine.httpx, "AsyncClient", _Fake)


async def test_fetch_ledger_tools_happy_path():
    """Owner-api returns a ledger; helper extracts unique tool names
    in arrival order. Request is hotkey-signed."""
    captured_headers: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_headers.update(dict(request.headers))
        return httpx.Response(
            200,
            json={
                "job_id": "job-abc",
                "n_calls": 3,
                "tool_calls": [
                    {"tool_name": "web_search"},
                    {"tool_name": "url_fetch"},
                    {"tool_name": "web_search"},  # duplicate — dedup
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    with _patch_engine_async_client(transport):
        tools = await _fetch_ledger_tools(
            "job-abc", owner_url="http://owner.test", signer=_stub_signer(),
        )
    assert captured_headers.get("authorization") == "Hotkey hk-test-validator"
    assert tools == ["web_search", "url_fetch"]


async def test_fetch_ledger_tools_no_job_id_returns_empty():
    tools = await _fetch_ledger_tools(
        "", owner_url="http://owner.test", signer=_stub_signer(),
    )
    assert tools == []


async def test_fetch_ledger_tools_5xx_returns_empty():
    """Owner-api error → empty list (composite gets 0
    tool_attestation_factor for tasks where required_tool is set —
    fail-safe behavior)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "transient"})

    transport = httpx.MockTransport(handler)
    with _patch_engine_async_client(transport):
        tools = await _fetch_ledger_tools(
            "job-abc", owner_url="http://owner.test", signer=_stub_signer(),
        )
    assert tools == []


async def test_fetch_ledger_tools_network_error_returns_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated")

    transport = httpx.MockTransport(handler)
    with _patch_engine_async_client(transport):
        tools = await _fetch_ledger_tools(
            "job-abc", owner_url="http://owner.test", signer=_stub_signer(),
        )
    assert tools == []


async def test_fetch_ledger_tools_empty_ledger():
    """Job had no tool calls — empty list, no error."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"job_id": "job-x", "n_calls": 0, "tool_calls": []},
        )

    transport = httpx.MockTransport(handler)
    with _patch_engine_async_client(transport):
        tools = await _fetch_ledger_tools(
            "job-x", owner_url="http://owner.test", signer=_stub_signer(),
        )
    assert tools == []


# -- _build_oracle_layer --------------------------------------------------


def test_build_oracle_layer_returns_none_when_no_creds(monkeypatch):
    """Without any provider creds configured, the layer returns
    (None, None). All three_oracle items will fall back to disputed
    in ``_enrich_task_oracle``."""
    for prefix in (
        "EIREL_VALIDATOR_ORACLE_OPENAI_",
        "EIREL_VALIDATOR_ORACLE_GEMINI_",
        "EIREL_VALIDATOR_ORACLE_GROK_",
        "EIREL_VALIDATOR_RECONCILER_",
    ):
        for suffix in ("BASE_URL", "API_KEY", "MODEL"):
            monkeypatch.delenv(f"{prefix}{suffix}", raising=False)
    fanout, reconciler = _build_oracle_layer()
    assert fanout is None
    assert reconciler is None


def test_build_oracle_layer_returns_none_when_only_one_oracle_configured(monkeypatch):
    """Only OpenAI configured, no Gemini/Grok → fewer than 2 oracles,
    layer returns None (plurality voting not possible with 1 vote)."""
    for prefix in (
        "EIREL_VALIDATOR_ORACLE_GEMINI_",
        "EIREL_VALIDATOR_ORACLE_GROK_",
    ):
        for suffix in ("BASE_URL", "API_KEY", "MODEL"):
            monkeypatch.delenv(f"{prefix}{suffix}", raising=False)
    monkeypatch.setenv(
        "EIREL_VALIDATOR_ORACLE_OPENAI_BASE_URL", "http://openai.test",
    )
    monkeypatch.setenv("EIREL_VALIDATOR_ORACLE_OPENAI_API_KEY", "tok")
    monkeypatch.setenv("EIREL_VALIDATOR_ORACLE_OPENAI_MODEL", "gpt-5.4")
    monkeypatch.setenv(
        "EIREL_VALIDATOR_RECONCILER_BASE_URL", "http://chutes.test",
    )
    monkeypatch.setenv("EIREL_VALIDATOR_RECONCILER_API_KEY", "tok")
    monkeypatch.setenv(
        "EIREL_VALIDATOR_RECONCILER_MODEL", "zai-org/GLM-5.1-TEE",
    )
    fanout, reconciler = _build_oracle_layer()
    assert fanout is None
    assert reconciler is None


def test_build_oracle_layer_succeeds_with_two_oracles_plus_reconciler(monkeypatch):
    """Two oracles + reconciler all configured → layer comes up.
    Grok missing is the realistic case (Grok had more downtime
    historically); fanout still works with 2-of-3."""
    monkeypatch.delenv("EIREL_VALIDATOR_ORACLE_GROK_BASE_URL", raising=False)
    monkeypatch.delenv("EIREL_VALIDATOR_ORACLE_GROK_API_KEY", raising=False)
    monkeypatch.setenv(
        "EIREL_VALIDATOR_ORACLE_OPENAI_BASE_URL", "http://openai.test",
    )
    monkeypatch.setenv("EIREL_VALIDATOR_ORACLE_OPENAI_API_KEY", "tok")
    monkeypatch.setenv("EIREL_VALIDATOR_ORACLE_OPENAI_MODEL", "gpt-5.4")
    monkeypatch.setenv(
        "EIREL_VALIDATOR_ORACLE_GEMINI_BASE_URL", "http://gemini.test",
    )
    monkeypatch.setenv("EIREL_VALIDATOR_ORACLE_GEMINI_API_KEY", "tok")
    monkeypatch.setenv(
        "EIREL_VALIDATOR_ORACLE_GEMINI_MODEL", "gemini-3.1-pro-preview",
    )
    monkeypatch.setenv(
        "EIREL_VALIDATOR_RECONCILER_BASE_URL", "http://chutes.test",
    )
    monkeypatch.setenv("EIREL_VALIDATOR_RECONCILER_API_KEY", "tok")
    monkeypatch.setenv(
        "EIREL_VALIDATOR_RECONCILER_MODEL", "zai-org/GLM-5.1-TEE",
    )
    fanout, reconciler = _build_oracle_layer()
    assert fanout is not None
    assert reconciler is not None
    assert fanout.vendors == ["openai", "gemini"]


def test_build_oracle_layer_returns_none_when_reconciler_missing(monkeypatch):
    """All 3 oracles configured but no reconciler creds → no layer.
    Reconciler is the synthesis step; without it the oracles are
    unusable."""
    for prefix in (
        "EIREL_VALIDATOR_ORACLE_OPENAI_",
        "EIREL_VALIDATOR_ORACLE_GEMINI_",
        "EIREL_VALIDATOR_ORACLE_GROK_",
    ):
        monkeypatch.setenv(f"{prefix}BASE_URL", "http://test")
        monkeypatch.setenv(f"{prefix}API_KEY", "tok")
        monkeypatch.setenv(f"{prefix}MODEL", "test-model")
    monkeypatch.delenv("EIREL_VALIDATOR_RECONCILER_BASE_URL", raising=False)
    monkeypatch.delenv("EIREL_VALIDATOR_RECONCILER_API_KEY", raising=False)
    fanout, reconciler = _build_oracle_layer()
    assert fanout is None
    assert reconciler is None
