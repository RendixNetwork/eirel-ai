from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx

from control_plane.owner_api.evaluation.scoring_manager import ScoringManager


def _settings(proxy_url: str = "http://provider-proxy.test") -> SimpleNamespace:
    return SimpleNamespace(
        provider_proxy_url=proxy_url,
        provider_proxy_token="internal-token",
    )


def _owner(proxy_url: str = "http://provider-proxy.test") -> SimpleNamespace:
    return SimpleNamespace(settings=_settings(proxy_url), db=None)


def test_charge_trace_gate_penalty_hits_provider_proxy(monkeypatch):
    captured: dict[str, Any] = {}

    def _fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return httpx.Response(
            200,
            json={
                "cost_usd_used": 5.50,
                "max_usd_budget": 30.0,
                "reason": "trace_gate_fail",
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", _fake_post)

    manager = ScoringManager(_owner())
    ok = manager.charge_trace_gate_penalty(
        "dep-abc", amount_usd=0.50, reason="trace_gate_fail"
    )
    assert ok is True
    assert captured["url"] == "http://provider-proxy.test/v1/jobs/miner-dep-abc/charge_penalty"
    assert captured["json"] == {"reason": "trace_gate_fail", "amount_usd": 0.50}
    assert captured["headers"] == {"Authorization": "Bearer internal-token"}


def test_charge_trace_gate_penalty_zero_amount_is_noop(monkeypatch):
    called = {"count": 0}

    def _fake_post(*args, **kwargs):
        called["count"] += 1
        return httpx.Response(200, request=httpx.Request("POST", "http://x"))

    monkeypatch.setattr(httpx, "post", _fake_post)

    manager = ScoringManager(_owner())
    assert manager.charge_trace_gate_penalty("dep", amount_usd=0.0) is False
    assert called["count"] == 0


def test_charge_trace_gate_penalty_returns_false_when_proxy_unconfigured(monkeypatch):
    def _fake_post(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("httpx.post should not be called")

    monkeypatch.setattr(httpx, "post", _fake_post)

    manager = ScoringManager(_owner(proxy_url=""))
    assert manager.charge_trace_gate_penalty("dep", amount_usd=0.50) is False


def test_charge_trace_gate_penalty_swallows_network_errors(monkeypatch):
    def _fake_post(url, json, headers, timeout):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", _fake_post)

    manager = ScoringManager(_owner())
    # Must not raise — penalty-charging failures should be logged but
    # never break the scoring pipeline.
    assert manager.charge_trace_gate_penalty("dep", amount_usd=0.50) is False
