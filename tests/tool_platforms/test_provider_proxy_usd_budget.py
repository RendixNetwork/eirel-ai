from __future__ import annotations

import importlib

import pytest
from httpx import ASGITransport, AsyncClient

from tool_platforms.provider_proxy.app import create_app


def _fake_dispatch(*, provider, model, payload, timeout_seconds):
    prompt_tokens = max(1, len(str(payload.get("messages", []))) // 4)
    completion_tokens = 20
    return {
        "id": "resp-usd",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _base_headers(budget: str = "1.00") -> dict[str, str]:
    return {
        "Authorization": "Bearer master-token",
        "X-Eirel-Job-Id": "job-usd-1",
        "X-Eirel-Run-Budget-Usd": budget,
    }


def _base_payload() -> dict:
    return {
        "provider": "chutes",
        "model": "chutes-test",
        "payload": {
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 100,
        },
        "max_requests": 100,
        "max_total_tokens": 100000,
        "max_wall_clock_seconds": 300,
        "per_request_timeout_seconds": 30,
    }


def _patch_redis_client(monkeypatch) -> None:
    """Back the store with fakeredis so tests don't need a real Redis."""
    import fakeredis.aioredis
    proxy_app = importlib.import_module("tool_platforms.provider_proxy.app")
    monkeypatch.setattr(
        proxy_app,
        "_make_redis_client",
        lambda _url: fakeredis.aioredis.FakeRedis(decode_responses=True),
    )
    # Suppress the background Chutes pricing refresh — tests shouldn't
    # hit the live llm.chutes.ai endpoint (flaky + slow + offline CI).
    monkeypatch.setenv("EIREL_CHUTES_PRICING_REFRESH_ENABLED", "false")


@pytest.fixture()
def _patch_env(monkeypatch):
    monkeypatch.setenv("EIREL_PROVIDER_PROXY_MASTER_TOKEN", "master-token")
    monkeypatch.setenv("EIREL_PROVIDER_CHUTES_API_KEY", "proxy-key")
    proxy_app = importlib.import_module("tool_platforms.provider_proxy.app")
    monkeypatch.setattr(proxy_app, "_dispatch_provider_request", _fake_dispatch)
    _patch_redis_client(monkeypatch)


def _timeout_dispatch(*, provider, model, payload, timeout_seconds):
    import socket
    raise socket.timeout("upstream provider timeout (test)")


async def test_proxy_refunds_estimate_when_upstream_raises(monkeypatch):
    """Failed /chat/completions must NOT leave an orphan estimate on
    ``cost_usd_used``.  Before this fix, a 60s Chutes timeout added the
    pre-reserved estimate to cost_usd_used but never wrote an actual
    cost to cost_by_provider — so the delta showed up as ghost spend in
    every downstream report.
    """
    monkeypatch.setenv("EIREL_PROVIDER_PROXY_MASTER_TOKEN", "master-token")
    monkeypatch.setenv("EIREL_PROVIDER_CHUTES_API_KEY", "proxy-key")
    proxy_app = importlib.import_module("tool_platforms.provider_proxy.app")
    monkeypatch.setattr(proxy_app, "_dispatch_provider_request", _timeout_dispatch)
    _patch_redis_client(monkeypatch)

    app = create_app()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # ASGITransport re-raises unhandled exceptions from the app
            # rather than wrapping them in an HTTP response; we just need
            # to observe that the proxy's in-memory state was rolled back.
            with pytest.raises(Exception):
                await client.post(
                    "/v1/chat/completions",
                    json=_base_payload(),
                    headers=_base_headers("5.00"),
                )

            cost_resp = await client.get(
                "/v1/jobs/job-usd-1/cost",
                headers={"Authorization": "Bearer master-token"},
            )
            body = cost_resp.json()
            # Estimate refunded after upstream failure — cost_usd_used
            # is back to zero and cost_by_provider stays empty.
            assert body["cost_usd_used"] == pytest.approx(0.0, abs=1e-9)
            assert body["llm_cost_usd"] == pytest.approx(0.0, abs=1e-9)
            assert body["tool_cost_usd"] == pytest.approx(0.0, abs=1e-9)
            assert not body["per_provider"]


async def test_proxy_accepts_request_under_budget(_patch_env):
    app = create_app()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json=_base_payload(),
                headers=_base_headers("10.00"),
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["usage"]["cost_usd"] >= 0.0
            assert body["usage"]["cost_usd_used"] >= 0.0
            assert body["usage"]["cost_remaining_usd"] >= 0.0


async def test_proxy_429s_when_budget_exhausted(_patch_env):
    app = create_app()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json=_base_payload(),
                headers=_base_headers("0.0000001"),
            )
            assert resp.status_code == 429
            assert "run budget exhausted" in resp.json()["detail"]


async def test_actual_vs_estimated_reconciliation(_patch_env):
    app = create_app()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json=_base_payload(),
                headers=_base_headers("10.00"),
            )
            assert resp.status_code == 200
            cost_resp = await client.get(
                "/v1/jobs/job-usd-1/cost",
                headers={"Authorization": "Bearer master-token"},
            )
            assert cost_resp.status_code == 200
            cost = cost_resp.json()
            assert cost["cost_usd_used"] > 0.0
            assert cost["per_provider"].get("chutes", 0.0) > 0.0


async def test_get_cost_endpoint_returns_snapshot(_patch_env):
    app = create_app()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json=_base_payload(),
                headers=_base_headers("5.00"),
            )
            assert resp.status_code == 200
            cost_resp = await client.get(
                "/v1/jobs/job-usd-1/cost",
                headers={"Authorization": "Bearer master-token"},
            )
            assert cost_resp.status_code == 200
            body = cost_resp.json()
            assert "cost_usd_used" in body
            assert "max_usd_budget" in body
            assert body["max_usd_budget"] == pytest.approx(5.00)
            assert "cost_rejections" in body
            assert "per_provider" in body
            # Split surfaces for DeploymentScoreRecord population.
            # After one successful LLM call and no tool charges:
            #   * llm_cost_usd comes from the bare ``chutes`` entry
            #   * tool_cost_usd is zero (no ``tool:`` prefixed entries)
            assert "llm_cost_usd" in body
            assert "tool_cost_usd" in body
            assert body["llm_cost_usd"] == pytest.approx(
                body["per_provider"].get("chutes", 0.0)
            )
            assert body["tool_cost_usd"] == pytest.approx(0.0, abs=1e-9)


async def test_charge_tool_increments_cost(_patch_env):
    app = create_app()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/v1/chat/completions",
                json=_base_payload(),
                headers=_base_headers("5.00"),
            )
            cost_before = (await client.get(
                "/v1/jobs/job-usd-1/cost",
                headers={"Authorization": "Bearer master-token"},
            )).json()["cost_usd_used"]

            charge_resp = await client.post(
                "/v1/jobs/job-usd-1/charge_tool",
                json={"tool_name": "web_search", "amount_usd": 0.001},
                headers={"Authorization": "Bearer master-token"},
            )
            assert charge_resp.status_code == 200
            assert charge_resp.json()["cost_usd_used"] > cost_before


async def test_charge_tool_429s_when_over_budget(_patch_env):
    app = create_app()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/v1/chat/completions",
                json=_base_payload(),
                headers=_base_headers("5.00"),
            )
            charge_resp = await client.post(
                "/v1/jobs/job-usd-1/charge_tool",
                json={"tool_name": "web_search", "amount_usd": 100.0},
                headers={"Authorization": "Bearer master-token"},
            )
            assert charge_resp.status_code == 429
            assert "run budget exhausted" in charge_resp.json()["detail"]


async def test_missing_budget_header_rejected(_patch_env):
    app = create_app()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            headers = {
                "Authorization": "Bearer master-token",
                "X-Eirel-Job-Id": "job-no-budget",
            }
            resp = await client.post(
                "/v1/chat/completions",
                json=_base_payload(),
                headers=headers,
            )
            assert resp.status_code == 400
            assert "missing run budget header" in resp.json()["detail"]
