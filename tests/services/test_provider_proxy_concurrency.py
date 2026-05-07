"""Concurrency-hardening tests for provider_proxy.

Two concerns:
  1. Concurrent ``charge_tool`` / ``reserve_estimate`` calls under one
     job_id must not over-reserve past the run budget.
  2. New ``reserve_batch_estimate`` endpoint must atomically commit or
     reject N estimates as one operation, with per-span attribution
     surfaced in ``cost_by_span``.

The fakeredis backend forces the Python fallback path (no Lua), which
is the path most likely to race in production deployments where Redis
is replaced by a Lua-less alternative.
"""
from __future__ import annotations

import asyncio
import importlib

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture(autouse=True)
def _patch_redis_client(monkeypatch):
    import fakeredis.aioredis
    proxy_app = importlib.import_module("tool_platforms.provider_proxy.app")
    monkeypatch.setattr(
        proxy_app,
        "_make_redis_client",
        lambda _url: fakeredis.aioredis.FakeRedis(decode_responses=True),
    )
    monkeypatch.setenv("EIREL_CHUTES_PRICING_REFRESH_ENABLED", "false")


def _build_payload(provider: str = "chutes", model: str = "chutes-test"):
    return {
        "provider": provider,
        "model": model,
        "payload": {"messages": [{"role": "user", "content": "hello"}]},
        "max_requests": 100,
        "max_total_tokens": 200_000,
        "max_wall_clock_seconds": 300,
        "per_request_timeout_seconds": 30,
    }


def _fake_dispatch(*, provider, model, payload, timeout_seconds):
    """Sync stub — production calls it via asyncio.to_thread."""
    return {
        "id": "resp",
        "provider": provider,
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }


@pytest.mark.asyncio
async def test_concurrent_charge_tool_does_not_over_reserve(monkeypatch):
    """50 concurrent charge_tool calls under a tight budget — only the
    ones that fit before exhaustion are accepted; the rest are 429s.
    Final cost_usd_used must never exceed max_usd_budget."""
    monkeypatch.setenv("EIREL_PROVIDER_PROXY_MASTER_TOKEN", "tok")
    monkeypatch.setenv("EIREL_PROVIDER_CHUTES_API_KEY", "k")
    monkeypatch.setenv("EIREL_PROVIDER_CHUTES_BASE_URL", "http://provider.local/x")
    proxy_app = importlib.import_module("tool_platforms.provider_proxy.app")
    monkeypatch.setattr(proxy_app, "_dispatch_provider_request", _fake_dispatch)

    app = proxy_app.create_app()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            # Seed the job with a tight budget via one initial chat call.
            seed = await client.post(
                "/v1/chat/completions",
                json=_build_payload(),
                headers={
                    "Authorization": "Bearer tok",
                    "X-Eirel-Job-Id": "job-stress",
                    "X-Eirel-Run-Budget-Usd": "0.05",  # ~5 cents total
                },
            )
            assert seed.status_code in (200, 429)

            # Fire 50 concurrent charges of 0.005 USD each — at most 10
            # can fit under 0.05 (less, given the seed's reservation).
            async def one_charge(i: int):
                return await client.post(
                    f"/v1/jobs/job-stress/charge_tool",
                    json={
                        "tool_name": "web_search",
                        "amount_usd": 0.005,
                        "span_id": f"span-{i}",
                    },
                    headers={"Authorization": "Bearer tok"},
                )

            results = await asyncio.gather(*(one_charge(i) for i in range(50)))
            # Some should be 200, some 429 — but never an error type.
            statuses = [r.status_code for r in results]
            assert set(statuses).issubset({200, 429})

            # Read final cost — must not exceed the 0.05 cap.
            cost = await client.get(
                "/v1/jobs/job-stress/cost",
                headers={"Authorization": "Bearer tok"},
            )
            assert cost.status_code == 200
            payload = cost.json()
            assert payload["cost_usd_used"] <= 0.05 + 1e-9, (
                f"cost {payload['cost_usd_used']} blew through cap 0.05"
            )
            # cost_rejections must reflect the rejected calls.
            assert payload["cost_rejections"] >= 1


@pytest.mark.asyncio
async def test_charge_tool_records_per_span_cost(monkeypatch):
    monkeypatch.setenv("EIREL_PROVIDER_PROXY_MASTER_TOKEN", "tok")
    monkeypatch.setenv("EIREL_PROVIDER_CHUTES_API_KEY", "k")
    monkeypatch.setenv("EIREL_PROVIDER_CHUTES_BASE_URL", "http://provider.local/x")
    proxy_app = importlib.import_module("tool_platforms.provider_proxy.app")
    monkeypatch.setattr(proxy_app, "_dispatch_provider_request", _fake_dispatch)

    app = proxy_app.create_app()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            # Seed the job to establish max_usd_budget.
            await client.post(
                "/v1/chat/completions",
                json=_build_payload(),
                headers={
                    "Authorization": "Bearer tok",
                    "X-Eirel-Job-Id": "job-span",
                    "X-Eirel-Run-Budget-Usd": "10.0",
                },
            )

            # Two charges under different spans.
            r1 = await client.post(
                "/v1/jobs/job-span/charge_tool",
                json={"tool_name": "web_search", "amount_usd": 0.01, "span_id": "span-a"},
                headers={"Authorization": "Bearer tok"},
            )
            assert r1.status_code == 200, r1.text
            r2 = await client.post(
                "/v1/jobs/job-span/charge_tool",
                json={"tool_name": "sandbox", "amount_usd": 0.02, "span_id": "span-b"},
                headers={"Authorization": "Bearer tok"},
            )
            assert r2.status_code == 200

            cost = await client.get(
                "/v1/jobs/job-span/cost",
                headers={"Authorization": "Bearer tok"},
            )
            payload = cost.json()
            per_span = payload.get("per_span") or {}
            assert per_span.get("span-a") == pytest.approx(0.01)
            assert per_span.get("span-b") == pytest.approx(0.02)
            buckets = payload.get("per_span_buckets") or {}
            # Bucket-level field carries the tool name suffix.
            assert any(k.startswith("span-a::tool:") for k in buckets)
            assert any(k.startswith("span-b::tool:") for k in buckets)


@pytest.mark.asyncio
async def test_reserve_batch_estimate_atomic_rollback(monkeypatch):
    """When the sum exceeds budget, NONE of the estimates are committed."""
    monkeypatch.setenv("EIREL_PROVIDER_PROXY_MASTER_TOKEN", "tok")
    monkeypatch.setenv("EIREL_PROVIDER_CHUTES_API_KEY", "k")
    monkeypatch.setenv("EIREL_PROVIDER_CHUTES_BASE_URL", "http://provider.local/x")
    proxy_app = importlib.import_module("tool_platforms.provider_proxy.app")
    monkeypatch.setattr(proxy_app, "_dispatch_provider_request", _fake_dispatch)

    app = proxy_app.create_app()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            # Seed the job with budget 0.10 so 5 × 0.05 (= 0.25) clearly
            # blows the cap once we factor in the seed's reservation.
            await client.post(
                "/v1/chat/completions",
                json=_build_payload(),
                headers={
                    "Authorization": "Bearer tok",
                    "X-Eirel-Job-Id": "job-batch",
                    "X-Eirel-Run-Budget-Usd": "0.10",
                },
            )

            cost_before = (await client.get(
                "/v1/jobs/job-batch/cost",
                headers={"Authorization": "Bearer tok"},
            )).json()["cost_usd_used"]

            batch_resp = await client.post(
                "/v1/jobs/job-batch/reserve_batch_estimate",
                json={
                    "max_usd_budget": 0.10,
                    "estimates": [
                        {
                            "estimated_cost": 0.05,
                            "estimated_tokens": 100,
                            "provider": "chutes",
                            "model": "chutes-test",
                            "span_id": f"span-{i}",
                        }
                        for i in range(5)
                    ],
                },
                headers={"Authorization": "Bearer tok"},
            )
            assert batch_resp.status_code == 429, batch_resp.text

            cost_after = (await client.get(
                "/v1/jobs/job-batch/cost",
                headers={"Authorization": "Bearer tok"},
            )).json()["cost_usd_used"]
            # Atomic rollback: cost is unchanged.
            assert cost_after == pytest.approx(cost_before)


@pytest.mark.asyncio
async def test_reserve_batch_estimate_commits_when_under_budget(monkeypatch):
    monkeypatch.setenv("EIREL_PROVIDER_PROXY_MASTER_TOKEN", "tok")
    monkeypatch.setenv("EIREL_PROVIDER_CHUTES_API_KEY", "k")
    monkeypatch.setenv("EIREL_PROVIDER_CHUTES_BASE_URL", "http://provider.local/x")
    proxy_app = importlib.import_module("tool_platforms.provider_proxy.app")
    monkeypatch.setattr(proxy_app, "_dispatch_provider_request", _fake_dispatch)

    app = proxy_app.create_app()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            # Generous budget for this case.
            await client.post(
                "/v1/chat/completions",
                json=_build_payload(),
                headers={
                    "Authorization": "Bearer tok",
                    "X-Eirel-Job-Id": "job-batch-ok",
                    "X-Eirel-Run-Budget-Usd": "5.0",
                },
            )

            batch_resp = await client.post(
                "/v1/jobs/job-batch-ok/reserve_batch_estimate",
                json={
                    "max_usd_budget": 5.0,
                    "estimates": [
                        {
                            "estimated_cost": 0.10,
                            "estimated_tokens": 100,
                            "provider": "chutes",
                            "model": "chutes-test",
                            "span_id": f"span-{i}",
                        }
                        for i in range(3)
                    ],
                },
                headers={"Authorization": "Bearer tok"},
            )
            assert batch_resp.status_code == 200, batch_resp.text
            body = batch_resp.json()
            assert sorted(body["reserved"]) == ["span-0", "span-1", "span-2"]

            cost = (await client.get(
                "/v1/jobs/job-batch-ok/cost",
                headers={"Authorization": "Bearer tok"},
            )).json()
            # Per-span buckets exist for all three reserved entries.
            per_span = cost.get("per_span") or {}
            assert {"span-0", "span-1", "span-2"}.issubset(set(per_span))
