from __future__ import annotations

import json
from typing import Any

import httpx
from httpx import ASGITransport, AsyncClient

from tool_platforms.x_tool_service.app import create_app, generate_job_token


def _fake_x_transport(tweets: list[dict[str, Any]]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": tweets,
                "includes": {
                    "users": [
                        {
                            "id": t.get("author_id", "u1"),
                            "username": t.get("_username", "testuser"),
                            "verified": t.get("_verified", False),
                        }
                        for t in tweets
                    ]
                },
                "meta": {"result_count": len(tweets)},
            },
        )

    return httpx.MockTransport(handler)


async def test_x_tool_service_enforces_auth_and_budget(monkeypatch):
    monkeypatch.setenv("EIREL_X_TOOL_API_TOKEN", "x-token")
    monkeypatch.setenv("EIREL_X_BEARER_TOKEN", "bearer-xxx")
    tweets = [
        {
            "id": "tweet-1",
            "author_id": "u1",
            "_username": "analyst",
            "_verified": True,
            "text": "NVIDIA Q4 revenue beat expectations.",
            "created_at": "2026-01-15T12:00:00Z",
            "public_metrics": {"retweet_count": 10, "like_count": 50, "reply_count": 3},
        }
    ]
    app = create_app(x_transport=_fake_x_transport(tweets))
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            unauth = await client.post(
                "/v1/search",
                json={"query": "NVIDIA"},
                headers={"X-Eirel-Job-Id": "job-1", "X-Eirel-Max-Requests": "2"},
            )
            assert unauth.status_code == 401

            headers = {
                "Authorization": "Bearer x-token",
                "X-Eirel-Job-Id": "job-1",
                "X-Eirel-Max-Requests": "2",
            }
            result = await client.post("/v1/search", json={"query": "NVIDIA"}, headers=headers)
            assert result.status_code == 200
            data = result.json()
            assert len(data["tweets"]) == 1
            assert data["tweets"][0]["tweet_id"] == "tweet-1"
            assert data["retrieval_ledger_id"] == "ledger:job-1"
            assert data["tweets"][0]["content_sha256"]

            result2 = await client.post("/v1/search", json={"query": "AMD"}, headers=headers)
            assert result2.status_code == 200

            rejected = await client.post("/v1/search", json={"query": "Intel"}, headers=headers)
            assert rejected.status_code == 429

            usage = await client.get(
                "/v1/jobs/job-1/usage",
                headers={"Authorization": "Bearer x-token"},
            )
            assert usage.status_code == 200
            assert usage.json()["request_count"] == 2
            assert usage.json()["tool_counts"]["x_search"] == 2


async def test_x_tool_service_healthz_and_metrics(monkeypatch):
    monkeypatch.setenv("EIREL_X_BEARER_TOKEN", "bearer-xxx")
    app = create_app(x_transport=_fake_x_transport([]))
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            health = await client.get("/healthz")
            assert health.status_code == 200
            assert health.json() == {"status": "ok"}

            metrics = await client.get("/metrics")
            assert metrics.status_code == 200
            assert "eirel_x_tool_requests_total" in metrics.text


async def test_x_tool_service_missing_job_id(monkeypatch):
    monkeypatch.setenv("EIREL_X_BEARER_TOKEN", "bearer-xxx")
    app = create_app(x_transport=_fake_x_transport([]))
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            result = await client.post("/v1/search", json={"query": "test"})
            assert result.status_code == 400


async def test_x_tool_service_per_job_token_auth(monkeypatch):
    monkeypatch.setenv("EIREL_X_TOOL_API_TOKEN", "master-token")
    monkeypatch.setenv("EIREL_X_BEARER_TOKEN", "bearer-xxx")
    app = create_app(x_transport=_fake_x_transport([]))
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            job_token = generate_job_token("master-token", "job-99")
            headers = {
                "Authorization": f"Bearer {job_token}",
                "X-Eirel-Job-Id": "job-99",
                "X-Eirel-Max-Requests": "5",
            }
            result = await client.post("/v1/search", json={"query": "test"}, headers=headers)
            assert result.status_code == 200
