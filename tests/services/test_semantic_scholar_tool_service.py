from __future__ import annotations

import json
from typing import Any

import httpx
from httpx import ASGITransport, AsyncClient

from tool_platforms.semantic_scholar_tool_service.app import (
    create_app,
    generate_job_token,
)


_SEARCH_RESPONSE = {
    "total": 12345,
    "offset": 0,
    "next": 2,
    "data": [
        {
            "paperId": "abc123",
            "title": "Scaling Laws Revisited: A 2026 Perspective",
            "abstract": "We revisit Chinchilla scaling laws in light of recent frontier models.",
            "authors": [
                {"authorId": "1", "name": "Alice Smith"},
                {"authorId": "2", "name": "Bob Jones"},
            ],
            "year": 2026,
            "venue": "NeurIPS",
            "citationCount": 42,
            "influentialCitationCount": 7,
            "externalIds": {"ArXiv": "2601.12345", "DOI": "10.1000/abc"},
            "openAccessPdf": {"url": "https://example.org/paper.pdf"},
            "url": "https://www.semanticscholar.org/paper/abc123",
        },
        {
            "paperId": "def456",
            "title": "Post-Quantum Cryptography in Cloud Environments",
            "abstract": "Survey of PQC migration status across major cloud providers.",
            "authors": [{"authorId": "3", "name": "Carol Lee"}],
            "year": 2025,
            "venue": "USENIX Security",
            "citationCount": 17,
            "influentialCitationCount": 2,
            "externalIds": {"ArXiv": "2601.67890"},
            "openAccessPdf": None,
            "url": "https://www.semanticscholar.org/paper/def456",
        },
    ],
}

_BATCH_RESPONSE = [
    {
        "paperId": "abc123",
        "title": "Scaling Laws Revisited: A 2026 Perspective",
        "abstract": "Abstract one.",
        "authors": [{"authorId": "1", "name": "Alice Smith"}],
        "year": 2026,
        "venue": "NeurIPS",
        "citationCount": 42,
        "influentialCitationCount": 7,
        "externalIds": {"ArXiv": "2601.12345"},
        "openAccessPdf": None,
        "url": "",
    }
]


def _fake_search_transport(body: dict[str, Any] | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/paper/search"):
            return httpx.Response(200, json=body or _SEARCH_RESPONSE)
        if request.url.path.endswith("/paper/batch"):
            return httpx.Response(200, json=_BATCH_RESPONSE)
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def _fake_429_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"error": "rate limited"},
            headers={"Retry-After": "2"},
        )

    return httpx.MockTransport(handler)


async def test_service_search_happy_path(monkeypatch):
    monkeypatch.setenv("EIREL_SEMANTIC_SCHOLAR_TOOL_API_TOKEN", "s2-token")
    app = create_app(semantic_scholar_transport=_fake_search_transport())
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            headers = {
                "Authorization": "Bearer s2-token",
                "X-Eirel-Job-Id": "job-1",
                "X-Eirel-Max-Requests": "2",
            }
            resp = await client.post(
                "/v1/search",
                json={
                    "query": "scaling laws",
                    "year": "2024-",
                    "fields_of_study": ["Computer Science"],
                    "max_results": 5,
                },
                headers=headers,
            )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_matching"] == 12345
    assert len(body["papers"]) == 2
    first = body["papers"][0]
    assert first["paper_id"] == "abc123"
    assert first["arxiv_id"] == "2601.12345"
    assert first["doi"] == "10.1000/abc"
    assert first["venue"] == "NeurIPS"
    assert first["citation_count"] == 42
    assert first["open_access_pdf_url"] == "https://example.org/paper.pdf"
    assert first["authors"] == ["Alice Smith", "Bob Jones"]
    assert first["year"] == 2026
    assert first["content_sha256"]
    assert body["retrieval_ledger_id"] == "ledger:job-1"


async def test_service_enforces_auth(monkeypatch):
    monkeypatch.setenv("EIREL_SEMANTIC_SCHOLAR_TOOL_API_TOKEN", "s2-token")
    app = create_app(semantic_scholar_transport=_fake_search_transport())
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            unauth = await client.post(
                "/v1/search",
                json={"query": "x"},
                headers={"X-Eirel-Job-Id": "job-1"},
            )
    assert unauth.status_code == 401


async def test_service_per_job_token_auth(monkeypatch):
    monkeypatch.setenv("EIREL_SEMANTIC_SCHOLAR_TOOL_API_TOKEN", "master-token")
    app = create_app(semantic_scholar_transport=_fake_search_transport())
    job_token = generate_job_token("master-token", "job-xyz")
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.post(
                "/v1/search",
                json={"query": "scaling"},
                headers={
                    "Authorization": f"Bearer {job_token}",
                    "X-Eirel-Job-Id": "job-xyz",
                },
            )
    assert resp.status_code == 200


async def test_service_budget_enforcement():
    app = create_app(semantic_scholar_transport=_fake_search_transport())
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            headers = {
                "X-Eirel-Job-Id": "job-budget",
                "X-Eirel-Max-Requests": "2",
            }
            first = await client.post("/v1/search", json={"query": "a"}, headers=headers)
            second = await client.post("/v1/search", json={"query": "b"}, headers=headers)
            third = await client.post("/v1/search", json={"query": "c"}, headers=headers)
    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
    assert "budget" in third.json()["detail"]


async def test_service_missing_job_id():
    app = create_app(semantic_scholar_transport=_fake_search_transport())
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.post("/v1/search", json={"query": "x"})
    assert resp.status_code == 400


async def test_service_healthz_and_metrics():
    app = create_app(semantic_scholar_transport=_fake_search_transport())
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            health = await client.get("/healthz")
            assert health.status_code == 200
            assert health.json() == {"status": "ok"}

            headers = {"X-Eirel-Job-Id": "job-m"}
            await client.post("/v1/search", json={"query": "x"}, headers=headers)
            metrics = await client.get("/metrics")
    assert metrics.status_code == 200
    assert "eirel_semantic_scholar_tool_requests_total 1" in metrics.text


async def test_service_batch_endpoint():
    app = create_app(semantic_scholar_transport=_fake_search_transport())
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.post(
                "/v1/batch",
                json={"paper_ids": ["abc123", "def456"]},
                headers={"X-Eirel-Job-Id": "job-b"},
            )
    assert resp.status_code == 200
    body = resp.json()
    assert body["result_count"] == 1
    assert body["papers"][0]["paper_id"] == "abc123"


async def test_service_upstream_429_maps_to_503():
    app = create_app(semantic_scholar_transport=_fake_429_transport())
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.post(
                "/v1/search",
                json={"query": "too hot"},
                headers={"X-Eirel-Job-Id": "job-429"},
            )
    assert resp.status_code == 503
    assert resp.headers.get("Retry-After") == "2"
    assert "rate limited" in resp.json()["detail"].lower()


async def test_service_sends_api_key_header_when_configured(monkeypatch):
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        captured["url"] = str(request.url)
        return httpx.Response(200, json=_SEARCH_RESPONSE)

    monkeypatch.setenv("EIREL_SEMANTIC_SCHOLAR_API_KEY", "my-s2-key")
    app = create_app(semantic_scholar_transport=httpx.MockTransport(handler))
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.post(
                "/v1/search",
                json={"query": "transformers"},
                headers={"X-Eirel-Job-Id": "job-key"},
            )
    assert resp.status_code == 200
    assert captured["headers"]["x-api-key"] == "my-s2-key"
    # Query forwarded with the expected fields param.
    assert "paperId" in captured["url"]
    assert "query=transformers" in captured["url"]
