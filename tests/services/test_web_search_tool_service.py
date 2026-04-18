from __future__ import annotations

import json

import httpx
from httpx import ASGITransport, AsyncClient

from tool_platforms.web_search_tool_service.app import (
    ResearchCatalogStore,
    ResearchDocumentRecord,
    create_app,
    generate_job_token,
)


async def test_research_tool_service_enforces_auth_and_request_budgets(monkeypatch):
    monkeypatch.setenv("EIREL_RESEARCH_TOOL_API_TOKEN", "tool-token")
    app = create_app(
        ResearchCatalogStore(
            documents={
                "doc-1": ResearchDocumentRecord(
                    document_id="doc-1",
                    title="WHO measles coverage",
                    url="https://who.int/example/measles-coverage",
                    snippet="Coverage reduces outbreaks.",
                    content="Coverage reduces outbreaks.\nCoverage gaps still matter.",
                )
            }
        ),
        backend="catalog",
    )
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            unauthorized = await client.post(
                "/v1/search",
                json={"query": "measles", "top_k": 3},
                headers={"X-Eirel-Job-Id": "job-1", "X-Eirel-Max-Requests": "2"},
            )
            assert unauthorized.status_code == 401

            headers = {
                "Authorization": "Bearer tool-token",
                "X-Eirel-Job-Id": "job-1",
                "X-Eirel-Max-Requests": "2",
            }
            search = await client.post("/v1/search", json={"query": "measles", "top_k": 3}, headers=headers)
            assert search.status_code == 200
            assert search.json()["documents"][0]["document_id"] == "doc-1"
            assert search.json()["retrieval_ledger_id"] == "ledger:job-1"

            open_page = await client.post("/v1/open-page", json={"document_id": "doc-1"}, headers=headers)
            assert open_page.status_code == 200
            assert "Coverage gaps still matter." in open_page.json()["content"]
            assert open_page.json()["content_hash"]
            assert open_page.json()["canonical_url"] == "https://who.int/example/measles-coverage"

            rejected = await client.post("/v1/find-on-page", json={"document_id": "doc-1", "pattern": "coverage"}, headers=headers)
            assert rejected.status_code == 429

            usage = await client.get("/v1/jobs/job-1/usage", headers={"Authorization": "Bearer tool-token"})
            assert usage.status_code == 200
            assert usage.json()["request_count"] == 2
            assert usage.json()["tool_counts"]["search"] == 1
            assert usage.json()["tool_counts"]["open_page"] == 1
            assert usage.json()["retrieval_ledger_id"] == "ledger:job-1"


async def test_research_tool_service_exposes_retrieval_ledger(monkeypatch):
    monkeypatch.setenv("EIREL_RESEARCH_TOOL_API_TOKEN", "tool-token")
    app = create_app(
        ResearchCatalogStore(
            documents={
                "doc-1": ResearchDocumentRecord(
                    document_id="doc-1",
                    title="WHO measles coverage 2025",
                    url="https://www.who.int/example/measles-coverage?ref=nav",
                    snippet="Coverage reduces outbreaks in 2025.",
                    content="Coverage reduces outbreaks in 2025.\nCoverage gaps still matter.",
                    metadata={"published_at": "2025-03-12"},
                )
            }
        ),
        backend="catalog",
    )
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            headers = {
                "Authorization": "Bearer tool-token",
                "X-Eirel-Job-Id": "job-2",
                "X-Eirel-Max-Requests": "4",
            }
            await client.post("/v1/search", json={"query": "measles coverage", "top_k": 3}, headers=headers)
            await client.post("/v1/open-page", json={"document_id": "doc-1"}, headers=headers)
            find = await client.post("/v1/find-on-page", json={"document_id": "doc-1", "pattern": "coverage"}, headers=headers)
            assert find.status_code == 200
            ledger = await client.get("/v1/jobs/job-2/ledger", headers={"Authorization": "Bearer tool-token"})
            assert ledger.status_code == 200
            payload = ledger.json()
            assert payload["retrieval_ledger_id"] == "ledger:job-2"
            assert payload["searches"][0]["query"] == "measles coverage"
            assert payload["opened_pages"][0]["canonical_url"] == "https://who.int/example/measles-coverage"
            assert payload["opened_pages"][0]["published_at"] == "2025-03-12"
            assert payload["opened_pages"][0]["support_spans"] == ["Coverage reduces outbreaks in 2025.", "Coverage gaps still matter."]
            assert payload["find_on_page_events"][0]["pattern"] == "coverage"


async def test_research_tool_service_brave_search_reranks_preferred_domains(monkeypatch):
    monkeypatch.setenv("EIREL_RESEARCH_TOOL_API_TOKEN", "tool-token")
    monkeypatch.setenv("EIREL_BRAVE_SEARCH_API_KEY", "brave-token")

    def _search_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/web/search")
        return httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {
                            "title": "General blog coverage",
                            "url": "https://example.com/post",
                            "description": "Commentary on the topic.",
                        },
                        {
                            "title": "Primary source update",
                            "url": "https://openai.com/index/policy-update",
                            "description": "Primary source statement.",
                            "page_age": "2026-03-20",
                        },
                    ]
                }
            },
        )

    app = create_app(
        backend="brave_live_web",
        search_transport=httpx.MockTransport(_search_handler),
    )
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            headers = {
                "Authorization": "Bearer tool-token",
                "X-Eirel-Job-Id": "job-brave-1",
                "X-Eirel-Max-Requests": "4",
            }
            response = await client.post(
                "/v1/search",
                json={
                    "query": "policy update",
                    "top_k": 2,
                    "preferred_domain_families": ["openai.com"],
                },
                headers=headers,
            )
            assert response.status_code == 200
            payload = response.json()
            assert payload["documents"][0]["url"] == "https://openai.com/index/policy-update"
            assert payload["documents"][0]["metadata"]["search_provider"] == "brave"
            assert payload["documents"][0]["metadata"]["domain"] == "openai.com"
            assert payload["documents"][0]["document_id"].startswith("web-")


async def test_research_tool_service_live_open_page_and_find_on_page_use_opened_text(monkeypatch):
    monkeypatch.setenv("EIREL_RESEARCH_TOOL_API_TOKEN", "tool-token")
    monkeypatch.setenv("EIREL_BRAVE_SEARCH_API_KEY", "brave-token")

    def _search_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {
                            "title": "Policy update",
                            "url": "https://openai.com/index/policy-update",
                            "description": "Latest policy update.",
                            "page_age": "2026-03-20",
                        }
                    ]
                }
            },
        )

    html = """
    <html>
      <head>
        <title>Policy update</title>
        <meta property="article:published_time" content="2026-03-20" />
      </head>
      <body>
        <nav>nav links</nav>
        <article>
          <p>Current policy update confirms incident summaries are published quarterly.</p>
          <p>Additional commentary should be treated carefully.</p>
        </article>
      </body>
    </html>
    """

    def _fetch_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            content=html.encode(),
            request=request,
        )

    app = create_app(
        backend="brave_live_web",
        search_transport=httpx.MockTransport(_search_handler),
        fetch_transport=httpx.MockTransport(_fetch_handler),
    )
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            headers = {
                "Authorization": "Bearer tool-token",
                "X-Eirel-Job-Id": "job-brave-2",
                "X-Eirel-Max-Requests": "5",
            }
            search = await client.post(
                "/v1/search",
                json={"query": "policy update", "top_k": 1},
                headers=headers,
            )
            document_id = search.json()["documents"][0]["document_id"]

            not_opened = await client.post(
                "/v1/find-on-page",
                json={"document_id": document_id, "pattern": "quarterly"},
                headers=headers,
            )
            assert not_opened.status_code == 409

            opened = await client.post(
                "/v1/open-page",
                json={"document_id": document_id},
                headers=headers,
            )
            assert opened.status_code == 200
            opened_payload = opened.json()
            assert opened_payload["canonical_url"] == "https://openai.com/index/policy-update"
            assert opened_payload["content_type"] == "text/html"
            assert opened_payload["http_status"] == 200
            assert opened_payload["date_confidence"] > 0.0
            assert "Current policy update confirms" in opened_payload["content"]

            found = await client.post(
                "/v1/find-on-page",
                json={"document_id": document_id, "pattern": "quarterly"},
                headers=headers,
            )
            assert found.status_code == 200
            assert found.json()["support_spans"] == [
                "Current policy update confirms incident summaries are published quarterly."
            ]

            ledger = await client.get("/v1/jobs/job-brave-2/ledger", headers={"Authorization": "Bearer tool-token"})
            payload = ledger.json()
            assert payload["opened_pages"][0]["content_type"] == "text/html"
            assert payload["opened_pages"][0]["date_confidence"] > 0.0
            assert payload["opened_pages"][0]["support_spans"] == [
                "Current policy update confirms incident summaries are published quarterly."
            ]


async def test_research_tool_service_live_open_page_marks_unsupported_content(monkeypatch):
    monkeypatch.setenv("EIREL_RESEARCH_TOOL_API_TOKEN", "tool-token")
    monkeypatch.setenv("EIREL_BRAVE_SEARCH_API_KEY", "brave-token")

    def _search_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {
                            "title": "PDF report",
                            "url": "https://example.com/report.pdf",
                            "description": "Binary report",
                        }
                    ]
                }
            },
        )

    def _fetch_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/pdf"},
            content=b"%PDF-1.4 binary",
            request=request,
        )

    app = create_app(
        backend="brave_live_web",
        search_transport=httpx.MockTransport(_search_handler),
        fetch_transport=httpx.MockTransport(_fetch_handler),
    )
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            headers = {
                "Authorization": "Bearer tool-token",
                "X-Eirel-Job-Id": "job-brave-3",
                "X-Eirel-Max-Requests": "4",
            }
            search = await client.post(
                "/v1/search",
                json={"query": "report", "top_k": 1},
                headers=headers,
            )
            document_id = search.json()["documents"][0]["document_id"]
            opened = await client.post(
                "/v1/open-page",
                json={"document_id": document_id},
                headers=headers,
            )
            assert opened.status_code == 200
            payload = opened.json()
            assert payload["content"] == ""
            assert payload["content_type"] == "application/pdf"
            assert payload["extraction_confidence"] == 0.0
            assert payload["metadata"]["open_error_kind"] == "unsupported_content_type"


async def test_per_job_scoped_token_auth(monkeypatch):
    monkeypatch.setenv("EIREL_RESEARCH_TOOL_API_TOKEN", "master-token")
    app = create_app(
        ResearchCatalogStore(
            documents={
                "doc-1": ResearchDocumentRecord(
                    document_id="doc-1",
                    title="Test doc",
                    url="https://example.com/test",
                    snippet="Test snippet.",
                    content="Test content.",
                )
            }
        ),
        backend="catalog",
    )
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            job_id = "miner-sub-123"
            job_token = generate_job_token("master-token", job_id)

            # Per-job scoped token with matching job ID works
            scoped_headers = {
                "Authorization": f"Bearer {job_token}",
                "X-Eirel-Job-Id": job_id,
                "X-Eirel-Max-Requests": "5",
            }
            search = await client.post(
                "/v1/search",
                json={"query": "test", "top_k": 1},
                headers=scoped_headers,
            )
            assert search.status_code == 200

            # Master token still works
            master_headers = {
                "Authorization": "Bearer master-token",
                "X-Eirel-Job-Id": job_id,
                "X-Eirel-Max-Requests": "5",
            }
            search2 = await client.post(
                "/v1/search",
                json={"query": "test", "top_k": 1},
                headers=master_headers,
            )
            assert search2.status_code == 200

            # Per-job token with wrong job ID is rejected
            wrong_job_headers = {
                "Authorization": f"Bearer {job_token}",
                "X-Eirel-Job-Id": "wrong-job-id",
                "X-Eirel-Max-Requests": "5",
            }
            rejected = await client.post(
                "/v1/search",
                json={"query": "test", "top_k": 1},
                headers=wrong_job_headers,
            )
            assert rejected.status_code == 401

            # Random token is rejected
            bad_token_headers = {
                "Authorization": "Bearer totally-wrong-token",
                "X-Eirel-Job-Id": job_id,
                "X-Eirel-Max-Requests": "5",
            }
            rejected2 = await client.post(
                "/v1/search",
                json={"query": "test", "top_k": 1},
                headers=bad_token_headers,
            )
            assert rejected2.status_code == 401
