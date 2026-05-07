"""Tests for the URL-fetch tool service."""
from __future__ import annotations

import asyncio

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from tool_platforms.url_fetch_tool_service.app import create_app
from tool_platforms.url_fetch_tool_service.extractor import extract_text
from tool_platforms.url_fetch_tool_service.ssrf import (
    UrlFetchSSRFError,
    validate_url,
)


# -- Extractor (pure) -------------------------------------------------------


def test_extractor_strips_boilerplate_and_picks_main():
    html = """
    <html><head><title>My Page</title></head>
    <body>
      <nav>navigation links</nav>
      <header>header bar</header>
      <main>
        <h1>Main heading</h1>
        <p>This is the article body.</p>
        <a href="/about">About</a>
      </main>
      <footer>footer junk</footer>
      <script>console.log('hi')</script>
    </body></html>
    """
    result = extract_text(html, base_url="https://example.com/post")
    assert result.title == "My Page"
    assert "navigation links" not in result.content
    assert "header bar" not in result.content
    assert "footer junk" not in result.content
    assert "console.log" not in result.content
    assert "Main heading" in result.content
    assert "article body" in result.content
    assert any(link["href"] == "https://example.com/about" for link in result.links)


def test_extractor_falls_back_to_body_without_main():
    html = "<html><body><p>plain body content</p></body></html>"
    result = extract_text(html)
    assert "plain body content" in result.content


def test_extractor_caps_max_chars():
    html = "<html><body>" + ("x" * 200_000) + "</body></html>"
    result = extract_text(html, max_chars=1_000)
    assert len(result.content) <= 1_010  # 1000 + ellipsis


def test_extractor_skips_javascript_and_anchor_links():
    html = """
    <html><body>
      <a href="javascript:void(0)">skip</a>
      <a href="#section">skip too</a>
      <a href="https://example.com/real">keep</a>
    </body></html>
    """
    result = extract_text(html, base_url="https://example.com/x")
    hrefs = [link["href"] for link in result.links]
    assert "https://example.com/real" in hrefs
    assert all("javascript:" not in h for h in hrefs)
    assert all(not h.endswith("#section") for h in hrefs)


# -- SSRF guard -------------------------------------------------------------


def test_ssrf_blocks_unsupported_scheme():
    with pytest.raises(UrlFetchSSRFError, match="scheme"):
        validate_url("file:///etc/passwd")
    with pytest.raises(UrlFetchSSRFError, match="scheme"):
        validate_url("gopher://example.com/")


def test_ssrf_blocks_loopback_ip():
    with pytest.raises(UrlFetchSSRFError, match="private"):
        validate_url("http://127.0.0.1/")
    with pytest.raises(UrlFetchSSRFError, match="private"):
        validate_url("http://[::1]/")


def test_ssrf_blocks_rfc1918():
    with pytest.raises(UrlFetchSSRFError, match="private"):
        validate_url("http://10.0.0.5/")
    with pytest.raises(UrlFetchSSRFError, match="private"):
        validate_url("http://192.168.1.1/")
    with pytest.raises(UrlFetchSSRFError, match="private"):
        validate_url("http://172.16.0.1/")


def test_ssrf_blocks_known_metadata_hosts():
    with pytest.raises(UrlFetchSSRFError, match="blocklist"):
        validate_url("http://metadata.google.internal/computeMetadata/")


def test_ssrf_rejects_url_with_no_host():
    with pytest.raises(UrlFetchSSRFError, match="hostname"):
        validate_url("http:///")


# -- Service: auth ----------------------------------------------------------


@pytest.fixture
def fake_redis_ledger(monkeypatch):
    """Force the in-memory job ledger so tests don't need Redis."""
    monkeypatch.setenv("REDIS_URL", "")  # empty → InMemoryJobLedger


async def test_service_rejects_missing_auth(monkeypatch, fake_redis_ledger):
    monkeypatch.setenv("EIREL_URL_FETCH_API_TOKEN", "tok")
    app = create_app()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.post(
                "/v1/fetch",
                json={"url": "https://example.com"},
                headers={"X-Eirel-Job-Id": "job-1"},
            )
    assert resp.status_code == 401


async def test_service_accepts_master_bearer(monkeypatch, fake_redis_ledger):
    """Auth passes; SSRF then rejects example.com? No — example.com is public.
    But we don't actually want to hit the network in tests, so we use a stub
    transport. That's covered in the fetch-roundtrip tests below; here we
    just confirm auth passes and a downstream-failure doesn't 401."""
    monkeypatch.setenv("EIREL_URL_FETCH_API_TOKEN", "tok")
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, content=b"<html><body>ok</body></html>",
                                   headers={"content-type": "text/html"}),
    )
    app = create_app(transport=transport)
    async with app.router.lifespan_context(app):
        asgi = ASGITransport(app=app)
        async with AsyncClient(transport=asgi, base_url="http://testserver") as client:
            resp = await client.post(
                "/v1/fetch",
                json={"url": "https://example.com"},
                headers={
                    "Authorization": "Bearer tok",
                    "X-Eirel-Job-Id": "job-1",
                },
            )
    assert resp.status_code == 200, resp.text


# -- Service: SSRF integration ---------------------------------------------


async def test_service_ssrf_blocks_loopback(monkeypatch, fake_redis_ledger):
    monkeypatch.setenv("EIREL_URL_FETCH_API_TOKEN", "tok")
    app = create_app()
    async with app.router.lifespan_context(app):
        asgi = ASGITransport(app=app)
        async with AsyncClient(transport=asgi, base_url="http://testserver") as client:
            resp = await client.post(
                "/v1/fetch",
                json={"url": "http://127.0.0.1/"},
                headers={"Authorization": "Bearer tok", "X-Eirel-Job-Id": "job-1"},
            )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert detail["error"] == "ssrf_blocked"


# -- Service: fetch roundtrip -----------------------------------------------


async def test_service_html_roundtrip(monkeypatch, fake_redis_ledger):
    monkeypatch.setenv("EIREL_URL_FETCH_API_TOKEN", "tok")
    html = (
        b"<html><head><title>T</title></head>"
        b"<body><main><h1>Hello</h1><p>World.</p>"
        b"<a href='/next'>next</a></main></body></html>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=html, headers={"content-type": "text/html; charset=utf-8"},
        )

    transport = httpx.MockTransport(handler)
    app = create_app(transport=transport)
    async with app.router.lifespan_context(app):
        asgi = ASGITransport(app=app)
        async with AsyncClient(transport=asgi, base_url="http://testserver") as client:
            resp = await client.post(
                "/v1/fetch",
                json={"url": "https://example.com/post"},
                headers={"Authorization": "Bearer tok", "X-Eirel-Job-Id": "job-1"},
            )
    assert resp.status_code == 200
    body = resp.json()
    assert body["title"] == "T"
    assert "Hello" in body["content"]
    assert "World." in body["content"]
    assert body["status_code"] == 200
    assert body["content_type"].startswith("text/html")
    assert any(link["href"].endswith("/next") for link in body["links"])
    assert body["truncated"] is False


async def test_service_size_cap_truncates(monkeypatch, fake_redis_ledger):
    monkeypatch.setenv("EIREL_URL_FETCH_API_TOKEN", "tok")
    monkeypatch.setenv("EIREL_URL_FETCH_MAX_RESPONSE_BYTES", "256")
    big_body = b"<html><body>" + (b"X" * 10_000) + b"</body></html>"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=big_body, headers={"content-type": "text/html"},
        )

    transport = httpx.MockTransport(handler)
    app = create_app(transport=transport)
    async with app.router.lifespan_context(app):
        asgi = ASGITransport(app=app)
        async with AsyncClient(transport=asgi, base_url="http://testserver") as client:
            resp = await client.post(
                "/v1/fetch",
                json={"url": "https://example.com/big"},
                headers={"Authorization": "Bearer tok", "X-Eirel-Job-Id": "job-1"},
            )
    assert resp.status_code == 200
    body = resp.json()
    assert body["truncated"] is True
    assert body["bytes_read"] <= 512  # cap + last chunk overflow


async def test_service_per_host_rate_limit(monkeypatch, fake_redis_ledger):
    monkeypatch.setenv("EIREL_URL_FETCH_API_TOKEN", "tok")
    monkeypatch.setenv("EIREL_URL_FETCH_PER_HOST_RATE", "2")
    monkeypatch.setenv("EIREL_URL_FETCH_PER_HOST_WINDOW_SECONDS", "60")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html><body>x</body></html>",
                              headers={"content-type": "text/html"})

    transport = httpx.MockTransport(handler)
    app = create_app(transport=transport)
    async with app.router.lifespan_context(app):
        asgi = ASGITransport(app=app)
        async with AsyncClient(transport=asgi, base_url="http://testserver") as client:
            statuses = []
            for _ in range(4):
                resp = await client.post(
                    "/v1/fetch",
                    json={"url": "https://example.com/x"},
                    headers={
                        "Authorization": "Bearer tok",
                        "X-Eirel-Job-Id": "job-rl",
                    },
                )
                statuses.append(resp.status_code)
                if resp.status_code == 429:
                    body = resp.json()["detail"]
                    assert body["error"] == "per_host_rate_limit_exceeded"
    # First two within the window pass; subsequent ones rate-limit.
    assert statuses[:2] == [200, 200]
    assert 429 in statuses[2:]


async def test_service_quota_rejects_after_limit(monkeypatch, fake_redis_ledger):
    monkeypatch.setenv("EIREL_URL_FETCH_API_TOKEN", "tok")
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, content=b"<html><body>x</body></html>",
                                   headers={"content-type": "text/html"}),
    )
    app = create_app(transport=transport)
    async with app.router.lifespan_context(app):
        asgi = ASGITransport(app=app)
        async with AsyncClient(transport=asgi, base_url="http://testserver") as client:
            statuses = []
            for _ in range(3):
                resp = await client.post(
                    "/v1/fetch",
                    json={"url": "https://example.com/x"},
                    headers={
                        "Authorization": "Bearer tok",
                        "X-Eirel-Job-Id": "job-q",
                        "X-Eirel-Max-Requests": "2",
                    },
                )
                statuses.append(resp.status_code)
    assert statuses == [200, 200, 429]


async def test_service_upstream_error_returns_502(monkeypatch, fake_redis_ledger):
    monkeypatch.setenv("EIREL_URL_FETCH_API_TOKEN", "tok")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("upstream down")

    transport = httpx.MockTransport(handler)
    app = create_app(transport=transport)
    async with app.router.lifespan_context(app):
        asgi = ASGITransport(app=app)
        async with AsyncClient(transport=asgi, base_url="http://testserver") as client:
            resp = await client.post(
                "/v1/fetch",
                json={"url": "https://example.com/dead"},
                headers={"Authorization": "Bearer tok", "X-Eirel-Job-Id": "job-d"},
            )
    assert resp.status_code == 502
    assert resp.json()["detail"]["error"] == "fetch_failed"


async def test_service_non_html_returns_raw_text(monkeypatch, fake_redis_ledger):
    monkeypatch.setenv("EIREL_URL_FETCH_API_TOKEN", "tok")
    transport = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            content=b'{"k":"v"}',
            headers={"content-type": "application/json"},
        ),
    )
    app = create_app(transport=transport)
    async with app.router.lifespan_context(app):
        asgi = ASGITransport(app=app)
        async with AsyncClient(transport=asgi, base_url="http://testserver") as client:
            resp = await client.post(
                "/v1/fetch",
                json={"url": "https://example.com/api/data.json"},
                headers={"Authorization": "Bearer tok", "X-Eirel-Job-Id": "job-j"},
            )
    body = resp.json()
    assert body["content_type"].startswith("application/json")
    assert body["title"] == ""
    assert "k" in body["content"] and "v" in body["content"]
    assert body["links"] == []


async def test_service_healthz_metrics_unauthenticated(monkeypatch, fake_redis_ledger):
    monkeypatch.setenv("EIREL_URL_FETCH_API_TOKEN", "tok")
    app = create_app()
    async with app.router.lifespan_context(app):
        asgi = ASGITransport(app=app)
        async with AsyncClient(transport=asgi, base_url="http://testserver") as client:
            health = await client.get("/healthz")
            assert health.status_code == 200
            assert health.json()["status"] == "ok"
            metrics = await client.get("/metrics")
            assert metrics.status_code == 200
            assert "eirel_url_fetch_requests_total" in metrics.text
