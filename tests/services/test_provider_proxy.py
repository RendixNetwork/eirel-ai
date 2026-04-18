from __future__ import annotations

import importlib
import json

import pytest
from httpx import ASGITransport, AsyncClient

from tool_platforms.provider_proxy.app import (
    create_app,
    _build_provider_request,
    _openai_payload_to_anthropic,
    _anthropic_response_to_openai,
    _extract_upstream_usage,
)


@pytest.fixture(autouse=True)
def _patch_redis_client(monkeypatch):
    """Back the provider-proxy store with fakeredis for every test in this
    module so no test needs a live Redis container.  Lua falls back to a
    Python path automatically when fakeredis is used (see redis_store.py).
    Also suppress the Chutes pricing refresh loop so tests don't hit the
    live llm.chutes.ai endpoint."""
    import fakeredis.aioredis
    proxy_app = importlib.import_module("tool_platforms.provider_proxy.app")
    monkeypatch.setattr(
        proxy_app,
        "_make_redis_client",
        lambda _url: fakeredis.aioredis.FakeRedis(decode_responses=True),
    )
    monkeypatch.setenv("EIREL_CHUTES_PRICING_REFRESH_ENABLED", "false")


@pytest.mark.asyncio
async def test_provider_proxy_enforces_request_budget(monkeypatch):
    monkeypatch.setenv("EIREL_PROVIDER_PROXY_MASTER_TOKEN", "master-token")
    monkeypatch.setenv("EIREL_PROVIDER_CHUTES_API_KEY", "proxy-key")
    monkeypatch.setenv("EIREL_PROVIDER_CHUTES_BASE_URL", "http://provider.local/v1/chat/completions")

    proxy_app = importlib.import_module("tool_platforms.provider_proxy.app")

    def fake_dispatch_provider_request(*, provider, model, payload, timeout_seconds):
        return {
            "id": "resp-1",
            "provider": provider,
            "model": model,
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        }

    monkeypatch.setattr(proxy_app, "_dispatch_provider_request", fake_dispatch_provider_request)

    app = create_app()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            payload = {
                "provider": "chutes",
                "model": "chutes-test",
                "payload": {"messages": [{"role": "user", "content": "hello"}]},
                "max_requests": 1,
                "max_total_tokens": 1000,
                "max_wall_clock_seconds": 300,
                "per_request_timeout_seconds": 30,
            }
            headers = {
                "Authorization": "Bearer master-token",
                "X-Eirel-Job-Id": "job-1",
                "X-Eirel-Run-Budget-Usd": "30.0",
            }
            first = await client.post("/v1/chat/completions", json=payload, headers=headers)
            second = await client.post("/v1/chat/completions", json=payload, headers=headers)

            assert first.status_code == 200
            assert second.status_code == 200

            metrics = await client.get("/metrics")
            assert metrics.status_code == 200
            assert 'eirel_provider_proxy_requests_total 2' in metrics.text
            assert 'eirel_provider_proxy_provider_requests_total{provider="chutes"} 2' in metrics.text

            summary = await client.get("/v1/operators/summary")
            assert summary.status_code == 200
            summary_payload = summary.json()
            assert summary_payload["active_job_count"] == 1
            assert "job-1" in summary_payload["active_jobs"]

            usage = await client.get(
                "/v1/jobs/job-1/usage",
                headers={"Authorization": "Bearer master-token"},
            )
            assert usage.status_code == 200
            usage_payload = usage.json()
            assert usage_payload["request_count"] == 2
            assert usage_payload["provider_request_counts"]["chutes"] == 2
            assert usage_payload["model_request_counts"]["chutes:chutes-test"] == 2


@pytest.mark.asyncio
async def test_provider_proxy_rejects_unsupported_provider(monkeypatch):
    monkeypatch.setenv("EIREL_PROVIDER_PROXY_MASTER_TOKEN", "master-token")
    app = create_app()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "provider": "unknown",
                    "model": "bad",
                    "payload": {"messages": []},
                },
                headers={
                    "Authorization": "Bearer master-token",
                    "X-Eirel-Job-Id": "job-2",
                },
            )
            assert response.status_code == 400


@pytest.mark.asyncio
async def test_provider_proxy_mock_mode_handles_missing_provider_key(monkeypatch):
    monkeypatch.setenv("EIREL_PROVIDER_PROXY_MASTER_TOKEN", "master-token")
    monkeypatch.delenv("EIREL_PROVIDER_CHUTES_API_KEY", raising=False)
    monkeypatch.setenv("EIREL_PROVIDER_PROXY_ALLOW_MOCK", "true")

    app = create_app()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "provider": "chutes",
                    "model": "chutes-test",
                    "payload": {"messages": [{"role": "user", "content": "hello from staging"}]},
                },
                headers={
                    "Authorization": "Bearer master-token",
                    "X-Eirel-Job-Id": "job-mock",
                    "X-Eirel-Run-Budget-Usd": "30.0",
                },
            )
            assert response.status_code == 200
            payload = response.json()
            assert payload["upstream_response"]["choices"][0]["message"]["content"].startswith(
                "Staging mock provider response"
            )


# ── _build_provider_request ──────────────────────────────────────────────────


def test_build_provider_request_openai_uses_bearer_auth():
    headers, body = _build_provider_request(
        provider="openai",
        model="gpt-4o",
        api_key="sk-test",
        payload={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert headers["Authorization"] == "Bearer sk-test"
    assert headers["Content-Type"] == "application/json"
    assert "x-api-key" not in headers
    assert body["model"] == "gpt-4o"
    assert body["messages"] == [{"role": "user", "content": "hi"}]


def test_build_provider_request_anthropic_uses_x_api_key():
    headers, body = _build_provider_request(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        api_key="sk-ant-test",
        payload={
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hello"},
            ],
            "max_tokens": 512,
        },
    )
    assert headers["x-api-key"] == "sk-ant-test"
    assert headers["anthropic-version"] == "2023-06-01"
    assert "Authorization" not in headers
    # Body should be Anthropic format (system extracted to top-level)
    assert body["system"] == "You are helpful."
    assert body["model"] == "claude-sonnet-4-20250514"
    assert all(m["role"] != "system" for m in body["messages"])
    assert body["max_tokens"] == 512


def test_build_provider_request_openrouter_and_chutes_use_bearer():
    for provider in ("openrouter", "chutes"):
        headers, body = _build_provider_request(
            provider=provider,
            model="some-model",
            api_key="key-123",
            payload={"messages": [{"role": "user", "content": "test"}]},
        )
        assert headers["Authorization"] == "Bearer key-123"
        assert body["model"] == "some-model"


# ── _openai_payload_to_anthropic ─────────────────────────────────────────────


def test_openai_to_anthropic_extracts_system():
    result = _openai_payload_to_anthropic("claude-sonnet-4-20250514", {
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "What is 2+2?"},
        ],
        "max_tokens": 100,
    })
    assert result["system"] == "Be concise."
    assert len(result["messages"]) == 1
    assert result["messages"][0] == {"role": "user", "content": "What is 2+2?"}
    assert result["max_tokens"] == 100
    assert result["model"] == "claude-sonnet-4-20250514"


def test_openai_to_anthropic_multiple_system_messages():
    result = _openai_payload_to_anthropic("model-x", {
        "messages": [
            {"role": "system", "content": "First."},
            {"role": "system", "content": "Second."},
            {"role": "user", "content": "Go"},
        ],
    })
    assert result["system"] == "First.\nSecond."
    assert len(result["messages"]) == 1


def test_openai_to_anthropic_defaults_max_tokens():
    result = _openai_payload_to_anthropic("model-x", {
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert result["max_tokens"] == 4096


def test_openai_to_anthropic_passes_temperature_and_stop():
    result = _openai_payload_to_anthropic("model-x", {
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.7,
        "top_p": 0.9,
        "stop": ["END", "STOP"],
    })
    assert result["temperature"] == 0.7
    assert result["top_p"] == 0.9
    assert result["stop_sequences"] == ["END", "STOP"]


def test_openai_to_anthropic_stop_string_to_list():
    result = _openai_payload_to_anthropic("model-x", {
        "messages": [{"role": "user", "content": "hi"}],
        "stop": "END",
    })
    assert result["stop_sequences"] == ["END"]


def test_openai_to_anthropic_no_system_message():
    result = _openai_payload_to_anthropic("model-x", {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"},
        ],
    })
    assert "system" not in result
    assert len(result["messages"]) == 3


# ── _anthropic_response_to_openai ────────────────────────────────────────────


def test_anthropic_response_to_openai_text_content():
    anthropic_response = {
        "id": "msg_123",
        "model": "claude-sonnet-4-20250514",
        "content": [{"type": "text", "text": "Hello world"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    result = _anthropic_response_to_openai(anthropic_response)
    assert result["id"] == "msg_123"
    assert result["object"] == "chat.completion"
    assert result["model"] == "claude-sonnet-4-20250514"
    assert result["choices"][0]["message"]["content"] == "Hello world"
    assert result["choices"][0]["finish_reason"] == "end_turn"
    assert result["usage"]["prompt_tokens"] == 10
    assert result["usage"]["completion_tokens"] == 5
    assert result["usage"]["total_tokens"] == 15
    assert "tool_calls" not in result["choices"][0]["message"]


def test_anthropic_response_to_openai_tool_use():
    anthropic_response = {
        "id": "msg_456",
        "model": "claude-sonnet-4-20250514",
        "content": [
            {"type": "text", "text": "Let me search for that."},
            {
                "type": "tool_use",
                "id": "call_abc",
                "name": "web_search",
                "input": {"query": "latest news"},
            },
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 20, "output_tokens": 15},
    }
    result = _anthropic_response_to_openai(anthropic_response)
    message = result["choices"][0]["message"]
    assert message["content"] == "Let me search for that."
    assert len(message["tool_calls"]) == 1
    tc = message["tool_calls"][0]
    assert tc["id"] == "call_abc"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "web_search"
    assert json.loads(tc["function"]["arguments"]) == {"query": "latest news"}
    assert result["choices"][0]["finish_reason"] == "tool_use"


def test_anthropic_response_to_openai_empty_content():
    result = _anthropic_response_to_openai({"content": [], "stop_reason": "stop", "usage": {}})
    assert result["choices"][0]["message"]["content"] == ""
    assert "tool_calls" not in result["choices"][0]["message"]


# ── _extract_upstream_usage ──────────────────────────────────────────────────


def test_extract_usage_openai_format():
    response = {
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    }
    usage = _extract_upstream_usage("openai", response)
    assert usage == {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "reasoning_tokens": 0}


def test_extract_usage_anthropic_already_normalized():
    # After _anthropic_response_to_openai, usage is already in OpenAI format
    response = {
        "usage": {"prompt_tokens": 80, "completion_tokens": 30, "total_tokens": 110}
    }
    usage = _extract_upstream_usage("anthropic", response)
    assert usage == {"prompt_tokens": 80, "completion_tokens": 30, "total_tokens": 110, "reasoning_tokens": 0}


def test_extract_usage_missing_usage_key():
    usage = _extract_upstream_usage("openai", {})
    assert usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0}


def test_extract_usage_non_dict_usage():
    usage = _extract_upstream_usage("openai", {"usage": "not-a-dict"})
    assert usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0}


def test_extract_usage_computes_total_if_zero():
    response = {
        "usage": {"prompt_tokens": 40, "completion_tokens": 20, "total_tokens": 0}
    }
    usage = _extract_upstream_usage("openai", response)
    assert usage["total_tokens"] == 60


# ── Integration: Anthropic dispatch through app ──────────────────────────────


@pytest.mark.asyncio
async def test_provider_proxy_rejects_blocked_providers(monkeypatch):
    """openai, anthropic, openrouter are blocked — only chutes is allowed."""
    monkeypatch.setenv("EIREL_PROVIDER_PROXY_MASTER_TOKEN", "master-token")
    app = create_app()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            for blocked_provider in ("openai", "anthropic", "openrouter"):
                response = await client.post(
                    "/v1/chat/completions",
                    json={
                        "provider": blocked_provider,
                        "model": "any-model",
                        "payload": {"messages": [{"role": "user", "content": "hi"}]},
                    },
                    headers={
                        "Authorization": "Bearer master-token",
                        "X-Eirel-Job-Id": "job-blocked",
                    },
                )
                assert response.status_code == 400, f"{blocked_provider} should be blocked"


@pytest.mark.asyncio
async def test_provider_proxy_anthropic_dispatch_full_flow(monkeypatch):
    # Anthropic is currently blocked but the conversion logic is kept for future use.
    # Temporarily allow it in this test to verify the full dispatch path.
    proxy_app = importlib.import_module("tool_platforms.provider_proxy.app")
    monkeypatch.setattr(proxy_app, "SUPPORTED_PROVIDER_IDS", ("chutes", "anthropic"))
    monkeypatch.setenv("EIREL_PROVIDER_PROXY_MASTER_TOKEN", "master-token")
    monkeypatch.setenv("EIREL_PROVIDER_ANTHROPIC_API_KEY", "sk-ant-key")

    proxy_app = importlib.import_module("tool_platforms.provider_proxy.app")

    captured_requests: list[tuple[dict, dict]] = []

    def fake_dispatch(*, provider, model, payload, timeout_seconds):
        # Capture what was sent, then return Anthropic-style response already normalized
        headers, body = _build_provider_request(
            provider=provider, model=model, api_key="fake", payload=payload,
        )
        captured_requests.append((headers, body))
        # Return already-normalized OpenAI format (as _dispatch_provider_request does)
        return {
            "id": "msg_test",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Anthropic response"},
                    "finish_reason": "end_turn",
                }
            ],
            "usage": {"prompt_tokens": 25, "completion_tokens": 10, "total_tokens": 35},
        }

    monkeypatch.setattr(proxy_app, "_dispatch_provider_request", fake_dispatch)

    app = create_app()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-20250514",
                    "payload": {
                        "messages": [
                            {"role": "system", "content": "Be helpful."},
                            {"role": "user", "content": "Hello"},
                        ],
                        "max_tokens": 256,
                    },
                },
                headers={
                    "Authorization": "Bearer master-token",
                    "X-Eirel-Job-Id": "job-ant",
                    "X-Eirel-Run-Budget-Usd": "30.0",
                },
            )

    assert response.status_code == 200
    body = response.json()

    # Verify response includes actual token usage and latency
    usage = body["usage"]
    assert usage["actual_prompt_tokens"] == 25
    assert usage["actual_completion_tokens"] == 10
    assert usage["actual_total_tokens"] == 35
    assert usage["latency_seconds"] >= 0.0
    assert usage["provider"] == "anthropic"
    assert usage["model"] == "claude-sonnet-4-20250514"

    # Verify the upstream response is passed through
    assert body["upstream_response"]["choices"][0]["message"]["content"] == "Anthropic response"

    # Verify dispatch saw Anthropic headers/body
    assert len(captured_requests) == 1
    headers, req_body = captured_requests[0]
    assert headers["x-api-key"] == "fake"
    assert headers["anthropic-version"] == "2023-06-01"
    assert "Authorization" not in headers
    assert req_body["system"] == "Be helpful."
    assert req_body["max_tokens"] == 256
    assert all(m["role"] != "system" for m in req_body["messages"])
