from __future__ import annotations

import httpx
import pytest

from tool_platforms._charge_tool import charge_tool_cost


class _RecordingTransport(httpx.AsyncBaseTransport):
    def __init__(self, status_code: int = 200) -> None:
        self.calls: list[dict] = []
        self._status_code = status_code

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        import json
        self.calls.append(
            {
                "url": str(request.url),
                "method": request.method,
                "headers": dict(request.headers),
                "body": json.loads(request.content.decode() or "{}"),
            }
        )
        return httpx.Response(self._status_code, json={"cost_usd_used": 0.003})


@pytest.fixture
def _patch_transport(monkeypatch):
    """Make httpx.AsyncClient use our mock transport regardless of kwargs."""
    recorder = _RecordingTransport()
    import tool_platforms._charge_tool as mod
    original = mod.httpx.AsyncClient

    def _patched(*args, **kwargs):
        kwargs["transport"] = recorder
        return original(*args, **kwargs)

    monkeypatch.setattr(mod.httpx, "AsyncClient", _patched)
    return recorder


async def test_posts_charge_tool_with_correct_payload(monkeypatch, _patch_transport):
    monkeypatch.setenv("EIREL_PROVIDER_PROXY_URL", "http://proxy")
    monkeypatch.setenv("EIREL_PROVIDER_PROXY_TOKEN", "proxy-token")

    await charge_tool_cost(
        job_id="miner-abc", tool_name="web_search", amount_usd=0.003,
    )

    assert len(_patch_transport.calls) == 1
    call = _patch_transport.calls[0]
    assert call["url"] == "http://proxy/v1/jobs/miner-abc/charge_tool"
    assert call["method"] == "POST"
    assert call["body"] == {"tool_name": "web_search", "amount_usd": 0.003}
    assert call["headers"]["authorization"] == "Bearer proxy-token"


async def test_skips_when_amount_is_zero(monkeypatch, _patch_transport):
    monkeypatch.setenv("EIREL_PROVIDER_PROXY_URL", "http://proxy")
    await charge_tool_cost(job_id="miner-abc", tool_name="sandbox", amount_usd=0.0)
    assert _patch_transport.calls == []


async def test_skips_when_job_id_missing(monkeypatch, _patch_transport):
    monkeypatch.setenv("EIREL_PROVIDER_PROXY_URL", "http://proxy")
    await charge_tool_cost(job_id=None, tool_name="web_search", amount_usd=0.003)
    assert _patch_transport.calls == []


async def test_skips_when_proxy_url_not_configured(monkeypatch, _patch_transport):
    monkeypatch.delenv("EIREL_PROVIDER_PROXY_URL", raising=False)
    await charge_tool_cost(job_id="miner-abc", tool_name="web_search", amount_usd=0.003)
    assert _patch_transport.calls == []


async def test_swallows_proxy_errors(monkeypatch):
    """Proxy outages must not break the tool service response."""
    monkeypatch.setenv("EIREL_PROVIDER_PROXY_URL", "http://proxy")
    import tool_platforms._charge_tool as mod
    original = mod.httpx.AsyncClient

    class _FailingTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            raise httpx.ConnectError("simulated outage")

    def _patched(*args, **kwargs):
        kwargs["transport"] = _FailingTransport()
        return original(*args, **kwargs)

    monkeypatch.setattr(mod.httpx, "AsyncClient", _patched)

    # Must not raise.
    await charge_tool_cost(job_id="miner-abc", tool_name="web_search", amount_usd=0.003)
