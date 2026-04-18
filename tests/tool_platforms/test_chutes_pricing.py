from __future__ import annotations

import httpx
import pytest

from shared.common.tool_pricing import (
    LLMPrice,
    get_dynamic_pricing,
    llm_cost_for,
    update_dynamic_pricing,
)
from tool_platforms.provider_proxy.chutes_pricing import (
    _parse_models_response,
    fetch_chutes_pricing,
    refresh_chutes_pricing_once,
)


@pytest.fixture(autouse=True)
def _clear_dynamic_pricing():
    """Every test starts with an empty overlay and ends the same way."""
    update_dynamic_pricing({})
    yield
    update_dynamic_pricing({})


def _fake_models_payload() -> dict:
    """Mirrors the real shape from https://llm.chutes.ai/v1/models ."""
    return {
        "object": "list",
        "data": [
            {
                "id": "moonshotai/Kimi-K2.5-TEE",
                "price": {
                    "input":  {"usd": 0.3827, "tao": 0.001477},
                    "output": {"usd": 1.72,   "tao": 0.00664},
                    "input_cache_read": {"usd": 0.19135, "tao": 0.000738},
                },
            },
            {
                "id": "Qwen/Qwen3-32B-TEE",
                "price": {
                    "input":  {"usd": 0.08, "tao": 0.000309},
                    "output": {"usd": 0.24, "tao": 0.000927},
                },
            },
            # Malformed row — must be skipped, not crash the parser.
            {"id": "broken/model", "price": {"input": {"usd": "not-a-number"}}},
            # Row without an id — skipped.
            {"price": {"input": {"usd": 1.0}, "output": {"usd": 2.0}}},
        ],
    }


# -- _parse_models_response -------------------------------------------------


def test_parse_extracts_input_and_output_usd_per_mtok():
    parsed = _parse_models_response(_fake_models_payload())
    assert parsed["chutes:moonshotai/Kimi-K2.5-TEE"] == LLMPrice(0.3827, 1.72)
    assert parsed["chutes:Qwen/Qwen3-32B-TEE"] == LLMPrice(0.08, 0.24)


def test_parse_skips_malformed_rows_without_raising():
    parsed = _parse_models_response(_fake_models_payload())
    # Only the two well-formed rows land; broken/model and id-less row dropped.
    assert "chutes:broken/model" not in parsed
    assert len(parsed) == 2


def test_parse_rejects_non_dict_root():
    with pytest.raises(ValueError):
        _parse_models_response([])


def test_parse_rejects_missing_data_array():
    with pytest.raises(ValueError):
        _parse_models_response({"object": "list"})


# -- fetch_chutes_pricing ---------------------------------------------------


async def test_fetch_hits_endpoint_and_maps_to_llmprice():
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_fake_models_payload())

    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        result = await fetch_chutes_pricing(
            url="http://mock/v1/models", client=client,
        )
    assert result["chutes:moonshotai/Kimi-K2.5-TEE"].input_per_mtok_usd == pytest.approx(0.3827)
    assert result["chutes:moonshotai/Kimi-K2.5-TEE"].output_per_mtok_usd == pytest.approx(1.72)


async def test_fetch_raises_on_http_error():
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(httpx.HTTPError):
            await fetch_chutes_pricing(
                url="http://mock/v1/models", client=client,
            )


# -- refresh_chutes_pricing_once -------------------------------------------


async def test_refresh_replaces_overlay_on_success(monkeypatch):
    # Pre-seed an old overlay that the refresh must replace entirely.
    update_dynamic_pricing({"chutes:stale/model": LLMPrice(99.0, 99.0)})

    async def _fake_fetch(*, url, timeout_seconds, client=None):
        return {"chutes:moonshotai/Kimi-K2.5-TEE": LLMPrice(0.3827, 1.72)}

    monkeypatch.setattr(
        "tool_platforms.provider_proxy.chutes_pricing.fetch_chutes_pricing",
        _fake_fetch,
    )
    n = await refresh_chutes_pricing_once()
    assert n == 1
    overlay = get_dynamic_pricing()
    assert "chutes:moonshotai/Kimi-K2.5-TEE" in overlay
    # Stale entry replaced — refresh is a full swap, not a merge, so a
    # removed model doesn't linger at a stale price.
    assert "chutes:stale/model" not in overlay


async def test_refresh_preserves_previous_overlay_on_network_failure(monkeypatch):
    # Seed a valid overlay, then force fetch to fail.
    update_dynamic_pricing({"chutes:kept/model": LLMPrice(1.0, 2.0)})

    async def _failing_fetch(**kw):
        raise httpx.ConnectError("simulated network down")

    monkeypatch.setattr(
        "tool_platforms.provider_proxy.chutes_pricing.fetch_chutes_pricing",
        _failing_fetch,
    )
    n = await refresh_chutes_pricing_once()
    assert n == 0
    # Overlay still present — we don't wipe it on a transient failure
    # because that would make every subsequent call fall back to the
    # possibly-stale static table.
    assert get_dynamic_pricing()["chutes:kept/model"].input_per_mtok_usd == pytest.approx(1.0)


async def test_refresh_preserves_overlay_on_empty_response(monkeypatch):
    update_dynamic_pricing({"chutes:kept/model": LLMPrice(1.0, 2.0)})

    async def _empty_fetch(**kw):
        return {}

    monkeypatch.setattr(
        "tool_platforms.provider_proxy.chutes_pricing.fetch_chutes_pricing",
        _empty_fetch,
    )
    n = await refresh_chutes_pricing_once()
    assert n == 0
    assert "chutes:kept/model" in get_dynamic_pricing()


# -- integration: dynamic overlay beats the static table -------------------


def test_llm_cost_for_uses_dynamic_overlay_when_present():
    # Static table has Kimi at $0.3827/M input.  Simulate Chutes raising
    # the rate to $0.50/M.  llm_cost_for must reflect the new rate
    # immediately after the overlay is updated.
    update_dynamic_pricing({
        "chutes:moonshotai/Kimi-K2.5-TEE": LLMPrice(0.50, 2.0),
    })
    cost = llm_cost_for(
        provider="chutes",
        model="moonshotai/Kimi-K2.5-TEE",
        prompt_tokens=1_000_000,
        completion_tokens=0,
    )
    assert cost == pytest.approx(0.50)


def test_llm_cost_for_falls_back_to_static_when_overlay_missing_model():
    # Overlay doesn't include Kimi; static table does.
    update_dynamic_pricing({"chutes:other/model": LLMPrice(1.0, 2.0)})
    cost = llm_cost_for(
        provider="chutes",
        model="moonshotai/Kimi-K2.5-TEE",
        prompt_tokens=1_000_000,
        completion_tokens=0,
    )
    # Static rate for Kimi.
    assert cost == pytest.approx(0.3827)
