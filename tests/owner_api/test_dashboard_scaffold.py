from __future__ import annotations

from httpx import AsyncClient

from control_plane.owner_api.app import app
from control_plane.owner_api.dashboard import queries
from control_plane.owner_api.dashboard.cache import TTLCache


EXPECTED_PATHS = {
    ("GET", "/api/v1/dashboard/overview"),
    ("GET", "/api/v1/dashboard/families"),
    ("GET", "/api/v1/dashboard/leaderboard"),
    ("GET", "/api/v1/dashboard/miners/{hotkey}"),
    ("GET", "/api/v1/dashboard/miners/{hotkey}/runs"),
    ("GET", "/api/v1/dashboard/miners/{hotkey}/runs/{run_id}"),
}


def test_dashboard_router_registers_all_six_endpoints():
    registered = {
        (method, route.path)
        for route in app.routes
        if hasattr(route, "path") and route.path.startswith("/api/v1/dashboard")
        for method in getattr(route, "methods", ()) or ()
    }
    missing = EXPECTED_PATHS - registered
    assert not missing, f"missing dashboard routes: {missing}"


def test_hotkey_short_uses_first_four_last_three():
    assert queries.shorten_hotkey("5FHneW46xGXgs5AMrxvJABC") == "5FHn...ABC"
    assert queries.shorten_hotkey("short") == "short"


def test_ttl_cache_expires_entries():
    cache: TTLCache[int] = TTLCache(default_ttl_seconds=1.0)
    cache.set(("k",), 42, ttl=10.0)
    assert cache.get(("k",)) == 42
    cache.set(("k2",), 99, ttl=0.0)
    assert cache.get(("k2",)) is None


async def test_dashboard_endpoints_return_501_until_phase_1b(client: AsyncClient):
    for method, path in EXPECTED_PATHS:
        concrete = path.format(hotkey="5FH...abc", run_id="run-x")
        params = {"family_id": "analyst"} if "family_id" in str(path) or path.endswith(("/leaderboard", "/runs")) or "/miners/" in path else None
        r = await client.request(method, concrete, params=params or {})
        assert r.status_code == 501, f"{method} {concrete} returned {r.status_code}: {r.text}"
