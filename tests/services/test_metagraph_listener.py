from __future__ import annotations

import json

from httpx import ASGITransport, AsyncClient

from validation.metagraph_listener.main import app


async def test_metagraph_listener_syncs_snapshot(monkeypatch, tmp_path):
    snapshot_path = tmp_path / "metagraph.json"
    snapshot_path.write_text(
        json.dumps(
            [
                {"hotkey": "neuron-hotkey-1", "uid": 0},
                {"hotkey": "neuron-hotkey-2", "uid": 1},
            ]
        )
    )
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'metagraph.db'}")
    monkeypatch.setenv("METAGRAPH_SNAPSHOT_PATH", str(snapshot_path))

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post("/v1/sync/run")
            assert response.status_code == 200
            payload = response.json()
            assert payload["status"] == "success"
            assert payload["neuron_count"] == 2

            status = await client.get("/v1/sync/status")
            assert status.status_code == 200
            assert status.json()["status"] == "success"


async def test_metagraph_listener_syncs_inline_payload(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'metagraph-inline.db'}")
    monkeypatch.setenv("METAGRAPH_SNAPSHOT_PATH", "")

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post(
                "/v1/sync/run",
                json={
                    "neurons": [
                        {"hotkey": "neuron-inline-hotkey", "uid": 7},
                    ]
                },
            )
            assert response.status_code == 200
            payload = response.json()
            assert payload["status"] == "success"
            assert payload["neuron_count"] == 1
