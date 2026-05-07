"""Sandbox session persistence: variables / imports survive across calls.

A long-lived worker subprocess keyed by ``session_id`` lets a
multi-step analysis build on prior state. The single test file
exercises three things end-to-end through the FastAPI app:

1. A variable defined in call 1 is readable in call 2 with the same
   session_id.
2. A different session_id sees a fresh kernel — the prior variable is
   not in scope.
3. The session_id round-trips through the response so the orchestrator
   can echo it back on subsequent turns.
"""
from __future__ import annotations

from httpx import ASGITransport, AsyncClient

from tool_platforms.sandbox_tool_service.app import create_app
from tool_platforms.sandbox_tool_service.backends import SubprocessBackend


async def _post(client: AsyncClient, code: str, session_id: str, job_id: str = "job-sess"):
    return await client.post(
        "/v1/execute",
        json={"code": code, "session_id": session_id},
        headers={
            "X-Eirel-Job-Id": job_id,
            "X-Eirel-Max-Requests": "20",
        },
    )


async def test_session_persists_variable_across_calls():
    backend = SubprocessBackend()
    app = create_app(backend=backend)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            r1 = await _post(client, "x = 42", "sess-A")
            assert r1.status_code == 200, r1.text
            assert r1.json()["exit_code"] == 0

            r2 = await _post(client, "print(x * 2)", "sess-A")
            assert r2.status_code == 200, r2.text
            body2 = r2.json()
            assert body2["exit_code"] == 0
            assert body2["stdout"].strip() == "84"
            assert body2["session_id"] == "sess-A"


async def test_session_imports_persist_across_calls():
    backend = SubprocessBackend()
    app = create_app(backend=backend)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            r1 = await _post(client, "import math", "sess-imp")
            assert r1.status_code == 200
            assert r1.json()["exit_code"] == 0

            r2 = await _post(client, "print(round(math.sqrt(2), 6))", "sess-imp")
            body2 = r2.json()
            assert body2["exit_code"] == 0
            assert body2["stdout"].strip() == "1.414214"


async def test_different_session_ids_get_fresh_kernels():
    backend = SubprocessBackend()
    app = create_app(backend=backend)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            r1 = await _post(client, "secret = 'shared'", "sess-X")
            assert r1.status_code == 200
            assert r1.json()["exit_code"] == 0

            r2 = await _post(client, "print(secret)", "sess-Y")
            body2 = r2.json()
            assert body2["exit_code"] != 0
            assert "NameError" in body2["stderr"]
            assert "shared" not in body2["stdout"]


async def test_oneshot_path_unchanged_when_no_session_id():
    """No session_id → existing single-shot behavior (no kernel state).

    Two calls without session_id should not share variables — call 2
    raises NameError. Confirms the legacy code path still works.
    """
    backend = SubprocessBackend()
    app = create_app(backend=backend)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            headers = {"X-Eirel-Job-Id": "job-oneshot", "X-Eirel-Max-Requests": "5"}
            await client.post(
                "/v1/execute",
                json={"code": "y = 9"},
                headers=headers,
            )
            r2 = await client.post(
                "/v1/execute",
                json={"code": "print(y)"},
                headers=headers,
            )
            body2 = r2.json()
            assert body2["exit_code"] != 0
            assert "NameError" in body2["stderr"]
            assert body2.get("session_id") is None


async def test_session_kernel_evicted_on_close():
    """SubprocessBackend.close() kills active sessions.

    The lifespan teardown calls backend.close(); after the lifespan
    exits, the subprocess should not still be running. Verify by
    checking the tracked session dict is empty post-close.
    """
    backend = SubprocessBackend()
    app = create_app(backend=backend)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            await _post(client, "z = 1", "sess-Z")
        assert "sess-Z" in backend._sessions
    # After lifespan exit, close() ran and emptied the session map.
    assert backend._sessions == {}
