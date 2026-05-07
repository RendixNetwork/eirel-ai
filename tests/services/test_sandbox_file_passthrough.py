"""Sandbox file I/O: files written in call 1 readable in call 2.

The session-persistent worker shares a working directory across calls.
Files written by user code show up in the response's ``files`` field
(base64-encoded for files under the per-file cap), and the same files
are still on disk for subsequent calls in the same session.
"""
from __future__ import annotations

import base64

from httpx import ASGITransport, AsyncClient

from tool_platforms.sandbox_tool_service.app import create_app
from tool_platforms.sandbox_tool_service.backends import SubprocessBackend


async def _post(client: AsyncClient, code: str, session_id: str | None = None):
    body: dict[str, object] = {"code": code}
    if session_id:
        body["session_id"] = session_id
    return await client.post(
        "/v1/execute",
        json=body,
        headers={
            "X-Eirel-Job-Id": "job-file",
            "X-Eirel-Max-Requests": "20",
        },
    )


async def test_files_written_during_session_returned_in_response():
    backend = SubprocessBackend()
    app = create_app(backend=backend)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            r1 = await _post(
                client,
                "open('out.txt', 'w').write('hello world')",
                "sess-files-A",
            )
            assert r1.status_code == 200, r1.text
            body1 = r1.json()
            assert body1["exit_code"] == 0
            files = body1["files"]
            written = next((f for f in files if f["path"] == "out.txt"), None)
            assert written is not None, files
            assert written["size"] == len("hello world")
            assert written["content_b64"] is not None
            assert (
                base64.b64decode(written["content_b64"]).decode("utf-8")
                == "hello world"
            )


async def test_file_persists_into_next_session_call():
    backend = SubprocessBackend()
    app = create_app(backend=backend)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            r1 = await _post(
                client,
                "open('shared.txt', 'w').write('persist me')",
                "sess-files-B",
            )
            assert r1.status_code == 200
            assert r1.json()["exit_code"] == 0

            r2 = await _post(
                client,
                "print(open('shared.txt').read())",
                "sess-files-B",
            )
            body2 = r2.json()
            assert body2["exit_code"] == 0
            assert body2["stdout"].strip() == "persist me"


async def test_files_listing_only_includes_changes_for_this_call():
    """Call 2's ``files`` does not echo files written by call 1.

    Snapshot-diff in the backend tracks mtime per relative path; an
    untouched file from a previous call is excluded from the response.
    Only newly-written or modified files show up.
    """
    backend = SubprocessBackend()
    app = create_app(backend=backend)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            r1 = await _post(
                client,
                "open('first.txt','w').write('one')",
                "sess-files-C",
            )
            assert r1.json()["exit_code"] == 0

            r2 = await _post(
                client,
                "open('second.txt','w').write('two')",
                "sess-files-C",
            )
            body2 = r2.json()
            assert body2["exit_code"] == 0
            paths = {f["path"] for f in body2["files"]}
            assert "second.txt" in paths
            assert "first.txt" not in paths


async def test_oneshot_with_no_attachments_returns_no_files():
    """No session_id and no attachments → no workdir, ``files`` is empty.

    Confirms the cheap fast path is preserved for the common case.
    """
    backend = SubprocessBackend()
    app = create_app(backend=backend)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            r = await client.post(
                "/v1/execute",
                json={"code": "print(1+1)"},
                headers={"X-Eirel-Job-Id": "job-no-fs"},
            )
            assert r.status_code == 200
            body = r.json()
            assert body["exit_code"] == 0
            assert body["files"] == []


async def test_large_file_returned_with_null_content_b64():
    """File over per-file cap surfaces path + size, content_b64=None.

    Tightening the cap on a fresh backend lets us verify the over-cap
    path without writing a real megabyte file.
    """
    backend = SubprocessBackend(file_size_cap=8, files_total_cap=1024)
    app = create_app(backend=backend)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            r = await _post(
                client,
                "open('big.bin','wb').write(b'x' * 64)",
                "sess-files-D",
            )
            body = r.json()
            assert body["exit_code"] == 0
            big = next((f for f in body["files"] if f["path"] == "big.bin"), None)
            assert big is not None
            assert big["size"] == 64
            assert big["content_b64"] is None
