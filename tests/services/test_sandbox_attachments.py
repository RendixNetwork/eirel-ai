"""Sandbox attachments: pre-existing files mounted into the workdir.

The ``attachments`` request field accepts ``[{filename, content_b64}]``.
Each file is decoded and dropped into the sandbox working directory
before user code runs. Typical caller: the orchestrator hydrating a
``ConsumerAttachment.blob_ref`` (e.g. a CSV the user uploaded) so the
agent can analyze it via pandas / stdlib parsers without a separate
download tool.
"""
from __future__ import annotations

import base64

from httpx import ASGITransport, AsyncClient

from tool_platforms.sandbox_tool_service.app import create_app
from tool_platforms.sandbox_tool_service.backends import SubprocessBackend


def _b64(s: str | bytes) -> str:
    if isinstance(s, str):
        s = s.encode("utf-8")
    return base64.b64encode(s).decode("ascii")


async def test_attachment_mounted_into_oneshot_workdir():
    """One-shot path with attachments: file mounted, user reads it."""
    backend = SubprocessBackend()
    app = create_app(backend=backend)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            r = await client.post(
                "/v1/execute",
                json={
                    "code": "print(open('greet.txt').read())",
                    "attachments": [
                        {"filename": "greet.txt", "content_b64": _b64("hello attachment")},
                    ],
                },
                headers={"X-Eirel-Job-Id": "job-att-1"},
            )
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["exit_code"] == 0
            assert body["stdout"].strip() == "hello attachment"


async def test_attachment_mounted_into_session_workdir():
    """Session path with attachments: same file persists for next call."""
    backend = SubprocessBackend()
    app = create_app(backend=backend)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            csv = "a,b,c\n1,2,3\n4,5,6\n"
            r1 = await client.post(
                "/v1/execute",
                json={
                    "code": (
                        "rows = open('data.csv').read().splitlines()[1:]\n"
                        "total = sum(int(x) for line in rows for x in line.split(','))\n"
                        "print(total)\n"
                    ),
                    "session_id": "sess-att",
                    "attachments": [
                        {"filename": "data.csv", "content_b64": _b64(csv)},
                    ],
                },
                headers={"X-Eirel-Job-Id": "job-att-2", "X-Eirel-Max-Requests": "10"},
            )
            assert r1.status_code == 200, r1.text
            body1 = r1.json()
            assert body1["exit_code"] == 0
            assert body1["stdout"].strip() == "21"

            # Attachment is unchanged → not echoed in files
            assert all(f["path"] != "data.csv" for f in body1["files"])

            # Second call in same session can still read the file
            r2 = await client.post(
                "/v1/execute",
                json={
                    "code": "print(len(open('data.csv').read()))",
                    "session_id": "sess-att",
                },
                headers={"X-Eirel-Job-Id": "job-att-2", "X-Eirel-Max-Requests": "10"},
            )
            body2 = r2.json()
            assert body2["exit_code"] == 0
            assert body2["stdout"].strip() == str(len(csv))


async def test_attachment_path_traversal_rejected():
    """Filenames are basename'd — '../../etc/passwd' becomes 'passwd'."""
    backend = SubprocessBackend()
    app = create_app(backend=backend)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            r = await client.post(
                "/v1/execute",
                json={
                    "code": (
                        "import os\n"
                        "names = sorted(os.listdir('.'))\n"
                        "print(','.join(names))\n"
                    ),
                    "attachments": [
                        {
                            "filename": "../../etc/passwd",
                            "content_b64": _b64("not really sensitive"),
                        },
                    ],
                },
                headers={"X-Eirel-Job-Id": "job-att-3"},
            )
            body = r.json()
            assert body["exit_code"] == 0
            # If the basename strip worked, we have 'passwd' inside the workdir,
            # not anywhere up the tree. listdir('.') is the workdir.
            files_in_workdir = body["stdout"].strip()
            # ``passwd`` is mounted because basename('.../passwd') == 'passwd'
            assert "passwd" in files_in_workdir.split(",")


async def test_attachment_with_invalid_base64_skipped():
    """Bad base64 → attachment silently skipped, code still runs."""
    backend = SubprocessBackend()
    app = create_app(backend=backend)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            r = await client.post(
                "/v1/execute",
                json={
                    "code": "print('ran')",
                    "attachments": [
                        {"filename": "bad.bin", "content_b64": "@@@not-b64@@@"},
                    ],
                },
                headers={"X-Eirel-Job-Id": "job-att-4"},
            )
            body = r.json()
            assert body["exit_code"] == 0
            assert body["stdout"].strip() == "ran"


async def test_attachment_binary_roundtrip():
    """Binary payloads survive the base64 → file → read trip."""
    backend = SubprocessBackend()
    app = create_app(backend=backend)
    raw = bytes(range(256))
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            r = await client.post(
                "/v1/execute",
                json={
                    "code": (
                        "data = open('blob.bin','rb').read()\n"
                        "print(len(data), data[0], data[-1])\n"
                    ),
                    "attachments": [
                        {"filename": "blob.bin", "content_b64": _b64(raw)},
                    ],
                },
                headers={"X-Eirel-Job-Id": "job-att-5"},
            )
            body = r.json()
            assert body["exit_code"] == 0
            assert body["stdout"].strip() == "256 0 255"
