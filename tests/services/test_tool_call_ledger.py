"""Tests for the orchestrator tool-call ledger.

Two layers:

  1. ``record_tool_call`` helper — best-effort POST to the owner-api
     endpoint; missing config / missing job_id silently no-op so a
     standalone tool service stays available.
  2. End-to-end: a tool-service call via httpx.MockTransport actually
     reaches a stub owner-api receiver — proves the wire is hooked up.
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from tool_platforms._record_tool_call import (
    digest_result,
    hash_args,
    record_tool_call,
)


# -- Pure helpers -----------------------------------------------------------


def test_hash_args_canonical_and_stable():
    a = {"q": "hello", "n": 5}
    b = {"n": 5, "q": "hello"}
    assert hash_args(a) == hash_args(b)
    assert len(hash_args(a)) == 64  # sha256 hex


def test_hash_args_distinguishes_payloads():
    assert hash_args({"q": "a"}) != hash_args({"q": "b"})


def test_digest_result_truncates_long_strings():
    long = "x" * 4096
    digest = digest_result(long)
    assert len(digest) <= 600  # stays under the 512 cap


def test_digest_result_serializes_non_strings():
    digest = digest_result({"k": [1, 2, 3]})
    assert "k" in digest


def test_digest_result_handles_none():
    assert digest_result(None) == ""


# -- Fire-and-forget no-op cases --------------------------------------------


async def test_record_tool_call_silently_skips_when_no_job_id(monkeypatch):
    """A direct curl / smoke test without X-Eirel-Job-Id is allowed."""
    monkeypatch.setenv("EIREL_OWNER_API_URL", "http://owner.test")
    monkeypatch.setenv("EIREL_INTERNAL_SERVICE_TOKEN", "tok")

    # No transport configured + no job_id = no HTTP call attempted.
    await record_tool_call(
        job_id=None, tool_name="web_search", args={"q": "x"},
    )
    # If we got here without raising, the no-op path works.


async def test_record_tool_call_silently_skips_when_no_owner_api_url(monkeypatch):
    """Missing EIREL_OWNER_API_URL = local-only service; never blocks the tool."""
    monkeypatch.delenv("EIREL_OWNER_API_URL", raising=False)
    await record_tool_call(
        job_id="job-1", tool_name="web_search", args={"q": "x"},
    )


async def test_record_tool_call_swallows_network_errors(monkeypatch):
    """Owner-api unreachable = log warning, but never raise.

    The tool-service caller treats this as fire-and-forget.
    """
    # Point at an unreachable host; a real httpx call would raise — assert
    # the helper swallows it.
    await record_tool_call(
        job_id="job-1", tool_name="web_search", args={"q": "x"},
        owner_api_url="http://127.0.0.1:1",  # unreachable
        owner_api_token="tok",
    )
    # Got here = swallowed correctly.


# -- End-to-end smoke -------------------------------------------------------


async def test_record_tool_call_reaches_owner_api(monkeypatch):
    """The helper's payload arrives at the owner-api endpoint with the
    right shape and bearer token."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization", "")
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(201, json={"id": "row-1", "ok": True})

    transport = httpx.MockTransport(handler)
    # Patch the helper to use our transport. Easiest: monkey-patch
    # httpx.AsyncClient default to use our transport.
    real_async_client = httpx.AsyncClient

    class StubAsyncClient(real_async_client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr("tool_platforms._record_tool_call.httpx.AsyncClient", StubAsyncClient)

    await record_tool_call(
        job_id="job-42",
        tool_name="web_search",
        args={"query": "claude code", "top_k": 5},
        result={"backend": "brave", "n_results": 3},
        latency_ms=120,
        cost_usd=0.003,
        status_str="ok",
        owner_api_url="http://owner.test",
        owner_api_token="tok-secret",
    )

    assert captured["url"] == "http://owner.test/v1/internal/eval/tool_calls"
    assert captured["auth"] == "Bearer tok-secret"
    body = captured["body"]
    assert body["job_id"] == "job-42"
    assert body["tool_name"] == "web_search"
    assert body["args_hash"] == hash_args({"query": "claude code", "top_k": 5})
    assert body["args_json"] == {"query": "claude code", "top_k": 5}
    assert "brave" in body["result_digest"]
    assert body["latency_ms"] == 120
    assert body["cost_usd"] == 0.003
    assert body["status"] == "ok"
