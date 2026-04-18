from __future__ import annotations

"""Tests for orchestrator platform tools — registry, code_exec, web_search,
image_gen, file_manager, memory_recall."""

import os
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestration.orchestrator.platform_tools.base import ToolResult
from orchestration.orchestrator.platform_tools.tools_registry import ToolsRegistry
from orchestration.orchestrator.platform_tools.code_executor import CodeExecutorTool
from orchestration.orchestrator.platform_tools.web_search import WebSearchTool
from orchestration.orchestrator.platform_tools.image_gen import ImageGenTool
from orchestration.orchestrator.platform_tools.file_manager import FileManagerTool
from orchestration.orchestrator.platform_tools.memory_tool import MemoryRecallTool


# ===================================================================
# ToolsRegistry tests
# ===================================================================

def test_default_tools_registered():
    registry = ToolsRegistry()
    tools = registry.available_tools()
    assert "code_exec" in tools
    assert "web_search" in tools
    assert "file_manager" in tools
    assert "image_gen" in tools
    assert "memory_recall" in tools


def test_lookup_by_name():
    registry = ToolsRegistry()
    tool = registry.get("code_exec")
    assert tool is not None
    assert tool.name == "code_exec"


async def test_invoke_unknown_tool():
    registry = ToolsRegistry()
    result = await registry.invoke("nonexistent_tool", {})
    assert result.success is False
    assert "unknown tool" in result.error


async def test_invoke_dispatches_correctly():
    registry = ToolsRegistry()
    # code_exec with empty code should return error (no external service needed)
    result = await registry.invoke("code_exec", {"code": ""})
    assert result.success is False
    assert "no code" in result.error


# ===================================================================
# CodeExecutorTool tests
# ===================================================================

async def test_code_exec_success(monkeypatch):
    tool = CodeExecutorTool()
    sandbox_response = {"stdout": "42\n", "stderr": "", "exit_code": 0}

    async def _fake_post(self, url, **kwargs):
        return httpx.Response(200, json=sandbox_response, request=httpx.Request("POST", str(url)))

    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)
    result = await tool.invoke(params={"code": "print(42)", "language": "python"})
    assert result.success is True
    assert result.output["stdout"] == "42\n"
    assert result.output["exit_code"] == 0


async def test_code_exec_sandbox_500(monkeypatch):
    tool = CodeExecutorTool()

    async def _fake_post(self, url, **kwargs):
        resp = httpx.Response(500, json={"error": "boom"}, request=httpx.Request("POST", str(url)))
        resp.raise_for_status()

    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)
    result = await tool.invoke(params={"code": "x"})
    assert result.success is False


async def test_code_exec_connect_error(monkeypatch):
    tool = CodeExecutorTool()

    async def _fake_post(self, url, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)
    result = await tool.invoke(params={"code": "x"})
    assert result.success is False


async def test_code_exec_no_code():
    tool = CodeExecutorTool()
    result = await tool.invoke(params={"code": "", "language": "python"})
    assert result.success is False
    assert "no code" in result.error


# ===================================================================
# WebSearchTool tests
# ===================================================================

async def test_web_search_success(monkeypatch):
    tool = WebSearchTool()
    search_response = {
        "results": [
            {"title": "Result 1", "url": "https://example.com", "snippet": "..."},
        ]
    }

    async def _fake_post(self, url, **kwargs):
        return httpx.Response(200, json=search_response, request=httpx.Request("POST", str(url)))

    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)
    result = await tool.invoke(params={"query": "test search"})
    assert result.success is True
    assert len(result.output["results"]) == 1


async def test_web_search_connect_error(monkeypatch):
    tool = WebSearchTool()

    async def _fake_post(self, url, **kwargs):
        raise httpx.ConnectError("unavailable")

    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)
    result = await tool.invoke(params={"query": "test"})
    assert result.success is False


async def test_web_search_empty_query():
    tool = WebSearchTool()
    result = await tool.invoke(params={"query": ""})
    assert result.success is False
    assert "no search query" in result.error


# ===================================================================
# ImageGenTool tests
# ===================================================================

async def test_image_gen_success(monkeypatch):
    monkeypatch.setattr(
        "orchestration.orchestrator.platform_tools.image_gen.PROVIDER_PROXY_URL",
        "http://fake-provider:9000",
    )
    tool = ImageGenTool()
    gen_response = {"data": [{"url": "https://img.example.com/1.png"}]}

    async def _fake_post(self, url, **kwargs):
        return httpx.Response(200, json=gen_response, request=httpx.Request("POST", str(url)))

    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)
    result = await tool.invoke(params={"prompt": "a cat"})
    assert result.success is True
    assert len(result.output["images"]) == 1


async def test_image_gen_not_configured(monkeypatch):
    monkeypatch.setattr(
        "orchestration.orchestrator.platform_tools.image_gen.PROVIDER_PROXY_URL", ""
    )
    tool = ImageGenTool()
    result = await tool.invoke(params={"prompt": "a cat"})
    assert result.success is False
    assert "not configured" in result.error


async def test_image_gen_connect_error(monkeypatch):
    monkeypatch.setattr(
        "orchestration.orchestrator.platform_tools.image_gen.PROVIDER_PROXY_URL",
        "http://fake-provider:9000",
    )
    tool = ImageGenTool()

    async def _fake_post(self, url, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)
    result = await tool.invoke(params={"prompt": "a cat"})
    assert result.success is False


# ===================================================================
# FileManagerTool tests
# ===================================================================

async def test_file_write_and_read_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "orchestration.orchestrator.platform_tools.file_manager.FILE_STORAGE_ROOT",
        str(tmp_path),
    )
    tool = FileManagerTool()
    write_result = await tool.invoke(params={
        "action": "write",
        "session_id": "s1",
        "filename": "test.txt",
        "content": "hello world",
    })
    assert write_result.success is True

    read_result = await tool.invoke(params={
        "action": "read",
        "session_id": "s1",
        "filename": "test.txt",
    })
    assert read_result.success is True
    assert read_result.output["content"] == "hello world"


async def test_file_list(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "orchestration.orchestrator.platform_tools.file_manager.FILE_STORAGE_ROOT",
        str(tmp_path),
    )
    tool = FileManagerTool()
    await tool.invoke(params={"action": "write", "session_id": "s2", "filename": "a.txt", "content": "a"})
    await tool.invoke(params={"action": "write", "session_id": "s2", "filename": "b.txt", "content": "b"})
    list_result = await tool.invoke(params={"action": "list", "session_id": "s2"})
    assert list_result.success is True
    assert list_result.output["count"] == 2


async def test_file_read_nonexistent(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "orchestration.orchestrator.platform_tools.file_manager.FILE_STORAGE_ROOT",
        str(tmp_path),
    )
    tool = FileManagerTool()
    result = await tool.invoke(params={"action": "read", "session_id": "s3", "filename": "nope.txt"})
    assert result.success is False
    assert "not found" in result.error


async def test_file_path_traversal_safety(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "orchestration.orchestrator.platform_tools.file_manager.FILE_STORAGE_ROOT",
        str(tmp_path),
    )
    tool = FileManagerTool()
    write_result = await tool.invoke(params={
        "action": "write",
        "session_id": "s4",
        "filename": "../../etc/passwd",
        "content": "nope",
    })
    # Should strip path components — file should be created as "passwd" inside session dir
    assert write_result.success is True
    assert "passwd" in write_result.output["filename"]
    # Should not have written outside session dir
    assert not (tmp_path / ".." / ".." / "etc" / "passwd").exists()


# ===================================================================
# MemoryRecallTool tests
# ===================================================================

async def test_memory_recall_with_context_history():
    tool = MemoryRecallTool()
    history = [
        {"content": "the weather is sunny today"},
        {"content": "cats are cute animals"},
        {"content": "python programming language"},
    ]
    result = await tool.invoke(params={
        "query": "weather forecast today",
        "context_history": history,
    })
    assert result.success is True
    ctx = result.output["conversation_context"]
    assert len(ctx) >= 1
    assert "weather" in ctx[0]["content"]


async def test_memory_recall_retrieval_unavailable(monkeypatch):
    tool = MemoryRecallTool()

    async def _fake_post(self, url, **kwargs):
        raise httpx.ConnectError("service down")

    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)
    result = await tool.invoke(params={"query": "test"})
    # Should gracefully degrade — still succeed with empty knowledge
    assert result.success is True
    assert result.output["retrieved_knowledge"] == []
