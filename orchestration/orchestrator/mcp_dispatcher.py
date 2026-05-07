"""Orchestrator-side MCP tool dispatcher.

Miners never see MCP. The dispatcher runs at the
:class:`ProductOrchestrator` boundary, decides (via a small LLM) which
of the user's active connections + tools to invoke for the turn,
calls the relay, and injects results into ``request.metadata.mcp_tool_results``.
The miner agent reasons over augmented context — no tool, no token,
no URL.

Cost shape:
  * Zero active connections → no LLM call, no relay calls.
  * Active connections, prompt has no tool match → 1 LLM (mini-planner)
    call, 0 relay calls.
  * Active connections, mini-planner picks N tools → 1 LLM call +
    N parallel relay calls (capped per-call timeout, total budget
    capped). Each call writes one :class:`ConsumerMcpToolCall` audit
    row.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

import httpx
from sqlalchemy import select

from shared.common.database import Database
from shared.common.models import (
    ConsumerMcpConnection,
    ConsumerMcpToolCall,
    McpIntegration,
)

_logger = logging.getLogger(__name__)

__all__ = [
    "DispatcherLLM",
    "MCPCallResult",
    "MCPRelayClient",
    "MCPToolDescriptor",
    "MCPToolDispatcher",
    "PendingMCPCall",
]


# Defaults — overridable via env in the orchestrator.
DEFAULT_PER_CALL_TIMEOUT_SECONDS: float = 10.0
DEFAULT_TOTAL_BUDGET_SECONDS: float = 30.0
DEFAULT_MAX_TOOLS_PER_TURN: int = 4


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text or "").strip()


@dataclass(frozen=True, slots=True)
class MCPToolDescriptor:
    """One tool the dispatcher can pick from across the user's connections."""

    connection_id: str
    integration_slug: str
    integration_id: str
    capabilities_hash: str
    tool_name: str
    description: str
    parameters_schema: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PendingMCPCall:
    """One tool call the mini-planner has decided to execute."""

    connection_id: str
    integration_slug: str
    capabilities_hash: str
    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MCPCallResult:
    """Outcome of one relayed MCP call, ready for envelope injection."""

    connection_id: str
    integration_slug: str
    tool_name: str
    args: dict[str, Any]
    ok: bool
    result_summary: str
    latency_ms: int
    cost_usd: float
    error: str | None = None


class DispatcherLLM(Protocol):
    """Minimal interface for the mini-planner LLM call.

    Matches :meth:`AgentProviderClient.chat_completions` so the
    orchestrator can pass any provider client.
    """

    async def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class MCPRelayClient(Protocol):
    """Async client for ``mcp_relay_service``.

    The orchestrator default is :class:`HTTPRelayClient`; tests inject
    a fake that resolves to canned :class:`MCPCallResult` items without
    network.
    """

    async def call(
        self,
        *,
        connection_id: str,
        tool_name: str,
        args: dict[str, Any],
        capabilities_hash: str | None,
        timeout_seconds: float,
    ) -> "tuple[bool, dict[str, Any] | None, str | None, int, float]": ...


class HTTPRelayClient:
    """Default relay client over HTTP."""

    __slots__ = ("_base_url", "_token", "_transport")

    def __init__(
        self,
        *,
        base_url: str,
        internal_service_token: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = internal_service_token
        self._transport = transport

    async def call(
        self,
        *,
        connection_id: str,
        tool_name: str,
        args: dict[str, Any],
        capabilities_hash: str | None,
        timeout_seconds: float,
    ) -> tuple[bool, dict[str, Any] | None, str | None, int, float]:
        kwargs: dict[str, Any] = {"timeout": timeout_seconds}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        body = {"tool_name": tool_name, "args": args}
        if capabilities_hash:
            body["capabilities_hash"] = capabilities_hash
        try:
            async with httpx.AsyncClient(**kwargs) as client:
                resp = await client.post(
                    f"{self._base_url}/v1/relay/connections/{connection_id}/call",
                    json=body, headers=headers,
                )
                resp.raise_for_status()
                payload = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            return False, None, f"relay_error: {exc}", 0, 0.0
        ok = bool(payload.get("ok"))
        result = payload.get("result") if ok else None
        error = None if ok else str(payload.get("error") or "")
        latency_ms = int(payload.get("latency_ms") or 0)
        cost_usd = float(payload.get("cost_usd") or 0.0)
        return ok, result, error, latency_ms, cost_usd


class MCPToolDispatcher:
    """Decide-and-call MCP tools for one chat turn.

    Construct with a :class:`Database` (for catalog + connection
    lookup), an :class:`MCPRelayClient`, and an optional
    :class:`DispatcherLLM` for the mini-planner. Without an LLM, the
    dispatcher returns no calls — useful for tests of the
    "passthrough when no MCP" path.
    """

    def __init__(
        self,
        *,
        database: Database,
        relay_client: MCPRelayClient,
        planner_llm: DispatcherLLM | None = None,
        per_call_timeout_seconds: float = DEFAULT_PER_CALL_TIMEOUT_SECONDS,
        total_budget_seconds: float = DEFAULT_TOTAL_BUDGET_SECONDS,
        max_tools_per_turn: int = DEFAULT_MAX_TOOLS_PER_TURN,
        planner_instruction: str | None = None,
    ) -> None:
        self._db = database
        self._relay = relay_client
        self._planner = planner_llm
        self._per_call_timeout = per_call_timeout_seconds
        self._total_budget = total_budget_seconds
        self._max_tools = max_tools_per_turn
        self._planner_instruction = planner_instruction or _DEFAULT_PLANNER_INSTRUCTION

    # -- discovery ---------------------------------------------------------

    def available_tools(self, *, user_id: str) -> list[MCPToolDescriptor]:
        """Flatten the active (connection, integration) tool surface."""
        out: list[MCPToolDescriptor] = []
        with self._db.sessionmaker() as session:
            stmt = (
                select(ConsumerMcpConnection, McpIntegration)
                .join(
                    McpIntegration,
                    McpIntegration.id == ConsumerMcpConnection.integration_id,
                )
                .where(
                    ConsumerMcpConnection.user_id == user_id,
                    ConsumerMcpConnection.status == "active",
                    McpIntegration.status == "active",
                )
            )
            for conn, integration in session.execute(stmt).all():
                tools = integration.capabilities_json or []
                for tool in tools:
                    if not isinstance(tool, dict):
                        continue
                    name = tool.get("name")
                    if not isinstance(name, str) or not name:
                        continue
                    out.append(MCPToolDescriptor(
                        connection_id=conn.id,
                        integration_slug=integration.slug,
                        integration_id=integration.id,
                        capabilities_hash=integration.capabilities_hash or "",
                        tool_name=name,
                        description=str(tool.get("description") or ""),
                        parameters_schema=tool.get("parameters_schema") or {},
                    ))
        return out

    # -- planning ----------------------------------------------------------

    async def decide_calls(
        self,
        *,
        prompt: str,
        history: Sequence[dict[str, Any]] | None,
        available: Sequence[MCPToolDescriptor],
    ) -> list[PendingMCPCall]:
        """Mini-planner: return zero or more calls to execute.

        ``history`` may be ``None`` (single-turn). The planner sees the
        most recent few entries plus the current prompt; it returns a
        JSON array of ``{integration_slug, tool_name, args}``.
        """
        if not available:
            return []
        if self._planner is None:
            return []
        catalog_lines = []
        by_key: dict[tuple[str, str], MCPToolDescriptor] = {}
        for desc in available[: self._max_tools * 8]:  # bound prompt size
            key = (desc.integration_slug, desc.tool_name)
            by_key[key] = desc
            catalog_lines.append(
                f"- {desc.integration_slug}.{desc.tool_name}: "
                f"{desc.description[:200]}"
            )
        catalog = "\n".join(catalog_lines)
        history_msgs = list(history or [])[-4:]
        try:
            response = await self._planner.chat_completions(
                {
                    "messages": [
                        {"role": "system", "content": (
                            f"{self._planner_instruction}\n\n"
                            f"Available MCP tools:\n{catalog}\n\n"
                            f"Pick at most {self._max_tools} tools. "
                            "Reply ONLY with a JSON array; empty array "
                            "if no tool helps."
                        )},
                        *history_msgs,
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 256,
                }
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            _logger.warning("mcp dispatcher planner call failed: %s", exc)
            return []
        try:
            content = response["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            return []
        plan = self._parse_plan(content)
        out: list[PendingMCPCall] = []
        for item in plan[: self._max_tools]:
            slug = item.get("integration_slug")
            tname = item.get("tool_name")
            if not isinstance(slug, str) or not isinstance(tname, str):
                continue
            desc = by_key.get((slug, tname))
            if desc is None:
                continue
            args = item.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            out.append(PendingMCPCall(
                connection_id=desc.connection_id,
                integration_slug=desc.integration_slug,
                capabilities_hash=desc.capabilities_hash,
                tool_name=desc.tool_name,
                args=args,
            ))
        return out

    @staticmethod
    def _parse_plan(text: str) -> list[dict[str, Any]]:
        cleaned = _strip_fences(text)
        if not cleaned:
            return []
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start < 0 or end < start:
            return []
        try:
            raw = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return []
        if not isinstance(raw, list):
            return []
        return [item for item in raw if isinstance(item, dict)]

    # -- execution ---------------------------------------------------------

    async def execute_calls(
        self,
        *,
        user_id: str,
        calls: Sequence[PendingMCPCall],
        conversation_id: str | None = None,
        message_id: str | None = None,
    ) -> list[MCPCallResult]:
        """Run all calls in parallel; persist one audit row per call.

        Total budget is enforced via ``asyncio.wait_for`` over the whole
        gather; per-call timeouts are forwarded to the relay client.
        Each call writes one :class:`ConsumerMcpToolCall` row regardless
        of outcome.
        """
        if not calls:
            return []

        async def _one(call: PendingMCPCall) -> MCPCallResult:
            t0 = time.perf_counter()
            try:
                ok, result, error, latency_ms, cost_usd = await self._relay.call(
                    connection_id=call.connection_id,
                    tool_name=call.tool_name,
                    args=call.args,
                    capabilities_hash=call.capabilities_hash or None,
                    timeout_seconds=self._per_call_timeout,
                )
            except Exception as exc:  # noqa: BLE001 — best-effort
                latency_ms = int((time.perf_counter() - t0) * 1000)
                ok, result, error = False, None, f"dispatcher_error: {exc}"
                cost_usd = 0.0
            summary = ""
            if ok and result is not None:
                summary = json.dumps(result, default=str)[:1000]
            return MCPCallResult(
                connection_id=call.connection_id,
                integration_slug=call.integration_slug,
                tool_name=call.tool_name,
                args=dict(call.args),
                ok=ok,
                result_summary=summary,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                error=error,
            )

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*(_one(c) for c in calls)),
                timeout=self._total_budget,
            )
        except asyncio.TimeoutError:
            # Budget exceeded — surface a synthetic timeout per call so
            # the audit log captures the partial state.
            results = [
                MCPCallResult(
                    connection_id=c.connection_id,
                    integration_slug=c.integration_slug,
                    tool_name=c.tool_name,
                    args=dict(c.args),
                    ok=False,
                    result_summary="",
                    latency_ms=int(self._total_budget * 1000),
                    cost_usd=0.0,
                    error="dispatcher_total_budget_exceeded",
                )
                for c in calls
            ]

        # Persist audit rows.
        with self._db.sessionmaker() as session:
            for result in results:
                session.add(ConsumerMcpToolCall(
                    conversation_id=conversation_id,
                    message_id=message_id,
                    connection_id=result.connection_id,
                    tool_name=result.tool_name,
                    args_json=dict(result.args),
                    result_digest=result.result_summary or (result.error or ""),
                    latency_ms=result.latency_ms,
                    cost_usd=result.cost_usd,
                    error=result.error,
                ))
            session.commit()

        return list(results)


_DEFAULT_PLANNER_INSTRUCTION = (
    "You are a tool-use planner. Given the user's prompt and the list "
    "of available MCP tools, decide which (if any) tools to call to "
    "answer well. Return a JSON array of "
    '{"integration_slug": str, "tool_name": str, "args": object}. '
    "Return [] if no tool is helpful for this turn. Prefer fewer "
    "tools over more; only include args you can derive directly from "
    "the prompt."
)
