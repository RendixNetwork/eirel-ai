"""Graph-runtime execution coordinator.

Drives one :class:`GraphPlan` against a chosen miner. Owner-api's
``POST /runtime/{deployment_id}/v1/agent/infer/stream`` already does
the heavy lifting (hotkey-signed auth, runtime_kind detection, trace
tee to eiretes, cost reconciliation), so this module is mostly a
thin client + event filter.

Two surface areas:

  * :meth:`GraphExecutor.invoke` — unary call returning the full
    :class:`AgentInvocationResponse`.
  * :meth:`GraphExecutor.astream` — async iterator yielding
    consumer-safe NDJSON dicts. **Drops** ``trace`` frames on the way
    out; those have already been teed to eiretes server-side via the
    runtime proxy.

Both honor ``thread_id`` for multi-turn continuity: pass the same
``thread_id`` on follow-up turns and the picker pins the same
deployment so the SDK's checkpointer hits its own thread state.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

from orchestration.orchestrator.graph_plan import GraphPlan
from orchestration.orchestrator.miner_picker import MinerCandidate, MinerPicker

_logger = logging.getLogger(__name__)

__all__ = ["GraphExecutor", "ConsumerStreamEvent"]


# Event taxonomy emitted by the graph SDK — these survive the trip
# from miner pod → owner-api → orchestrator. The orchestrator passes
# everything except ``trace`` through to the consumer.
_PASSTHROUGH_EVENTS = frozenset({"delta", "tool_call", "tool_result", "citation", "checkpoint", "done"})
_DROP_EVENTS = frozenset({"trace"})


class ConsumerStreamEvent(dict):
    """Marker dict subclass — purely for type clarity at call sites."""


class GraphExecutor:
    """Drives a :class:`GraphPlan` against owner-api's streaming runtime."""

    def __init__(
        self,
        *,
        miner_picker: MinerPicker,
        owner_api_url: str | None = None,
        internal_service_token: str | None = None,
        timeout_seconds: float = 1800.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._picker = miner_picker
        self._owner_api_url = (
            owner_api_url or os.getenv("OWNER_API_URL", "http://owner-api:8000")
        ).rstrip("/")
        self._internal_token = (
            internal_service_token
            or os.getenv("EIREL_INTERNAL_SERVICE_TOKEN", "")
        )
        self._timeout = timeout_seconds
        self._transport = transport

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._internal_token:
            headers["Authorization"] = f"Bearer {self._internal_token}"
        return headers

    def _client_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"timeout": self._timeout}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return kwargs

    def _build_payload(
        self,
        *,
        prompt: str,
        history: list[dict[str, Any]] | None,
        thread_id: str | None,
        resume_token: str | None,
        mode: str,
        web_search: bool,
        run_budget_usd: float | None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        # Standard AgentInvocationRequest envelope. GraphAgent (in eirel
        # SDK) accepts this directly via BaseAgent inheritance — no
        # graph-specific envelope needed at this layer.
        body: dict[str, Any] = {
            "prompt": prompt,
            "history": list(history or []),
            "mode": mode,
            "web_search": web_search,
            "turn_id": thread_id,
        }
        if resume_token:
            body["resume_token"] = resume_token
        if metadata or run_budget_usd is not None:
            body_metadata = dict(metadata or {})
            if run_budget_usd is not None:
                body_metadata.setdefault("run_budget_usd", float(run_budget_usd))
            body["metadata"] = body_metadata
        return body

    async def _pick(
        self,
        *,
        plan: GraphPlan,
        thread_id: str | None,
    ) -> MinerCandidate:
        # Plans are single-step today; pick once for the only step.
        family_id = plan.steps[0].family_id
        return self._picker.pick(family_id=family_id, thread_id=thread_id)

    async def invoke(
        self,
        *,
        plan: GraphPlan,
        prompt: str,
        history: list[dict[str, Any]] | None = None,
        thread_id: str | None = None,
        resume_token: str | None = None,
        mode: str = "instant",
        web_search: bool = False,
        run_budget_usd: float | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Unary invocation. Returns the miner's
        :class:`AgentInvocationResponse` body verbatim.

        Useful for the orchestrator's non-streaming path (consumer
        APIs that aren't NDJSON-capable) and for tests. Streaming
        consumers should use :meth:`astream` to forward token-level
        deltas.
        """
        miner = await self._pick(plan=plan, thread_id=thread_id)
        payload = self._build_payload(
            prompt=prompt,
            history=history,
            thread_id=thread_id,
            resume_token=resume_token,
            mode=mode,
            web_search=web_search,
            run_budget_usd=run_budget_usd,
            metadata=metadata,
        )
        path = (
            f"/v1/internal/runs/{run_id}/deployments/{miner.deployment_id}/v1/agent/infer"
            if run_id
            else f"/runtime/{miner.deployment_id}/v1/agent/infer"
        )
        url = f"{self._owner_api_url}{path}"
        async with httpx.AsyncClient(**self._client_kwargs()) as client:
            resp = await client.post(url, json=payload, headers=self._headers())
            resp.raise_for_status()
            response_body = resp.json()
        # Annotate with picker metadata so eiretes / leaderboard have
        # provenance for the served response without having to re-query.
        meta = response_body.get("metadata")
        if not isinstance(meta, dict):
            meta = {}
        meta.setdefault("orchestrator", {
            "deployment_id": miner.deployment_id,
            "miner_hotkey": miner.miner_hotkey,
            "runtime_kind": miner.runtime_kind,
            "thread_id": thread_id,
            "plan_id": plan.plan_id,
            "selection": plan.selection.to_dict(),
        })
        response_body["metadata"] = meta
        return response_body

    async def astream(
        self,
        *,
        plan: GraphPlan,
        prompt: str,
        history: list[dict[str, Any]] | None = None,
        thread_id: str | None = None,
        resume_token: str | None = None,
        mode: str = "instant",
        web_search: bool = False,
        run_budget_usd: float | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[ConsumerStreamEvent]:
        """Stream NDJSON events from the miner pod through to the consumer.

        ``trace`` events are dropped — they've already been teed to
        eiretes by the owner-api runtime proxy. Everything else
        (delta / tool_call / tool_result / citation / checkpoint /
        done) passes through. The terminal ``done`` event picks up an
        ``orchestrator`` block in its ``metadata`` so the consumer
        can audit which miner served the response.
        """
        miner = await self._pick(plan=plan, thread_id=thread_id)
        payload = self._build_payload(
            prompt=prompt,
            history=history,
            thread_id=thread_id,
            resume_token=resume_token,
            mode=mode,
            web_search=web_search,
            run_budget_usd=run_budget_usd,
            metadata=metadata,
        )
        path = (
            f"/v1/internal/runs/{run_id}/deployments/{miner.deployment_id}/v1/agent/infer/stream"
            if run_id
            else f"/runtime/{miner.deployment_id}/v1/agent/infer/stream"
        )
        url = f"{self._owner_api_url}{path}"
        async with httpx.AsyncClient(**self._client_kwargs()) as client:
            async with client.stream(
                "POST", url, json=payload, headers=self._headers()
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        frame = json.loads(line)
                    except json.JSONDecodeError:
                        # Pass garbage through verbatim — the proxy
                        # already exists to enforce the wire contract;
                        # consumers see exactly what arrived.
                        yield ConsumerStreamEvent({"event": "raw", "line": line})
                        continue
                    if not isinstance(frame, dict):
                        continue
                    event = frame.get("event")
                    if event in _DROP_EVENTS:
                        continue
                    if event not in _PASSTHROUGH_EVENTS:
                        # Unknown future event types — forward
                        # verbatim. Consumers either understand them or
                        # ignore them, but we never silently drop.
                        yield ConsumerStreamEvent(frame)
                        continue
                    if event == "done":
                        meta = frame.get("metadata")
                        if not isinstance(meta, dict):
                            meta = {}
                        meta.setdefault("orchestrator", {
                            "deployment_id": miner.deployment_id,
                            "miner_hotkey": miner.miner_hotkey,
                            "runtime_kind": miner.runtime_kind,
                            "thread_id": thread_id,
                            "plan_id": plan.plan_id,
                            "selection": plan.selection.to_dict(),
                        })
                        frame["metadata"] = meta
                    yield ConsumerStreamEvent(frame)
