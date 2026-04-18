"""Miner registry client — fetches serving miners from owner-api."""

from __future__ import annotations

import os
from typing import Any

import httpx

from shared.contracts.models import MinerRegistryEntry


class RegistryUnavailableError(RuntimeError):
    pass


async def fetch_registry() -> tuple[dict[str, list[MinerRegistryEntry]], dict[str, Any]]:
    """Fetch the miner registry (serving + candidate) and workflow composition registry."""
    base_url = os.getenv("OWNER_API_URL", "http://owner-api:8000").rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            serving_response = await client.get(f"{base_url}/v1/internal/registry")
            serving_response.raise_for_status()
            serving_payload = serving_response.json()
            candidate_response = await client.get(f"{base_url}/v1/internal/candidate-registry")
            candidate_response.raise_for_status()
            candidate_payload = candidate_response.json()
            workflow_response = await client.get(f"{base_url}/v1/internal/workflow-composition/registry")
            workflow_response.raise_for_status()
            workflow_payload = workflow_response.json()
    except Exception as exc:
        raise RegistryUnavailableError(f"serving registry unavailable: {exc}") from exc
    payload: dict[str, list[dict[str, Any]]] = {}
    for family_id in set(serving_payload) | set(candidate_payload):
        serving_items = serving_payload.get(family_id) or []
        candidate_items = candidate_payload.get(family_id) or []
        payload[family_id] = list(serving_items or candidate_items)
    return (
        {
            family_id: [MinerRegistryEntry.model_validate(item) for item in items]
            for family_id, items in payload.items()
        },
        dict(workflow_payload or {}),
    )


def workflow_composition_payload(
    workflow_composition_registry: dict[str, Any] | None,
    *,
    workflow_spec_id: str | None,
) -> dict[str, Any]:
    if not workflow_spec_id or not isinstance(workflow_composition_registry, dict):
        return {}
    payload = workflow_composition_registry.get(workflow_spec_id)
    return dict(payload) if isinstance(payload, dict) else {}


def workflow_composition_node_registry(
    workflow_composition_registry: dict[str, Any] | None,
    *,
    workflow_spec_id: str | None,
) -> dict[str, MinerRegistryEntry]:
    payload = workflow_composition_payload(
        workflow_composition_registry,
        workflow_spec_id=workflow_spec_id,
    )
    if not payload:
        return {}
    selected = payload.get("selected_node_map")
    if not isinstance(selected, dict):
        return {}
    resolved: dict[str, MinerRegistryEntry] = {}
    for node_id, item in selected.items():
        if not isinstance(item, dict):
            continue
        endpoint = str(item.get("endpoint") or "").strip()
        family_id = item.get("family_id")
        hotkey = str(item.get("miner_hotkey") or item.get("hotkey") or "").strip()
        if not endpoint or not family_id or not hotkey:
            continue
        resolved[str(node_id)] = MinerRegistryEntry(
            hotkey=hotkey,
            family_id=str(family_id),
            endpoint=endpoint,
            latency_score=1.0,
            quality_score=float(item.get("serving_selection_score") or 0.0),
            metadata={
                "workflow_spec_id": workflow_spec_id,
                "workflow_composition": True,
                "workflow_composition_source": payload.get("selection_reason"),
                "workflow_composition_revision": payload.get("source_serving_release_id"),
                "workflow_composition_reason": payload.get("selection_reason"),
                "source_deployment_id": item.get("deployment_id"),
                "source_submission_id": item.get("submission_id"),
                "node_id": item.get("node_id"),
                "role_id": item.get("role_id"),
            },
        )
    return resolved
