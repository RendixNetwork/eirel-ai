"""Helpers for canonicalizing + hashing an integration's tool surface."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

__all__ = ["canonicalize_tools", "hash_capabilities"]


def canonicalize_tools(tools: Sequence[Any]) -> list[dict[str, Any]]:
    """Return a sorted, normalized list of tool descriptors.

    Each entry is reduced to ``{name, description, parameters_schema}``.
    Sorted by ``name`` so two equivalent surfaces hash identically
    regardless of source ordering.
    """
    out: list[dict[str, Any]] = []
    for item in tools:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name:
            continue
        out.append({
            "name": name,
            "description": str(item.get("description") or ""),
            "parameters_schema": item.get("parameters_schema") or {},
        })
    out.sort(key=lambda t: t["name"])
    return out


def hash_capabilities(tools: Sequence[Any]) -> str:
    """Stable sha256 hex digest over the canonicalized tool list."""
    canonical = canonicalize_tools(tools)
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
