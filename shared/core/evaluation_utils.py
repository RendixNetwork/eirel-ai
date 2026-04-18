"""Shared utility functions for the eiretes evaluation framework."""

from __future__ import annotations

from typing import Any


def safe_dict(value: Any) -> dict[str, Any]:
    """Return value if it's a dict, otherwise return an empty dict."""
    return value if isinstance(value, dict) else {}


def safe_list(value: Any) -> list[Any]:
    """Return value if it's a list, otherwise return an empty list."""
    return value if isinstance(value, list) else []
