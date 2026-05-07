"""URL-fetch tool service.

Subnet-owned HTTP service that miner agents call to read the contents
of a public URL. Mirrors ChatGPT's ``browser`` and Claude's ``web_fetch``
tools at the wire level: input is a URL, output is extracted text +
metadata.
"""
from __future__ import annotations

from tool_platforms.url_fetch_tool_service.app import create_app, main

__all__ = ["create_app", "main"]
