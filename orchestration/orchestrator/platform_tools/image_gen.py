"""Platform tool: image_gen

Generates images via external provider APIs (DALL-E, Stable Diffusion).
This is a platform-level tool because image generation is a deterministic
API call, not a specialist competition task.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from orchestration.orchestrator.platform_tools.base import PlatformTool, ToolResult

_logger = logging.getLogger(__name__)

PROVIDER_PROXY_URL = os.getenv("EIREL_PROVIDER_PROXY_URL", "")
PROVIDER_PROXY_TOKEN = os.getenv("EIREL_PROVIDER_PROXY_TOKEN", "")
IMAGE_GEN_TIMEOUT = float(os.getenv("IMAGE_GEN_TIMEOUT_SECONDS", "60"))


class ImageGenTool(PlatformTool):
    @property
    def name(self) -> str:
        return "image_gen"

    @property
    def description(self) -> str:
        return "Generate images from text prompts using AI image generation models."

    async def execute(self, *, params: dict[str, Any]) -> ToolResult:
        prompt = params.get("prompt", "")
        if not prompt.strip():
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="no prompt provided",
            )

        if not PROVIDER_PROXY_URL:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="image generation not configured (no provider proxy)",
            )

        payload = {
            "prompt": prompt,
            "model": params.get("model", "dall-e-3"),
            "size": params.get("size", "1024x1024"),
            "n": min(params.get("n", 1), 4),
        }
        headers: dict[str, str] = {}
        if PROVIDER_PROXY_TOKEN:
            headers["Authorization"] = f"Bearer {PROVIDER_PROXY_TOKEN}"

        try:
            async with httpx.AsyncClient(timeout=IMAGE_GEN_TIMEOUT) as client:
                resp = await client.post(
                    f"{PROVIDER_PROXY_URL.rstrip('/')}/v1/images/generations",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
                return ToolResult(
                    tool_name=self.name,
                    success=True,
                    output={
                        "images": data.get("data", []),
                        "prompt": prompt,
                    },
                )
        except httpx.HTTPStatusError as exc:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"provider proxy returned {exc.response.status_code}",
            )
        except httpx.ConnectError:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="provider proxy unavailable",
            )
