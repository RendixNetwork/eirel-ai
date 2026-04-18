from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai_dict(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(slots=True)
class NormalizedToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class NormalizedAssistantMessage:
    content: str | None
    tool_calls: list[NormalizedToolCall]
    finish_reason: str | None
