from __future__ import annotations

import json
from typing import Any

from shared.core.protocol_types import (
    NormalizedAssistantMessage,
    NormalizedToolCall,
    ToolSpec,
)


def build_chat_completion_request(
    *,
    messages: list[dict[str, Any]],
    tools: list[ToolSpec] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    seed: int | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = [tool.to_openai_dict() for tool in tools]
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    if seed is not None:
        payload["seed"] = seed
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    return payload


def normalize_chat_completion_response(payload: dict[str, Any]) -> NormalizedAssistantMessage:
    if not isinstance(payload, dict):
        raise RuntimeError("malformed invoke response: payload must be a JSON object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("malformed invoke response: missing choices[0]")
    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError("malformed invoke response: choices[0] must be an object")
    message = first.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("malformed invoke response: missing choices[0].message")
    if message.get("role") != "assistant":
        raise RuntimeError("malformed invoke response: assistant role required")
    raw_content = message.get("content")
    # Reasoning models (e.g. Kimi-K2.5-TEE) may return content=null with
    # reasoning_content containing the actual output.
    if raw_content is None:
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            raw_content = reasoning
    if raw_content is not None and not isinstance(raw_content, str):
        raise RuntimeError("malformed invoke response: assistant content must be a string or null")
    raw_tool_calls = message.get("tool_calls") or []
    if not isinstance(raw_tool_calls, list):
        raise RuntimeError("malformed invoke response: tool_calls must be a list")
    tool_calls: list[NormalizedToolCall] = []
    for index, item in enumerate(raw_tool_calls):
        if not isinstance(item, dict):
            raise RuntimeError(
                f"malformed invoke response: tool_calls[{index}] must be an object"
            )
        if item.get("type") != "function":
            raise RuntimeError(
                f"malformed invoke response: tool_calls[{index}] type must be 'function'"
            )
        function = item.get("function")
        if not isinstance(function, dict):
            raise RuntimeError(
                f"malformed invoke response: tool_calls[{index}].function missing"
            )
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(name, str) or not name:
            raise RuntimeError(
                f"malformed invoke response: tool_calls[{index}].function.name missing"
            )
        if not isinstance(arguments, str):
            raise RuntimeError(
                f"malformed invoke response: tool_calls[{index}].function.arguments must be a string"
            )
        try:
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"malformed invoke response: tool_calls[{index}] arguments must be valid JSON"
            ) from exc
        if not isinstance(parsed_arguments, dict):
            raise RuntimeError(
                f"malformed invoke response: tool_calls[{index}] arguments must decode to an object"
            )
        tool_calls.append(
            NormalizedToolCall(
                id=str(item.get("id") or f"tool-call-{index + 1}"),
                name=name,
                arguments=parsed_arguments,
            )
        )
    if raw_content is None and not tool_calls:
        raise RuntimeError(
            "malformed invoke response: assistant must return content or tool_calls"
        )
    finish_reason = first.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise RuntimeError("malformed invoke response: finish_reason must be a string or null")
    return NormalizedAssistantMessage(
        content=raw_content.strip() if isinstance(raw_content, str) else None,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
    )
