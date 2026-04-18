from __future__ import annotations

import pytest

from shared.core.protocol import (
    build_chat_completion_request,
    normalize_chat_completion_response,
)
from shared.core.protocol_types import ToolSpec


def test_build_chat_completion_request_supports_tools_and_seed():
    payload = build_chat_completion_request(
        messages=[{"role": "user", "content": "hello"}],
        tools=[
            ToolSpec(
                name="calculator",
                description="math",
                parameters={"type": "object", "properties": {}},
            )
        ],
        tool_choice="auto",
        seed=7,
        max_tokens=32,
    )

    assert payload["messages"][0]["content"] == "hello"
    assert payload["tools"][0]["function"]["name"] == "calculator"
    assert payload["tool_choice"] == "auto"
    assert payload["seed"] == 7
    assert payload["max_tokens"] == 32
    assert payload["temperature"] == 0.0


def test_normalize_chat_completion_response_accepts_content_reply():
    normalized = normalize_chat_completion_response(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello there"},
                    "finish_reason": "stop",
                }
            ]
        }
    )

    assert normalized.content == "Hello there"
    assert normalized.tool_calls == []
    assert normalized.finish_reason == "stop"


def test_normalize_chat_completion_response_accepts_tool_calls():
    normalized = normalize_chat_completion_response(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "calculator",
                                    "arguments": '{"expression":"2+2"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )

    assert normalized.content is None
    assert normalized.tool_calls[0].name == "calculator"
    assert normalized.tool_calls[0].arguments == {"expression": "2+2"}


def test_normalize_chat_completion_response_rejects_invalid_payload():
    with pytest.raises(RuntimeError, match="assistant role required"):
        normalize_chat_completion_response(
            {"choices": [{"message": {"role": "user", "content": "bad"}}]}
        )
