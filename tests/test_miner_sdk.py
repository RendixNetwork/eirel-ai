from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from eirel.compliance import run_public_compliance_suite_async
from eirel.helpers import build_tool_call, content_response, tool_call_response, validate_request
from eirel.models import ChatCompletionRequest
from eirel.sample_server import create_app


def test_validate_request_accepts_supported_subset():
    payload = {
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "tool_choice": "auto",
        "temperature": 0,
        "seed": 7,
        "max_tokens": 32,
    }

    request = validate_request(payload)

    assert isinstance(request, ChatCompletionRequest)
    assert request.messages[0].content == "hello"
    assert request.tools[0].function.name == "calculator"


def test_validate_request_rejects_invalid_message_shape():
    with pytest.raises(ValueError, match="tool messages must include tool_call_id"):
        validate_request({"messages": [{"role": "tool", "content": "x"}]})


def test_content_response_helper_formats_valid_response():
    response = content_response("hello")
    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.content == "hello"
    assert response.choices[0].finish_reason == "stop"


def test_tool_call_response_helper_formats_valid_response():
    response = tool_call_response(
        [build_tool_call(tool_id="call-1", name="calculator", arguments={"expression": "2+2"})]
    )
    assert response.choices[0].message.tool_calls[0].function.name == "calculator"
    assert response.choices[0].finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_public_compliance_suite_passes_against_sample_server(monkeypatch):
    # The public compliance suite exercises the raw chat-completions contract,
    # so bypass inbound validator auth for the in-process fixture.
    monkeypatch.setenv("EIREL_DISABLE_REQUEST_AUTH", "1")
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        results = await run_public_compliance_suite_async(
            lambda payload: _post_json(client, payload)
        )

    assert all(item["passed"] for item in results)


async def _post_json(client: AsyncClient, payload: dict) -> dict:
    response = await client.post("/v1/chat/completions", json=payload)
    response.raise_for_status()
    return response.json()
