from __future__ import annotations

import httpx
import pytest

from shared.core.evaluation_models import JudgeResult
from shared.core.judge_client import JudgeServiceClient


def _mock_judge_response() -> dict:
    return {
        "model": "test-model",
        "rubric_name": "test_rubric",
        "score": 0.82,
        "rationale": "Good analysis",
        "latency_seconds": 1.5,
        "dimension_scores": {"quality": 0.9},
        "constraint_flags": [],
        "usage": {},
        "metadata": {},
    }


def test_judge_call(monkeypatch):
    """JudgeServiceClient.judge() calls the sidecar and returns JudgeResult."""

    def _mock_post(self, url, **kwargs):
        assert "/v1/judge" in str(url)
        body = kwargs.get("json", {})
        assert body["family_id"] == "general_chat"
        assert body["prompt"] == "test prompt"
        assert body["mode"] == "instant"
        return httpx.Response(200, json=_mock_judge_response(), request=httpx.Request("POST", str(url)))

    monkeypatch.setattr(httpx.Client, "post", _mock_post)

    client = JudgeServiceClient(base_url="http://fake:8095")
    result = client.judge(
        family_id="general_chat",
        prompt="test prompt",
        response_excerpt="test response",
    )
    assert isinstance(result, JudgeResult)
    assert result.score == 0.82
    assert result.model == "test-model"
    client.close()


def test_extract_claims_call(monkeypatch):
    """JudgeServiceClient.extract_research_claims() calls the sidecar."""

    def _mock_post(self, url, **kwargs):
        assert "/v1/extract-claims" in str(url)
        return httpx.Response(200, json={"claims": [], "metadata": {}}, request=httpx.Request("POST", str(url)))

    monkeypatch.setattr(httpx.Client, "post", _mock_post)

    client = JudgeServiceClient(base_url="http://fake:8095")
    result = client.extract_research_claims(
        prompt="test",
        report_markdown="# Report",
    )
    assert "claims" in result
    client.close()


def test_healthcheck_returns_service_info(monkeypatch):
    def _mock_get(self, url, **kwargs):
        assert "/healthz" in str(url)
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "judge_model": "test-model",
                "rubric_version": "family_rubric_v2",
            },
            request=httpx.Request("GET", str(url)),
        )

    monkeypatch.setattr(httpx.Client, "get", _mock_get)
    client = JudgeServiceClient(base_url="http://fake:8095")
    info = client.healthcheck(expected_rubric_version="family_rubric_v2")
    assert info["judge_model"] == "test-model"
    client.close()


def test_healthcheck_detects_rubric_version_drift(monkeypatch):
    def _mock_get(self, url, **kwargs):
        return httpx.Response(
            200,
            json={"status": "ok", "judge_model": "m", "rubric_version": "family_rubric_v3"},
            request=httpx.Request("GET", str(url)),
        )

    monkeypatch.setattr(httpx.Client, "get", _mock_get)
    client = JudgeServiceClient(base_url="http://fake:8095")
    with pytest.raises(RuntimeError, match="rubric_version drift"):
        client.healthcheck(expected_rubric_version="family_rubric_v2")
    client.close()


def test_fetch_catalog_returns_families(monkeypatch):
    def _mock_get(self, url, **kwargs):
        assert "/v1/catalog" in str(url)
        return httpx.Response(
            200,
            json={
                "rubric_version": "general_chat_rubric_v1",
                "judge_model": "test-model",
                "families": {
                    "general_chat": {"mode": "instant", "dimensions": []},
                },
            },
            request=httpx.Request("GET", str(url)),
        )

    monkeypatch.setattr(httpx.Client, "get", _mock_get)
    client = JudgeServiceClient(base_url="http://fake:8095")
    catalog = client.fetch_catalog()
    assert catalog["rubric_version"] == "general_chat_rubric_v1"
    assert "general_chat" in catalog["families"]
    client.close()
