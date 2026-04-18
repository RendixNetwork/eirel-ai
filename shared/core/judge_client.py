from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

from shared.core.evaluation_models import JudgeResult

_logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = {429, 502, 503, 504}
_MAX_RETRIES = 2
_BASE_BACKOFF_SECONDS = 1.0


class JudgeServiceClient:
    """Thin HTTP client that calls the eiretes judge sidecar service.

    Replaces direct ``LLMJudgeClient`` usage — all judge configuration
    (model, API keys, rubric version, ensemble settings) lives in the
    eiretes service environment, not here.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
    ):
        self.base_url = (
            base_url
            or os.getenv("EIREL_JUDGE_SERVICE_URL", "http://eiretes-judge:8095")
        ).rstrip("/")
        self.timeout_seconds = float(
            timeout_seconds
            if timeout_seconds is not None
            else os.getenv("EIREL_JUDGE_TIMEOUT_SECONDS", "60")
        )
        self._client = httpx.Client(timeout=self.timeout_seconds)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> JudgeServiceClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _request_with_retry(
        self,
        *,
        path: str,
        json_body: dict[str, Any],
    ) -> httpx.Response:
        url = f"{self.base_url}{path}"
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = self._client.post(url, json=json_body)
                if resp.status_code in _RETRYABLE_STATUS_CODES and attempt < _MAX_RETRIES:
                    backoff = _BASE_BACKOFF_SECONDS * (2 ** attempt)
                    _logger.warning(
                        "judge %s returned %d, retrying in %.1fs (attempt %d/%d)",
                        path, resp.status_code, backoff, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(backoff)
                    continue
                resp.raise_for_status()
                return resp
            except httpx.ConnectError as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    backoff = _BASE_BACKOFF_SECONDS * (2 ** attempt)
                    _logger.warning(
                        "judge connection error for %s: %s, retrying in %.1fs (attempt %d/%d)",
                        path, exc, backoff, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(backoff)
                else:
                    _logger.error("judge connection failed after %d retries: %s", _MAX_RETRIES, exc)
                    raise
        raise last_exc  # type: ignore[misc]

    def judge(
        self,
        *,
        family_id: str,
        prompt: str,
        response_excerpt: str,
        mode: str = "instant",
        rubric_variant: str | None = None,
    ) -> JudgeResult:
        """Synchronous judge call. ``mode`` is one of ``instant`` / ``thinking``."""
        t0 = time.monotonic()
        resp = self._request_with_retry(
            path="/v1/judge",
            json_body={
                "family_id": family_id,
                "prompt": prompt,
                "response_excerpt": response_excerpt,
                "mode": mode,
                "rubric_variant": rubric_variant,
            },
        )
        latency = time.monotonic() - t0
        _logger.debug(
            "judge call: family=%s variant=%s latency=%.2fs",
            family_id, rubric_variant, latency,
        )
        return JudgeResult.model_validate(resp.json())

    def extract_research_claims(
        self,
        *,
        prompt: str,
        report_markdown: str,
        expert_guidance: list[str] | None = None,
        required_report_sections: list[str] | None = None,
        claim_index: list[dict[str, Any]] | None = None,
        batch_size: int = 20,
    ) -> dict[str, Any]:
        """Synchronous claim extraction matching ``LLMJudgeClient.extract_research_claims()``."""
        t0 = time.monotonic()
        resp = self._request_with_retry(
            path="/v1/extract-claims",
            json_body={
                "prompt": prompt,
                "report_markdown": report_markdown,
                "expert_guidance": expert_guidance or [],
                "required_report_sections": required_report_sections or [],
                "claim_index": claim_index or [],
                "batch_size": batch_size,
            },
        )
        latency = time.monotonic() - t0
        _logger.debug("extract_research_claims: latency=%.2fs", latency)
        return resp.json()

    def healthcheck(self, *, expected_rubric_version: str | None = None) -> dict[str, Any]:
        """Ping ``/healthz``; optionally assert the rubric version matches ours.

        Call from lifespan startup so the first judge call isn't also the first
        time the service URL is exercised. Raises :class:`httpx.HTTPError` if the
        sidecar is unreachable and ``RuntimeError`` if the rubric version has
        drifted.
        """
        resp = self._client.get(f"{self.base_url}/healthz")
        resp.raise_for_status()
        data = resp.json()
        _logger.info(
            "judge healthcheck ok: model=%s rubric=%s",
            data.get("judge_model"),
            data.get("rubric_version"),
        )
        if expected_rubric_version and data.get("rubric_version") != expected_rubric_version:
            raise RuntimeError(
                f"judge rubric_version drift: service reports "
                f"{data.get('rubric_version')!r} but expected {expected_rubric_version!r}"
            )
        return data

    def fetch_catalog(self) -> dict[str, Any]:
        """Fetch the live rubric catalog from the judge service.

        Returns ``{"rubric_version", "judge_model", "families": {...}}``.
        Consumers should cache this and re-fetch when rubric_version changes.
        """
        resp = self._client.get(f"{self.base_url}/v1/catalog")
        resp.raise_for_status()
        return resp.json()
