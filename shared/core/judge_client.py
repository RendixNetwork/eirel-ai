from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

from shared.core.evaluation_models import AgreementJudgeOutput, VERDICT_SCORES

_logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = {429, 502, 503, 504}
_MAX_RETRIES = 2
_BASE_BACKOFF_SECONDS = 1.0


class JudgeServiceClient:
    """HTTP client for the eiretes outcome-only agreement judge.

    Calls ``POST /v1/judge/agreement`` and adapts the response into the
    shared ``AgreementJudgeOutput`` model. The judge itself returns just a
    verdict + rationale; this client derives the scalar ``agreement_score``
    from the verdict using ``VERDICT_SCORES``.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        transport: httpx.BaseTransport | None = None,
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
        if transport is not None:
            self._client = httpx.Client(timeout=self.timeout_seconds, transport=transport)
        else:
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

    def judge_agreement(
        self,
        *,
        family_id: str,
        prompt: str,
        response_a: str,
        response_b: str,
        task_mode: str | None = None,
        task_category: str | None = None,
        swap: bool = False,
    ) -> AgreementJudgeOutput:
        """Call the eiretes agreement judge and return ``AgreementJudgeOutput``.

        A is the candidate agent, B is the OpenAI baseline reference.
        Citations must have already been stripped from both responses by
        the caller — the judge never sees them.
        """
        t0 = time.monotonic()
        resp = self._request_with_retry(
            path="/v1/judge/agreement",
            json_body={
                "family_id": family_id,
                "prompt": prompt,
                "response_a": response_a,
                "response_b": response_b,
                "task_mode": task_mode,
                "task_category": task_category,
                "swap": swap,
            },
        )
        latency = time.monotonic() - t0
        _logger.debug(
            "agreement judge call: family=%s task_mode=%s category=%s swap=%s latency=%.2fs",
            family_id, task_mode, task_category, swap, latency,
        )
        body = resp.json()
        verdict = body.get("verdict")
        if verdict not in VERDICT_SCORES:
            verdict = "error"
        return AgreementJudgeOutput(
            verdict=verdict,
            agreement_score=body.get("agreement_score", VERDICT_SCORES.get(verdict, 0.0)),
            rationale=body.get("rationale", ""),
            swap_applied=body.get("swap_applied", False),
            model=body.get("model", ""),
            rubric_name=body.get("rubric_name", ""),
            metadata=body.get("metadata", {}),
        )

    def healthcheck(self, *, expected_rubric_version: str | None = None) -> dict[str, Any]:
        """Ping ``/healthz``; optionally assert the rubric version matches ours."""
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
        """Fetch the live rubric catalog from the judge service."""
        resp = self._client.get(f"{self.base_url}/v1/catalog")
        resp.raise_for_status()
        return resp.json()
