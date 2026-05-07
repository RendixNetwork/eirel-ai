from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

_logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = {429, 502, 503, 504}
_MAX_RETRIES = 2
_BASE_BACKOFF_SECONDS = 1.0


def _judge_base_url() -> str:
    return os.getenv("EIREL_JUDGE_SERVICE_URL", "http://eiretes-judge:8095").rstrip("/")


class JudgeServiceClient:
    """HTTP client for the eiretes reference-based eval judge.

    Calls ``POST /v1/judge/eval`` (LLM-as-judge against expected_answer)
    and ``POST /v1/judge/eval/composite`` (pure-function multiplicative
    composite). Eiretes is a pure judge service — the validator engine
    is the dispatch coordinator; this client just brokers the per-item
    judge calls.
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
            else os.getenv("EIREL_EVAL_JUDGE_TIMEOUT_SECONDS", "60")
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

    def judge_eval(
        self,
        *,
        bundle: dict[str, Any],
        expected_answer: str,
        must_not_claim: list[str] | None = None,
        required_tool: str | None = None,
        oracle_source: str = "deterministic",
    ) -> dict[str, Any]:
        """Reference-based judge call against eiretes' ``/v1/judge/eval``.

        The validator engine builds ``bundle`` (a JudgeInputBundle dict
        with task-shape fields + candidate response in ``answers[0]``)
        and passes per-call extras alongside. ``expected_answer`` +
        ``must_not_claim`` come from the validator's per-task in-memory
        cache populated by the oracle/reconciler enrichment phase (or
        from ``task.expected_output`` for ``deterministic`` items).

        ``oracle_source`` ∈ {``three_oracle``, ``deterministic``}
        signals which outcomes the judge is allowed to return —
        ``disputed`` is only valid for ``three_oracle`` items.

        Returns a dict ``{outcome, failure_mode, guidance}``. Outcome ∈
        {correct, partial, wrong, hallucinated, refused, disputed}.
        """
        body: dict[str, Any] = {
            "bundle": bundle,
            "expected_answer": expected_answer,
            "must_not_claim": list(must_not_claim or []),
            "required_tool": required_tool,
            "oracle_source": oracle_source,
        }
        resp = self._request_with_retry(path="/v1/judge/eval", json_body=body)
        return resp.json()

    def judge_multi(
        self,
        *,
        bundle: dict[str, Any],
        applicable_metrics: list[str],
        expected_answer: str | None = None,
        candidate_citations: list[str] | None = None,
    ) -> dict[str, Any]:
        """Outer-metric judge call against eiretes' ``/v1/judge/multi``.

        ``bundle.answers`` MUST be ``[candidate_response]`` (single
        candidate). Per-call extras: ``expected_answer``,
        ``candidate_citations``, ``applicable_metrics``.

        Returns a dict whose keys are the requested dimensions
        (``grounded_correctness`` / ``retrieval_quality`` /
        ``instruction_safety``); each value is ``{score, rationale}``.
        Missing keys mean the dimension wasn't requested or the judge
        chose to skip it (caller should treat as N/A and re-normalize).
        """
        body: dict[str, Any] = {
            "bundle": bundle,
            "applicable_metrics": list(applicable_metrics),
            "candidate_citations": list(candidate_citations or []),
        }
        if expected_answer is not None:
            body["expected_answer"] = expected_answer
        resp = self._request_with_retry(path="/v1/judge/multi", json_body=body)
        return resp.json()

    def judge_pairwise(
        self,
        *,
        bundle: dict[str, Any],
        expected_answer: str | None = None,
    ) -> dict[str, Any]:
        """Pairwise preference call against eiretes' ``/v1/judge/pairwise``.

        ``bundle.answers`` MUST be ``[answer_a, answer_b]``. Position-
        bias defense (running twice with A/B swapped, OR random per-task
        A/B assignment) is the caller's responsibility — eiretes judges
        one ordering at a time.

        ``expected_answer`` (optional) is the consensus / deterministic
        gold; when supplied the judge anchors preference on factual
        agreement, not style alone.

        Returns ``{winner: "A"|"B"|"tie", confidence, reason,
        category_scores}``.
        """
        body: dict[str, Any] = {"bundle": bundle}
        if expected_answer is not None and expected_answer.strip():
            body["expected_answer"] = expected_answer
        resp = self._request_with_retry(path="/v1/judge/pairwise", json_body=body)
        return resp.json()

    def judge_eval_composite(
        self,
        *,
        outcome: str,
        failure_mode: str | None = None,
        candidate_response: str = "",
        must_not_claim: list[str] | None = None,
        required_tool: str | None = None,
        ledger_tools: list[str] | None = None,
        latency_ms: int = 0,
        cost_usd: float = 0.0,
        latency_budget_ms: int | None = None,
        cost_budget_usd: float | None = None,
        cost_floor_usd: float | None = None,
        grounded_correctness_score: float | None = None,
        instruction_safety_score: float | None = None,
        pairwise_preference_score: float | None = None,
    ) -> dict[str, Any]:
        """Compute the multiplicative composite via eiretes.

        No LLM call server-side — pure-function shape mirrored from
        ``eiretes.eval.composite``. Validator passes its outcome from
        ``judge_eval`` plus the orchestrator-attested ledger tools and
        per-task cost; gets back the composite score with knockout
        reasons populated.
        """
        body: dict[str, Any] = {
            "outcome": outcome,
            "failure_mode": failure_mode,
            "candidate_response": candidate_response,
            "must_not_claim": list(must_not_claim or []),
            "required_tool": required_tool,
            "ledger_tools": list(ledger_tools or []),
            "latency_ms": int(max(0, latency_ms)),
            "cost_usd": float(max(0.0, cost_usd)),
        }
        if latency_budget_ms is not None:
            body["latency_budget_ms"] = int(latency_budget_ms)
        if cost_budget_usd is not None:
            body["cost_budget_usd"] = float(cost_budget_usd)
        if cost_floor_usd is not None:
            body["cost_floor_usd"] = float(cost_floor_usd)
        if grounded_correctness_score is not None:
            body["grounded_correctness_score"] = float(grounded_correctness_score)
        if instruction_safety_score is not None:
            body["instruction_safety_score"] = float(instruction_safety_score)
        if pairwise_preference_score is not None:
            body["pairwise_preference_score"] = float(pairwise_preference_score)
        resp = self._request_with_retry(
            path="/v1/judge/eval/composite", json_body=body,
        )
        return resp.json()

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
