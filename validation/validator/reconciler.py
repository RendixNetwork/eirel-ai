"""Three-oracle reconciler.

Single LLM call (Chutes-hosted ``zai-org/GLM-5.1-TEE``) that
synthesizes 3 oracle answers into:

  * ``consensus_claims`` — atomic facts ALL 3 (or all available)
    oracles support.
  * ``majority_claims`` — facts ≥2/3 oracles support, with the
    supporting-vendor list per claim.
  * ``minority_claims`` — facts only 1 oracle supports (dropped from
    ``expected_claims`` but kept for telemetry).
  * ``must_not_claim_extras`` — atomic claims any oracle explicitly
    contradicts; ADDITIVE to the template-time floor.
  * ``oracle_status`` — ``consensus`` | ``majority`` | ``disputed``.
  * ``disagreement_note`` — one-line summary.

Failure path: malformed JSON, LLM error, or fewer than 2 successful
oracles → ``ReconciledOracle(oracle_status="disputed",
expected_claims=[], must_not_claim=template_floor)``. Items still go
to judging; ``disputed`` outcomes contribute 0.5.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from validation.validator.eval_config import reconciler_config
from validation.validator.oracles.base import OracleGrounding
from validation.validator.oracles.fanout import (
    successful_groundings,
    vendor_status_map,
)
from validation.validator.providers.openai_compatible import (
    OpenAICompatibleClient,
)
from validation.validator.providers.types import (
    ProviderError,
    ProviderTimeout,
)

_logger = logging.getLogger(__name__)


OracleConsensusStatus = Literal[
    "consensus", "majority", "disputed", "deterministic",
]


@dataclass(frozen=True)
class ReconciledOracle:
    """Output of the reconciler — consumed by ``_judge_miner``.

    ``expected_claims`` is the set the judge scores satisfaction
    against (consensus + majority claims combined). ``must_not_claim``
    is the union of template floor + reconciler extras. Telemetry
    fields (``consensus_claims`` / ``majority_claims`` /
    ``minority_claims`` / ``vendor_status``) are persisted on the
    TaskMinerResult row for per-vendor agreement-rate analysis.
    """

    expected_claims: list[str] = field(default_factory=list)
    must_not_claim: list[str] = field(default_factory=list)
    oracle_status: OracleConsensusStatus = "disputed"
    disagreement_note: str | None = None
    consensus_claims: list[str] = field(default_factory=list)
    majority_claims: list[dict[str, Any]] = field(default_factory=list)
    minority_claims: list[dict[str, Any]] = field(default_factory=list)
    vendor_status: dict[str, str] = field(default_factory=dict)
    reconciler_latency_ms: int = 0
    reconciler_cost_usd: float | None = None
    # URLs each vendor cited during their grounded call. Persisted
    # alongside ``vendor_status`` so the dashboard can show "where did
    # the oracle get this answer." Empty for ``deterministic`` items
    # and for vendors that errored / returned no citations.
    vendor_citations: dict[str, list[str]] = field(default_factory=dict)
    # Raw answer text per vendor, kept verbatim from each oracle's
    # ``OracleGrounding.raw_text``. Used as the pairwise-comparator
    # reference instead of running a separate OpenAI baseline call —
    # the ``EIREL_VALIDATOR_PAIRWISE_REFERENCE_VENDOR`` env var picks
    # which vendor's answer becomes the comparator. Empty for
    # ``deterministic`` items (where the pool's pre-baked answer in
    # ``expected_claims[0]`` serves as the reference instead).
    vendor_answers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_deterministic(
        cls,
        *,
        answer: str,
        must_not_claim_floor: Iterable[str] = (),
    ) -> "ReconciledOracle":
        """Build a ``ReconciledOracle`` for ``oracle_source=deterministic``
        items. The pool's built-in grader (live_endpoint, sandbox_python,
        span F1, regex) is the truth; no oracle/reconciler call needed.

        ``answer`` is the pool's pre-baked gold; the judge consumes it
        as the single member of ``expected_claims``. ``must_not_claim``
        carries only the template floor (no reconciler extras possible
        without an LLM call).
        """
        cleaned_answer = (answer or "").strip()
        return cls(
            expected_claims=[cleaned_answer] if cleaned_answer else [],
            must_not_claim=_dedupe_preserving_order(must_not_claim_floor),
            oracle_status="deterministic",
            consensus_claims=[cleaned_answer] if cleaned_answer else [],
            disagreement_note=None,
            vendor_status={},
        )


_RECONCILER_SYSTEM_PROMPT = """You synthesize oracle answers for an \
agent-evaluation pipeline. Up to 3 independent oracles answered the \
same user prompt. Your job is to extract atomic factual claims and \
classify how strongly the oracles support each.

Decompose each oracle's answer into ATOMIC claims (one fact each). \
For each unique claim across the answers, classify:

  * consensus_claims — atomic claim ALL successful oracles support \
    explicitly or by clear paraphrase. Plain strings.
  * majority_claims — atomic claim ≥2 oracles support but not all. \
    Objects with ``claim`` (string) and ``supporting_oracles`` \
    (list of vendor names).
  * minority_claims — atomic claim only 1 oracle supports. Objects \
    with ``claim`` and ``supporting_oracle`` (single vendor).
  * must_not_claim_extras — atomic claims any oracle EXPLICITLY \
    contradicts another's. Plain strings, phrased as the false claim \
    (NOT the correction).

Then set ``oracle_status``:
  * "consensus" — every claim has ALL-oracle support (every claim is \
    in consensus_claims; majority_claims may be empty).
  * "majority" — at least one claim is in majority_claims (≥2/3 \
    support but not unanimous).
  * "disputed" — no claim has ≥2-oracle support, OR the oracles \
    explicitly contradict each other on the user's question.

``disagreement_note`` is one short sentence summarizing where the \
oracles diverged (or null if there's no meaningful disagreement).

Return strict JSON matching the response schema exactly."""


_RECONCILER_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "consensus_claims", "majority_claims", "minority_claims",
        "must_not_claim_extras", "oracle_status",
    ],
    "properties": {
        "consensus_claims": {
            "type": "array", "items": {"type": "string"},
        },
        "majority_claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["claim", "supporting_oracles"],
                "properties": {
                    "claim": {"type": "string"},
                    "supporting_oracles": {
                        "type": "array", "items": {"type": "string"},
                    },
                },
            },
        },
        "minority_claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["claim", "supporting_oracle"],
                "properties": {
                    "claim": {"type": "string"},
                    "supporting_oracle": {"type": "string"},
                },
            },
        },
        "must_not_claim_extras": {
            "type": "array", "items": {"type": "string"},
        },
        "oracle_status": {
            "type": "string",
            "enum": ["consensus", "majority", "disputed"],
        },
        "disagreement_note": {"type": ["string", "null"]},
    },
}


class Reconciler:
    """Wraps the Chutes-hosted GLM-5.1-TEE reconciler call.

    One instance per validator process (re-uses an httpx session).
    Construction without an explicit ``client`` builds one from
    ``reconciler_config()``. Tests pass a pre-built client backed by
    ``httpx.MockTransport``.
    """

    def __init__(
        self,
        client: OpenAICompatibleClient | None = None,
    ) -> None:
        self._client = client or OpenAICompatibleClient(reconciler_config())

    async def aclose(self) -> None:
        await self._client.aclose()

    async def reconcile(
        self,
        *,
        prompt: str,
        groundings: list[OracleGrounding],
        must_not_claim_floor: Iterable[str] = (),
    ) -> ReconciledOracle:
        """Reconcile a list of oracle groundings into ``expected_claims``.

        ``must_not_claim_floor`` carries any template-time forbidden
        claims; the final ``must_not_claim`` is their union with the
        reconciler-derived extras.
        """
        floor_list = _dedupe_preserving_order(must_not_claim_floor)
        ok = successful_groundings(groundings)
        statuses = vendor_status_map(groundings)
        # Capture each vendor's citations regardless of whether they
        # ended up in the consensus — operators want to audit who
        # cited what even on disputed runs.
        vendor_citations = {
            g.vendor: list(g.citations)
            for g in groundings
            if g.citations
        }
        # Per-vendor raw answers — preserved so the engine can use
        # one of them as the pairwise comparator (replaces the legacy
        # OpenAI baseline call). Only successful groundings carry
        # usable text.
        vendor_answers = {
            g.vendor: g.raw_text
            for g in ok
            if g.raw_text
        }

        if len(ok) < 2:
            # 0-1 oracle survived: not enough signal for plurality
            # voting. Fail-safe → disputed with template floor only.
            return ReconciledOracle(
                expected_claims=[],
                must_not_claim=floor_list,
                oracle_status="disputed",
                disagreement_note=(
                    f"only {len(ok)}/{len(groundings)} oracles "
                    f"returned usable answers"
                ),
                vendor_status=statuses,
                vendor_citations=vendor_citations,
                vendor_answers=vendor_answers,
            )

        try:
            user_prompt = _build_reconciler_user_prompt(prompt, ok)
            resp = await self._client.complete_structured(
                system=_RECONCILER_SYSTEM_PROMPT,
                user=user_prompt,
                response_schema=_RECONCILER_RESPONSE_SCHEMA,
                schema_name="reconciler_output",
            )
        except (ProviderTimeout, ProviderError) as exc:
            _logger.warning(
                "reconciler call failed: %s (vendors=%s)", exc, statuses,
            )
            return ReconciledOracle(
                expected_claims=[],
                must_not_claim=floor_list,
                oracle_status="disputed",
                disagreement_note=f"reconciler_error: {exc}",
                vendor_status=statuses,
                vendor_citations=vendor_citations,
                vendor_answers=vendor_answers,
            )

        try:
            parsed = json.loads(resp.text)
        except json.JSONDecodeError as exc:
            _logger.warning(
                "reconciler returned malformed JSON: %s (raw=%s)",
                exc, resp.text[:512],
            )
            return ReconciledOracle(
                expected_claims=[],
                must_not_claim=floor_list,
                oracle_status="disputed",
                disagreement_note=f"reconciler_malformed_json: {exc}",
                vendor_status=statuses,
                vendor_citations=vendor_citations,
                vendor_answers=vendor_answers,
                reconciler_latency_ms=resp.latency_ms,
                reconciler_cost_usd=resp.usage_usd,
            )

        consensus = _string_list(parsed.get("consensus_claims"))
        majority = _claim_list(parsed.get("majority_claims"), "supporting_oracles")
        minority = _claim_list(parsed.get("minority_claims"), "supporting_oracle")
        extras = _string_list(parsed.get("must_not_claim_extras"))
        status = parsed.get("oracle_status")
        if status not in ("consensus", "majority", "disputed"):
            _logger.warning(
                "reconciler returned invalid oracle_status=%r; downgrading to disputed",
                status,
            )
            status = "disputed"
        note = parsed.get("disagreement_note")
        if not isinstance(note, str):
            note = None

        # ``expected_claims`` = consensus + majority (claim text only).
        # The judge scores miner satisfaction against this set;
        # minority claims are dropped from scoring but kept for
        # telemetry above.
        expected_claims = list(consensus)
        for entry in majority:
            text = entry.get("claim")
            if isinstance(text, str) and text:
                expected_claims.append(text)
        expected_claims = _dedupe_preserving_order(expected_claims)

        # ``must_not_claim`` = union(template_floor, reconciler_extras),
        # deduplicated preserving floor order so floor stays first.
        must_not_claim = _dedupe_preserving_order(list(floor_list) + extras)

        return ReconciledOracle(
            expected_claims=expected_claims,
            must_not_claim=must_not_claim,
            oracle_status=status,
            disagreement_note=note,
            consensus_claims=consensus,
            majority_claims=majority,
            minority_claims=minority,
            vendor_status=statuses,
            vendor_citations=vendor_citations,
            vendor_answers=vendor_answers,
            reconciler_latency_ms=resp.latency_ms,
            reconciler_cost_usd=resp.usage_usd,
        )


# -- prompt building ------------------------------------------------------


def _build_reconciler_user_prompt(
    prompt: str, groundings: list[OracleGrounding],
) -> str:
    payload = {
        "user_prompt": prompt,
        "oracle_answers": [
            {"vendor": g.vendor, "answer": g.raw_text}
            for g in groundings
        ],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


# -- response shape coercion ---------------------------------------------


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [v.strip() for v in value if isinstance(v, str) and v.strip()]


def _claim_list(value: Any, oracle_key: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        claim = item.get("claim")
        if not isinstance(claim, str) or not claim.strip():
            continue
        oracle_value = item.get(oracle_key)
        if oracle_key.endswith("s") and isinstance(oracle_value, list):
            oracle_value = [
                v for v in oracle_value if isinstance(v, str)
            ]
        elif not isinstance(oracle_value, str):
            continue
        out.append({"claim": claim.strip(), oracle_key: oracle_value})
    return out


def _dedupe_preserving_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        cleaned = (item or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


__all__ = [
    "OracleConsensusStatus",
    "ReconciledOracle",
    "Reconciler",
]
