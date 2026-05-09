from __future__ import annotations

import logging
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx

logger = logging.getLogger(__name__)

from shared.common.bittensor_signing import load_signer
from shared.common.security import sha256_hex
from eirel.groups import ensure_active_family_id
from shared.core.evaluation_models import MinerBenchmarkTarget
from shared.scoring.multi_metric import (
    applicable_metrics as _multi_metric_applicable,
    assemble_task_score as _multi_metric_assemble,
    derive_task_type as _multi_metric_derive_task_type,
    score_latency_cost as _multi_metric_latency_cost,
    score_tool_routing as _multi_metric_tool_routing,
)
# C4: judge runs server-side via owner-api judge proxy — no direct
# JudgeServiceClient import needed here anymore.
from shared.contracts.models import MinerRegistryEntry
from validation.validator import metrics as _metrics
from validation.validator.eval_config import (
    gemini_oracle_config,
    grok_oracle_config,
    openai_oracle_config,
    reconciler_config,
)
from validation.validator.oracles import (
    GeminiOracle,
    GrokOracle,
    OpenAIOracle,
    OracleClient,
    OracleContext,
    OracleFanout,
)
from validation.validator.providers.gemini import GeminiClient
from validation.validator.providers.openai_compatible import (
    OpenAICompatibleClient,
)
from validation.validator.reconciler import ReconciledOracle, Reconciler


def _owner_api_url() -> str:
    return os.getenv("OWNER_API_URL", "http://owner-api:8000").rstrip("/")


# One-shot guard so the comparator-choice banner only logs at startup,
# not on every poll iteration of ``run_distributed_benchmarks``.
_PAIRWISE_BANNER_LOGGED: dict[str, bool] = {"done": False}


@dataclass(frozen=True)
class _SyntheticBaselineResponse:
    """Pairwise-comparator reference, sourced from the chosen oracle's
    cached answer. ``source_vendor`` is preserved for telemetry +
    ``baseline_response_json``; cost/latency stay on the dataclass to
    keep ``baseline_response_json`` schema-stable for older readers.
    """

    response_text: str
    citations: list[dict[str, Any]]
    cost_usd: float
    latency_seconds: float
    source_vendor: str

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        del mode  # parity with Pydantic BaseModel.model_dump signature
        return {
            "response_text": self.response_text,
            "citations": list(self.citations),
            "cost_usd": self.cost_usd,
            "latency_seconds": self.latency_seconds,
            "source_vendor": self.source_vendor,
        }


def _select_pairwise_reference(
    *, reconciled: "ReconciledOracle", preferred_vendor: str,
) -> tuple[str, str]:
    """Pick the comparator text + tag where it came from.

    Order:
      1. ``preferred_vendor`` (env-configured oracle, default openai)
      2. Any other vendor's raw answer (round-robin among the 3)
      3. ``expected_claims[0]`` (consensus / majority / deterministic gold)
      4. Empty string with source ``"none"``
    """
    answers = reconciled.vendor_answers or {}
    text = (answers.get(preferred_vendor) or "").strip()
    if text:
        return text, preferred_vendor
    for vendor, raw in answers.items():
        if raw and raw.strip():
            return raw.strip(), f"{vendor}_fallback"
    if reconciled.expected_claims:
        first = (reconciled.expected_claims[0] or "").strip()
        if first:
            tag = (
                "deterministic"
                if reconciled.oracle_status == "deterministic"
                else "consensus_claim"
            )
            return first, tag
    return "", "none"


def _build_oracle_layer() -> tuple[OracleFanout | None, Reconciler | None]:
    """Lazy-init the 3-oracle fanout + Chutes reconciler.

    Each provider config is checked independently; missing creds for a
    given vendor mean that oracle is skipped (Grok-down precedent —
    fanout degrades gracefully). If fewer than 2 oracles are
    configured OR the reconciler is missing, the fanout returns None
    and the validator falls back to deterministic-only scoring for
    every task (regardless of oracle_source tag).

    Reconciler config defaults to ``zai-org/GLM-5.1-TEE`` via Chutes
    — same model used by the eiretes judge roles for TEE attestation.
    """
    oracle_clients: list[OracleClient] = []
    openai_cfg = openai_oracle_config()
    if openai_cfg.configured:
        oracle_clients.append(
            OpenAIOracle(client=OpenAICompatibleClient(openai_cfg)),
        )
    gemini_cfg = gemini_oracle_config()
    if gemini_cfg.configured:
        oracle_clients.append(
            GeminiOracle(client=GeminiClient(gemini_cfg)),
        )
    grok_cfg = grok_oracle_config()
    if grok_cfg.configured:
        oracle_clients.append(
            GrokOracle(client=OpenAICompatibleClient(grok_cfg)),
        )

    rec_cfg = reconciler_config()
    if len(oracle_clients) < 2 or not rec_cfg.configured:
        # Not enough vendors for plurality voting OR reconciler not
        # available. Three_oracle items will fall back to deterministic
        # paths (or surface as ``oracle_status="disputed"`` with empty
        # expected_claims).
        if oracle_clients:
            for client in oracle_clients:
                # Tear down half-built oracle clients so we don't leak
                # httpx sessions when the layer is not actually used.
                # Synchronous close not available; leak is bounded
                # because clients are lazy-init only when called.
                pass
        return None, None

    fanout = OracleFanout(oracle_clients)
    reconciler = Reconciler(
        client=OpenAICompatibleClient(rec_cfg),
    )
    return fanout, reconciler


async def _enrich_task_oracle(
    task_obj: Any,
    *,
    fanout: OracleFanout | None,
    reconciler: Reconciler | None,
) -> ReconciledOracle:
    """Produce a ``ReconciledOracle`` for one task.

    For ``oracle_source="three_oracle"``: runs the configured oracle
    fanout in parallel + Chutes reconciler. Falls back to disputed
    when fanout/reconciler aren't configured.

    For ``oracle_source="deterministic"`` or unset: builds a minimal
    ``ReconciledOracle`` from the task's pre-baked
    ``expected_output.answer`` and ``expected_output.must_not_claim``.
    No LLM calls.
    """
    expected_output = getattr(task_obj, "expected_output", None) or {}
    answer = str(expected_output.get("answer") or "").strip()
    must_not_claim_floor = list(expected_output.get("must_not_claim") or [])
    oracle_source = getattr(task_obj, "oracle_source", None)

    # Deterministic path: pool's grader produced the gold; no oracle
    # call needed.
    if oracle_source != "three_oracle":
        return ReconciledOracle.from_deterministic(
            answer=answer, must_not_claim_floor=must_not_claim_floor,
        )

    # Three-oracle path: degrade to disputed if the layer isn't wired.
    if fanout is None or reconciler is None:
        logger.warning(
            "three_oracle task %s but oracle layer not configured; "
            "falling back to disputed with template floor",
            getattr(task_obj, "task_id", "?"),
        )
        return ReconciledOracle(
            expected_claims=[],
            must_not_claim=must_not_claim_floor,
            oracle_status="disputed",
            disagreement_note="oracle_layer_not_configured",
        )

    context = OracleContext(
        task_id=str(getattr(task_obj, "task_id", "")),
        prompt=str(getattr(task_obj, "prompt", "") or ""),
        conversation_recent=list(getattr(task_obj, "turns", None) or []),
        attached_document=str(expected_output.get("attached_document") or "") or None,
        category=str(getattr(task_obj, "category", "") or "") or None,
    )
    groundings = await fanout.run(context)
    return await reconciler.reconcile(
        prompt=context.prompt,
        groundings=groundings,
        must_not_claim_floor=must_not_claim_floor,
    )


async def _fetch_ledger_tools(
    job_id: str, *, owner_url: str, signer,
) -> list[str]:
    """Fetch tool names invoked under ``job_id`` from the orchestrator
    ledger. Returns the deduped list in arrival order.

    Authenticates with the validator's hotkey signature — the owner-api
    gates ``/v1/internal/eval/job_ledger`` on ``validator_dependency``,
    so any registered active validator can read the ledger for any job
    they're scoring. Missing job_id → empty list. Network errors also
    return [] (fail-safe: composite's tool_attestation factor will be 0
    for required_tool tasks if the ledger is unreachable).
    """
    if not job_id:
        return []
    path = f"/v1/internal/eval/job_ledger?job_id={job_id}"
    url = f"{owner_url}{path}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                url,
                headers=_signed_headers(
                    signer=signer, method="GET", path=path, body=b"",
                ),
            )
        if resp.status_code != 200:
            logger.warning(
                "ledger fetch returned %d for job_id=%s: %s",
                resp.status_code, job_id, (resp.text or "")[:200],
            )
            return []
        payload = resp.json()
    except Exception as exc:
        logger.warning(
            "ledger fetch failed for job_id=%s: %s", job_id, exc,
        )
        return []
    tool_names: list[str] = []
    seen: set[str] = set()
    for row in payload.get("tool_calls") or []:
        name = str((row or {}).get("tool_name") or "").strip()
        if name and name not in seen:
            seen.add(name)
            tool_names.append(name)
    return tool_names


def _hydrate_agent_inputs(task_payload: dict[str, Any]) -> dict[str, Any]:
    """Fold top-level task fields into the ``inputs`` dict sent to the miner.

    The benchmark task JSON carries ``mode`` and ``web_search`` at the top
    level, but the miner agent reads its knobs from ``inputs.mode`` and
    ``inputs.web_search``. Without this bridge, every task is invoked with
    defaults (``mode=instant``, ``web_search=false``) regardless of the
    task spec, so the miner's web-search code path never runs during
    evaluation and citations stay empty. An existing value in ``inputs``
    always wins; ``allowed_tools`` is kept as a legacy fallback for older
    datasets that haven't been migrated to the explicit boolean.
    """
    base = dict(task_payload.get("inputs") or {})
    if "mode" not in base:
        mode = task_payload.get("mode")
        if isinstance(mode, str) and mode:
            base["mode"] = mode
    if "web_search" not in base:
        raw = task_payload.get("web_search")
        if isinstance(raw, bool):
            base["web_search"] = raw
        else:
            allowed = task_payload.get("allowed_tools") or []
            if isinstance(allowed, list) and "web_search" in allowed:
                base["web_search"] = True
    return base


def _rewrite_benchmark_endpoint_for_host(endpoint: str) -> str:
    rewrite_base = (
        os.getenv("EIREL_BENCHMARK_ENDPOINT_BASE_URL")
        or os.getenv("API_GATEWAY_PUBLIC_URL")
        or ""
    ).strip()
    if not rewrite_base:
        return endpoint
    parsed_endpoint = urlparse(endpoint)
    parsed_base = urlparse(rewrite_base)
    if not parsed_endpoint.scheme or not parsed_endpoint.netloc:
        return endpoint
    if not parsed_base.scheme or not parsed_base.netloc:
        return endpoint
    if parsed_endpoint.hostname not in {"api-gateway", "owner-api"}:
        return endpoint
    return urlunparse(
        (
            parsed_base.scheme,
            parsed_base.netloc,
            parsed_endpoint.path,
            parsed_endpoint.params,
            parsed_endpoint.query,
            parsed_endpoint.fragment,
        )
    )


def _load_validator_signer():
    return load_signer(
        mnemonic=os.getenv("EIREL_VALIDATOR_MNEMONIC") or None,
        wallet_name=os.getenv("EIREL_VALIDATOR_WALLET_NAME") or None,
        hotkey_name=os.getenv("EIREL_VALIDATOR_HOTKEY_NAME") or None,
        wallet_path=os.getenv("EIREL_VALIDATOR_WALLET_PATH") or None,
    )


def _signed_headers(*, signer, method: str, path: str, body: bytes) -> dict[str, str]:
    return signer.signed_headers(method, path, sha256_hex(body))


def _pairwise_miner_score(*, winner: str, miner_position: str) -> float:
    """Map a single pairwise call's winner to a miner-perspective score.

    ``miner_position`` is whichever of "A"/"B" the miner's answer was
    placed in for that call. The two-call swap ensures each call sees
    the miner in a different position; averaging the two miner-perspective
    scores cancels position bias.
    """
    w = (winner or "").strip().lower()
    if w == "tie":
        return 0.5
    if (w == "a" and miner_position == "A") or (w == "b" and miner_position == "B"):
        return 1.0
    return 0.0


def _build_pairwise_prompt(task_obj: Any) -> str:
    """Render the user prompt for the pairwise judge.

    Single-turn tasks use ``task_obj.prompt`` directly. Multi-turn
    fixtures fold the conversation into a chronological transcript so
    the judge sees the same context the candidates saw.
    """
    prompt = getattr(task_obj, "prompt", "") or ""
    turns = getattr(task_obj, "turns", None) or []
    if not turns:
        return prompt
    rendered: list[str] = []
    for turn in turns:
        if isinstance(turn, dict):
            user_msg = turn.get("user")
            asst_msg = turn.get("assistant")
        else:
            user_msg = getattr(turn, "user", None)
            asst_msg = getattr(turn, "assistant", None)
        if user_msg:
            rendered.append(f"USER: {user_msg}")
        if asst_msg:
            rendered.append(f"ASSISTANT: {asst_msg}")
    if prompt:
        rendered.append(f"USER: {prompt}")
    return "\n".join(rendered) if rendered else prompt


def _extract_answer_text(run) -> str:
    """Pull the miner's final-answer text out of a BenchmarkTaskRun.

    The pairwise judge sees ONLY the final answer — no tool_calls, no
    citations, no trace, no envelope metadata. Returning a dict-repr
    string here is a critical failure: it lets the judge identify which
    side is the miner from formatting alone (the OpenAI baseline returns
    clean prose), which collapses the swap defense.

    The current graph SDK puts the answer at ``response.output.answer``.
    Older shapes used ``output.content`` / ``output.text`` / a list of
    text blocks. We try them in order and DO NOT fall through to a
    repr — if no shape matches, return empty string and let the row
    surface as a miner-side defect.
    """
    resp = getattr(run, "response", None) or {}
    if not isinstance(resp, dict):
        return ""
    output = resp.get("output")
    if isinstance(output, dict):
        # Current shape — eirel.graph runtime emits {output: {answer: "..."}}
        for key in ("answer", "content", "text"):
            value = output.get(key)
            if isinstance(value, str) and value:
                return value
        # Some agents stream content as a list of {text: "..."} blocks
        content = output.get("content")
        if isinstance(content, list):
            parts = [
                (b.get("text") if isinstance(b, dict) else str(b))
                for b in content
                if b
            ]
            joined = "\n\n".join(p for p in parts if isinstance(p, str) and p)
            if joined:
                return joined
    elif isinstance(output, str) and output:
        return output
    # Last-resort fallbacks — top-level fields some BaseAgent miners use.
    for key in ("output_text", "response_text", "text", "content"):
        value = resp.get(key)
        if isinstance(value, str) and value:
            return value
    # Intentionally NOT falling through to str(resp): leaking the full
    # envelope dict as the candidate string lets the judge identify the
    # miner by its formatting and breaks the pairwise defense.
    return ""


def _extract_miner_citations(run) -> list[dict[str, Any]]:
    """Pull the miner's cited URLs out of a BenchmarkTaskRun for dashboard
    display only. These do NOT participate in scoring.

    The agent invocation helper surfaces citations and tool_calls at the
    *top* of ``response`` (alongside ``output``, ``status``, ``metadata``),
    not nested under ``output``. This extractor checks the top-level
    keys first, then falls back to ``response.output.*`` for older
    miner SDKs that emit there.
    """
    resp = getattr(run, "response", None) or {}
    if not isinstance(resp, dict):
        return []
    output = resp.get("output") if isinstance(resp.get("output"), dict) else {}

    citations: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(url: str, title: str) -> None:
        url = (url or "").strip()
        if not url or url in seen:
            return
        seen.add(url)
        citations.append({"url": url, "title": (title or "").strip()})

    # Structured citations — top-level (current SDK shape) or nested
    # under ``output`` (legacy fallback). The SDK envelope emits
    # citations as either bare URL strings (``["https://...", ...]``)
    # OR dicts with ``url``/``title`` keys; handle both.
    for source in (resp.get("citations"), output.get("citations")):
        if isinstance(source, list):
            for c in source:
                if isinstance(c, str):
                    _add(c, "")
                elif isinstance(c, dict):
                    _add(str(c.get("url") or ""), str(c.get("title") or ""))

    # Tool-call results — same dual location as above. ``tool_name`` is
    # the canonical key; older shapes used ``tool``.
    for source in (resp.get("tool_calls"), output.get("tool_calls")):
        if not isinstance(source, list):
            continue
        for tc in source:
            if not isinstance(tc, dict):
                continue
            kind = str(tc.get("tool_name") or tc.get("tool") or "").strip()
            if kind not in ("web_search", "url_fetch"):
                continue
            result_block = tc.get("result") or {}
            if not isinstance(result_block, dict):
                continue
            results = result_block.get("results") or []
            if not isinstance(results, list):
                continue
            for r in results:
                if isinstance(r, dict) and r.get("url"):
                    _add(str(r.get("url") or ""), str(r.get("title") or ""))
    return citations


# ---------------------------------------------------------------------------
# Distributed evaluation: autonomous polling loop
# ---------------------------------------------------------------------------

_POLL_INTERVAL_SECONDS = float(
    os.getenv("EIREL_VALIDATOR_POLL_INTERVAL_SECONDS")
    or os.getenv("VALIDATOR_POLL_INTERVAL_SECONDS")
    or "30"
)
_BATCH_SIZE = int(os.getenv("EIREL_VALIDATOR_BATCH_SIZE", "1"))
_MAX_PARALLEL = int(os.getenv("EIREL_VALIDATOR_MAX_PARALLEL", "3"))

# Miner latency SLA enforced at scoring time. A `completed` miner response
# whose total wall-clock latency exceeds the mode's budget is stamped
# `latency_violation` (counts as a loss in aggregation, regardless of
# content). Tasks without a recognized mode skip the gate. Network-level
# errors are not double-penalized.
#
# Mode-specific completion budgets:
#   instant: 120s — miners should pick a cheap/non-thinking model.
#   thinking: 600s — full reasoning + tool use allowed.
_INSTANT_MINER_LATENCY_BUDGET_SECONDS = float(
    os.getenv("EIREL_MINER_INSTANT_LATENCY_BUDGET_SECONDS", "120")
)
_THINKING_MINER_LATENCY_BUDGET_SECONDS = float(
    os.getenv("EIREL_MINER_THINKING_LATENCY_BUDGET_SECONDS", "600")
)
_ACTIVE_FAMILIES = [
    f.strip()
    for f in os.getenv("EIREL_VALIDATOR_ACTIVE_FAMILIES", "analyst,builder,verifier").split(",")
    if f.strip()
]


async def run_validator_loop() -> None:
    """Autonomous background loop: claim → evaluate → submit → repeat.

    Polls the owner-api for available tasks across all active families.
    When tasks are available, evaluates them in parallel and submits
    results.  When no tasks are available, sleeps and retries.

    Runs indefinitely until the process is stopped.
    """
    import asyncio

    signer = _load_validator_signer()
    owner_url = _owner_api_url()
    judge_model = os.getenv("EIREL_EVAL_JUDGE_MODEL", "local-rubric-judge")
    rubric_version = os.getenv("EIREL_VALIDATOR_RUBRIC_VERSION", "family_rubric_v2")

    logger.info(
        "validator loop starting: families=%s poll_interval=%.0fs batch_size=%d max_parallel=%d",
        _ACTIVE_FAMILIES, _POLL_INTERVAL_SECONDS, _BATCH_SIZE, _MAX_PARALLEL,
    )

    # Periodic idle heartbeat. Empty claim cycles used to be silent —
    # the operator couldn't tell the validator was alive without
    # poking the DB. Log a one-line "idle" every Nth empty cycle
    # (default 10 ≈ 5 minutes at 30s poll). Set
    # ``EIREL_VALIDATOR_IDLE_HEARTBEAT_EVERY=0`` to disable.
    _idle_heartbeat_every = int(
        os.getenv("EIREL_VALIDATOR_IDLE_HEARTBEAT_EVERY", "10")
    )
    _idle_cycles = 0

    while True:
        found_work = False
        for family_id in _ACTIVE_FAMILIES:
            try:
                result = await run_distributed_benchmarks(
                    family_id=family_id,
                    batch_size=_BATCH_SIZE,
                    max_parallel=_MAX_PARALLEL,
                    rubric_version=rubric_version,
                    judge_model=judge_model,
                )
                if result["total_claimed"] > 0:
                    found_work = True
                    logger.info(
                        "validator loop: family=%s claimed=%d submitted=%d failed=%d "
                        "baseline_failed=%d",
                        family_id, result["total_claimed"],
                        result["total_submitted"], result["total_failed"],
                        result.get("total_baseline_failed", 0),
                    )
            except Exception:
                logger.exception("validator loop error: family=%s", family_id)

        _metrics.validator_loop_last_success_timestamp_seconds.labels(
            loop_name="validator"
        ).set_to_current_time()
        if not found_work:
            _idle_cycles += 1
            if (
                _idle_heartbeat_every > 0
                and _idle_cycles % _idle_heartbeat_every == 0
            ):
                logger.info(
                    "validator loop: idle (no claimable tasks) families=%s "
                    "idle_cycles=%d",
                    _ACTIVE_FAMILIES, _idle_cycles,
                )
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        else:
            _idle_cycles = 0


# ---------------------------------------------------------------------------
# Weight-setting background loop
# ---------------------------------------------------------------------------

# 180 blocks × ~12s/block ≈ 36 minutes
_WEIGHT_SET_INTERVAL_BLOCKS = int(os.getenv("EIREL_WEIGHT_SET_INTERVAL_BLOCKS", "180"))
_WEIGHT_SET_INTERVAL_SECONDS = _WEIGHT_SET_INTERVAL_BLOCKS * 12
_MIN_STAKE_TO_SET_WEIGHTS = int(os.getenv("EIREL_MIN_STAKE_TO_SET_WEIGHTS", "5000"))


async def run_weight_setting_loop() -> None:
    """Background loop: poll owner-api for weights, set on-chain every cycle.

    Cadence is configurable via ``EIREL_WEIGHT_SET_INTERVAL_BLOCKS``
    (default 180 blocks ≈ 36 minutes). On each tick we re-publish the
    current target-run winners even if the target run hasn't changed
    since the last publication — Bittensor expects a ``set_weights``
    every window to keep vtrust from decaying.

    The owner-api's ``/v1/weights`` returns ``{hotkey: weight}`` for the
    latest *completed* run (run-(N-1) while run N is in progress). This
    loop syncs the metagraph to resolve hotkey→UID, assigns unallocated
    family weight to UID 0 (to burn alpha), and calls ``set_weights()``
    with retry, circuit breaker, and chain verification.
    """
    import asyncio
    import time as _time
    import bittensor as bt
    from validation.validator.chain_verifier import verify_weights_on_chain

    signer = _load_validator_signer()
    owner_url = _owner_api_url()
    network = os.getenv("BITTENSOR_NETWORK", "finney")
    netuid = int(os.getenv("BITTENSOR_NETUID", "0"))

    # Circuit breaker state: opens after consecutive failures to avoid
    # hammering the chain when it is consistently rejecting transactions.
    _max_attempts = 3
    _backoff_base = 5.0
    _cb_failure_threshold = 3
    _cb_recovery_timeout = 120.0
    _consecutive_failures = 0
    _circuit_opened_at: float = 0.0

    logger.info(
        "weight-setting loop starting: interval=%d blocks (~%ds) network=%s netuid=%d",
        _WEIGHT_SET_INTERVAL_BLOCKS, _WEIGHT_SET_INTERVAL_SECONDS, network, netuid,
    )

    while True:
        try:
            # --- Circuit breaker check ---
            if _consecutive_failures >= _cb_failure_threshold:
                elapsed = _time.monotonic() - _circuit_opened_at
                if elapsed < _cb_recovery_timeout:
                    logger.warning(
                        "weight-setting: circuit open (%d failures), recovery in %.0fs",
                        _consecutive_failures, _cb_recovery_timeout - elapsed,
                    )
                    await asyncio.sleep(min(30.0, _cb_recovery_timeout - elapsed))
                    continue
                logger.info("weight-setting: circuit half-open, retrying")

            # 1. Fetch {hotkey: weight} from owner-api. If owner-api is
            #    unreachable we still want to set weights on-chain (burning
            #    to UID 0) so the validator keeps its vtrust and doesn't
            #    skip the weight-setting window.
            path = "/v1/weights"
            owner_api_down = False
            data: dict[str, Any] = {}
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    response = await client.get(
                        f"{owner_url}{path}",
                        headers=_signed_headers(signer=signer, method="GET", path=path, body=b""),
                    )
                    response.raise_for_status()
                    data = response.json()
            except httpx.HTTPError as exc:
                logger.warning(
                    "weight-setting: owner-api fetch failed (%s) — falling back to burn-to-uid-0",
                    exc,
                )
                owner_api_down = True

            if owner_api_down:
                current_run_id = None
                weights_by_hotkey: dict[str, float] = {}
                family_winners: list[dict[str, Any]] = []
            else:
                if not data.get("ready"):
                    # No completed run yet; nothing to publish. Skipping here
                    # is fine because the chain hasn't started expecting
                    # weights from us.
                    logger.info("weight-setting: no weights ready, sleeping")
                    await asyncio.sleep(_WEIGHT_SET_INTERVAL_SECONDS)
                    continue

                current_run_id = data.get("run_id")
                # Bittensor expects set_weights every ~180 blocks regardless
                # of whether the target run id or winner set repeats — missing
                # a window decays vtrust. The owner-api's /v1/weights already
                # returns the latest completed run (run-(N-1) while run N is
                # open), so we just re-publish the same weights each cycle
                # until a newer run completes. Do NOT dedup on run_id here.
                weights_by_hotkey = data.get("weights", {})
                family_winners = data.get("family_winners", [])

            # 2. Sync metagraph to resolve hotkey → UID
            subtensor = bt.Subtensor(network=network)
            wallet = bt.Wallet(
                name=os.getenv("EIREL_VALIDATOR_WALLET_NAME"),
                hotkey=os.getenv("EIREL_VALIDATOR_HOTKEY_NAME"),
                path=os.getenv("EIREL_VALIDATOR_WALLET_PATH"),
            )
            metagraph = subtensor.metagraph(netuid=netuid, lite=True)
            if not metagraph.hotkeys:
                logger.error("weight-setting: metagraph returned empty hotkeys, skipping")
                await asyncio.sleep(_WEIGHT_SET_INTERVAL_SECONDS)
                continue
            uid_by_hotkey = {str(hk): uid for uid, hk in enumerate(metagraph.hotkeys)}

            # 3. Build UID weight vector
            uid_weights: dict[int, float] = {}
            total_assigned = 0.0

            for hotkey, weight in weights_by_hotkey.items():
                uid = uid_by_hotkey.get(hotkey)
                if uid is not None:
                    uid_weights[uid] = uid_weights.get(uid, 0.0) + float(weight)
                    total_assigned += float(weight)
                else:
                    logger.warning(
                        "weight-setting: hotkey %s not found in metagraph, weight goes to UID 0",
                        hotkey[:16],
                    )

            # 4. Burn the remainder of the weight vector to UID 0.
            #
            # The on-chain ``set_weights`` call normalises whatever vector
            # we publish so its components sum to 1.0 — i.e. emission is
            # distributed in proportion to the published weights, no matter
            # what their absolute magnitude. So if owner-api says
            # ``general_chat:0.5`` and we publish only ``[0.5]`` against
            # the winner, Bittensor renormalises that to ``[1.0]`` and the
            # winner ends up with 100% of subnet emission — defeating the
            # operator's intent to burn 50%.
            #
            # The fix: always publish a vector that sums to 1.0. Any
            # fraction not assigned to a winner — for any reason —
            # becomes burn:
            #
            #   * The operator-configured "non-family" share — when
            #     ``EIREL_FAMILY_WEIGHTS`` sums to less than 1.0 (e.g.
            #     ``general_chat:0.5`` → 50% intentional burn).
            #   * Allocated-to-family-but-no-winner — when a family was
            #     assigned weight but produced no qualifying winner this
            #     run (gate didn't pass).
            #   * Winner-not-in-metagraph — already excluded from
            #     ``total_assigned`` above; gets folded into burn here.
            #
            # All three collapse into ``burn = 1.0 - total_assigned``.
            total_family_weight = sum(
                float(fw.get("family_weight", 0.0))
                for fw in family_winners
            )
            burn = max(0.0, 1.0 - total_assigned)
            if burn > 0:
                uid_weights[0] = uid_weights.get(0, 0.0) + burn

            if not uid_weights:
                # Total fallback — owner-api returned nothing actionable.
                # Burn-everything keeps us alive on chain (vtrust intact).
                uid_weights[0] = 1.0
            elif total_assigned > 1.0:
                logger.warning(
                    "weight-setting: total_assigned=%.4f exceeds 1.0; "
                    "EIREL_FAMILY_WEIGHTS likely misconfigured (sum > 1.0)",
                    total_assigned,
                )

            uids = sorted(uid_weights.keys())
            weight_vals = [uid_weights[uid] for uid in uids]

            logger.info(
                "weight-setting: setting weights for %d UIDs run=%s mode=%s "
                "uids=%s weights=%s assigned=%.4f burn=%.4f family_total=%.4f",
                len(uids),
                current_run_id,
                "owner_api_down_burn" if owner_api_down else "normal",
                uids,
                [round(w, 4) for w in weight_vals],
                total_assigned,
                uid_weights.get(0, 0.0) if 0 in uid_weights else 0.0,
                total_family_weight,
            )

            # 5. Set weights on-chain with retry + exponential backoff
            success = False
            last_message = ""
            last_exc: BaseException | None = None
            _submission_started_at = time.perf_counter()
            _submission_mode = "submitted"
            for attempt in range(1, _max_attempts + 1):
                try:
                    result = subtensor.set_weights(
                        wallet=wallet,
                        netuid=netuid,
                        uids=uids,
                        weights=weight_vals,
                        wait_for_inclusion=True,
                        wait_for_finalization=False,
                        version_key=0,
                    )
                    # set_weights returns (success: bool, message: str)
                    if isinstance(result, tuple):
                        success, last_message = bool(result[0]), str(result[1])
                    else:
                        success, last_message = bool(result), str(result)
                    last_exc = None

                    if success:
                        logger.info(
                            "weight-setting: set_weights succeeded on attempt %d run=%s msg=%s",
                            attempt, current_run_id, last_message,
                        )
                        break
                    logger.warning(
                        "weight-setting: set_weights returned failure on attempt %d/%d: %s",
                        attempt, _max_attempts, last_message,
                    )
                except Exception as exc:
                    logger.warning(
                        "weight-setting: set_weights exception on attempt %d/%d: %s",
                        attempt, _max_attempts, exc,
                    )
                    last_message = str(exc)
                    last_exc = exc

                if attempt < _max_attempts:
                    backoff = _backoff_base * (2 ** (attempt - 1))
                    await asyncio.sleep(backoff)

            _submission_elapsed = time.perf_counter() - _submission_started_at
            _family_label = "all"  # loop publishes all families in one call
            _metrics.weight_setter_submission_duration_seconds.labels(
                family=_family_label, mode=_submission_mode
            ).observe(_submission_elapsed)
            _metrics.weight_setter_submissions_total.labels(
                family=_family_label,
                mode=_submission_mode,
                status="success" if success else "failed",
            ).inc()
            if success:
                _metrics.weight_setter_last_success_timestamp_seconds.labels(
                    family=_family_label
                ).set_to_current_time()
            else:
                error_type = _metrics.classify_chain_error(last_message, last_exc)
                _metrics.weight_setter_chain_errors_total.labels(error_type=error_type).inc()

            if not success:
                _consecutive_failures += 1
                if _consecutive_failures >= _cb_failure_threshold:
                    _circuit_opened_at = _time.monotonic()
                    logger.error(
                        "weight-setting: circuit breaker opened after %d consecutive failures: %s",
                        _consecutive_failures, last_message,
                    )
                await asyncio.sleep(min(60.0, _WEIGHT_SET_INTERVAL_SECONDS))
                continue

            # Reset circuit breaker on success
            _consecutive_failures = 0
            _metrics.validator_loop_last_success_timestamp_seconds.labels(
                loop_name="weight_setter"
            ).set_to_current_time()

            # 6. Verify weights landed on-chain
            try:
                verification = verify_weights_on_chain(
                    subtensor=subtensor,
                    netuid=netuid,
                    wallet_hotkey=wallet.hotkey.ss58_address,
                    expected_uids=uids,
                    expected_weights=weight_vals,
                )
                if verification.get("verified"):
                    logger.info("weight-setting: chain verification passed run=%s", current_run_id)
                else:
                    logger.warning(
                        "weight-setting: chain verification failed run=%s mismatches=%s",
                        current_run_id, verification.get("mismatches") or verification.get("reason"),
                    )
            except Exception:
                logger.warning("weight-setting: chain verification error", exc_info=True)

        except Exception:
            logger.exception("weight-setting loop error")
            # Sleep shorter on transient errors (e.g. owner-api down)
            await asyncio.sleep(min(60.0, _WEIGHT_SET_INTERVAL_SECONDS))
            continue

        await asyncio.sleep(_WEIGHT_SET_INTERVAL_SECONDS)


async def run_distributed_benchmarks(
    *,
    run_id: str | None = None,
    family_id: str,
    batch_size: int = 1,
    max_parallel: int = 2,
    rubric_version: str = "pairwise_general_chat_v1",
    judge_model: str = "local-rubric-judge",
) -> dict[str, Any]:
    """Claim task-level evaluations, fan out to all miners + OpenAI baseline,
    pairwise-judge, and submit the full batch of per-miner results.

    Redesigned flow (see plan `im-gonna-update-the-reflective-curry.md`):
      1. Claim a batch of tasks (task-level lease, one row per task).
      2. For each claimed task, in parallel under `max_parallel`:
         a. Fan out to every miner listed on the claim + call OpenAI baseline.
            Wall clock cap 3 minutes; per-miner failures don't kill the task.
         b. Pairwise-judge each miner response vs the baseline, with A/B
            position randomized per miner.
         c. Submit one result payload carrying all per-miner verdicts.
      3. Repeat until no pending tasks remain.

    If the OpenAI baseline call fails, the task is released back to pending
    via ``mark_baseline_failed`` so another validator can try.
    """
    import asyncio
    import json as _json
    import secrets

    from shared.benchmark._invocation import _invoke_task
    from shared.benchmark._judge import build_judge_excerpt
    from shared.core.evaluation_models import BenchmarkTaskRun
    from shared.core.judge_client import JudgeServiceClient
    from validation.validator.eval_config import pairwise_reference_vendor

    family_id = ensure_active_family_id(family_id)
    signer = _load_validator_signer()
    owner_url = _owner_api_url()

    # Pairwise judge client reused across tasks. The pairwise
    # comparator reads the chosen oracle's cached answer (no separate
    # baseline call); ``pairwise_reference_vendor()`` is the env knob.
    judge_client = JudgeServiceClient()
    pairwise_vendor = pairwise_reference_vendor()
    # Log the comparator choice once per process lifetime — this
    # function runs every poll cycle (default 30s) and the banner is
    # otherwise pure log noise. Per-task ``baseline=ok source=...``
    # lines surface the per-call vendor afterwards.
    if not _PAIRWISE_BANNER_LOGGED["done"]:
        logger.info(
            "pairwise comparator: oracle vendor=%s (baseline call disabled)",
            pairwise_vendor,
        )
        _PAIRWISE_BANNER_LOGGED["done"] = True

    # Oracle enrichment layer — runs at task-claim time, produces
    # ``ReconciledOracle`` per task with ``expected_claims`` +
    # ``must_not_claim`` for the EvalJudge to score against. Wired
    # only when the validator has API keys for ≥2 oracles + Chutes
    # reconciler; otherwise three_oracle items fall back to disputed.
    oracle_fanout, reconciler = _build_oracle_layer()

    total_claimed = 0
    total_submitted = 0
    total_failed = 0
    total_baseline_failed = 0
    _benchmark_started_at = time.perf_counter()
    _metrics.benchmark_runs_started_total.labels(family=family_id).inc()
    _benchmark_outcome = "success"

    # Outer wall-clock budget for the fan-out (miners + OpenAI baseline).
    # Single-turn ceiling is one mode-budget (thinking=600s) plus headroom
    # for the baseline call. Multi-turn fixtures replay N user turns, so
    # the budget scales with N; default 1500s comfortably fits a 3-turn
    # thinking-mode fixture (3 × ~500s wall clock). Override with
    # ``EIREL_FAN_OUT_TIMEOUT_SECONDS`` if you ship longer multi-turn
    # scripts. Per-turn budget enforcement still happens in ``_judge_miner``.
    _FAN_OUT_TIMEOUT_SECONDS = float(
        os.getenv("EIREL_FAN_OUT_TIMEOUT_SECONDS", "1500")
    )

    async def _invoke_one_miner(
        miner: dict[str, Any], task_obj: Any, task_id: str,
    ) -> tuple[str, BenchmarkTaskRun]:
        miner_target = MinerBenchmarkTarget(
            hotkey=miner["hotkey"],
            endpoint=miner["endpoint"],
            stake=0,
            metadata={"auth_headers": miner.get("auth_headers", {})},
        )
        try:
            run = await _invoke_task(
                miner=miner_target, task=task_obj, timeout_seconds=_FAN_OUT_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            run = BenchmarkTaskRun(
                task_id=task_id, family_id=family_id,
                prompt=task_obj.prompt, expected_output=task_obj.expected_output,
                response={}, status="failed", error=str(exc), metadata={},
            )
        return miner["hotkey"], run

    async def _evaluate_task(task_claim: dict[str, Any]) -> str:
        """Evaluate one task across all miners. Returns 'ok',
        'baseline_failed', or 'submit_failed'."""
        task_evaluation_id = task_claim["task_evaluation_id"]
        task_payload = task_claim["task_payload"]
        task_id = task_claim["task_id"]
        miners = task_claim.get("miners") or []
        if not miners:
            return "ok"  # nothing to evaluate
        task_started_at = time.perf_counter()

        class _TaskProxy:
            pass
        task_obj = _TaskProxy()
        task_obj.task_id = task_id
        task_obj.family_id = family_id
        task_obj.prompt = task_payload.get("prompt", "")
        task_obj.expected_output = task_payload.get("expected_output", {})
        task_obj.inputs = _hydrate_agent_inputs(task_payload)
        task_obj.metadata = task_payload.get("metadata", {})
        task_obj.execution_mode = task_payload.get("execution_mode")
        # Multi-turn fixtures carry a ``turns`` array of {user, assistant?}.
        # Pass through to the invocation helper unchanged (it handles
        # both single-turn and multi-turn cases). Single-turn tasks have
        # turns=None and use ``prompt``.
        task_obj.turns = task_payload.get("turns")
        # ``category`` drives multi-metric task-type derivation
        # (web_required / rag_required / sandbox_required / etc).
        # ``allowed_tools`` is a fallback signal for tool_routing when
        # ``category`` is missing.
        task_obj.category = str(task_payload.get("category") or "")
        task_obj.allowed_tools = list(task_payload.get("allowed_tools") or [])
        task_mode = str(task_obj.inputs.get("mode") or "instant")
        # Per-task web-search flag, mirroring the end-user toggle in the chat
        # UI. Missing field defaults to False so a baseline never silently
        # searches — task authors must opt in per task.
        web_search_flag = bool(
            task_payload.get("web_search")
            or (task_obj.metadata or {}).get("web_search")
            or False
        )

        # Per-task lifecycle: emit a CLAIMED log so the operator can see
        # exactly what's about to be invoked (mode, multi-turn vs
        # single-turn, miner count) without correlating DB rows manually.
        turn_count = len(task_obj.turns) if task_obj.turns else 1
        logger.info(
            "task_eval=%s task=%s mode=%s web_search=%s turns=%d miners=%d CLAIMED",
            task_evaluation_id[:8], task_id, task_mode, web_search_flag,
            turn_count, len(miners),
        )

        # Build the baseline's user-only conversation script. The
        # baseline replays the same user prompts as the miner so the
        # judge compares two answers conditioned on the same user
        # intent. For scripted-assistant turns we feed the same canned
        # exchange into the baseline's history (parity with the miner).
        # We then replay against the baseline turn-by-turn and let it
        # build its own assistant history, which is what gets compared
        # at the final live turn.
        # Augment task_obj with the bundle's ``oracle_source`` tag so
        # the enrichment helper can branch correctly. Not all bundles
        # carry this field; default to None (= deterministic).
        task_obj.oracle_source = task_payload.get("oracle_source")

        # Oracle enrichment runs in parallel with miner dispatch — it
        # makes 3 frontier-LLM calls + 1 reconciler call (only for
        # three_oracle items), and the result is consumed by every
        # ``_judge_miner`` call for this task. Cache lifetime = this
        # task evaluation.
        reconciled_task = asyncio.create_task(
            _enrich_task_oracle(
                task_obj, fanout=oracle_fanout, reconciler=reconciler,
            )
        )

        miner_tasks = [
            asyncio.create_task(_invoke_one_miner(m, task_obj, task_id)) for m in miners
        ]

        try:
            async with asyncio.timeout(_FAN_OUT_TIMEOUT_SECONDS):
                miner_runs = await asyncio.gather(*miner_tasks, return_exceptions=True)
        except TimeoutError:
            logger.warning(
                "fan-out exceeded %.0fs for task_eval=%s",
                _FAN_OUT_TIMEOUT_SECONDS, task_evaluation_id,
            )
            for t in miner_tasks:
                if not t.done():
                    t.cancel()
            return "submit_failed"

        # Pairwise comparator: pick the chosen oracle's answer as the
        # reference instead of paying for a separate OpenAI baseline
        # call. Fallback chain handles vendors that errored or
        # ``deterministic`` items where no oracle ran.
        try:
            reconciled_for_baseline = await reconciled_task
        except Exception as enrich_exc:
            logger.warning(
                "oracle enrichment crashed for task=%s: %s; "
                "falling back to disputed (no comparator text available)",
                task_id, enrich_exc,
            )
            reconciled_for_baseline = ReconciledOracle(
                expected_claims=[],
                must_not_claim=list(
                    (task_obj.expected_output or {}).get("must_not_claim") or []
                ),
                oracle_status="disputed",
                disagreement_note=f"enrichment_crashed: {enrich_exc!r}",
            )
        baseline_text, baseline_source = _select_pairwise_reference(
            reconciled=reconciled_for_baseline,
            preferred_vendor=pairwise_vendor,
        )
        # Synthetic ``baseline`` shim that still satisfies the rest of
        # the engine's expected fields (cost, latency, citations).
        # ``cost_usd=0.0`` because reusing the cached oracle answer
        # adds no incremental spend.
        baseline = _SyntheticBaselineResponse(
            response_text=baseline_text,
            citations=[],
            cost_usd=0.0,
            latency_seconds=0.0,
            source_vendor=baseline_source,
        )
        logger.info(
            "task_eval=%s baseline=ok source=%s",
            task_evaluation_id[:8],
            baseline_source,
        )

        # Per-miner fan-out outcome. miner_runs is a list of either
        # ``(hotkey, BenchmarkTaskRun)`` tuples or raised exceptions
        # (return_exceptions=True on the gather). Cost is server-side
        # ground truth: owner-api stamped X-Eirel-Job-Id per task,
        # provider-proxy ledgered LLM spend under that tag, and
        # owner-api injected the lookup result into the miner's
        # done-chunk metadata before forwarding back here.
        for entry in miner_runs:
            if isinstance(entry, BaseException):
                logger.warning(
                    "task_eval=%s miner=? fan_out_exception=%s",
                    task_evaluation_id[:8], entry,
                )
                continue
            try:
                hk, run = entry  # type: ignore[misc]
            except Exception:
                continue
            run_meta = (run.metadata or {}) if hasattr(run, "metadata") else {}
            response_meta = (
                (run.response or {}).get("metadata") or {}
                if hasattr(run, "response") else {}
            )
            llm_cost = float(response_meta.get("proxy_cost_usd") or 0.0)
            cost_str = (
                f" llm_cost=$?  (no ledger entry)"
                if response_meta.get("proxy_cost_absent")
                else f" llm_cost=${llm_cost:.4f}"
            )
            logger.info(
                "task_eval=%s miner=%s status=%s latency=%.2fs streamed=%s%s%s",
                task_evaluation_id[:8],
                str(hk)[:12] if hk else "?",
                getattr(run, "status", "?"),
                float(run_meta.get("latency_seconds") or 0.0),
                run_meta.get("streamed"),
                cost_str,
                f" error=\"{run.error[:80]}\"" if getattr(run, "error", None) else "",
            )

        task_category = (
            task_payload.get("category")
            or (task_obj.metadata or {}).get("category")
        )

        # For multi-turn fixtures the judge sees the **final user turn**
        # as the prompt. Earlier turns provide context but are not what
        # the agent is being graded on. For single-turn tasks this is
        # just task_obj.prompt.
        judge_prompt = task_obj.prompt
        if task_obj.turns:
            for raw in reversed(task_obj.turns):
                user_text = raw.get("user") if isinstance(raw, dict) else getattr(raw, "user", None)
                if isinstance(user_text, str) and user_text:
                    judge_prompt = user_text
                    break

        # Pick the latency budget for this task based on its declared mode.
        # Tasks without a recognized mode skip the gate entirely (None budget).
        if task_mode == "instant":
            latency_budget: float | None = _INSTANT_MINER_LATENCY_BUDGET_SECONDS
        elif task_mode == "thinking":
            latency_budget = _THINKING_MINER_LATENCY_BUDGET_SECONDS
        else:
            latency_budget = None

        # Oracle enrichment was awaited above (during pairwise
        # reference selection). Reuse that resolved value here so the
        # judge layer has the same ``ReconciledOracle`` the comparator
        # was derived from.
        reconciled_for_task = reconciled_for_baseline

        # Agreement-judge each miner concurrently with position randomization.
        # The judge sees only the final answer text — citations are stripped.
        async def _judge_miner(miner_run: tuple[str, BenchmarkTaskRun]) -> dict[str, Any]:
            miner_hotkey, run = miner_run
            miner_citations = _extract_miner_citations(run)
            # ``latency_seconds`` is the sum across turns (the wall clock
            # the user actually waited). For multi-turn fixtures the
            # mode-budget SLA is enforced **per turn** (each turn must
            # finish within the budget), so we read the per-turn max
            # from the invocation helper. Single-turn runs report the
            # same value under both keys.
            run_meta = run.metadata or {}
            miner_latency = float(run_meta.get("latency_seconds") or 0.0)
            max_turn_latency = float(
                run_meta.get("max_turn_latency_seconds") or miner_latency
            )
            # Per-task LLM cost the miner incurred against the subnet
            # provider-proxy. Owner-api injects this into the miner's
            # done-chunk metadata after looking it up server-side from
            # the proxy ledger — never from miner self-report. We pull
            # it out of the response payload here and pass it along to
            # the submit endpoint for storage.
            response_meta = (run.response or {}).get("metadata") or {}
            proxy_cost_usd = float(response_meta.get("proxy_cost_usd") or 0.0)
            if run.status != "completed":
                return {
                    "miner_hotkey": miner_hotkey,
                    "miner_response": run.model_dump(mode="json"),
                    "miner_citations": miner_citations,
                    "judge_output": None,
                    "agreement_score": 0.0,
                    "verdict": "error",
                    "miner_latency_seconds": miner_latency,
                    "latency_seconds": 0.0,
                    "proxy_cost_usd": proxy_cost_usd,
                    "judge_cost_usd": 0.0,
                }
            try:
                miner_answer = _extract_answer_text(run)
                # Pairwise preference judge against the OpenAI baseline.
                # Single call per task with a *random* A/B assignment:
                # the miner's answer goes into slot A or slot B with
                # 50/50 probability, decided fresh per task. The judge
                # cannot tell which side is the miner (so long as the
                # candidate text doesn't itself reveal it — see
                # ``_extract_answer_text``, which strips the envelope).
                # Score ∈ {1.0 win, 0.5 tie, 0.0 loss} from the miner's
                # perspective after position remap.
                #
                # We deliberately do NOT run a second swapped call and
                # average — opposite-call disagreement (judge picks the
                # same slot regardless of who's there) gets averaged
                # into 0.5 and conflates real ties with positional drift.
                # One call with random assignment forces the judge to
                # commit to one verdict per task and keeps the defense
                # auditable from the persisted ``miner_position`` field.
                judge_started = time.perf_counter()
                pairwise_prompt = _build_pairwise_prompt(task_obj)
                miner_position = "A" if secrets.randbelow(2) == 0 else "B"
                if miner_position == "A":
                    answer_a, answer_b = miner_answer, baseline_text
                else:
                    answer_a, answer_b = baseline_text, miner_answer
                pairwise_bundle = {
                    "question": pairwise_prompt,
                    "answers": [answer_a, answer_b],
                }
                # Anchor the pairwise judge on the consensus /
                # deterministic gold so it can reward correctness +
                # style instead of style alone. Empty for runs where
                # the reconciler couldn't produce expected_claims
                # (e.g. all 3 oracles errored) — judge falls back to
                # legacy "no factuality assumed" mode.
                pairwise_expected_answer: str | None = None
                if reconciled_for_task.expected_claims:
                    first_claim = (
                        reconciled_for_task.expected_claims[0] or ""
                    ).strip()
                    if first_claim:
                        pairwise_expected_answer = first_claim
                pw_call = await asyncio.to_thread(
                    judge_client.judge_pairwise,
                    bundle=pairwise_bundle,
                    expected_answer=pairwise_expected_answer,
                )
                preference_score = _pairwise_miner_score(
                    winner=str(pw_call.get("winner") or ""),
                    miner_position=miner_position,
                )
                # Map score to the legacy verdict bucket the existing DB
                # column / dashboard expects.
                if preference_score >= 0.999:
                    verdict = "matches"
                elif preference_score <= 0.001:
                    verdict = "contradicts"
                else:
                    verdict = "partially_matches"
                agreement_score = preference_score
                judge_meta: dict[str, Any] = {
                    "pairwise_preference_score": preference_score,
                    "miner_position": miner_position,
                    "winner": pw_call.get("winner"),
                    "confidence": pw_call.get("confidence"),
                    "reason": pw_call.get("reason"),
                    "category_scores": pw_call.get("category_scores"),
                }
                # ── Multi-metric outer-dimension scoring ────────────
                # Pairwise gives the dominant 0.40-weight signal. The
                # remaining ~0.60 weight is split across grounded /
                # retrieval / safety (LLM-judged in one /v1/judge/multi
                # call) plus tool_routing / latency_cost / (sandbox-only)
                # computation_correctness (deterministic, validator-local).
                # Non-applicable dimensions for this task type re-normalize
                # out of the final task_score.
                task_category = str(getattr(task_obj, "category", "") or "")
                task_type = _multi_metric_derive_task_type(task_category)
                applicable = _multi_metric_applicable(task_type)
                expected_output = (
                    getattr(task_obj, "expected_output", {}) or {}
                )
                expected_answer = str(expected_output.get("answer") or "").strip()
                must_not_claim_text = (
                    "; ".join(str(x) for x in (expected_output.get("must_not_claim") or []))
                )
                constraints = must_not_claim_text or None

                # Deterministic dimensions
                miner_response_payload = run.response or {}
                tool_calls_emitted = miner_response_payload.get("tool_calls") or []
                tools_called = [
                    str((t.get("tool") or t.get("name") or "")).strip()
                    for t in tool_calls_emitted
                    if isinstance(t, dict)
                ]
                tool_routing_score = (
                    _multi_metric_tool_routing(
                        task_type=task_type,
                        tools_called=tools_called,
                        has_citations=bool(miner_citations),
                    )
                    if "tool_routing" in applicable else None
                )
                latency_cost_score = (
                    _multi_metric_latency_cost(
                        miner_latency_seconds=miner_latency,
                        mode_budget_seconds=latency_budget,
                        proxy_cost_usd=proxy_cost_usd,
                        cost_budget_usd=None,  # per-task cost cap not
                                                # yet wired; latency leg
                                                # alone is informative.
                    )
                    if "latency_cost" in applicable else None
                )

                # Sandbox tasks: simple deterministic exact-match for
                # computation_correctness when the bundle ships an
                # ``expected_output.answer``. Substring match against
                # the candidate response keeps it forgiving for
                # whitespace / formatting drift.
                computation_correctness_score: float | None = None
                if "computation_correctness" in applicable:
                    if expected_answer:
                        ans_norm = expected_answer.strip().lower()
                        cand_norm = (miner_answer or "").strip().lower()
                        computation_correctness_score = (
                            1.0 if ans_norm and ans_norm in cand_norm else 0.0
                        )
                    else:
                        computation_correctness_score = None  # N/A

                # LLM-judged outer dimensions in one call
                outer_dims = sorted(
                    {"grounded_correctness", "retrieval_quality", "instruction_safety"}
                    & applicable
                )
                multi_resp: dict[str, Any] = {}
                if outer_dims:
                    citations_list = [
                        str(c.get("url") or c.get("href") or "").strip()
                        for c in (miner_citations or [])
                        if isinstance(c, dict)
                    ]
                    citations_list = [c for c in citations_list if c]
                    multi_bundle: dict[str, Any] = {
                        "question": pairwise_prompt,
                        "answers": [miner_answer],
                    }
                    if constraints:
                        multi_bundle["constraints"] = constraints
                    try:
                        multi_resp = await asyncio.to_thread(
                            judge_client.judge_multi,
                            bundle=multi_bundle,
                            expected_answer=expected_answer or None,
                            candidate_citations=citations_list,
                            applicable_metrics=outer_dims,
                        )
                    except Exception as multi_exc:
                        logger.warning(
                            "multi judge failed for miner=%s task_eval=%s: %s",
                            miner_hotkey[:16], task_evaluation_id, multi_exc,
                        )
                        multi_resp = {}

                def _multi_score(name: str) -> float | None:
                    payload = multi_resp.get(name) if isinstance(multi_resp, dict) else None
                    if not isinstance(payload, dict):
                        return None
                    score = payload.get("score")
                    try:
                        return float(score) if score is not None else None
                    except (TypeError, ValueError):
                        return None

                grounded_score = _multi_score("grounded_correctness")
                retrieval_score = _multi_score("retrieval_quality")
                safety_score = _multi_score("instruction_safety")

                breakdown = _multi_metric_assemble(
                    task_type=task_type,
                    raw_scores={
                        "pairwise_preference_score": preference_score,
                        "grounded_correctness": grounded_score,
                        "retrieval_quality": retrieval_score,
                        "computation_correctness": computation_correctness_score,
                        "tool_routing": tool_routing_score,
                        "instruction_safety": safety_score,
                        "latency_cost": latency_cost_score,
                    },
                )

                # ── EvalJudge + composite (the new ranking signal) ─────
                # Use the cached reconciled-oracle output (expected
                # claims + must_not_claim) as the judge's reference.
                # For three_oracle tasks: validator-side reconciler
                # produced the expected_claims at task-claim time; for
                # deterministic tasks: from_deterministic wrapped the
                # pool's pre-baked answer.
                #
                # The composite multiplicatively combines outcome
                # (correct/partial/wrong/...) × tool_attestation ×
                # efficiency × hallucination_knockout ×
                # cost_attestation_knockout. final_task_score is the
                # composite, replacing the legacy weighted-sum.
                expected_claims_text = "\n".join(
                    reconciled_for_task.expected_claims
                )
                must_not_claim_for_judge = list(
                    reconciled_for_task.must_not_claim
                )
                eval_required_tool = (
                    str(expected_output.get("required_tool") or "").strip()
                    or None
                )
                oracle_source_for_judge = (
                    "three_oracle"
                    if reconciled_for_task.oracle_status == "consensus"
                    or reconciled_for_task.oracle_status == "majority"
                    else "deterministic"
                    if reconciled_for_task.oracle_status == "deterministic"
                    else "three_oracle"  # disputed → still treat as oracle path
                )
                eval_bundle = {
                    "question": judge_prompt,
                    "answers": [miner_answer],
                }
                if must_not_claim_for_judge:
                    eval_bundle["constraints"] = "; ".join(
                        must_not_claim_for_judge
                    )

                eval_outcome_str: str = "wrong"
                eval_failure_mode: str | None = None
                eval_guidance: str = ""
                eval_resp: dict[str, Any] = {}
                try:
                    eval_resp = await asyncio.to_thread(
                        judge_client.judge_eval,
                        bundle=eval_bundle,
                        expected_answer=expected_claims_text or expected_answer or "(no expected answer)",
                        must_not_claim=must_not_claim_for_judge,
                        required_tool=eval_required_tool,
                        oracle_source=oracle_source_for_judge,
                    )
                    eval_outcome_str = str(eval_resp.get("outcome") or "wrong")
                    eval_failure_mode = eval_resp.get("failure_mode")
                    eval_guidance = str(eval_resp.get("guidance") or "")
                except Exception as eval_exc:
                    logger.warning(
                        "eval judge failed for miner=%s task_eval=%s: %s",
                        miner_hotkey[:16], task_evaluation_id, eval_exc,
                    )

                # Server-attested ledger: which tools did this miner
                # actually invoke under its job_id? Composite's
                # tool_attestation_factor uses this — fabricated
                # tool_call frames in the miner's response don't
                # count, only what the orchestrator's tool services
                # actually executed.
                miner_job_id = str(
                    response_meta.get("job_id")
                    or run_meta.get("job_id")
                    or ""
                )
                ledger_tools = await _fetch_ledger_tools(
                    miner_job_id,
                    owner_url=owner_url,
                    signer=signer,
                )

                composite_resp: dict[str, Any] = {}
                try:
                    composite_resp = await asyncio.to_thread(
                        judge_client.judge_eval_composite,
                        outcome=eval_outcome_str,
                        failure_mode=eval_failure_mode,
                        candidate_response=miner_answer,
                        must_not_claim=must_not_claim_for_judge,
                        required_tool=eval_required_tool,
                        ledger_tools=ledger_tools,
                        latency_ms=int(max(0.0, miner_latency) * 1000),
                        cost_usd=proxy_cost_usd,
                        latency_budget_ms=(
                            int(latency_budget * 1000)
                            if latency_budget else None
                        ),
                        cost_budget_usd=None,
                        cost_floor_usd=None,
                        # Outer-dimension gates: ``grounded`` ≥ 0.60,
                        # ``safety`` ≥ 0.80. ``None`` for tasks where
                        # the multi-judge call didn't run or that
                        # dimension is N/A — gate bypassed in either
                        # case so missing data never zeros the score.
                        grounded_correctness_score=grounded_score,
                        instruction_safety_score=safety_score,
                        # Pairwise win-rate vs the OpenAI baseline —
                        # ±0.10 bonus on top of outcome_score. Becomes
                        # the tiebreaker between equally-correct miners.
                        pairwise_preference_score=preference_score,
                    )
                except Exception as comp_exc:
                    logger.warning(
                        "composite judge failed for miner=%s task_eval=%s: %s",
                        miner_hotkey[:16], task_evaluation_id, comp_exc,
                    )
                composite_score = float(
                    composite_resp.get("composite") or 0.0
                )
                composite_knockout_reason = composite_resp.get("knockout_reason")
                knockout_reasons_list: list[str] = (
                    [str(composite_knockout_reason)]
                    if composite_knockout_reason else []
                )

                # Excerpts persisted alongside the EvalJudge outcome —
                # owner-api derives the durable EvalFeedback row from
                # this metadata when it accepts the task-result POST.
                eval_prompt_excerpt = str(judge_prompt or "")[:200]
                eval_response_excerpt = str(miner_answer or "")[:500]

                judge_result_payload: dict[str, Any] = {
                    "verdict": verdict,
                    "agreement_score": agreement_score,
                    "rationale": str(pw_call.get("reason") or ""),
                    "swap_applied": True,
                    "model": "eiretes_pairwise_judge",
                    "rubric_name": "pairwise_v1",
                    "metadata": {
                        **judge_meta,
                        "task_type": task_type,
                        "multi_judge_response": multi_resp or {},
                        # Legacy weighted-sum value, retained for parity
                        # comparison during shadow-mode rollout. The new
                        # ``composite_score`` (below) is what populates
                        # ``final_task_score`` going forward.
                        "weighted_sum_score": breakdown.final_task_score,
                        "final_task_score": composite_score,
                        "applied_weights": breakdown.applied_weights,
                        "applicable_metrics": breakdown.applicable_metrics,
                        # EvalJudge + composite fields. Owner-api reads
                        # these on receipt of the task-result POST and
                        # upserts an EvalFeedback row server-side.
                        "eval_outcome": eval_outcome_str,
                        "eval_failure_mode": eval_failure_mode,
                        "eval_guidance": eval_guidance,
                        "eval_prompt_excerpt": eval_prompt_excerpt,
                        "eval_response_excerpt": eval_response_excerpt,
                        "eval_knockout_reasons": knockout_reasons_list,
                        "composite_score": composite_score,
                        "composite_knockout_reason": composite_resp.get(
                            "knockout_reason"
                        ),
                        "oracle_status": reconciled_for_task.oracle_status,
                        "oracle_disagreement_note": (
                            reconciled_for_task.disagreement_note
                        ),
                        "vendor_status": reconciled_for_task.vendor_status,
                        "vendor_citations": reconciled_for_task.vendor_citations,
                        "ledger_tools": ledger_tools,
                    },
                }
                judge_latency = max(0.0, time.perf_counter() - judge_started)
                # eiretes-judge surfaces ``cost_usd`` at the response
                # root of each /v1/judge/{pairwise,multi,eval} endpoint
                # (eiretes ≥ 0.2.1). One miner-judgment runs ALL THREE
                # endpoints, so per-miner judge cost = pairwise + multi
                # + eval. ``None`` on any endpoint means the upstream
                # didn't surface token counts (older payload shape) —
                # treated as $0 in the sum so we don't crash.
                def _cost_usd(payload: dict[str, Any] | None) -> float:
                    if not isinstance(payload, dict):
                        return 0.0
                    raw = payload.get("cost_usd")
                    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                        return float(raw)
                    return 0.0

                judge_cost_usd = (
                    _cost_usd(pw_call)
                    + _cost_usd(multi_resp)
                    + _cost_usd(eval_resp)
                )
                # Mode-specific completion-time SLA: any *single turn*
                # exceeding the budget counts as a violation. We gate on
                # the per-turn max so a fixture that has 3 well-behaved
                # turns + 1 over-budget turn is still flagged.
                if latency_budget is not None and max_turn_latency > latency_budget:
                    logger.info(
                        "latency_violation miner=%s task_eval=%s mode=%s "
                        "max_turn=%.2fs total=%.2fs budget=%.2fs prior_verdict=%s",
                        miner_hotkey[:16], task_evaluation_id, task_mode,
                        max_turn_latency, miner_latency, latency_budget, verdict,
                    )
                    verdict = "latency_violation"
                    agreement_score = 0.0
                return {
                    "miner_hotkey": miner_hotkey,
                    "miner_response": run.model_dump(mode="json"),
                    "miner_citations": miner_citations,
                    "judge_output": judge_result_payload,
                    "agreement_score": agreement_score,
                    "verdict": verdict,
                    "miner_latency_seconds": miner_latency,
                    "latency_seconds": judge_latency,
                    "proxy_cost_usd": proxy_cost_usd,
                    "judge_cost_usd": judge_cost_usd,
                    # Multi-metric per-task scoring.
                    "task_type": task_type,
                    "pairwise_preference_score": breakdown.dimension_scores.get(
                        "pairwise_preference_score",
                    ),
                    "grounded_correctness": breakdown.dimension_scores.get(
                        "grounded_correctness",
                    ),
                    "retrieval_quality": breakdown.dimension_scores.get(
                        "retrieval_quality",
                    ),
                    "tool_routing": breakdown.dimension_scores.get("tool_routing"),
                    "instruction_safety": breakdown.dimension_scores.get(
                        "instruction_safety",
                    ),
                    "latency_cost": breakdown.dimension_scores.get("latency_cost"),
                    "computation_correctness": breakdown.dimension_scores.get(
                        "computation_correctness",
                    ),
                    # final_task_score is the new multiplicative
                    # composite — not the legacy weighted-sum. The
                    # weighted-sum value still surfaces as
                    # ``judge_output.metadata.weighted_sum_score`` for
                    # parity comparison during rollout.
                    "final_task_score": composite_score,
                    "applied_weights": breakdown.applied_weights,
                    "applicable_metrics": breakdown.applicable_metrics,
                    # Fields surfacing the EvalJudge + composite + oracle
                    # status path. ``oracle_status`` is the validator-side
                    # reconciler's verdict on the 3 oracles
                    # (consensus / majority / disputed) or
                    # "deterministic" for non-three_oracle items.
                    "composite_score": composite_score,
                    "eval_outcome": eval_outcome_str,
                    "eval_failure_mode": eval_failure_mode,
                    "eval_guidance": eval_guidance,
                    "oracle_status": reconciled_for_task.oracle_status,
                }
            except Exception as exc:
                logger.warning(
                    "agreement judge failed for miner=%s task_eval=%s: %s",
                    miner_hotkey[:16], task_evaluation_id, exc,
                )
                return {
                    "miner_hotkey": miner_hotkey,
                    "miner_response": run.model_dump(mode="json"),
                    "miner_citations": miner_citations,
                    "judge_output": None,
                    "agreement_score": 0.0,
                    "verdict": "error",
                    "miner_latency_seconds": miner_latency,
                    "latency_seconds": 0.0,
                    "proxy_cost_usd": proxy_cost_usd,
                    "judge_cost_usd": 0.0,
                }

        # Materialize the gather results (handle exceptions from miner fan-out)
        resolved_runs: list[tuple[str, BenchmarkTaskRun]] = []
        for miner, result in zip(miners, miner_runs):
            if isinstance(result, BaseException):
                resolved_runs.append((
                    miner["hotkey"],
                    BenchmarkTaskRun(
                        task_id=task_id, family_id=family_id,
                        prompt=task_obj.prompt, expected_output=task_obj.expected_output,
                        response={}, status="failed", error=str(result), metadata={},
                    ),
                ))
            else:
                resolved_runs.append(result)

        judge_results = await asyncio.gather(*[_judge_miner(r) for r in resolved_runs])

        # Verdict tally — one-line summary so an operator scrolling the
        # validator log can see "task X scored 2/3 matches" at a glance,
        # without having to query the DB.
        verdict_counts: dict[str, int] = {}
        for r in judge_results:
            v = str(r.get("verdict") or "error")
            verdict_counts[v] = verdict_counts.get(v, 0) + 1
        verdict_summary = " ".join(
            f"{k}={v}" for k, v in sorted(verdict_counts.items())
        )
        logger.info(
            "task_eval=%s verdicts: %s",
            task_evaluation_id[:8], verdict_summary or "(none)",
        )

        # Per-task cost roll-up. Four components:
        #   (1) oracle layer = per-vendor oracle costs + reconciler.
        #       Validator-paid; reused across every miner judged for
        #       this task. Submitted to owner-api as
        #       ``oracle_cost_usd`` for the validator-cost dashboard.
        #   (2) miners = sum of provider-proxy ledger lookups
        #       (miner-paid; informational here).
        #   (3) judge = sum of eiretes-judge per-judgment costs across
        #       miners (validator-paid; persisted per (task, miner)).
        #   (4) baseline = legacy ``_SyntheticBaselineResponse.cost_usd``
        #       shim — always 0 since the comparator is reused from
        #       a cached oracle answer.
        baseline_cost = float(getattr(baseline, "cost_usd", 0.0) or 0.0)
        oracle_layer_cost = sum(
            float(c or 0.0)
            for c in (reconciled_for_baseline.vendor_costs or {}).values()
        ) + float(reconciled_for_baseline.reconciler_cost_usd or 0.0)
        miners_cost = sum(
            float(r.get("proxy_cost_usd") or 0.0) for r in judge_results
        )
        judge_cost = sum(
            float(r.get("judge_cost_usd") or 0.0) for r in judge_results
        )
        logger.info(
            "task_eval=%s cost: oracle=$%.4f baseline=$%.4f miners=$%.4f "
            "judge=$%.4f validator_paid=$%.4f",
            task_evaluation_id[:8],
            oracle_layer_cost, baseline_cost, miners_cost, judge_cost,
            oracle_layer_cost + judge_cost,
        )

        # Submit the batch
        result_path = f"/v1/families/{family_id}/task-evaluations/{task_evaluation_id}/result"
        result_body = {
            "baseline_response": baseline.model_dump(mode="json"),
            "miner_results": judge_results,
            "validator_hotkey": signer.hotkey,
            "judge_model": judge_model,
            "oracle_cost_usd": oracle_layer_cost,
        }
        result_body_bytes = _json.dumps(result_body).encode()
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{owner_url}{result_path}",
                    content=result_body_bytes,
                    headers={
                        **_signed_headers(signer=signer, method="POST", path=result_path, body=result_body_bytes),
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
            elapsed = time.perf_counter() - task_started_at
            logger.info(
                "task_eval=%s SUBMITTED total=%.2fs",
                task_evaluation_id[:8], elapsed,
            )
            return "ok"
        except Exception as exc:
            logger.warning("submit failed for task_eval=%s: %s", task_evaluation_id, exc)
            return "submit_failed"

    sem = asyncio.Semaphore(max_parallel)

    async def _bounded_evaluate(task_claim: dict[str, Any]) -> str:
        async with sem:
            return await _evaluate_task(task_claim)

    try:
        while True:
            claim_path = f"/v1/families/{family_id}/tasks/claim"
            claim_body: dict[str, Any] = {"batch_size": batch_size}
            if run_id:
                claim_body["run_id"] = run_id
            claim_body_bytes = _json.dumps(claim_body).encode()

            async with httpx.AsyncClient(timeout=30.0) as client:
                claim_response = await client.post(
                    f"{owner_url}{claim_path}",
                    content=claim_body_bytes,
                    headers={
                        **_signed_headers(signer=signer, method="POST", path=claim_path, body=claim_body_bytes),
                        "Content-Type": "application/json",
                    },
                )
                if claim_response.status_code == 404:
                    break
                claim_response.raise_for_status()
                claim_data = claim_response.json()

            tasks = claim_data.get("tasks", [])
            if not tasks:
                break

            total_claimed += len(tasks)

            results = await asyncio.gather(
                *[_bounded_evaluate(t) for t in tasks],
                return_exceptions=True,
            )
            for r in results:
                if r == "ok":
                    total_submitted += 1
                elif r == "baseline_failed":
                    total_baseline_failed += 1
                else:
                    total_failed += 1
    finally:
        judge_client.close()

    if total_claimed > 0 and (total_failed + total_baseline_failed) == total_claimed:
        _benchmark_outcome = "failed"
    elif total_claimed > 0 and (total_failed + total_baseline_failed) > 0:
        _benchmark_outcome = "partial"
    _metrics.benchmark_run_duration_seconds.labels(family=family_id).observe(
        time.perf_counter() - _benchmark_started_at
    )
    _metrics.benchmark_runs_completed_total.labels(
        family=family_id, outcome=_benchmark_outcome
    ).inc()

    return {
        "run_id": run_id,
        "family_id": family_id,
        "total_claimed": total_claimed,
        "total_submitted": total_submitted,
        "total_failed": total_failed,
        "total_baseline_failed": total_baseline_failed,
    }


async def _release_baseline_failed(
    *, task_evaluation_id: str, signer, owner_url: str,
) -> None:
    """POST to owner-api to release a claim after baseline failure."""
    import json as _json
    path = f"/v1/families/general_chat/task-evaluations/{task_evaluation_id}/baseline-failed"
    body = _json.dumps({}).encode()
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            await client.post(
                f"{owner_url}{path}",
                content=body,
                headers={
                    **_signed_headers(signer=signer, method="POST", path=path, body=body),
                    "Content-Type": "application/json",
                },
            )
    except Exception as exc:
        logger.warning("failed to release baseline-failed task=%s: %s", task_evaluation_id, exc)
