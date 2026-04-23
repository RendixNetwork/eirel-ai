from __future__ import annotations

import logging
import os
import time
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx

logger = logging.getLogger(__name__)

from shared.common.bittensor_signing import load_signer
from shared.common.security import sha256_hex
from eirel.groups import ensure_active_family_id
from shared.core.evaluation_models import MinerBenchmarkTarget
# C4: judge runs server-side via owner-api judge proxy — no direct
# JudgeServiceClient import needed here anymore.
from shared.contracts.models import MinerRegistryEntry
from validation.validator import metrics as _metrics


def _owner_api_url() -> str:
    return os.getenv("OWNER_API_URL", "http://owner-api:8000").rstrip("/")


def _hydrate_agent_inputs(task_payload: dict[str, Any]) -> dict[str, Any]:
    """Fold top-level task fields into the ``inputs`` dict sent to the miner.

    The benchmark task JSON carries ``mode`` and ``allowed_tools`` at the
    top level but the miner agent reads its knobs from ``inputs.mode`` and
    ``inputs.web_search``. Without this bridge, every task is invoked with
    the defaults (``mode=instant``, ``web_search=false``) regardless of
    the task spec, so the web-search code path never runs during
    evaluation. An existing value in ``inputs`` always wins.
    """
    base = dict(task_payload.get("inputs") or {})
    if "mode" not in base:
        mode = task_payload.get("mode")
        if isinstance(mode, str) and mode:
            base["mode"] = mode
    if "web_search" not in base:
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


def _extract_answer_text(run) -> str:
    """Pull the miner's final-answer text out of a BenchmarkTaskRun.

    The agreement judge sees ONLY the final answer — no tool_calls, no
    citations, no trace. Most miner SDKs put the answer at
    ``response.output.content`` (dict) or ``response.output.text``.
    Fall back to str(response) when structure is unknown.
    """
    resp = getattr(run, "response", None) or {}
    if isinstance(resp, dict):
        output = resp.get("output") or {}
        if isinstance(output, dict):
            content = output.get("content") or output.get("text")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # some SDKs emit content as a list of text blocks
                return "\n\n".join(
                    (b.get("text") if isinstance(b, dict) else str(b))
                    for b in content if b
                )
        # Fallback: flatten the whole response dict as text without URLs.
        text = resp.get("text") or resp.get("content") or ""
        if isinstance(text, str) and text:
            return text
    return str(resp) if resp else ""


def _extract_miner_citations(run) -> list[dict[str, Any]]:
    """Pull the miner's cited URLs out of a BenchmarkTaskRun for dashboard
    display only. These do NOT participate in scoring.

    Looks in two likely locations:
      * ``response.output.citations`` — structured list of citation dicts
      * ``response.output.tool_calls`` of type ``web_search`` — URL results
    Returns an empty list when no citations are present.
    """
    resp = getattr(run, "response", None) or {}
    if not isinstance(resp, dict):
        return []
    output = resp.get("output") or {}
    if not isinstance(output, dict):
        return []
    citations: list[dict[str, Any]] = []
    structured = output.get("citations")
    if isinstance(structured, list):
        for c in structured:
            if isinstance(c, dict):
                citations.append({
                    "url": str(c.get("url") or ""),
                    "title": str(c.get("title") or ""),
                })
    tool_calls = output.get("tool_calls")
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            if not isinstance(tc, dict) or tc.get("tool_name") != "web_search":
                continue
            for result in (tc.get("result") or {}).get("results", []) or []:
                if isinstance(result, dict) and result.get("url"):
                    citations.append({
                        "url": str(result.get("url") or ""),
                        "title": str(result.get("title") or ""),
                    })
    return citations


async def _resolve_targets(
    *,
    run_id: str,
    family_id: str,
    miners: list[MinerRegistryEntry],
    rubric_version: str,
    judge_model: str,
) -> tuple[list[MinerRegistryEntry], dict[str, object], str, dict[str, object] | None, dict[str, object] | None, str | None]:
    del miners
    family_id = ensure_active_family_id(family_id)
    signer = _load_validator_signer()
    path = f"/v1/families/{family_id}/targets"
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(
            f"{_owner_api_url()}{path}",
            params={
                "run_id": run_id,
                "benchmark_version": "family_benchmark_v2",
                "rubric_version": rubric_version,
                "judge_model": judge_model,
            },
            headers=_signed_headers(signer=signer, method="GET", path=path, body=b""),
        )
        response.raise_for_status()
        payload = response.json()
    evaluation_bundle = (
        payload.get("evaluation_bundle") if isinstance(payload.get("evaluation_bundle"), dict) else None
    )
    if evaluation_bundle is None:
        raise ValueError(f"owner target response missing evaluation_bundle for family {family_id}")
    benchmark_version = str(payload.get("benchmark_version") or "family_benchmark_v2")
    if evaluation_bundle.get("benchmark_version"):
        benchmark_version = str(evaluation_bundle.get("benchmark_version"))
    retrieval_environment = payload.get("retrieval_environment")
    if not isinstance(retrieval_environment, dict):
        retrieval_environment = (
            evaluation_bundle.get("retrieval_environment")
            if isinstance(evaluation_bundle.get("retrieval_environment"), dict)
            else None
        )
    judge_config = payload.get("judge_config")
    if not isinstance(judge_config, dict):
        judge_config = (
            evaluation_bundle.get("judge_config")
            if isinstance(evaluation_bundle.get("judge_config"), dict)
            else None
        )
    policy_version = payload.get("policy_version")
    if policy_version is None and evaluation_bundle.get("policy_version") is not None:
        policy_version = str(evaluation_bundle.get("policy_version"))
    elif policy_version is not None:
        policy_version = str(policy_version)
    return (
        [
            MinerRegistryEntry.model_validate(
                {
                    **item,
                    "endpoint": _rewrite_benchmark_endpoint_for_host(str(item.get("endpoint") or "")),
                }
            )
            for item in list(payload.get("members") or [])
            if isinstance(item, dict)
        ],
        evaluation_bundle,
        benchmark_version,
        retrieval_environment,
        judge_config,
        policy_version,
    )


# ---------------------------------------------------------------------------
# Distributed evaluation: autonomous polling loop
# ---------------------------------------------------------------------------

_POLL_INTERVAL_SECONDS = float(
    os.getenv("EIREL_VALIDATOR_POLL_INTERVAL_SECONDS")
    or os.getenv("VALIDATOR_POLL_INTERVAL_SECONDS")
    or "30"
)
_BATCH_SIZE = int(os.getenv("EIREL_VALIDATOR_BATCH_SIZE", "3"))
_MAX_PARALLEL = int(os.getenv("EIREL_VALIDATOR_MAX_PARALLEL", "3"))
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
    judge_model = os.getenv("EIREL_JUDGE_MODEL", "local-rubric-judge")
    rubric_version = os.getenv("EIREL_VALIDATOR_RUBRIC_VERSION", "family_rubric_v2")

    logger.info(
        "validator loop starting: families=%s poll_interval=%.0fs batch_size=%d max_parallel=%d",
        _ACTIVE_FAMILIES, _POLL_INTERVAL_SECONDS, _BATCH_SIZE, _MAX_PARALLEL,
    )

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
                        "validator loop: family=%s claimed=%d submitted=%d failed=%d",
                        family_id, result["total_claimed"],
                        result["total_submitted"], result["total_failed"],
                    )
            except Exception:
                logger.exception("validator loop error: family=%s", family_id)

        _metrics.validator_loop_last_success_timestamp_seconds.labels(
            loop_name="validator"
        ).set_to_current_time()
        if not found_work:
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)


# ---------------------------------------------------------------------------
# Weight-setting background loop
# ---------------------------------------------------------------------------

# 180 blocks × ~12s/block ≈ 36 minutes
_WEIGHT_SET_INTERVAL_BLOCKS = int(os.getenv("EIREL_WEIGHT_SET_INTERVAL_BLOCKS", "180"))
_WEIGHT_SET_INTERVAL_SECONDS = _WEIGHT_SET_INTERVAL_BLOCKS * 12
_MIN_STAKE_TO_SET_WEIGHTS = int(os.getenv("EIREL_MIN_STAKE_TO_SET_WEIGHTS", "5000"))


async def run_weight_setting_loop() -> None:
    """Background loop: poll owner-api for weights, set on-chain every ~180 blocks.

    The owner-api returns ``{hotkey: weight}`` for family winners.
    This loop syncs the metagraph to resolve hotkey→UID, assigns
    unallocated family weight to UID 0 (to burn alpha), and calls
    ``set_weights()`` with retry, circuit breaker, and chain verification.
    """
    import asyncio
    import time as _time
    import bittensor as bt
    from validation.weight_setter.chain_verifier import verify_weights_on_chain

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
    _last_published_run_id: str | None = None

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
                    logger.info("weight-setting: no weights ready, sleeping")
                    await asyncio.sleep(_WEIGHT_SET_INTERVAL_SECONDS)
                    continue

                current_run_id = data.get("run_id")
                if current_run_id and current_run_id == _last_published_run_id:
                    logger.info(
                        "weight-setting: run %s already published, skipping",
                        current_run_id,
                    )
                    await asyncio.sleep(_WEIGHT_SET_INTERVAL_SECONDS)
                    continue

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

            # 4. Unassigned family weight goes to UID 0 (burns alpha)
            total_family_weight = sum(
                float(fw.get("family_weight", 0.0))
                for fw in family_winners
            )
            unassigned = max(0.0, total_family_weight - total_assigned)
            if unassigned > 0:
                uid_weights[0] = uid_weights.get(0, 0.0) + unassigned

            if not uid_weights:
                uid_weights[0] = total_family_weight or 1.0

            uids = sorted(uid_weights.keys())
            weight_vals = [uid_weights[uid] for uid in uids]

            logger.info(
                "weight-setting: setting weights for %d UIDs run=%s mode=%s uids=%s weights=%s",
                len(uids),
                current_run_id,
                "owner_api_down_burn" if owner_api_down else "normal",
                uids,
                [round(w, 4) for w in weight_vals],
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
            _last_published_run_id = current_run_id
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
    batch_size: int = 2,
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
    from validation.validator.openai_baseline import (
        OpenAIBaselineClient,
        OpenAIBaselineError,
    )

    family_id = ensure_active_family_id(family_id)
    signer = _load_validator_signer()
    owner_url = _owner_api_url()

    # Pairwise judge + OpenAI baseline clients are reused across tasks.
    judge_client = JudgeServiceClient()
    baseline_client = OpenAIBaselineClient()

    total_claimed = 0
    total_submitted = 0
    total_failed = 0
    total_baseline_failed = 0
    _benchmark_started_at = time.perf_counter()
    _metrics.benchmark_runs_started_total.labels(family=family_id).inc()
    _benchmark_outcome = "success"

    _FAN_OUT_TIMEOUT_SECONDS = 180.0

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
        task_mode = str(task_obj.inputs.get("mode") or "instant")

        # Fan out: all miners + baseline concurrently, bounded by wall clock
        baseline_task = asyncio.create_task(
            baseline_client.generate(prompt=task_obj.prompt)
        )
        miner_tasks = [
            asyncio.create_task(_invoke_one_miner(m, task_obj, task_id)) for m in miners
        ]

        try:
            async with asyncio.timeout(_FAN_OUT_TIMEOUT_SECONDS):
                miner_runs = await asyncio.gather(*miner_tasks, return_exceptions=True)
                baseline = await baseline_task
        except OpenAIBaselineError as exc:
            logger.warning(
                "baseline failed for task_eval=%s: %s; releasing task",
                task_evaluation_id, exc,
            )
            for t in miner_tasks:
                t.cancel()
            await _release_baseline_failed(
                task_evaluation_id=task_evaluation_id, signer=signer, owner_url=owner_url,
            )
            return "baseline_failed"
        except TimeoutError:
            logger.warning(
                "fan-out exceeded %.0fs for task_eval=%s",
                _FAN_OUT_TIMEOUT_SECONDS, task_evaluation_id,
            )
            for t in (*miner_tasks, baseline_task):
                if not t.done():
                    t.cancel()
            return "submit_failed"

        baseline_text = baseline.response_text
        task_category = task_payload.get("category") or (task_obj.metadata or {}).get("category")

        # Agreement-judge each miner concurrently with position randomization.
        # The judge sees only the final answer text — citations are stripped.
        async def _judge_miner(miner_run: tuple[str, BenchmarkTaskRun]) -> dict[str, Any]:
            miner_hotkey, run = miner_run
            miner_citations = _extract_miner_citations(run)
            if run.status != "completed":
                return {
                    "miner_hotkey": miner_hotkey,
                    "miner_response": run.model_dump(mode="json"),
                    "miner_citations": miner_citations,
                    "judge_output": None,
                    "agreement_score": 0.0,
                    "verdict": "error",
                    "latency_seconds": 0.0,
                }
            try:
                miner_answer = _extract_answer_text(run)
                swap = bool(secrets.randbits(1))
                judge_started = time.perf_counter()
                judge_result = await asyncio.to_thread(
                    judge_client.judge_agreement,
                    family_id=family_id,
                    prompt=task_obj.prompt,
                    response_a=miner_answer,
                    response_b=baseline_text,
                    task_mode=task_mode,
                    task_category=task_category,
                    swap=swap,
                )
                judge_latency = max(0.0, time.perf_counter() - judge_started)
                return {
                    "miner_hotkey": miner_hotkey,
                    "miner_response": run.model_dump(mode="json"),
                    "miner_citations": miner_citations,
                    "judge_output": judge_result.model_dump(mode="json"),
                    "agreement_score": float(judge_result.agreement_score or 0.0),
                    "verdict": judge_result.verdict,
                    "latency_seconds": judge_latency,
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
                    "latency_seconds": 0.0,
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

        # Submit the batch
        result_path = f"/v1/families/{family_id}/task-evaluations/{task_evaluation_id}/result"
        result_body = {
            "baseline_response": baseline.model_dump(mode="json"),
            "miner_results": judge_results,
            "validator_hotkey": signer.hotkey,
            "judge_model": judge_model,
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
        await baseline_client.aclose()

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
