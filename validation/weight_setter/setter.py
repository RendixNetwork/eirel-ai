from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import bittensor as bt

from shared.common.circuit_breaker import CircuitBreaker, CircuitOpenError
from shared.common.tracing import get_tracer
from control_plane.owner_api.managed import fixed_family_weight, DEFAULT_FIXED_FAMILY_WEIGHTS
from shared.contracts.models import FamilyScoreSnapshot
from validation.weight_setter.chain_verifier import verify_weights_on_chain

logger = logging.getLogger(__name__)
_tracer = get_tracer(__name__)
_chain_circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=120.0)

_DEFAULT_MAX_ATTEMPTS = 3
_DEFAULT_BACKOFF_BASE_SECONDS = 5.0
_DYNAMIC_WEIGHT_ALPHA = float(os.getenv("EIREL_DYNAMIC_WEIGHT_ALPHA", "0.7"))


def compute_dynamic_family_weight(
    family_id: str,
    *,
    query_volume_share: float = 0.0,
    alpha: float | None = None,
) -> float:
    """Blend base family weight with usage-based share.

    ``final_weight = alpha * base_weight + (1 - alpha) * usage_share``

    When ``query_volume_share`` is 0.0, falls back to the base weight.
    """
    if alpha is None:
        alpha = _DYNAMIC_WEIGHT_ALPHA
    alpha = max(0.0, min(1.0, alpha))
    base_weight = fixed_family_weight(family_id)
    if query_volume_share <= 0.0:
        return base_weight
    return round(alpha * base_weight + (1.0 - alpha) * query_volume_share, 6)


def normalize_family_weights(weights: dict[str, float]) -> dict[str, float]:
    """Normalize a dict of family weights so they sum to 1.0."""
    total = sum(weights.values())
    if total <= 0:
        return weights
    return {k: round(v / total, 6) for k, v in weights.items()}


def build_weight_submission(
    snapshot: FamilyScoreSnapshot,
    *,
    query_volume_share: float = 0.0,
) -> dict[str, object]:
    family_weight = compute_dynamic_family_weight(
        snapshot.family_id,
        query_volume_share=query_volume_share,
    )
    allocation_mode = (
        "dynamic_family_weights_v1"
        if query_volume_share > 0.0
        else "fixed_family_weights_v1"
    )
    return {
        "run_id": snapshot.run_id,
        "family_id": snapshot.family_id,
        "family_weight": family_weight,
        "weights": snapshot.normalized_weights,
        "scaled_weights": {
            hotkey: weight * family_weight
            for hotkey, weight in snapshot.normalized_weights.items()
        },
        "query_volume_share": query_volume_share,
        "rubric_version": snapshot.rubric_version,
        "allocation_mode": allocation_mode,
    }


@dataclass(slots=True)
class WeightSubmissionConfig:
    network: str
    netuid: int
    wallet_name: str | None
    hotkey_name: str | None
    wallet_path: str | None


def submission_config_from_env() -> WeightSubmissionConfig:
    # Weight-setter is run by the validator operator and should share the
    # validator's wallet. Names resolve in this order:
    #   1. EIREL_VALIDATOR_* (canonical — same vars the validator-engine uses)
    #   2. EIREL_WEIGHT_SETTER_* (explicit override, useful when the setter
    #      wallet is deliberately different from the engine's)
    #   3. WEIGHT_SETTER_* (legacy, kept for backward compat)
    def _pick(*names: str) -> str | None:
        for name in names:
            value = os.getenv(name)
            if value:
                return value
        return None

    return WeightSubmissionConfig(
        network=(
            _pick("EIREL_WEIGHT_SETTER_NETWORK", "WEIGHT_SETTER_NETWORK", "BITTENSOR_NETWORK")
            or "finney"
        ),
        netuid=int(
            _pick("EIREL_WEIGHT_SETTER_NETUID", "WEIGHT_SETTER_NETUID", "BITTENSOR_NETUID")
            or "0"
        ),
        wallet_name=_pick(
            "EIREL_VALIDATOR_WALLET_NAME",
            "EIREL_WEIGHT_SETTER_WALLET_NAME",
            "WEIGHT_SETTER_WALLET_NAME",
        ),
        hotkey_name=_pick(
            "EIREL_VALIDATOR_HOTKEY_NAME",
            "EIREL_WEIGHT_SETTER_HOTKEY_NAME",
            "WEIGHT_SETTER_HOTKEY_NAME",
        ),
        wallet_path=_pick(
            "EIREL_VALIDATOR_WALLET_PATH",
            "EIREL_WEIGHT_SETTER_WALLET_PATH",
            "WEIGHT_SETTER_WALLET_PATH",
        ),
    )


def submit_weight_submission(
    payload: dict[str, object],
    *,
    config: WeightSubmissionConfig | None = None,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    backoff_base_seconds: float = _DEFAULT_BACKOFF_BASE_SECONDS,
    verify: bool = True,
) -> dict[str, object]:
    with _tracer.start_as_current_span("submit_weight_submission") as span:
        span.set_attribute("weight.family_id", str(payload.get("family_id", "")))
        span.set_attribute("weight.run_id", str(payload.get("run_id", "")))
        result = _submit_weight_submission_inner(
            payload, config=config, max_attempts=max_attempts,
            backoff_base_seconds=backoff_base_seconds, verify=verify,
        )
        span.set_attribute("weight.verified", bool(result.get("verified")))
        span.set_attribute("weight.attempt_count", int(result.get("attempt_count", 0)))
        return result


def _submit_weight_submission_inner(
    payload: dict[str, object],
    *,
    config: WeightSubmissionConfig | None = None,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    backoff_base_seconds: float = _DEFAULT_BACKOFF_BASE_SECONDS,
    verify: bool = True,
) -> dict[str, object]:
    config = config or submission_config_from_env()
    if not config.wallet_name or not config.hotkey_name:
        raise ValueError("weight setter wallet configuration is missing")
    weights_payload = payload.get("scaled_weights") or payload.get("weights") or {}
    if not isinstance(weights_payload, dict) or not weights_payload:
        raise ValueError("weight payload is empty")
    subtensor = bt.Subtensor(network=config.network)
    wallet = bt.Wallet(
        name=config.wallet_name,
        hotkey=config.hotkey_name,
        path=config.wallet_path,
    )
    metagraph = subtensor.metagraph(netuid=config.netuid, lite=True)
    uid_by_hotkey = {str(hotkey): uid for uid, hotkey in enumerate(metagraph.hotkeys)}
    uids: list[int] = []
    weights: list[float] = []
    skipped_hotkeys: list[str] = []
    for hotkey, value in weights_payload.items():
        hotkey_str = str(hotkey)
        if hotkey_str not in uid_by_hotkey:
            skipped_hotkeys.append(hotkey_str)
            continue
        uids.append(int(uid_by_hotkey[hotkey_str]))
        weights.append(float(value))
    if not uids:
        raise ValueError("no weight hotkeys resolved to metagraph uids")

    # --- Circuit breaker guard ---
    chain_key = f"chain:{config.network}:{config.netuid}"
    cb_state = _chain_circuit_breaker.state(chain_key)
    if cb_state.value == "open":
        raise CircuitOpenError(chain_key, _chain_circuit_breaker.recovery_timeout)

    # --- Retry loop with exponential backoff ---
    attempt_results: list[dict[str, object]] = []
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            result = subtensor.set_weights(
                wallet=wallet,
                netuid=config.netuid,
                uids=uids,
                weights=weights,
                wait_for_inclusion=True,
                wait_for_finalization=False,
                version_key=0,
            )
            chain_result = result if isinstance(result, (dict, list, str, int, float, bool)) else str(result)
            attempt_results.append({"attempt": attempt, "status": "success", "chain_result": chain_result})
            _chain_circuit_breaker.record_success(chain_key)
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            attempt_results.append({"attempt": attempt, "status": "failed", "error": str(exc)})
            _chain_circuit_breaker.record_failure(chain_key)
            logger.warning("weight submission attempt %d/%d failed: %s", attempt, max_attempts, exc)
            if attempt < max_attempts:
                backoff = backoff_base_seconds * (2 ** (attempt - 1))
                time.sleep(backoff)

    if last_error is not None:
        raise RuntimeError(
            f"weight submission failed after {max_attempts} attempts: {last_error}"
        ) from last_error

    # --- Chain verification ---
    verification: dict[str, object] = {}
    if verify:
        try:
            wallet_hotkey = wallet.hotkey.ss58_address
            verification = verify_weights_on_chain(
                subtensor=subtensor,
                netuid=config.netuid,
                wallet_hotkey=wallet_hotkey,
                expected_uids=uids,
                expected_weights=weights,
            )
        except Exception as exc:
            logger.warning("chain verification failed: %s", exc)
            verification = {"verified": False, "error": str(exc)}

    return {
        "submitted": True,
        "verified": verification.get("verified", False),
        "network": config.network,
        "netuid": config.netuid,
        "uids": uids,
        "weight_count": len(weights),
        "skipped_hotkeys": skipped_hotkeys,
        "attempt_count": len(attempt_results),
        "attempts": attempt_results,
        "verification": verification,
    }
