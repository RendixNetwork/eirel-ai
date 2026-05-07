"""Prometheus metrics for the validator-engine process.

The validator-engine process hosts three long-running loops:

* ``run_validator_loop`` — claim → evaluate → submit cycle.
* ``run_weight_setting_loop`` — set on-chain weights on a schedule.
* ``run_distributed_benchmarks`` — called per family from the validator loop.

This module defines a dedicated :class:`CollectorRegistry` so the
:meth:`validation.validator.main.metrics` handler can emit everything in
text format without interfering with the process-wide default registry.
"""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

registry = CollectorRegistry()

# -- Loop heartbeat --------------------------------------------------------

validator_loop_last_success_timestamp_seconds = Gauge(
    "eirel_validator_last_successful_loop_timestamp_seconds",
    "Unix timestamp of the last successful iteration of a named loop.",
    ["loop_name"],
    registry=registry,
)

# -- Benchmark runs --------------------------------------------------------

benchmark_runs_started_total = Counter(
    "eirel_validator_benchmark_runs_started_total",
    "Distributed benchmark invocations started by the validator.",
    ["family"],
    registry=registry,
)

benchmark_runs_completed_total = Counter(
    "eirel_validator_benchmark_runs_completed_total",
    "Distributed benchmark invocations completed, by outcome.",
    ["family", "outcome"],
    registry=registry,
)

benchmark_run_duration_seconds = Histogram(
    "eirel_validator_benchmark_run_duration_seconds",
    "End-to-end duration of a distributed benchmark invocation.",
    ["family"],
    buckets=(0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600, 1800),
    registry=registry,
)

# -- Judge invocations -----------------------------------------------------

judge_invocations_total = Counter(
    "eirel_validator_judge_invocations_total",
    "Judge invocations (via the owner-api judge proxy).",
    ["judge_model", "outcome"],
    registry=registry,
)

judge_duration_seconds = Histogram(
    "eirel_validator_judge_duration_seconds",
    "Per-task judge call duration.",
    ["family"],
    buckets=(0.25, 0.5, 1, 2, 5, 10, 30, 60, 120),
    registry=registry,
)

judge_disagreements_total = Counter(
    "eirel_validator_judge_disagreements_total",
    "Ensemble judge disagreements (remains 0 when single-judge mode is active).",
    ["family"],
    registry=registry,
)

# -- Oracle / reconciler / composite ----------------------------------------

oracle_grounding_outcomes_total = Counter(
    "eirel_validator_oracle_grounding_outcomes_total",
    "Per-vendor oracle outcomes (status ∈ {ok, error, blocked}).",
    ["vendor", "status"],
    registry=registry,
)

oracle_status_per_run_total = Counter(
    "eirel_validator_oracle_status_per_run_total",
    "Reconciler verdicts per task (consensus / majority / disputed / "
    "deterministic).",
    ["family", "oracle_status"],
    registry=registry,
)

reconciler_disputed_rate = Gauge(
    "eirel_validator_reconciler_disputed_rate",
    "Fraction of three_oracle tasks in the most recent run that ended "
    "in oracle_status=disputed. Spikes (>20%) indicate an oracle "
    "calibration issue worth manual review.",
    ["family"],
    registry=registry,
)

judge_json_malformations_total = Counter(
    "eirel_validator_judge_json_malformations_total",
    "Judge calls that returned malformed JSON before any retry. Rate "
    "above 2% triggers operator alert; JSON-repair retry recovers the "
    "90-98% band but below 90% means swap the model.",
    ["judge_role"],
    registry=registry,
)

eval_outcome_total = Counter(
    "eirel_validator_eval_outcome_total",
    "EvalJudge outcomes per (task, miner) pair (correct / partial / "
    "wrong / hallucinated / refused / disputed).",
    ["family", "outcome"],
    registry=registry,
)

composite_score_distribution = Histogram(
    "eirel_validator_composite_score",
    "Distribution of composite scores per (task, miner) pair. "
    "Bimodal in healthy state (most miners near 0.0 or near 1.0); "
    "uniform mid-band suggests scoring is uncalibrated.",
    ["family"],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=registry,
)

composite_knockouts_total = Counter(
    "eirel_validator_composite_knockouts_total",
    "Composite scores that were zeroed by a multiplicative knockout "
    "(tool_attestation / hallucination / cost_attestation).",
    ["family", "knockout_factor"],
    registry=registry,
)

oracle_layer_calls_total = Counter(
    "eirel_validator_oracle_layer_calls_total",
    "Per-vendor oracle / reconciler call totals (for cost auditing).",
    ["vendor"],
    registry=registry,
)

rank_parity_spearman_gauge = Gauge(
    "eirel_validator_rank_parity_spearman",
    "Spearman rank correlation between legacy weighted-sum and new "
    "composite scoring across the most recent run's miners. "
    "Rollout gate: ≥0.85 across 2 cycles before flipping default.",
    ["family"],
    registry=registry,
)


def record_oracle_grounding(vendor: str, status: str) -> None:
    """Bump the per-vendor grounding-outcome counter. Called from the
    validator's oracle fanout layer once per (task, vendor) pair."""
    oracle_grounding_outcomes_total.labels(
        vendor=vendor, status=status,
    ).inc()


def record_oracle_status(family: str, oracle_status: str) -> None:
    """Bump the reconciler-verdict counter. Called once per task at
    enrichment-phase completion."""
    oracle_status_per_run_total.labels(
        family=family, oracle_status=oracle_status,
    ).inc()


def record_eval_outcome(family: str, outcome: str) -> None:
    oracle_outcome = (outcome or "unknown").strip().lower()
    eval_outcome_total.labels(
        family=family, outcome=oracle_outcome,
    ).inc()


def record_composite_score(family: str, score: float) -> None:
    composite_score_distribution.labels(family=family).observe(
        max(0.0, min(1.0, float(score))),
    )


def record_composite_knockout(family: str, knockout_factor: str) -> None:
    """Bump the per-knockout counter when composite is zeroed.

    ``knockout_factor`` ∈ {``tool_attestation`` /
    ``hallucination_knockout`` / ``cost_attestation_knockout`` /
    ``outcome_zero``}.
    """
    composite_knockouts_total.labels(
        family=family, knockout_factor=knockout_factor,
    ).inc()


def record_judge_json_malformation(judge_role: str) -> None:
    """Bump on every judge call that returned malformed JSON before
    any retry. Used to drive the >2% alert threshold."""
    judge_json_malformations_total.labels(judge_role=judge_role).inc()


def record_oracle_call(vendor: str, n: int = 1) -> None:
    """Bump the per-vendor call total; ``n`` lets a single record
    cover multiple calls (e.g. retry loops)."""
    oracle_layer_calls_total.labels(vendor=vendor).inc(int(max(1, n)))


def record_disputed_rate(family: str, rate: float) -> None:
    reconciler_disputed_rate.labels(family=family).set(
        max(0.0, min(1.0, float(rate))),
    )


def record_rank_parity(family: str, spearman: float) -> None:
    rank_parity_spearman_gauge.labels(family=family).set(
        max(-1.0, min(1.0, float(spearman))),
    )

# -- Weight-setter metrics (loop runs in validator-engine process) --------

weight_setter_submissions_total = Counter(
    "eirel_weight_setter_submissions_total",
    "set_weights() attempts emitted by the autonomous weight-setting loop.",
    ["family", "mode", "status"],
    registry=registry,
)

weight_setter_submission_duration_seconds = Histogram(
    "eirel_weight_setter_submission_duration_seconds",
    "Time from loop trigger to set_weights() return, in seconds.",
    ["family", "mode"],
    buckets=(0.5, 1, 2, 5, 10, 30, 60, 120),
    registry=registry,
)

weight_setter_last_success_timestamp_seconds = Gauge(
    "eirel_weight_setter_last_success_timestamp_seconds",
    "Unix timestamp of the most recent successful set_weights().",
    ["family"],
    registry=registry,
)

weight_setter_chain_errors_total = Counter(
    "eirel_weight_setter_chain_errors_total",
    "set_weights() failures grouped by classified error bucket.",
    ["error_type"],
    registry=registry,
)


_RATE_LIMIT_MARKERS = ("too soon", "rate limit", "rate-limit")
_AUTH_MARKERS = ("unauthor", "signer", "permission")
_STAKE_MARKERS = ("insufficient stake", "not enough stake")


def classify_chain_error(message: str = "", exc: BaseException | None = None) -> str:
    """Collapse free-text chain errors into a bounded label set."""
    if exc is not None:
        name = type(exc).__name__.lower()
        if "timeout" in name:
            return "timeout"
        if "connection" in name or "network" in name:
            return "connection"
    msg = (message or "").lower()
    if any(m in msg for m in _RATE_LIMIT_MARKERS):
        return "rate_limited"
    if any(m in msg for m in _AUTH_MARKERS):
        return "auth"
    if any(m in msg for m in _STAKE_MARKERS):
        return "insufficient_stake"
    if exc is not None:
        return "exception"
    return "other"
