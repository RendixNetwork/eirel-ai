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
