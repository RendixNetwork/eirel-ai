"""Shared result type for calibration gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


GateStatus = Literal["pass", "fail", "marginal"]


@dataclass(frozen=True)
class GateResult:
    """Outcome of one calibration gate run.

    ``status`` ∈ {pass, fail, marginal}. ``marginal`` is the band
    where the underlying metric is below the strict acceptance bar
    but above the "swap the model" bar — operators should enable
    JSON-repair retry or look at fixtures before promoting to prod.

    ``details`` carries gate-specific telemetry (per-fixture results,
    per-vendor breakdown, etc.) for the deploy record.
    """

    name: str
    status: GateStatus
    measured_rate: float
    threshold: float
    n_samples: int
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "pass"
