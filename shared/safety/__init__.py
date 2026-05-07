"""Orchestrator-boundary safety guards.

Two thin guards run by default at the :class:`ProductOrchestrator`
boundary:

  * :class:`PIIRedactionGuard` — regex SSN / CC / email / phone
    redaction on the inbound prompt and outbound assistant content.
  * :class:`PromptInjectionGuard` — regex denylist (with an optional
    LLM classifier escalation) on the inbound prompt.

The guards live at the orchestrator boundary, NOT inside the miner
graph. Same isolation principle as user data in eval mode: hardening
is a product-layer responsibility, miners compete on reasoning over
already-clean inputs.

The :class:`OrchestratorGuard` ABC is intentionally separate from
:class:`eirel.safety.Guard`. The SDK guard signature is shaped around
graph state mappings; the orchestrator guard takes a plain string
prompt or content. Mixing them via shimming creates more friction than
it saves.
"""
from __future__ import annotations

from shared.safety.guard import (
    GuardVerdict,
    OrchestratorGuard,
    Redaction,
)
from shared.safety.pii_redaction import PIIRedactionGuard
from shared.safety.prompt_injection import (
    PromptInjectionClassifier,
    PromptInjectionGuard,
)

__all__ = [
    "GuardVerdict",
    "OrchestratorGuard",
    "PIIRedactionGuard",
    "PromptInjectionClassifier",
    "PromptInjectionGuard",
    "Redaction",
]
