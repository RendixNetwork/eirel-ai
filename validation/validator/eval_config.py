"""Validator-side env-var taxonomy for the eval pipeline.

The validator process runs three external-LLM dependencies that previously
lived elsewhere:

  * **3 oracles** (OpenAI / Gemini / Grok) — invoked at task-claim time
    for items tagged ``oracle_source: three_oracle``. Each validator
    independently produces its own ground truth.
  * **Reconciler** (Chutes-hosted ``zai-org/GLM-5.1-TEE``) — synthesizes
    the 3 oracle answers into ``expected_claims`` + ``must_not_claim``
    extras for the judge.

This module is the single source of truth for the env vars these
dependencies consume. Importing from here keeps the providers / oracle
layer decoupled from raw ``os.getenv`` calls; tests can override via
``monkeypatch.setenv`` and trust the resolution is uniform.

Ops note: validator deployments now require API keys for OpenAI +
Gemini + Grok + Chutes. Distribute via existing secret management.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _strip(value: str | None) -> str:
    return (value or "").strip()


def _strip_or_none(value: str | None) -> str | None:
    cleaned = _strip(value)
    return cleaned or None


def _float_env(name: str, default: float) -> float:
    raw = _strip(os.getenv(name))
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"env {name}={raw!r} is not a valid float"
        ) from exc


def _bool_env(name: str, default: bool) -> bool:
    raw = _strip(os.getenv(name)).lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(
        f"env {name}={raw!r} is not a valid bool (expected 1/0/true/false)"
    )


@dataclass(frozen=True)
class ProviderConfig:
    """Per-provider config block. ``configured`` iff base_url + api_key + model are all set."""

    base_url: str
    api_key: str
    model: str
    timeout_seconds: float
    max_tokens: int

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)


_DEFAULT_TIMEOUT_SECONDS = 30.0
# Reasoning-mode oracles (gpt-5.x, gemini-3.x, grok-4.x) burn budget
# on internal thinking tokens before producing the visible JSON. 2048
# was tight enough that Gemini hit ``MAX_TOKENS`` on trivial prompts.
# 4096 leaves room for reasoning + a short canonical answer.
_DEFAULT_MAX_TOKENS = 4096


def _provider_config(prefix: str, *, default_model: str) -> ProviderConfig:
    """Resolve a ``ProviderConfig`` from ``EIREL_VALIDATOR_<PREFIX>_*`` envs.

    ``EIREL_VALIDATOR_<PREFIX>_BASE_URL`` / ``_API_KEY`` / ``_MODEL`` are required
    at use time; this helper loads whatever is present and the caller should
    check ``configured`` before invoking the provider.
    """
    base_url = _strip(os.getenv(f"{prefix}_BASE_URL"))
    api_key = _strip(os.getenv(f"{prefix}_API_KEY"))
    model = _strip(os.getenv(f"{prefix}_MODEL")) or default_model
    timeout_seconds = _float_env(
        f"{prefix}_TIMEOUT_SECONDS",
        _float_env(
            "EIREL_VALIDATOR_ORACLE_TIMEOUT_S", _DEFAULT_TIMEOUT_SECONDS
        ),
    )
    max_tokens_raw = _strip(os.getenv(f"{prefix}_MAX_TOKENS"))
    try:
        max_tokens = int(max_tokens_raw) if max_tokens_raw else _DEFAULT_MAX_TOKENS
    except ValueError as exc:
        raise RuntimeError(
            f"env {prefix}_MAX_TOKENS={max_tokens_raw!r} is not a valid int"
        ) from exc
    return ProviderConfig(
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
        max_tokens=max_tokens,
    )


def openai_oracle_config() -> ProviderConfig:
    """OpenAI oracle config (``EIREL_VALIDATOR_ORACLE_OPENAI_*``)."""
    return _provider_config(
        "EIREL_VALIDATOR_ORACLE_OPENAI", default_model="gpt-5.4",
    )


def gemini_oracle_config() -> ProviderConfig:
    """Gemini oracle config (``EIREL_VALIDATOR_ORACLE_GEMINI_*``)."""
    return _provider_config(
        "EIREL_VALIDATOR_ORACLE_GEMINI",
        default_model="gemini-3.1-pro-preview",
    )


def grok_oracle_config() -> ProviderConfig:
    """Grok oracle config (``EIREL_VALIDATOR_ORACLE_GROK_*``)."""
    return _provider_config(
        "EIREL_VALIDATOR_ORACLE_GROK", default_model="grok-4.3",
    )


def reconciler_config() -> ProviderConfig:
    """Chutes-hosted reconciler config (``EIREL_VALIDATOR_RECONCILER_*``).

    Default model ``zai-org/GLM-5.1-TEE``: TEE-attestable; same family
    used by all 3 internal eiretes judge roles for verifiable scoring.
    """
    return _provider_config(
        "EIREL_VALIDATOR_RECONCILER", default_model="zai-org/GLM-5.1-TEE",
    )


def oracle_parallel_enabled() -> bool:
    """Whether to fan out the 3 oracle calls in parallel (default: true).

    Sequential execution is supported for debugging individual vendor
    failures (set ``EIREL_VALIDATOR_ORACLE_PARALLEL=0``).
    """
    return _bool_env("EIREL_VALIDATOR_ORACLE_PARALLEL", default=True)


_VALID_PAIRWISE_VENDORS: frozenset[str] = frozenset({"openai", "gemini", "grok"})
_DEFAULT_PAIRWISE_VENDOR = "openai"


def pairwise_reference_vendor() -> str:
    """Which oracle's answer the pairwise judge sees as the comparator.

    Set via ``EIREL_VALIDATOR_PAIRWISE_REFERENCE_VENDOR ∈
    {openai, gemini, grok}``. Reusing one of the 3 oracle answers
    eliminates the per-task OpenAI baseline call (~$0.05/task * ~30
    tasks/cycle) — the oracle answers are already grounded via web
    search at task-claim time and cached per task.

    Falls back to ``openai`` for unknown values rather than raising,
    so a misconfigured env var degrades gracefully.
    """
    raw = _strip(os.getenv("EIREL_VALIDATOR_PAIRWISE_REFERENCE_VENDOR"))
    if not raw:
        return _DEFAULT_PAIRWISE_VENDOR
    lowered = raw.lower()
    if lowered not in _VALID_PAIRWISE_VENDORS:
        return _DEFAULT_PAIRWISE_VENDOR
    return lowered
