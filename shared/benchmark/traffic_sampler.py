from __future__ import annotations

"""Sample and anonymize real user traffic for use in evaluation (Item 14).

Converts recent ``TaskRequestRecord`` entries into ``BenchmarkTask``-compatible
dicts so the evaluation pipeline can blend curated, adversarial, and real
traffic into a single mixed evaluation set.
"""

import hashlib
import random
import re
from typing import Any


# PII patterns to redact
_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE_RE = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")
_IP_RE = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")


def _anonymize_text(text: str) -> str:
    """Strip PII patterns from text."""
    text = _EMAIL_RE.sub("[EMAIL_REDACTED]", text)
    text = _PHONE_RE.sub("[PHONE_REDACTED]", text)
    text = _IP_RE.sub("[IP_REDACTED]", text)
    return text


def _hash_user_id(user_id: str) -> str:
    """One-way hash user IDs for anonymization."""
    return hashlib.sha256(user_id.encode()).hexdigest()[:16]


def anonymize_task_record(record: dict[str, Any]) -> dict[str, Any]:
    """Anonymize a single task request record for evaluation use.

    Expects a dict with at least ``task_id``, ``raw_input``, ``user_id``,
    and optionally ``metadata_json``, ``routing_plan_json``.
    """
    anonymized = {
        "task_id": f"real_{record.get('task_id', 'unknown')}",
        "prompt": _anonymize_text(str(record.get("raw_input", ""))),
        "source": "real_traffic",
        "original_user_hash": _hash_user_id(str(record.get("user_id", ""))),
        "metadata": {
            "source": "real_traffic",
            "original_mode": record.get("mode", "unknown"),
            "original_status": record.get("status", "unknown"),
        },
    }

    # Extract family_id from routing plan if available
    routing_plan = record.get("routing_plan_json") or {}
    if isinstance(routing_plan, dict):
        family_id = routing_plan.get("family_id") or routing_plan.get("primary_family")
        if family_id:
            anonymized["family_id"] = str(family_id)

    return anonymized


def sample_traffic_records(
    records: list[dict[str, Any]],
    *,
    max_samples: int = 20,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Sample and anonymize traffic records for evaluation.

    Parameters:
        records: Raw task request records (as dicts).
        max_samples: Maximum number of samples to return.
        seed: Random seed for reproducibility.

    Returns:
        List of anonymized, benchmark-compatible task dicts.
    """
    if not records:
        return []

    # Filter to completed tasks with non-empty input
    eligible = [
        r for r in records
        if r.get("status") == "completed" and r.get("raw_input", "").strip()
    ]

    if not eligible:
        return []

    rng = random.Random(seed)
    sampled = rng.sample(eligible, min(max_samples, len(eligible)))

    return [anonymize_task_record(r) for r in sampled]


def load_mixed_evaluation_tasks(
    *,
    curated_tasks: list[dict[str, Any]],
    adversarial_tasks: list[dict[str, Any]] | None = None,
    real_traffic_records: list[dict[str, Any]] | None = None,
    curated_ratio: float = 0.30,
    adversarial_ratio: float = 0.30,
    real_ratio: float = 0.40,
    max_total: int = 50,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Blend curated, adversarial, and real traffic into a mixed evaluation set.

    Parameters:
        curated_tasks: Pre-built evaluation tasks.
        adversarial_tasks: Adversarial/stress-test tasks.
        real_traffic_records: Raw traffic records (will be sampled + anonymized).
        curated_ratio: Target fraction of curated tasks (default 0.30).
        adversarial_ratio: Target fraction of adversarial tasks (default 0.30).
        real_ratio: Target fraction of real traffic tasks (default 0.40).
        max_total: Maximum total tasks in the mixed set.
        seed: Random seed for reproducibility.

    Returns:
        Mixed list of evaluation tasks with ``source`` metadata.
    """
    rng = random.Random(seed)

    # Determine target counts
    n_curated = max(1, int(max_total * curated_ratio)) if curated_tasks else 0
    n_adversarial = max(1, int(max_total * adversarial_ratio)) if adversarial_tasks else 0
    n_real = max(1, int(max_total * real_ratio)) if real_traffic_records else 0

    # Sample from each source
    selected_curated = rng.sample(curated_tasks, min(n_curated, len(curated_tasks))) if curated_tasks else []
    for t in selected_curated:
        t.setdefault("source", "curated")

    selected_adversarial = rng.sample(adversarial_tasks, min(n_adversarial, len(adversarial_tasks))) if adversarial_tasks else []
    for t in selected_adversarial:
        t.setdefault("source", "adversarial")

    selected_real = sample_traffic_records(
        real_traffic_records or [],
        max_samples=n_real,
        seed=seed,
    )

    # Combine and shuffle
    mixed = selected_curated + selected_adversarial + selected_real
    rng.shuffle(mixed)

    return mixed[:max_total]
