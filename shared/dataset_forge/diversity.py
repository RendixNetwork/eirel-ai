"""Prompt diversity and history-aware deduplication.

The forge produces more candidate prompts than it needs, then this module
prunes:

- exact duplicates (after normalization),
- prompts with high token Jaccard overlap against any *currently selected*
  prompt or against any prompt from the previous N runs (history),
- prompts that would push the bundle past category-distribution caps.

A future iteration can layer sentence-embedding cosine on top of this. Today
the Jaccard layer is sufficient and avoids dragging in the
``sentence-transformers`` dependency.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_prompt(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.lower()))


def tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return intersection / union


@dataclass(slots=True)
class DiversityConstraints:
    max_jaccard_against_selected: float = 0.6
    max_jaccard_against_history: float = 0.7
    max_per_category: int = 2
    max_per_seed: int = 3


@dataclass(slots=True)
class CandidatePrompt:
    seed_id: str
    track: str
    category: str
    difficulty: str
    risk_tags: list[str]
    task_family: str
    topic: str
    prompt: str
    expected_output: dict[str, object] = field(default_factory=dict)
    allowed_tools: list[str] = field(default_factory=list)
    retrieval_constraints: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class DiversitySelectionResult:
    selected: list[CandidatePrompt]
    rejected: list[tuple[CandidatePrompt, str]]


def select_diverse(
    candidates: list[CandidatePrompt],
    *,
    target_count: int,
    history_prompts: list[str],
    constraints: DiversityConstraints | None = None,
) -> DiversitySelectionResult:
    """Greedy diverse selection.

    Walks ``candidates`` in order, accepting each one only if it does not
    collide with previously-accepted prompts or history, and if its category /
    seed quota has not been exhausted. Returns up to ``target_count`` selected
    prompts plus a parallel list of rejection reasons (useful for diagnostics).
    """
    constraints = constraints or DiversityConstraints()
    history_token_sets = [tokenize(p) for p in history_prompts]

    selected: list[CandidatePrompt] = []
    rejected: list[tuple[CandidatePrompt, str]] = []
    selected_token_sets: list[set[str]] = []
    seen_normalized: set[str] = set()
    category_counts: Counter[str] = Counter()
    seed_counts: Counter[str] = Counter()

    for candidate in candidates:
        if len(selected) >= target_count:
            rejected.append((candidate, "target_count_reached"))
            continue
        normalized = normalize_prompt(candidate.prompt)
        if not normalized:
            rejected.append((candidate, "empty_prompt"))
            continue
        if normalized in seen_normalized:
            rejected.append((candidate, "exact_duplicate"))
            continue
        if seed_counts[candidate.seed_id] >= constraints.max_per_seed:
            rejected.append((candidate, "max_per_seed"))
            continue
        if category_counts[candidate.category] >= constraints.max_per_category:
            rejected.append((candidate, "max_per_category"))
            continue
        tokens = tokenize(candidate.prompt)
        if any(
            jaccard(tokens, history_tokens) > constraints.max_jaccard_against_history
            for history_tokens in history_token_sets
        ):
            rejected.append((candidate, "history_collision"))
            continue
        if any(
            jaccard(tokens, prior_tokens) > constraints.max_jaccard_against_selected
            for prior_tokens in selected_token_sets
        ):
            rejected.append((candidate, "selected_collision"))
            continue
        selected.append(candidate)
        selected_token_sets.append(tokens)
        seen_normalized.add(normalized)
        category_counts[candidate.category] += 1
        seed_counts[candidate.seed_id] += 1

    return DiversitySelectionResult(selected=selected, rejected=rejected)
