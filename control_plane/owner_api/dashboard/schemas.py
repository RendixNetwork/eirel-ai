from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


TrendLiteral = Literal["up", "down", "stable", "new"]
ModeLiteral = Literal["instant", "thinking"]


class MinerMetrics(BaseModel):
    """Per-miner general_chat metrics.

    Post-agreement redesign: the primary signal is ``mean_agreement`` — the
    mean of per-task agreement scores against the OpenAI baseline reference.
    Verdict counts are exposed for dashboard drill-down (how many tasks the
    miner matched vs partially matched vs contradicted).
    """

    mean_agreement: float | None = None
    # Per-verdict counts across the miner's judged tasks.
    matches_count: int = 0
    partially_matches_count: int = 0
    not_applicable_count: int = 0
    contradicts_count: int = 0
    # Fraction of this miner's rows that ended in verdict=="error".
    error_rate: float | None = None
    # True iff ``error_rate`` is below the reliability threshold (0.30).
    reliable: bool | None = None


class CitationRef(BaseModel):
    """A single cited URL surfaced for dashboard display.

    Citations do not participate in scoring. They are extracted from the
    miner response (or the OpenAI baseline) and stored purely so operators
    can see what each agent cited alongside its verdict.
    """

    url: str
    title: str | None = None


class OverviewResponse(BaseModel):
    total_miners: int
    active_validators: int
    total_families: int
    netuid: int
    network: str
    current_run_id: str | None
    current_run_sequence: int | None
    current_run_status: str | None = None
    current_run_started_at: str | None = None
    current_run_ends_at: str | None = None
    queued_submissions: int = 0
    evaluating_submissions: int = 0
    completed_submissions: int = 0
    retired_submissions: int = 0
    build_failed_submissions: int = 0


class FamilySummary(BaseModel):
    id: str
    label: str
    weight: float
    active: bool


class FamiliesResponse(BaseModel):
    families: list[FamilySummary]


class LeaderboardEntry(BaseModel):
    rank: int
    hotkey: str
    hotkey_short: str
    # Agent identity surfaced from the submission's manifest.yaml. Lets
    # miners search by their own project name instead of memorising a
    # hotkey, and lets observers tell agents apart at a glance.
    agent_name: str | None = None
    agent_version: str | None = None
    # SHA-256 of the submission tarball as the subnet stored it. Miners
    # can `sha256sum` their local archive and compare to confirm the
    # subnet is running an unmodified copy of what they uploaded.
    artifact_sha256: str | None = None
    raw_score: float
    normalized_score: float | None
    is_serving_winner: bool
    is_running: bool = False
    trend: TrendLiteral
    previous_rank: int | None
    metrics: MinerMetrics
    epochs_participated: int
    win_count: int


class LeaderboardResponse(BaseModel):
    run_id: str | None
    run_sequence: int | None
    run_status: str | None = None
    family_id: str
    total: int
    entries: list[LeaderboardEntry] = Field(default_factory=list)


class QueuedSubmission(BaseModel):
    """One submission that has not yet appeared on a leaderboard.

    Covers anything pre-first-score: ``queued`` / ``building`` /
    ``evaluating`` / ``build_failed``. Excludes ``retired`` and any
    submission that already has a DeploymentScoreRecord (those belong on
    the leaderboard, not the queue).
    """

    submission_id: str
    hotkey: str
    hotkey_short: str
    family_id: str
    agent_name: str | None = None
    agent_version: str | None = None
    artifact_sha256: str | None = None
    status: str
    submitted_at: str | None
    submission_block: int | None = None


class QueuedSubmissionsResponse(BaseModel):
    total: int
    submissions: list[QueuedSubmission] = Field(default_factory=list)


class RunSummary(BaseModel):
    id: str
    sequence: int
    status: str
    started_at: str | None = None
    ends_at: str | None = None
    closed_at: str | None = None


class RunListResponse(BaseModel):
    runs: list[RunSummary] = Field(default_factory=list)


class MinerProfileResponse(BaseModel):
    hotkey: str
    hotkey_short: str
    uid: int | None
    family_id: str
    current_rank: int | None
    current_score: float | None
    current_weight: float | None
    lifetime_wins: int
    epochs_participated: int
    first_seen_at: str | None
    latest_metrics: MinerMetrics


class MinerRunSummary(BaseModel):
    run_id: str
    epoch_sequence: int
    status: str
    started_at: str
    closed_at: str | None
    rank: int | None
    raw_score: float | None
    normalized_score: float | None
    was_winner: bool


class MinerRunsResponse(BaseModel):
    miner_hotkey: str
    family_id: str
    runs: list[MinerRunSummary] = Field(default_factory=list)


class TaskEvaluation(BaseModel):
    """One miner's agreement result for a single task."""

    task_id: str
    mode: ModeLiteral | None
    category: str | None
    difficulty: str | None
    # Whether the task's prompt is expected to need live web search. Mirrors
    # the end-user toggle; the baseline uses this flag directly.
    web_search: bool = False
    validator_hotkey: str | None = None
    task_status: str | None  # "completed" | "failed"
    evaluated_at: str | None
    prompt: str | None
    # Number of user turns in the fixture. 1 for single-turn tasks; >1 for
    # multi-turn replay fixtures. Multi-turn tasks judge the *final*
    # assistant answer only — earlier turns build context but do not
    # produce a scored response.
    turn_count: int = 1
    # User-prompt sequence for multi-turn fixtures. Empty for single-turn
    # tasks (use ``prompt`` instead). Carries only user messages, not the
    # scripted-assistant entries — those exist for both miner and
    # baseline equally and are not interesting to display.
    user_turns: list[str] = Field(default_factory=list)
    miner_response: dict | None
    # OpenAI baseline text, extracted from TaskEvaluation.baseline_response_json.
    # Rendered side-by-side with the miner response on the dashboard so users
    # can see what the judge compared against.
    baseline_response_text: str | None = None
    # Agreement verdict: "matches" | "partially_matches" | "not_applicable"
    # | "contradicts" | "error".
    agreement_verdict: str | None = None
    # Scalar score derived from the verdict (0..1).
    agreement_score: float | None = None
    # Citations are informational only — not scored.
    miner_citations: list[CitationRef] = Field(default_factory=list)
    baseline_citations: list[CitationRef] = Field(default_factory=list)
    latency_ms: int | None
    judge_rationale: str | None


class RunDetailResponse(BaseModel):
    run_id: str
    epoch_sequence: int
    status: str
    official_score: float | None
    metrics: MinerMetrics
    tasks: list[TaskEvaluation] = Field(default_factory=list)
