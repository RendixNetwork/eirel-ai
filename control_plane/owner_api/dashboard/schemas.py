from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


TrendLiteral = Literal["up", "down", "stable", "new"]
ModeLiteral = Literal["instant", "thinking"]


class MinerMetrics(BaseModel):
    """Per-miner general_chat metrics. All values [0, 1] unless noted."""
    quality_mean: float | None = None
    latency_mean: float | None = None
    trace_gate_pass_rate: float | None = None
    honeytoken_count: int = 0
    instant_mean: float | None = None
    thinking_mean: float | None = None
    blended: float | None = None
    cost_efficiency: float | None = None


class JudgeDimensions(BaseModel):
    """Eiretes judge's 4-dimension rubric scores for general_chat."""
    goal_fulfillment: float | None = None
    correctness: float | None = None
    grounding: float | None = None
    conversation_coherence: float | None = None


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
    task_id: str
    task_index: int
    mode: ModeLiteral | None
    category: str | None
    difficulty: str | None
    validator_hotkey: str | None
    task_score: float | None
    task_status: str | None
    evaluated_at: str | None
    prompt: str | None
    miner_response: dict | None
    quality_score: float | None
    dimension_scores: JudgeDimensions
    latency_score: float | None
    latency_ms: int | None
    trace_gate_passed: bool | None
    honeytoken_cited: bool | None
    judge_rationale: str | None


class RunDetailResponse(BaseModel):
    run_id: str
    epoch_sequence: int
    status: str
    official_score: float | None
    metrics: MinerMetrics
    tasks: list[TaskEvaluation] = Field(default_factory=list)
