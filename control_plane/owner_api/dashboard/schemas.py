from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


TrendLiteral = Literal["up", "down", "stable", "new"]
ModeLiteral = Literal["instant", "thinking"]


class MinerMetrics(BaseModel):
    """Per-miner general_chat metrics.

    Multi-metric per-task scoring: the headline is ``mean_task_score`` —
    the mean of per-task ``final_task_score`` values, each computed as
    a weighted sum of up to 7 dimensions (pairwise + grounded +
    retrieval + tool_routing + safety + latency_cost +
    computation_correctness) with weights re-normalized to whichever
    dimensions applied for each task type.

    Per-dimension means show how the miner scores along each axis;
    ``tasks_with_<dim>`` give coverage (some dimensions are N/A for
    some task types). Legacy ``mean_agreement`` is preserved so older
    dashboard consumers don't break — it equals the mean of pairwise
    verdict scores via ``VERDICT_SCORES``.
    """

    # Headline + legacy
    mean_task_score: float | None = None
    mean_agreement: float | None = None
    # Per-dimension means (None when no task in this run scored the dimension)
    mean_pairwise_preference: float | None = None
    mean_grounded_correctness: float | None = None
    mean_retrieval_quality: float | None = None
    mean_tool_routing: float | None = None
    mean_instruction_safety: float | None = None
    mean_latency_cost: float | None = None
    mean_computation_correctness: float | None = None
    # Coverage counts — how many tasks scored each dimension
    tasks_with_pairwise: int = 0
    tasks_with_grounded: int = 0
    tasks_with_retrieval: int = 0
    tasks_with_tool_routing: int = 0
    tasks_with_safety: int = 0
    tasks_with_latency_cost: int = 0
    tasks_with_computation_correctness: int = 0
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


class PairwiseBreakdown(BaseModel):
    """Inner mechanics of how the pairwise score was computed.

    Single judge call per task with a randomized A/B assignment:
    ``miner_position`` records which slot held the miner's answer for
    THIS task (chosen uniformly at random per task; the judge cannot
    tell which side is the miner unless the candidate text leaks it).
    ``winner`` is what the judge picked. ``final_score`` is the
    miner-perspective score after position remap (1.0 win / 0.5 tie /
    0.0 loss). ``category_scores`` is the per-criterion 0-5 breakdown
    if the judge LLM emitted it.
    """

    final_score: float | None = None
    miner_position: Literal["A", "B"] | None = None
    winner: Literal["A", "B", "tie"] | None = None
    confidence: float | None = None
    reason: str | None = None
    category_scores: dict[str, dict[str, int]] | None = None


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
    # Multi-metric per-task score breakdown. All dimensions nullable —
    # N/A dimensions for a given task type re-normalize out of
    # ``final_task_score``. ``applied_weights`` records the actual
    # weights used after re-normalization; ``applicable_metrics`` is
    # the subset of dimensions that scored (non-N/A) for this task.
    task_type: str | None = None
    pairwise_preference_score: float | None = None
    grounded_correctness: float | None = None
    retrieval_quality: float | None = None
    tool_routing: float | None = None
    instruction_safety: float | None = None
    latency_cost: float | None = None
    computation_correctness: float | None = None
    final_task_score: float | None = None
    applied_weights: dict[str, float] | None = None
    applicable_metrics: list[str] | None = None
    # Inner mechanics of the pairwise score: both swap-calls plus the
    # averaged final. Populated from ``judge_output_json.metadata``.
    # Null on legacy rows (judged before the swap-and-average path
    # landed) and on rows where the judge call errored.
    pairwise_breakdown: PairwiseBreakdown | None = None
    # ── Oracle / EvalJudge / composite surfacing ────────────────────
    # Whether this task ran 3-oracle enrichment at the validator
    # (``three_oracle``) or used the pool's pre-baked deterministic
    # gold (``deterministic``). Null on legacy rows.
    oracle_source: Literal["three_oracle", "deterministic"] | None = None
    # Reconciler verdict on the 3 oracles for ``three_oracle`` items;
    # ``deterministic`` for everything else. Null on legacy rows.
    oracle_status: Literal[
        "consensus", "majority", "disputed", "deterministic"
    ] | None = None
    # One-line note on where the oracles diverged (when applicable).
    oracle_disagreement_note: str | None = None
    # Per-vendor up/down for the oracle fanout — surfaces Grok-circuit-
    # breaker activations + per-vendor agreement-with-majority telemetry.
    vendor_status: dict[str, str] | None = None
    # Multiplicative composite from EvalJudge + composite_score
    # endpoint. Replaces ``final_task_score`` as the canonical per-task
    # score; the legacy weighted-sum is kept in ``weighted_sum_score``
    # for parity comparison.
    composite_score: float | None = None
    composite_knockout_reason: str | None = None
    weighted_sum_score: float | None = None
    # EvalJudge per-task outcome + categorical guidance.
    eval_outcome: Literal[
        "correct", "partial", "wrong", "hallucinated",
        "refused", "disputed",
    ] | None = None
    eval_failure_mode: str | None = None
    eval_guidance: str | None = None
    # Server-attested tool usage from OrchestratorToolCallLog. Empty
    # list when the validator's ledger fetch returned nothing (no tool
    # calls OR ledger fetch unavailable / fail-safe).
    ledger_tools: list[str] = Field(default_factory=list)
    # Capability × domain matrix tags from the eirel-eval-pool sampler.
    # Null when the bundle wasn't rendered with --use-matrix-sampler.
    capability: str | None = None
    domain: str | None = None


class RunDetailResponse(BaseModel):
    run_id: str
    epoch_sequence: int
    status: str
    official_score: float | None
    metrics: MinerMetrics
    # Total task count in the run's evaluation bundle. Lets the frontend
    # render "judged / total" progress (judged = len(tasks)). Falls back
    # to len(tasks) if the bundle isn't readable for any reason.
    total_tasks: int = 0
    # The miner's submission ID for this run. Populated only when the
    # miner had a deployment scored in this run (DeploymentScoreRecord
    # lookup). Frontend uses it to build the public artifact-download +
    # source-viewer links once the run is closed.
    submission_id: str | None = None
    tasks: list[TaskEvaluation] = Field(default_factory=list)


class SubmissionFile(BaseModel):
    path: str
    size_bytes: int


class SubmissionFilesResponse(BaseModel):
    files: list[SubmissionFile] = Field(default_factory=list)


class ValidatorRunCost(BaseModel):
    """Per-validator cost breakdown for one run.

    Costs are validator-paid components only (oracle layer +
    eiretes-judge). Miner-paid ``proxy_cost_usd`` is NOT included —
    that's the miner's bill against the subnet provider-proxy and
    showing it under "validator cost" would conflate two payers.
    """

    validator_hotkey: str
    tasks_claimed: int
    tasks_evaluated: int
    oracle_cost_usd: float
    judge_cost_usd: float
    total_cost_usd: float


class ValidatorRunCostsResponse(BaseModel):
    run_id: str
    validators: list[ValidatorRunCost] = Field(default_factory=list)
    # Run-wide totals for quick "what did this run cost the validator
    # fleet" readouts on the dashboard.
    total_oracle_cost_usd: float = 0.0
    total_judge_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
