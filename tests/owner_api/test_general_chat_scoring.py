from __future__ import annotations

from control_plane.owner_api.evaluation.general_chat_scoring import (
    aggregate_miner_score,
    budget_for_mode,
    compute_latency_score,
    extract_attribution_claims,
    score_general_chat_conversation,
    verify_trace_integrity,
)
from shared.core.evaluation_models import (
    ConversationScore,
    ConversationTrace,
    ConversationTurn,
    ModeBudget,
    TraceEntry,
)


class _FakeJudge:
    def __init__(self, score: float = 0.82) -> None:
        self._score = score
        self.calls: list[dict] = []

    async def score_quality(self, *, conversation_history, response, mode):
        self.calls.append({"turns": len(conversation_history), "mode": mode})
        return self._score


def _trace_with(tool_name: str, query: str) -> ConversationTrace:
    return ConversationTrace(
        conversation_id="c1",
        entries=[
            TraceEntry(
                tool_name=tool_name,
                args={"query": query},
                result_digest="digest",
                latency_ms=150,
                cost_usd=0.001,
                metadata={"url": query},
            )
        ],
    )


def test_extract_attribution_claims_captures_urls_and_citations():
    text = (
        "I found this at https://example.com/page1 and also [1] https://example.org/article\n"
        "I searched for \"mixture of experts\" to find more."
    )
    claims = extract_attribution_claims(text)
    assert "https://example.com/page1" in claims
    assert "https://example.org/article" in claims
    assert any(c.startswith("search:") for c in claims)


def test_verify_trace_integrity_passes_when_all_claims_backed():
    trace = _trace_with("web_search", "https://example.com/page1")
    claims = ["https://example.com/page1"]
    assert verify_trace_integrity(claims, trace) is True


def test_verify_trace_integrity_fails_on_fabricated_citation():
    trace = _trace_with("web_search", "https://example.com/page1")
    claims = ["https://fake.url/not-in-trace"]
    assert verify_trace_integrity(claims, trace) is False


# -- Tier 0 hardening --------------------------------------------------------


def test_url_normalization_strips_query_and_fragment():
    trace = _trace_with("web_search", "https://example.com/page1")
    # Same canonical URL, extra query + fragment. Should still match.
    claims = ["https://example.com/page1?ref=tracking#section-2"]
    assert verify_trace_integrity(claims, trace) is True


def test_url_normalization_strips_trailing_slash_and_case():
    trace = _trace_with("web_search", "https://example.com/page1")
    claims = ["HTTPS://EXAMPLE.COM/page1/"]
    assert verify_trace_integrity(claims, trace) is True


def test_domain_gate_blocks_cross_site_substring_match():
    trace = ConversationTrace(
        conversation_id="c1",
        entries=[
            TraceEntry(
                tool_name="web_search",
                args={"url": "https://good.example/article"},
                result_digest="https://good.example/article",
                latency_ms=100,
                cost_usd=0.001,
            )
        ],
    )
    # Attacker cites a different domain whose path is a substring of the
    # legit URL. Must not satisfy the gate.
    claims = ["https://evil.example/good.example/article"]
    assert verify_trace_integrity(claims, trace) is False


def test_search_intent_without_search_tool_call_fails():
    # Empty trace = zero search tool calls. Search-intent claim must fail
    # the gate because nothing was actually searched.
    trace = ConversationTrace(conversation_id="c1", entries=[])
    claims = extract_attribution_claims("Per my research, AAPL is doing well.")
    assert "search:present" in claims
    assert verify_trace_integrity(claims, trace) is False


def test_search_intent_satisfied_by_any_search_tool():
    trace = _trace_with("semantic_scholar", "attention mechanism")
    claims = extract_attribution_claims("I looked up the attention mechanism paper.")
    assert "search:present" in claims
    assert verify_trace_integrity(claims, trace) is True


def test_authority_claim_without_trace_entry_fails():
    trace = _trace_with("web_search", "https://example.com/other")
    claims = extract_attribution_claims(
        "According to Bloomberg, the Fed cut rates yesterday."
    )
    assert any(c.startswith("authority:bloomberg") for c in claims)
    assert verify_trace_integrity(claims, trace) is False


def test_authority_claim_satisfied_when_entity_appears_in_trace():
    trace = ConversationTrace(
        conversation_id="c1",
        entries=[
            TraceEntry(
                tool_name="web_search",
                args={"url": "https://bloomberg.com/news/rates"},
                result_digest="https://bloomberg.com/news/rates",
                latency_ms=100,
                cost_usd=0.001,
                metadata={"source": "bloomberg"},
            )
        ],
    )
    claims = extract_attribution_claims(
        "According to Bloomberg, the Fed cut rates."
    )
    # "bloomberg" is in the trace's metadata/url fields (lowercased).
    assert verify_trace_integrity(claims, trace) is True


def test_announcement_pattern_detected_as_authority_claim():
    claims = extract_attribution_claims("OpenAI recently announced a new model.")
    assert any(c.startswith("authority:openai") for c in claims)


def test_entity_stopwords_do_not_trigger_authority_claims():
    # Casual pronouns and articles should not become authority claims.
    claims = extract_attribution_claims(
        "They recently announced something. This is great news."
    )
    assert not any(c.startswith("authority:") for c in claims)


def test_empty_response_produces_no_claims():
    assert extract_attribution_claims("") == []


def test_response_with_no_citations_passes_gate():
    trace = ConversationTrace(conversation_id="c1", entries=[])
    claims = extract_attribution_claims("The answer is 42. No sources needed.")
    assert verify_trace_integrity(claims, trace) is True


# -- Tier 1 body overlap ----------------------------------------------------


def _trace_with_excerpt(url: str, body_excerpt: str) -> ConversationTrace:
    return ConversationTrace(
        conversation_id="c-body",
        entries=[
            TraceEntry(
                tool_name="web_search",
                args={"url": url},
                result_digest=url,
                result_body_excerpt=body_excerpt.lower(),
                latency_ms=100,
                cost_usd=0.001,
                metadata={"url": url},
            )
        ],
    )


def test_body_overlap_passes_when_sentence_shares_content_words():
    url = "https://bank.example/report"
    body = (
        "quarterly earnings increased due to new mortgage originations and "
        "record deposits across all branches in the western region"
    )
    trace = _trace_with_excerpt(url, body)
    response = (
        f"According to the bank report at {url}, quarterly earnings "
        f"increased thanks to new mortgage originations and record deposits."
    )
    claims = extract_attribution_claims(response)
    assert verify_trace_integrity(claims, trace, response_text=response) is True


def test_body_overlap_fails_when_sentence_diverges_from_body():
    url = "https://bank.example/report"
    # Body talks about earnings/mortgages; miner fabricates a summary about
    # aliens and space travel. Same URL, wildly different content.
    body = (
        "quarterly earnings increased due to new mortgage originations and "
        "record deposits across all branches in the western region"
    )
    trace = _trace_with_excerpt(url, body)
    response = (
        f"According to {url}, alien spacecraft were spotted orbiting Jupiter "
        f"yesterday and astronomers confirmed extraterrestrial contact."
    )
    claims = extract_attribution_claims(response)
    assert verify_trace_integrity(claims, trace, response_text=response) is False


def test_body_overlap_skipped_when_excerpt_missing():
    # Older trace entries or tool services that don't populate the excerpt
    # should not be penalized — the gate falls back to URL-match only.
    url = "https://legacy.example/page"
    trace = ConversationTrace(
        conversation_id="c-legacy",
        entries=[
            TraceEntry(
                tool_name="web_search",
                args={"url": url},
                result_digest=url,
                result_body_excerpt="",
                latency_ms=100,
                cost_usd=0.001,
            )
        ],
    )
    response = f"See {url} — random unrelated content."
    claims = extract_attribution_claims(response)
    assert verify_trace_integrity(claims, trace, response_text=response) is True


def test_body_overlap_skipped_when_sentence_too_short():
    url = "https://a.example/x"
    body = "quarterly earnings increased due to new mortgage originations"
    trace = _trace_with_excerpt(url, body)
    # Single short citation with no surrounding prose — not enough content
    # words to judge. Don't fail the miner on this.
    response = f"{url}"
    claims = extract_attribution_claims(response)
    assert verify_trace_integrity(claims, trace, response_text=response) is True


def test_body_overlap_not_invoked_when_response_text_empty():
    url = "https://a.example/x"
    body = "completely different content"
    trace = _trace_with_excerpt(url, body)
    claims = [f"url:{url}"]
    # No response_text passed — overlap check skipped.
    assert verify_trace_integrity(claims, trace) is True


# -- Tier 4 trace-gate penalty ----------------------------------------------


async def test_penalty_stamped_on_score_when_gate_fails():
    judge = _FakeJudge(score=1.0)
    trace = _trace_with("web_search", "https://example.com/real")
    budget = ModeBudget(
        mode="instant",
        web_search=True,
        latency_seconds=20.0,
        output_tokens=1024,
        reasoning_tokens=0,
        cost_usd=0.10,
    )
    response = "I checked https://fabricated.example/fake and it says 42."
    score = await score_general_chat_conversation(
        conversation_history=[ConversationTurn(role="user", content="look it up")],
        response=response,
        trace=trace,
        budget=budget,
        judge_client=judge,
        trace_gate_penalty_usd=0.50,
    )
    assert score.trace_gate == 0.0
    assert score.total == 0.0
    assert score.trace_gate_penalty_usd == 0.50


async def test_penalty_zero_when_gate_passes():
    judge = _FakeJudge(score=0.9)
    trace = _trace_with("web_search", "https://example.com/real")
    budget = ModeBudget(
        mode="instant",
        web_search=True,
        latency_seconds=20.0,
        output_tokens=1024,
        reasoning_tokens=0,
        cost_usd=0.10,
    )
    response = "See https://example.com/real for details."
    score = await score_general_chat_conversation(
        conversation_history=[ConversationTurn(role="user", content="hi")],
        response=response,
        trace=trace,
        budget=budget,
        judge_client=judge,
        trace_gate_penalty_usd=0.50,
    )
    assert score.trace_gate == 1.0
    assert score.trace_gate_penalty_usd == 0.0


async def test_penalty_zero_when_disabled_even_if_gate_fails():
    judge = _FakeJudge(score=1.0)
    trace = _trace_with("web_search", "https://example.com/real")
    budget = ModeBudget(
        mode="instant",
        web_search=True,
        latency_seconds=20.0,
        output_tokens=1024,
        reasoning_tokens=0,
        cost_usd=0.10,
    )
    response = "I checked https://fake.example/x and it says 42."
    score = await score_general_chat_conversation(
        conversation_history=[ConversationTurn(role="user", content="hi")],
        response=response,
        trace=trace,
        budget=budget,
        judge_client=judge,
        trace_gate_penalty_usd=0.0,  # penalty disabled
    )
    assert score.trace_gate == 0.0
    assert score.trace_gate_penalty_usd == 0.0


# -- Tier 3 honeytoken detection -------------------------------------------


async def test_honeytoken_citation_forces_gate_fail_even_with_matching_trace():
    from shared.core.honeytokens import generate_honeytoken_set

    active = generate_honeytoken_set("run-ht-1", count=4)
    judge = _FakeJudge(score=1.0)
    # Build a trace that includes the honeytoken URL so the basic URL
    # check would pass — the honeytoken short-circuit must override.
    trace = ConversationTrace(
        conversation_id="c-ht",
        entries=[
            TraceEntry(
                tool_name="web_search",
                args={"url": active[0]},
                result_digest=active[0],
                result_body_excerpt="",
                latency_ms=100,
                cost_usd=0.001,
            )
        ],
    )
    budget = ModeBudget(
        mode="instant",
        web_search=True,
        latency_seconds=20.0,
        output_tokens=1024,
        reasoning_tokens=0,
        cost_usd=0.10,
    )
    response = f"See {active[0]} for background."
    score = await score_general_chat_conversation(
        conversation_history=[ConversationTurn(role="user", content="look up x")],
        response=response,
        trace=trace,
        budget=budget,
        judge_client=judge,
        active_honeytokens=active,
    )
    assert score.trace_gate == 0.0
    assert score.total == 0.0
    assert score.metadata["honeytoken_cited"] is True


async def test_honeytoken_not_in_active_set_falls_through_to_normal_gate():
    from shared.core.honeytokens import generate_honeytoken_set

    active = generate_honeytoken_set("run-other", count=4)
    judge = _FakeJudge(score=0.9)
    trace = _trace_with("web_search", "https://real.example/a")
    budget = ModeBudget(
        mode="instant",
        web_search=True,
        latency_seconds=20.0,
        output_tokens=1024,
        reasoning_tokens=0,
        cost_usd=0.10,
    )
    response = "See https://real.example/a for details."
    score = await score_general_chat_conversation(
        conversation_history=[ConversationTurn(role="user", content="hi")],
        response=response,
        trace=trace,
        budget=budget,
        judge_client=judge,
        active_honeytokens=active,
    )
    assert score.trace_gate == 1.0
    assert score.metadata["honeytoken_cited"] is False


def test_aggregate_miner_score_zeros_when_honeytoken_cited():
    good = _make_conv_score(0.9, mode="instant")
    bad = _make_conv_score(0.9, mode="thinking")
    # Flip the metadata flag to simulate a honeytoken hit.
    bad_dumped = bad.model_dump()
    bad_dumped["metadata"]["honeytoken_cited"] = True
    bad_flagged = ConversationScore.model_validate(bad_dumped)

    result = aggregate_miner_score(
        miner_hotkey="5HK",
        conversation_scores=[good, bad_flagged],
        run_budget_usd=30.0,
        run_cost_usd_used=5.0,
    )
    assert result.honeytoken_cited is True
    assert result.blended == 0.0
    # Mode means still computed — we don't hide the signal, just zero
    # the final score.
    assert result.instant_mean > 0.0
    assert result.thinking_mean > 0.0


def test_aggregate_miner_score_no_flag_when_all_conversations_clean():
    good_a = _make_conv_score(0.9, mode="instant")
    good_b = _make_conv_score(0.8, mode="thinking")
    result = aggregate_miner_score(
        miner_hotkey="5HK",
        conversation_scores=[good_a, good_b],
        run_budget_usd=30.0,
        run_cost_usd_used=5.0,
    )
    assert result.honeytoken_cited is False
    assert result.blended > 0.0


def test_compute_latency_score_linear_normalization():
    assert compute_latency_score(0, 1000) == 1.0
    assert compute_latency_score(500, 1000) == 0.5
    assert compute_latency_score(1000, 1000) == 0.0
    assert compute_latency_score(2000, 1000) == 0.0


# -- ModeBudget derivation -------------------------------------------------


def test_budget_for_mode_instant_has_tight_latency():
    budget = budget_for_mode("instant")
    assert budget.mode == "instant"
    assert budget.latency_seconds == 20.0
    assert budget.output_tokens == 1024
    assert budget.reasoning_tokens == 0
    assert budget.web_search is True


def test_budget_for_mode_thinking_has_relaxed_budget():
    budget = budget_for_mode("thinking")
    assert budget.mode == "thinking"
    assert budget.latency_seconds == 60.0
    assert budget.output_tokens == 4096
    assert budget.reasoning_tokens == 16384


def test_budget_for_mode_unknown_falls_back_to_instant():
    budget = budget_for_mode(None)
    assert budget.mode == "instant"
    budget = budget_for_mode("")
    assert budget.mode == "instant"
    budget = budget_for_mode("weird-mode")
    assert budget.mode == "instant"


def test_budget_for_mode_case_insensitive():
    assert budget_for_mode("INSTANT").mode == "instant"
    assert budget_for_mode("Thinking").mode == "thinking"


def test_budget_for_mode_respects_web_search_flag():
    assert budget_for_mode("instant", web_search=False).web_search is False
    assert budget_for_mode("thinking", web_search=True).web_search is True


async def test_score_general_chat_conversation_happy_path():
    judge = _FakeJudge(score=0.8)
    trace = _trace_with("web_search", "https://example.com/page1")
    budget = ModeBudget(
        mode="instant",
        web_search=True,
        latency_seconds=20.0,
        output_tokens=1024,
        reasoning_tokens=0,
        cost_usd=0.10,
    )
    turns = [ConversationTurn(role="user", content="Find me a source about X.")]
    score = await score_general_chat_conversation(
        conversation_history=turns,
        response="See https://example.com/page1 for details.",
        trace=trace,
        budget=budget,
        judge_client=judge,
    )
    assert score.trace_gate == 1.0
    assert score.quality == 0.8
    assert score.latency > 0.9
    # Cost is now 0.0 at conversation level (moved to miner aggregate)
    assert score.cost == 0.0
    # Total: trace_gate * (0.80*quality + 0.20*latency)
    # = 1.0 * (0.80*0.8 + 0.20*~0.9925) = 1.0 * (0.64 + ~0.1985) = ~0.8385
    assert 0.80 < score.total <= 1.0


async def test_score_general_chat_conversation_gate_zero_on_fake_citation():
    judge = _FakeJudge(score=1.0)
    trace = _trace_with("web_search", "https://example.com/page1")
    budget = ModeBudget(
        mode="thinking",
        web_search=False,
        latency_seconds=60.0,
        output_tokens=4096,
        reasoning_tokens=16384,
        cost_usd=0.50,
    )
    response = (
        "I checked https://fabricated.example/this-page-does-not-exist for the answer."
    )
    score = await score_general_chat_conversation(
        conversation_history=[ConversationTurn(role="user", content="hi")],
        response=response,
        trace=trace,
        budget=budget,
        judge_client=judge,
    )
    assert score.trace_gate == 0.0
    assert score.total == 0.0


# -- aggregate_miner_score with cost efficiency ----------------------------


def _make_conv_score(total: float, mode: str = "instant") -> ConversationScore:
    return ConversationScore(
        quality=total,
        latency=total,
        cost=0.0,
        trace_gate=1.0,
        total=total,
        per_dimension={},
        mode=mode,
    )


def test_aggregate_miner_score_zero_spend_gets_1x_multiplier():
    scores = [_make_conv_score(0.80, "instant")]
    result = aggregate_miner_score(
        miner_hotkey="m1",
        conversation_scores=scores,
        run_budget_usd=30.0,
        run_cost_usd_used=0.0,
    )
    assert result.cost_efficiency == 1.0
    # blended_quality = 0.6*0.80 + 0.4*0.0 = 0.48
    # miner_score = 0.48 * (0.80 + 0.20*1.0) = 0.48 * 1.0 = 0.48
    assert abs(result.blended - 0.48) < 1e-6


def test_aggregate_miner_score_budget_exhausted_gets_0_8x_multiplier():
    scores = [_make_conv_score(0.80, "instant")]
    result = aggregate_miner_score(
        miner_hotkey="m2",
        conversation_scores=scores,
        run_budget_usd=30.0,
        run_cost_usd_used=30.0,
    )
    assert result.cost_efficiency == 0.0
    # blended_quality = 0.6*0.80 = 0.48
    # miner_score = 0.48 * (0.80 + 0.20*0.0) = 0.48 * 0.80 = 0.384
    assert abs(result.blended - 0.384) < 1e-6


def test_aggregate_miner_score_overspend_clamps_cost_efficiency_to_zero():
    scores = [_make_conv_score(1.0, "instant")]
    result = aggregate_miner_score(
        miner_hotkey="m3",
        conversation_scores=scores,
        run_budget_usd=10.0,
        run_cost_usd_used=50.0,
    )
    assert result.cost_efficiency == 0.0
    # blended_quality = 0.6*1.0 = 0.6
    # miner_score = 0.6 * 0.80 = 0.48
    assert abs(result.blended - 0.48) < 1e-6


def test_aggregate_miner_score_cost_efficiency_clamp_at_one():
    scores = [_make_conv_score(1.0, "instant")]
    result = aggregate_miner_score(
        miner_hotkey="m4",
        conversation_scores=scores,
        run_budget_usd=30.0,
        run_cost_usd_used=0.0,
    )
    assert result.cost_efficiency == 1.0


def test_aggregate_miner_score_partial_spend():
    scores = [_make_conv_score(1.0, "instant")]
    result = aggregate_miner_score(
        miner_hotkey="m5",
        conversation_scores=scores,
        run_budget_usd=30.0,
        run_cost_usd_used=15.0,
    )
    # cost_efficiency = max(0, 1 - 15/30) = 0.5
    assert abs(result.cost_efficiency - 0.5) < 1e-6
    # blended_quality = 0.6*1.0 = 0.6
    # miner_score = 0.6 * (0.80 + 0.20*0.5) = 0.6 * 0.90 = 0.54
    assert abs(result.blended - 0.54) < 1e-6


def test_aggregate_miner_score_multiplicative_formula_correctness():
    instant = [_make_conv_score(0.90, "instant"), _make_conv_score(0.70, "instant")]
    thinking = [_make_conv_score(0.95, "thinking")]
    result = aggregate_miner_score(
        miner_hotkey="m6",
        conversation_scores=[*instant, *thinking],
        run_budget_usd=30.0,
        run_cost_usd_used=6.0,
    )
    # instant_mean = (0.90+0.70)/2 = 0.80
    assert abs(result.instant_mean - 0.80) < 1e-6
    # thinking_mean = 0.95
    assert abs(result.thinking_mean - 0.95) < 1e-6
    # blended_quality = 0.6*0.80 + 0.4*0.95 = 0.48 + 0.38 = 0.86
    blended_quality = 0.86
    # cost_efficiency = max(0, 1 - 6/30) = 0.80
    assert abs(result.cost_efficiency - 0.80) < 1e-6
    # miner_score = 0.86 * (0.80 + 0.20*0.80) = 0.86 * 0.96 = 0.8256
    expected = round(blended_quality * (0.80 + 0.20 * 0.80), 6)
    assert abs(result.blended - expected) < 1e-6


def test_aggregate_miner_score_handles_missing_mode_side():
    only_instant = [_make_conv_score(0.50, "instant")]
    result = aggregate_miner_score(
        miner_hotkey="m",
        conversation_scores=only_instant,
        run_budget_usd=30.0,
        run_cost_usd_used=0.0,
    )
    assert result.instant_mean == 0.50
    assert result.thinking_mean == 0.0
    # blended_quality = 0.6*0.5 = 0.30
    # miner_score = 0.30 * 1.0 = 0.30
    assert abs(result.blended - 0.30) < 1e-6


def test_aggregate_miner_score_populates_cost_fields():
    scores = [_make_conv_score(0.80, "instant")]
    result = aggregate_miner_score(
        miner_hotkey="m7",
        conversation_scores=scores,
        run_budget_usd=25.0,
        run_cost_usd_used=10.0,
    )
    assert result.run_budget_usd == 25.0
    assert result.run_cost_usd_used == 10.0
