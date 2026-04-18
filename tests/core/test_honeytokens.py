from __future__ import annotations

from shared.core.honeytokens import (
    HONEYTOKEN_MARKER,
    detect_honeytoken_citation,
    generate_honeytoken_set,
    inject_honeytokens_into_search_payload,
    is_honeytoken,
)


def test_generate_honeytoken_set_is_deterministic():
    a = generate_honeytoken_set("run-42", count=8)
    b = generate_honeytoken_set("run-42", count=8)
    assert a == b
    assert len(a) == 8
    assert len({url for url in a}) == 8  # all unique
    for url in a:
        assert url.startswith("https://")
        assert HONEYTOKEN_MARKER in url


def test_generate_honeytoken_set_varies_per_run_id():
    run_a = generate_honeytoken_set("run-1", count=5)
    run_b = generate_honeytoken_set("run-2", count=5)
    assert set(run_a) & set(run_b) == set()  # zero overlap


def test_generate_honeytoken_set_zero_count_is_empty():
    assert generate_honeytoken_set("run-x", count=0) == []


def test_is_honeytoken_matches_active_set_case_insensitive():
    active = ["https://archive-news.example/eirel-canary/00-abcdef1234"]
    assert is_honeytoken("HTTPS://ARCHIVE-NEWS.EXAMPLE/eirel-canary/00-abcdef1234", active)
    assert is_honeytoken(active[0] + ".", active)  # trailing punctuation
    assert is_honeytoken(active[0] + " ", active)  # trailing whitespace


def test_is_honeytoken_fails_closed_on_marker_match():
    # Even without the active set, a URL carrying the marker is flagged.
    fabricated = "https://arbitrary.example/eirel-canary/ff-deadbeef0"
    assert is_honeytoken(fabricated, None) is True
    assert is_honeytoken(fabricated, []) is True


def test_is_honeytoken_rejects_ordinary_urls():
    assert is_honeytoken("https://example.com/page", ["https://other.example/x"]) is False
    assert is_honeytoken("", ["https://other.example/x"]) is False
    assert is_honeytoken("not a url", None) is False


def test_detect_honeytoken_citation_catches_tagged_claim():
    active = generate_honeytoken_set("run-1", count=4)
    claims = [f"url:{active[2]}", "authority:bloomberg", "search:present"]
    assert detect_honeytoken_citation(claims, active) is True


def test_detect_honeytoken_citation_catches_bare_url_claim():
    active = generate_honeytoken_set("run-1", count=4)
    claims = [active[0], "url:https://real.example/a"]
    assert detect_honeytoken_citation(claims, active) is True


def test_detect_honeytoken_citation_ignores_non_url_claims():
    assert detect_honeytoken_citation(["search:present"], ["https://x.example/y"]) is False
    assert detect_honeytoken_citation(["authority:acme"], ["https://x.example/y"]) is False


def test_detect_honeytoken_citation_empty_claims():
    assert detect_honeytoken_citation([], ["https://x.example/y"]) is False
    assert detect_honeytoken_citation([], []) is False


def test_inject_honeytokens_appends_to_results_list_on_hit():
    active = generate_honeytoken_set("run-inject", count=4)
    payload = {
        "results": [
            {"title": "real 1", "url": "https://real.example/1"},
            {"title": "real 2", "url": "https://real.example/2"},
        ]
    }
    # rate=1.0 forces a hit every call.
    out = inject_honeytokens_into_search_payload(
        payload,
        active_set=active,
        conversation_id="c-1",
        call_index=0,
        rate=1.0,
    )
    assert len(out["results"]) == 3
    injected_url = out["results"][-1]["url"]
    assert injected_url in active
    assert HONEYTOKEN_MARKER in injected_url


def test_inject_honeytokens_skips_when_rate_misses():
    active = generate_honeytoken_set("run-x", count=4)
    payload = {"results": [{"url": "https://real.example/1"}]}
    out = inject_honeytokens_into_search_payload(
        payload,
        active_set=active,
        conversation_id="c-miss",
        call_index=0,
        rate=0.0,  # never inject
    )
    assert len(out["results"]) == 1


def test_inject_honeytokens_noop_on_non_dict():
    active = ["https://x.example/eirel-canary/00-aabb"]
    assert inject_honeytokens_into_search_payload(
        "not-a-dict",
        active_set=active,
        conversation_id="c",
        call_index=0,
        rate=1.0,
    ) == "not-a-dict"


def test_inject_honeytokens_noop_on_empty_active_set():
    payload = {"results": [{"url": "https://real.example/1"}]}
    out = inject_honeytokens_into_search_payload(
        payload,
        active_set=[],
        conversation_id="c",
        call_index=0,
        rate=1.0,
    )
    assert len(out["results"]) == 1  # no injection


def test_inject_honeytokens_noop_when_no_recognized_list_field():
    payload = {"answer": "42"}  # no results/items/data
    out = inject_honeytokens_into_search_payload(
        payload,
        active_set=["https://x.example/eirel-canary/00-aa"],
        conversation_id="c",
        call_index=0,
        rate=1.0,
    )
    assert out == {"answer": "42"}


def test_inject_honeytokens_deterministic_across_calls():
    active = generate_honeytoken_set("run-stable", count=4)
    payload_a = {"results": []}
    payload_b = {"results": []}
    inject_honeytokens_into_search_payload(
        payload_a, active_set=active, conversation_id="c-1", call_index=5, rate=1.0
    )
    inject_honeytokens_into_search_payload(
        payload_b, active_set=active, conversation_id="c-1", call_index=5, rate=1.0
    )
    assert payload_a == payload_b
