from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.validator.engine import _hydrate_agent_inputs


def test_hydrate_promotes_mode_from_top_level():
    payload = {"mode": "thinking", "inputs": {}}
    assert _hydrate_agent_inputs(payload) == {"mode": "thinking"}


def test_hydrate_sets_web_search_true_when_allowed_tools_has_it():
    payload = {"mode": "instant", "allowed_tools": ["web_search", "sandbox"]}
    assert _hydrate_agent_inputs(payload) == {"mode": "instant", "web_search": True}


def test_hydrate_leaves_web_search_false_when_allowed_tools_omits_it():
    payload = {"mode": "instant", "allowed_tools": ["sandbox"]}
    assert _hydrate_agent_inputs(payload) == {"mode": "instant"}


def test_hydrate_preserves_existing_inputs_overrides():
    payload = {
        "mode": "thinking",
        "allowed_tools": ["web_search"],
        "inputs": {"mode": "instant", "web_search": False, "custom_k": 3},
    }
    out = _hydrate_agent_inputs(payload)
    assert out["mode"] == "instant"
    assert out["web_search"] is False
    assert out["custom_k"] == 3


def test_hydrate_empty_payload_returns_empty_dict():
    assert _hydrate_agent_inputs({}) == {}


def test_hydrate_rejects_non_list_allowed_tools():
    payload = {"allowed_tools": "web_search"}  # wrong shape: string, not list
    assert _hydrate_agent_inputs(payload) == {}


def test_hydrate_does_not_mutate_input_payload():
    original_inputs = {"existing": 1}
    payload = {"mode": "thinking", "inputs": original_inputs}
    _hydrate_agent_inputs(payload)
    assert original_inputs == {"existing": 1}
    assert payload["inputs"] is original_inputs
