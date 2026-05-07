"""Tests for PIIRedactionGuard."""
from __future__ import annotations

import pytest

from shared.safety.pii_redaction import PIIRedactionGuard, _luhn


# A Luhn-valid 16-digit test card number (Visa test): 4242 4242 4242 4242
_VISA = "4242 4242 4242 4242"
_NON_CARD_DIGITS = "1234 5678 9012 3456"  # not Luhn-valid


def test_luhn_validates_visa_test_card():
    assert _luhn("4242424242424242") is True


def test_luhn_rejects_non_card_digits():
    assert _luhn("1234567890123456") is False


# -- pre_input -------------------------------------------------------------


async def test_pre_input_redacts_email():
    guard = PIIRedactionGuard()
    v = await guard.pre_input("contact me at alice@example.com please", {})
    assert v.allow is True
    assert "[REDACTED-EMAIL]" in v.redacted_text
    assert "alice@example.com" not in v.redacted_text
    kinds = {r.kind for r in v.redactions}
    assert kinds == {"email"}


async def test_pre_input_redacts_ssn():
    guard = PIIRedactionGuard()
    v = await guard.pre_input("my ssn is 123-45-6789 ok", {})
    assert v.allow is True
    assert "[REDACTED-SSN]" in v.redacted_text
    assert "123-45-6789" not in v.redacted_text


async def test_pre_input_redacts_luhn_valid_card_only():
    guard = PIIRedactionGuard()
    v = await guard.pre_input(f"card: {_VISA}", {})
    assert "[REDACTED-CC]" in v.redacted_text
    # Order id that happens to look like 16 digits should NOT be masked.
    v2 = await guard.pre_input(f"order: {_NON_CARD_DIGITS}", {})
    assert "[REDACTED-CC]" not in (v2.redacted_text or "")
    assert v2.redacted_text is None  # no matches → no redaction emitted


async def test_pre_input_redacts_phone_north_american():
    guard = PIIRedactionGuard()
    v = await guard.pre_input("reach me at (555) 123-4567 or 555-987-6543", {})
    assert v.redacted_text is not None
    assert v.redacted_text.count("[REDACTED-PHONE]") == 2


async def test_pre_input_no_match_returns_allow_no_redaction():
    guard = PIIRedactionGuard()
    v = await guard.pre_input("just a clean prompt about Python", {})
    assert v.allow is True
    assert v.redacted_text is None
    assert v.redactions == ()


async def test_pre_input_multiple_kinds_in_one_prompt():
    guard = PIIRedactionGuard()
    text = f"email alice@example.com ssn 123-45-6789 card {_VISA}"
    v = await guard.pre_input(text, {})
    assert v.redacted_text is not None
    kinds = {r.kind for r in v.redactions}
    assert kinds >= {"email", "ssn", "credit_card"}


async def test_pre_input_redaction_is_disableable():
    guard = PIIRedactionGuard(redact_input=False)
    v = await guard.pre_input("alice@example.com", {})
    assert v.allow is True
    assert v.redacted_text is None
    assert v.redactions == ()


# -- post_output ------------------------------------------------------------


async def test_post_output_also_redacts():
    guard = PIIRedactionGuard()
    v = await guard.post_output("here is what I found: bob@example.com", {})
    assert v.redacted_text is not None
    assert "[REDACTED-EMAIL]" in v.redacted_text


async def test_post_output_disableable():
    guard = PIIRedactionGuard(redact_output=False)
    v = await guard.post_output("bob@example.com", {})
    assert v.allow is True
    assert v.redacted_text is None
