"""Recognize selected equivalent refusal and restriction expressions."""

import pytest

from remove_ai_watermarks.text import verify_text_rewrite_facts


@pytest.mark.parametrize(
    ("original", "rewritten"),
    [
        (
            "The foreman refused to let the storm halt his work.",
            "The foreman would not let the storm stop his work.",
        ),
        (
            "The foreman said nothing and only gestured for the pilot to continue.",
            "The foreman said nothing and merely gestured for the pilot to continue.",
        ),
    ],
)
def test_equivalent_refusal_or_limitation_is_accepted(original: str, rewritten: str) -> None:
    assert verify_text_rewrite_facts(original, rewritten)


@pytest.mark.parametrize(
    ("original", "rewritten"),
    [
        (
            "The foreman refused to let the storm halt his work.",
            "The foreman let the storm halt his work.",
        ),
        (
            "The foreman merely gestured for the pilot to continue.",
            "The foreman gestured for the pilot to continue.",
        ),
    ],
)
def test_removing_a_refusal_or_limitation_is_rejected(original: str, rewritten: str) -> None:
    assert not verify_text_rewrite_facts(original, rewritten)
