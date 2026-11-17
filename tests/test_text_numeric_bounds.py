"""Numeric bounds must survive rewriting without turning a bound into negation."""

import pytest

from remove_ai_watermarks.text import verify_text_rewrite_facts


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("At least 20 visitors attended.", "No fewer than 20 visitors attended."),
        ("At most 20 visitors attended.", "No more than 20 visitors attended."),
        ("At least +20 visitors attended.", "At least 20 visitors attended."),
        ("At least 20.0 visitors attended.", "No less than twenty visitors attended."),
        ("No less than twenty visitors attended.", "At least twenty visitors attended."),
        ("No greater than 20 visitors attended.", "At most 20 visitors attended."),
        ("More than 20 visitors attended.", "Over 20 visitors attended."),
        ("Fewer than 20 visitors attended.", "Under 20 visitors attended."),
        ("About 20 visitors attended.", "Approximately 20 visitors attended."),
        ("Neighbors learned about one another.", "Neighbors learned about each other."),
        ("They talked about one of the best options.", "They discussed one of the best options."),
        ("\u041d\u0435 менее 20 посетителей пришли.", "Пришли как минимум 20 посетителей."),
        ("\u041d\u0435 более 20 посетителей пришли.", "Пришли максимум 20 посетителей."),
        ("Примерно 20 посетителей пришли.", "Пришли около 20 посетителей."),
    ],
)
def test_equivalent_numeric_bounds_are_accepted(source, candidate):
    assert verify_text_rewrite_facts(source, candidate)


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("At least 20 visitors attended.", "20 visitors attended."),
        ("20 visitors attended.", "At least 20 visitors attended."),
        ("At least 20 visitors attended.", "At most 20 visitors attended."),
        ("More than 20 visitors attended.", "At least 20 visitors attended."),
        ("At least -20 degrees were recorded.", "At least 20 degrees were recorded."),
        ("At most 20 visitors attended.", "Fewer than 20 visitors attended."),
        ("About 20 visitors attended.", "20 visitors attended."),
        ("20 visitors attended.", "Around 20 visitors attended."),
        ("At least one of the best options was selected.", "One of the best options was selected."),
        ("\u041d\u0435 менее 20 посетителей пришли.", "20 посетителей пришли."),
        ("\u041d\u0435 более 20 посетителей пришли.", "\u041d\u0435 менее 20 посетителей пришли."),
        ("Примерно 20 посетителей пришли.", "20 посетителей пришли."),
        (
            "At least 20 visitors and at most 50 staff attended.",
            "At most 20 visitors and at least 50 staff attended.",
        ),
        ("At least 20 visitors did not attend.", "No fewer than 20 visitors attended."),
        ("No visitors attended.", "Visitors attended."),
    ],
)
def test_numeric_bounds_and_real_negation_cannot_be_dropped_or_inverted(source, candidate):
    assert not verify_text_rewrite_facts(source, candidate)
