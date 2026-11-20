"""Distinguish completed handovers from offers involving the same item."""

import pytest

from remove_ai_watermarks.text import verify_text_rewrite_facts


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("The officer handed her a cup of tea and a map.", "The officer offered her a cup of tea and a chart."),
        ("They offered him a key.", "They handed him a key."),
        ("She offered them a sealed envelope.", "She handed them a sealed envelope."),
    ],
)
def test_offer_and_physical_handover_are_not_interchangeable(source: str, candidate: str) -> None:
    assert not verify_text_rewrite_facts(source, candidate)


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("The officer handed her a cup of tea.", "The officer passed her a cup of tea."),
        ("They handed him a key.", "They handed him the key."),
        ("She offered them a chance.", "She gave them a chance."),
        ("She offered him a book and handed him a book.", "She handed him a book."),
    ],
)
def test_equivalent_or_unpaired_handover_wording_does_not_trigger_guard(source: str, candidate: str) -> None:
    assert verify_text_rewrite_facts(source, candidate)
