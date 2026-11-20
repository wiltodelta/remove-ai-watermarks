"""Preserve explicitly qualified influence in selected matched English clauses."""

import pytest

from remove_ai_watermarks.text import verify_text_rewrite_facts


@pytest.mark.parametrize(
    ("source", "equivalent", "changed"),
    [
        (
            "Artists must check the paper. It's crucial to understand that the surface absorbency of their paper "
            "can significantly impact how the paint dries and how the final artwork appears.",
            "Artists must check the paper. Painters should understand that the level of surface absorption in their "
            "paper may significantly affect how the paint dries and how the final work appears.",
            "Artists must check the paper. Artists must recognize that the level of surface absorption in their "
            "paper directly influences how the paint dries and ultimately how the finished piece looks.",
        ),
        (
            "The filter might affect how the pump operates.",
            "The filter could influence how the pump operates.",
            "The filter affects how the pump operates.",
        ),
        (
            "In cold rooms, the coating can impact how the instrument responds.",
            "In cold rooms, the coating may strongly influence how the instrument responds.",
            "In cold rooms, the coating directly affects how the instrument responds.",
        ),
    ],
)
def test_matches_influence_claim_strength(source: str, equivalent: str, changed: str) -> None:
    assert verify_text_rewrite_facts(source, equivalent)
    assert not verify_text_rewrite_facts(source, changed)
    assert not verify_text_rewrite_facts(changed, source)


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("The filter might affect how the pump operates.", "The lid affects how the pump operates."),
        ("The filter might affect how the pump operates.", "The filter affects how the fan operates."),
        (
            "In cold rooms, the coating might affect how the instrument responds.",
            "In warm rooms, the coating affects how the instrument responds.",
        ),
        (
            "The surface absorbency of their paper may affect how the paint dries.",
            "The level of surface absorption in another paper influences how the paint dries.",
        ),
        (
            "The coating may affect how the instrument responds. The coating affects how the instrument responds.",
            "The coating affects how the instrument responds.",
        ),
        ("The filter can affect how the pump operates.", "The filter affects how the pump operates."),
    ],
)
def test_abstains_if_influence_frames_do_not_match(source: str, candidate: str) -> None:
    assert verify_text_rewrite_facts(source, candidate)
