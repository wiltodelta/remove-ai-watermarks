"""Check modal strength of matching conditional, coordinated English outcomes."""

import pytest

from remove_ai_watermarks.text import _changes_coordinated_outcome_qualification, verify_text_rewrite_facts


@pytest.mark.parametrize(
    ("source", "equivalent", "changed"),
    [
        (
            "Diverse climate conditions change the properties of a wooden door. "
            "In humid weather, a wooden door may expand and become heavier.",
            "Climate changes the physical features of a wooden door. "
            "In wet environments like humid weather, the door might expand and become heavier.",
            "Climate changes the physical features of a wooden door. "
            "In wet environments like humid weather, the door expands and becomes heavier.",
        ),
        (
            "In cold weather, a coating could crack and lose adhesion.",
            "In cold weather, the coating might crack and lose adhesion.",
            "In cold weather, the coating cracks and loses adhesion.",
        ),
        (
            "In high winds, an antenna may shake and stop working.",
            "In high winds, the antenna could shake and stop working.",
            "In high winds, the antenna shakes and stops working.",
        ),
        (
            "In warm weather, a cloth may dry and retain its shape.",
            "In warm weather, the cloth might dry and retain its shape.",
            "In warm weather, the cloth dries and retains its shape.",
        ),
    ],
)
def test_matches_modal_strength_for_coordinated_outcomes(source: str, equivalent: str, changed: str) -> None:
    assert verify_text_rewrite_facts(source, equivalent)
    assert not verify_text_rewrite_facts(source, changed)
    assert not verify_text_rewrite_facts(changed, source)


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        (
            "In cold weather, the coating may crack and lose adhesion.",
            "In hot weather, the coating cracks and loses adhesion.",
        ),
        (
            "In cold weather, the coating may crack and lose adhesion.",
            "In cold weather, the coating cracks and loses color.",
        ),
        (
            "In humid weather, a wooden door may expand and become heavier.",
            "In humid weather, a metal door expands and becomes heavier.",
        ),
        (
            "A wooden door is beside a metal door. In humid weather, the wooden door may expand and become heavier.",
            "A wooden door is beside a metal door. In humid weather, the door expands and becomes heavier.",
        ),
        (
            "In humid weather, a wooden door may expand and become heavier.",
            "In humid weather, the door expands and becomes heavier.",
        ),
        (
            "In cold weather, the coating may crack and lose adhesion. "
            "In cold weather, the coating cracks and loses adhesion.",
            "In cold weather, the coating cracks and loses adhesion.",
        ),
        (
            "In cold weather, the coating can crack and lose adhesion.",
            "In cold weather, the coating cracks and loses adhesion.",
        ),
    ],
)
def test_abstains_without_an_unambiguous_modal_contrast(source: str, candidate: str) -> None:
    assert verify_text_rewrite_facts(source, candidate)


def test_negated_condition_example_is_not_matched() -> None:
    source = "In humid weather, the door may expand and become heavier."
    candidate = "In environments not like humid weather, the door expands and becomes heavier."
    assert not _changes_coordinated_outcome_qualification(source, candidate)
