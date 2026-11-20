"""Equivalent necessity wording must not become a spurious rewrite refusal."""

import pytest

from remove_ai_watermarks.text import verify_text_rewrite_facts


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("They recorded materials needed for later repairs.", "They noted materials required for future repairs."),
        ("They recorded materials required for later repairs.", "They noted materials needed for future repairs."),
        ("The organizers need equipment.", "The organizers require equipment."),
        ("The workshop needs tools.", "The workshop requires tools."),
        ("Repairing the item required cotton.", "Fixing the item needed cotton."),
        ("The equipment is necessary.", "The equipment is needed."),
        (
            "Those routines were predictable, necessary, and unyielding.",
            "The rounds were predictable, indispensable and unyielding.",
        ),
        ("Reliable backup power is indispensable.", "Reliable backup power is necessary."),
    ],
)
def test_equivalent_requirement_forms_are_accepted(source, candidate):
    assert verify_text_rewrite_facts(source, candidate)


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("The workshop needed tools.", "The workshop used tools."),
        ("The workshop used tools.", "The workshop needed tools."),
        ("The organizers require equipment.", "The organizers want equipment."),
        ("The equipment is necessary.", "The equipment is optional."),
        ("Reliable backup power is indispensable.", "Reliable backup power is optional."),
    ],
)
def test_necessity_changes_are_refused(source, candidate):
    assert not verify_text_rewrite_facts(source, candidate)


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        (
            "It is essential to know the mechanism to understand the process.",
            "To understand the process, one must first grasp the mechanism.",
        ),
        (
            "The screen traps debris, which requires occasional cleaning with a brush.",
            "The screen accumulates debris, necessitating periodic cleaning with a brush.",
        ),
        (
            "They discussed the need for a guide to accompany the group.",
            "They discussed the necessity for a guide to accompany the group.",
        ),
    ],
)
def test_equivalent_requirements_with_distinct_syntax(source: str, candidate: str) -> None:
    assert verify_text_rewrite_facts(source, candidate)
    assert verify_text_rewrite_facts(candidate, source)


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        (
            "It is essential to know the code before opening the gate.",
            "Knowing the code before opening the gate is optional.",
        ),
        (
            "The screen traps debris, necessitating periodic cleaning.",
            "The screen traps debris, and cleaning is optional.",
        ),
        (
            "They discussed the necessity for a guide to accompany the group.",
            "They discussed the possibility of a guide accompanying the group.",
        ),
        (
            "Of the operators, one must first grasp the handle.",
            "Of the operators, two must first grasp the handle.",
        ),
    ],
)
def test_removed_requirements_with_distinct_syntax(source: str, candidate: str) -> None:
    assert not verify_text_rewrite_facts(source, candidate)


def test_a_requirement_elsewhere_does_not_hide_a_change_in_commitment() -> None:
    source = (
        "The attendant will be expected to sign the agreed upon access form before entry. "
        "The supervisor must record the badge number."
    )
    candidate = (
        "The attendant will be required to sign the agreed-upon access form prior to entry. "
        "The supervisor must record the badge number."
    )
    assert not verify_text_rewrite_facts(source, candidate)
    assert not verify_text_rewrite_facts(candidate, source)


def test_matching_commitments_allow_reordering_and_timing_paraphrases() -> None:
    source = (
        "The attendant will be expected to sign the access form before entry. "
        "The attendant will be required to return the badge."
    )
    candidate = (
        "The attendant will be required to return the badge. "
        "The attendant will be expected to sign the access form prior to entry."
    )
    assert verify_text_rewrite_facts(source, candidate)
    assert verify_text_rewrite_facts(candidate, source)
