"""Selected explicit signal sources must not become the signal recipients."""

import pytest

from remove_ai_watermarks.text import _guard_local_rewrite_candidate, verify_text_rewrite_facts


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("Borrowers receive a nod from the community.", "Borrowers give a nod to the community."),
        ("Borrowers receive a nod from the community.", "Borrowers offer a nod in return to the community."),
        ("Volunteers receive approval from the committee.", "Volunteers give approval to the committee."),
        ("Residents get permission from the owner.", "Residents give permission to the owner."),
        ("The team gets consent from the council.", "The team gives consent to the council."),
        ("Neighbors give a nod to the volunteers.", "Neighbors get a nod from the volunteers."),
        ("A nod comes with approval from the original owner.", "A nod comes with approval to the original owner."),
        ("They receive nods from the neighbors.", "They give nods to the neighbors."),
    ],
)
def test_explicit_signal_direction_changes_are_rejected(source: str, candidate: str) -> None:
    assert not verify_text_rewrite_facts(source, candidate)


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("Borrowers receive a nod from the community.", "A nod from the community reassures borrowers."),
        ("Neighbors give a nod to the volunteers.", "The neighbors offer a nod to the volunteers."),
        ("They get approval from the committee.", "Approval from the committee is given to them."),
        ("They get a nod from the community.", "The community gives them a nod."),
        ("They get a nod from the volunteers.", "They give a nod to the community."),
        ("They get a nod from the community. They give a nod to the community.", "They get a nod from the community."),
        ("They get a nod from the community.", "They get a nod from the community and give a nod to the community."),
        ("A nod from the community garden greets visitors.", "Visitors give a nod to the community."),
        ("A nod from the owner-shaped display greets visitors.", "Visitors give a nod to the owner."),
        ("Approval from the local council is necessary.", "Approval to the council is necessary."),
        ("They get a nod from the community. Consent goes to the community.", "They get a nod from the community."),
        ("They get a nod from the community.", "They get a nod at the community event."),
    ],
)
def test_preserved_or_unresolved_signal_direction_is_not_rejected(source: str, candidate: str) -> None:
    # An unresolved relation remains outside this narrow heuristic's claim.
    assert verify_text_rewrite_facts(source, candidate)


def test_native_candidate_guard_applies_signal_direction_check() -> None:
    source = "Borrowers receive a nod from the community."
    candidate = "Borrowers give a nod to the community."
    assert _guard_local_rewrite_candidate(source, source, {}, candidate, None) is None
