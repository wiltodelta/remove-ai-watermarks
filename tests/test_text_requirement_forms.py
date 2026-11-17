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
    ],
)
def test_necessity_changes_are_refused(source, candidate):
    assert not verify_text_rewrite_facts(source, candidate)
