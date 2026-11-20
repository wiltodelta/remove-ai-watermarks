"""Keep an explicit percentage endpoint distinct from a percentage change."""

import pytest

from remove_ai_watermarks import text


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("Customer satisfaction rose to 86%.", "Customer satisfaction rose by 86%."),
        ("The rate fell by 3 percent.", "The rate fell to 3 percent."),
        (
            "The final survey reported a satisfaction score increase to 86%.",
            "The final survey reported an increase in satisfaction of 86%.",
        ),
        ("Support rose to 80% and retention rose by 5%.", "Support rose by 80% and retention rose to 5%."),
    ],
)
def test_rejects_explicit_endpoint_and_change_swaps(source, candidate):
    assert not text.verify_text_rewrite_facts(source, candidate)


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("Customer satisfaction rose to 86%.", "Customer satisfaction increased to 86 percent."),
        ("The rate fell by 3 percent.", "The rate declined by three percent."),
        ("Satisfaction rose by 12%.", "There was an increase in satisfaction of 12%."),
        ("Satisfaction rose to 86%.", "Satisfaction reached 86%."),
        ("Support rose to 10% and retention rose by 10%.", "Support reached 10% and retention rose by 10%."),
    ],
)
def test_accepts_equivalent_and_unrecognized_percentage_roles(source, candidate):
    assert text.verify_text_rewrite_facts(source, candidate)
