"""Check contextual requirement wording for tools and instruments."""

import pytest

from remove_ai_watermarks.text import verify_text_rewrite_facts


def test_essential_and_indispensable_tools_have_the_same_requirement() -> None:
    source = "A needle and thread are basic but essential tools in our sewing kit."
    candidate = "A needle and thread serve as fundamental yet indispensable instruments within our sewing kit."
    assert verify_text_rewrite_facts(source, candidate)
    assert verify_text_rewrite_facts(candidate, source)


@pytest.mark.parametrize(
    ("required", "optional"),
    [
        ("These are essential tools for measuring pressure.", "These are useful instruments for measuring pressure."),
        ("An essential instrument positions the screen.", "A useful instrument positions the screen."),
    ],
)
def test_added_or_removed_tool_requirement_is_rejected(required: str, optional: str) -> None:
    assert not verify_text_rewrite_facts(required, optional)
    assert not verify_text_rewrite_facts(optional, required)
