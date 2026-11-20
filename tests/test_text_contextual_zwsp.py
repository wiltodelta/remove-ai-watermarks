"""Preserve contextual zero-width spaces without preserving adjacent marks."""

import pytest

from remove_ai_watermarks.text import identify_text, remove_text_metadata


@pytest.mark.parametrize(
    "clean",
    [
        "ภาษาไทย\u200bทดสอบ",
        "Example: `label = 'a\u200bb'`.",
        "Example:\r\n\r\n```python\r\nlabel = 'a\u200bb'\r\n```\r\n",
    ],
)
def test_preserves_contextual_space_and_removes_injected_prose_mark(clean: str) -> None:
    assert not identify_text(clean).has_ai_metadata
    assert remove_text_metadata(clean).cleaned_text == clean
    dirty = clean + "\n\nCheck dock\u200bside."
    result = remove_text_metadata(dirty)
    assert result.cleaned_text == clean + "\n\nCheck dockside."
    assert result.removed_codepoints == 1


@pytest.mark.parametrize(
    ("dirty", "expected"),
    [
        ("ภาษาไทย\u200b\u200bทดสอบ", "ภาษาไทยทดสอบ"),
        ("Example: `x = 'a\u200b\u2063b'`.", "Example: `x = 'ab'`."),
        ("Example: `x = 'a\u200b\u034f\u2060\u200bb'`.", "Example: `x = 'ab'`."),
        ("ภาษาไทย\u200btest", "ภาษาไทยtest"),
    ],
)
def test_ambiguous_context_does_not_exempt_marker_sequences(dirty: str, expected: str) -> None:
    assert remove_text_metadata(dirty).cleaned_text == expected
