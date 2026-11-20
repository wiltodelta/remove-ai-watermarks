"""Protect explicit no-break multiplication and division notation."""

import pytest

from remove_ai_watermarks.text import remove_text_metadata


@pytest.mark.parametrize(
    "expression", ["3\u2060\u00d7\u20607", "12\u2060\u00f7\u20603", "\u0663\u2060\u00d7\u2060\u0667"]
)
def test_preserves_math_joiners_and_removes_unrelated_mark(expression: str) -> None:
    clean = f"Write {expression} above the label."
    marked = clean.replace("label", "la\u2060bel")
    assert remove_text_metadata(clean).cleaned_text == clean
    result = remove_text_metadata(marked)
    assert result.cleaned_text == clean
    assert result.removed_codepoints == 1


@pytest.mark.parametrize("source", ["a\u2060\u00d7\u2060b", "3\u2060\u00d7\u2060\u20607", "dock\u2060side"])
def test_keeps_stripping_word_joiners_outside_the_supported_form(source: str) -> None:
    assert remove_text_metadata(source).cleaned_text == source.replace("\u2060", "")
