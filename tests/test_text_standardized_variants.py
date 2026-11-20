"""Preserve standardized glyph choices without retaining selector payloads."""

import pytest

from remove_ai_watermarks.text import identify_text, remove_text_metadata


@pytest.mark.parametrize(
    "sequence",
    ["0\ufe00", "\u2229\ufe00", "\u1000\ufe00", "\ua856\ufe00", "\U000131a0\ufe02", "\u4e0d\ufe00"],
)
def test_preserves_standardized_variant_and_removes_separate_marker(sequence: str) -> None:
    clean = f"Glyph {sequence}. Read the note."
    assert not identify_text(clean).has_ai_metadata
    assert remove_text_metadata(clean).cleaned_text == clean
    marked = clean.replace("note", "no\u200bte")
    result = remove_text_metadata(marked)
    assert result.cleaned_text == clean
    assert result.removed_codepoints == 1


@pytest.mark.parametrize(
    ("marked", "clean"),
    [
        ("A\ufe00", "A"),
        ("0\ufe01", "0"),
        ("\u4e00\ufe06", "\u4e00"),
        ("0\ufe00\ufe01", "0"),
    ],
)
def test_unlisted_pairs_and_selector_runs_are_not_preserved(marked: str, clean: str) -> None:
    assert remove_text_metadata(marked).cleaned_text == clean
