"""Protect discretionary breaks with a plausible typographic context."""

import pytest

from remove_ai_watermarks.text import identify_text, remove_text_metadata


@pytest.mark.parametrize(
    "value",
    [
        "co\u00adoperate",
        "tug\u00adgummi",
        "ge\u00adërfde",
        "cafe\u0301\u00adine",
        "dispara\u00ad\u2011te",
        "\u0438\u043d\u00ad\u0444\u043e\u0440\u043c\u0430\u0446\u0438\u044f",
        "\u0627\u0633\u00ad\u062a\u062e\u062f\u0627\u0645",
        "co\u00adop\u00aderate",
    ],
)
def test_preserves_intra_word_discretionary_break_and_removes_unrelated_mark(value: str) -> None:
    clean = f"{value} Memo."
    marked = f"{value} Me\u2063mo."
    assert not identify_text(clean).has_ai_metadata
    assert remove_text_metadata(clean).cleaned_text == clean

    result = remove_text_metadata(marked)
    assert result.cleaned_text == clean
    assert result.removed_codepoints == 1


@pytest.mark.parametrize(
    ("marked", "clean"),
    [
        ("\u00adMemo", "Memo"),
        ("Memo\u00ad.", "Memo."),
        ("12\u00ad34", "1234"),
        ("co\u00ad\u00adoperate", "cooperate"),
        ("a\u200bb\u00adc\u206cd", "abcd"),
        ("a\u00ad\u0301bc", "a\u0301bc"),
    ],
)
def test_strips_soft_hyphens_without_discretionary_context(marked: str, clean: str) -> None:
    assert remove_text_metadata(marked).cleaned_text == clean
