"""Preserve CGJ when it blocks canonical reordering of combining marks."""

import unicodedata

import pytest

from remove_ai_watermarks.text import identify_text, remove_text_metadata


@pytest.mark.parametrize(
    "glyph",
    [
        "\u05dc\u05b7\u034f\u05b4\u05dd",
        "t\u0361\u034f\u0301s",
        "\u0628\u0651\u034f\u064e",
        "D\u0307\u034f\u0323",
        "q\u0315\u0301\u034f\u0308\u0327",
    ],
)
def test_preserves_reordering_barrier_and_recovers_marked_text(glyph: str) -> None:
    left, right = glyph.split("\u034f")
    assert unicodedata.normalize("NFD", left + right) != (
        unicodedata.normalize("NFD", left) + unicodedata.normalize("NFD", right)
    )
    clean = f"Glyph {glyph}; Memo."
    assert not identify_text(clean).has_ai_metadata
    assert remove_text_metadata(clean).cleaned_text == clean

    windows = "\u200b\u034f\u2060\u200b"
    tags = "\U000e006d\U000e0065\U000e006d\U000e006f\U000e007f"
    binary = "\u2061\u2062" * 32
    variants = (
        (clean.replace("\u034f", "\u034f" + windows), "windows_ai_text_watermark", 4),
        (clean.replace("\u034f", tags + "\u034f"), "unicode_tags", 5),
        (clean.replace("Memo", binary + "Memo"), "groundtruth_mcp", 64),
        (clean.replace("Memo", "Me\u034fmo"), "zero_width_steganography", 1),
    )
    for marked, signal, removed in variants:
        result = remove_text_metadata(marked)
        assert signal in result.removed_signals
        assert result.cleaned_text == clean
        assert result.removed_codepoints == removed


@pytest.mark.parametrize(
    ("marked", "clean"),
    [
        ("Me\u034fmo.", "Memo."),
        ("Me\u034f\u034fmo.", "Memo."),
        ("q\u0301\u034f\u0308", "q\u0301\u0308"),
        ("q\u0323\u034f\u0301", "q\u0323\u0301"),
        ("Me\u200b\u034f\u2060\u200bmo.", "Memo."),
    ],
)
def test_removes_joiners_without_supported_reordering_context(marked: str, clean: str) -> None:
    assert remove_text_metadata(marked).cleaned_text == clean
