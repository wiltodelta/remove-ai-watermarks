"""Preserve explicit virama join controls and remove neighboring watermark payloads."""

import pytest

from remove_ai_watermarks.text import identify_text, remove_text_metadata


@pytest.mark.parametrize(
    "glyph",
    [
        "\u0917\u094d\u200d",
        "\u0930\u094d\u200d.",
        "\u0915\u093c\u094d\u200d Notes.",
        "\u0958\u094d\u200d!",
        "\u0978\u094d\u200d\nNotes.",
        "\u0d05\u0d23\u0d4d\u200d.",
        "\u0d05\u0d28\u0d4d\u200d",
        "\u0d05\u0d30\u0d4d\u200d Notes.",
        "\u0d05\u0d32\u0d4d\u200d.",
        "\u0d05\u0d33\u0d4d\u200d",
        "\ua892\ua8c4\u200d\ua8b0",
        "\ua8a0\ua8c4\u200d\ua892",
        "\u0915\u094d\u200d\u0937",
        "\u0915\u094d\u200c\u0937",
        "\u0b15\u200d\u0b4d\u0b37",
        "\u0b15\u0b4d\u200c\u0b37",
    ],
)
def test_preserves_virama_join_controls_in_clean_and_marked_text(glyph: str) -> None:
    clean = f"Memo {glyph}"
    joiner = "\u200d" if "\u200d" in glyph else "\u200c"
    assert remove_text_metadata(clean).cleaned_text == clean
    assert not identify_text(clean).has_ai_metadata

    windows = "\u200b\u034f\u2060\u200b"
    tags = "\U000e006d\U000e0065\U000e006d\U000e006f\U000e007f"
    binary = "\u2061\u2062" * 32
    variants = (
        (clean.replace(joiner, windows + joiner), "windows_ai_text_watermark", 4),
        (clean.replace(joiner, tags + joiner), "unicode_tags", 5),
        (clean.replace("Memo", binary + "Memo"), "groundtruth_mcp", 64),
        (clean.replace("Memo", "Me\u200dmo"), "zero_width_steganography", 1),
    )
    for marked, signal, removed in variants:
        result = remove_text_metadata(marked)
        assert result.cleaned_text == clean
        assert signal in result.removed_signals
        assert result.removed_codepoints == removed


@pytest.mark.parametrize(
    "marked",
    [
        "A\u094d\u200d.",
        "\u0905\u094d\u200d.",
        "\u0d05\u0d4d\u200d.",
        "\ua892\ua8c4\u200d.",
        "\ua892\ua8c4\u200dA.",
        "\u0915\u200d.",
        "\u0915\u094d\u200dA.",
        "Me\u200dmo.",
    ],
)
def test_removes_joiner_outside_supported_virama_context(marked: str) -> None:
    result = remove_text_metadata(marked)
    assert result.cleaned_text == marked.replace("\u200d", "")
    assert result.removed_codepoints == 1
