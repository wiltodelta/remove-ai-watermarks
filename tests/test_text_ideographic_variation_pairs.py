"""Preserve registered ideographic variation sequences and strip unregistered selectors."""

import pytest

from remove_ai_watermarks.text import identify_text, remove_text_metadata


@pytest.mark.parametrize(
    ("base", "selector"),
    [
        ("一", "\U000e0100"),
        ("㐄", "\U000e0102"),
        ("葛", "\U000e0109"),
        ("邊", "\U000e0114"),
        ("\U00020000", "\U000e0102"),
    ],
)
def test_registered_ideographic_variation_is_preserved(base: str, selector: str) -> None:
    clean = f"字形 {base}{selector} を確認します。"
    assert not identify_text(clean).has_ai_metadata
    marked = clean.replace(selector, selector + "\u200b\u034f\u2060\u200b")
    result = remove_text_metadata(marked)
    assert result.cleaned_text == clean
    assert result.removed_codepoints == 4


@pytest.mark.parametrize(
    ("base", "selector"),
    [
        ("一", "\U000e0101"),
        ("㐄", "\U000e0103"),
        ("葛", "\U000e010a"),
        ("邊", "\U000e0115"),
        ("\U00020000", "\U000e0103"),
        ("\uf900", "\U000e0100"),
    ],
)
def test_unregistered_ideographic_variation_is_removed(base: str, selector: str) -> None:
    result = remove_text_metadata(f"字形 {base}{selector} を確認します。")
    assert result.cleaned_text == f"字形 {base} を確認します。"
    assert result.removed_codepoints == 1
