"""Preserve registered emoji text and emoji presentations by exact pair."""

import pytest

from remove_ai_watermarks.text import identify_text, remove_text_metadata


@pytest.mark.parametrize("base", list("#*0123456789"))
@pytest.mark.parametrize("selector", ["\ufe0e", "\ufe0f"])
def test_registered_ascii_emoji_variations_are_valid_without_an_enclosure(base, selector):
    clean = f"Press {base}{selector} now."
    assert not identify_text(clean).has_ai_metadata
    marked = clean.replace("now", "n\u200bow")
    result = remove_text_metadata(marked)
    assert result.cleaned_text == clean
    assert result.removed_codepoints == 1


@pytest.mark.parametrize("base", list("\u2325\u24b6\u25a1\u26af\u26b2\u2b20"))
@pytest.mark.parametrize("selector", ["\ufe0e", "\ufe0f"])
def test_unregistered_presentation_selectors_are_removed(base, selector):
    result = remove_text_metadata(f"Glyph {base}{selector} in the sample.")
    assert result.cleaned_text == f"Glyph {base} in the sample."
    assert result.removed_codepoints == 1
