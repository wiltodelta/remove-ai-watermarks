"""Valid emoji keycaps retain their presentation selector during metadata cleanup."""

import pytest

from remove_ai_watermarks.text import remove_text_metadata


@pytest.mark.parametrize("base", list("#*0123456789"))
def test_preserves_emoji_keycap_and_removes_unrelated_invisible_marker(base: str) -> None:
    keycap = f"{base}\ufe0f\u20e3"
    result = remove_text_metadata(f"Press {keycap} to con\u200btinue.")
    assert result.cleaned_text == f"Press {keycap} to continue."
    assert result.removed_codepoints == 1


def test_unregistered_presentation_selector_is_removed_before_enclosure() -> None:
    assert remove_text_metadata("A\ufe0f\u20e3 next").cleaned_text == "A\u20e3 next"
