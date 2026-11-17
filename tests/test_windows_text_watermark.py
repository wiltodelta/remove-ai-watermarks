"""Published Windows AI Unicode format fixtures, not captured Windows model outputs."""

from __future__ import annotations

import pytest

from remove_ai_watermarks.text import identify_text, remove_text_metadata

# Microsoft Learn: /windows/ai/apis/text-watermarking, observed 2026-11-14.
MARK = "\u200b\u034f\u2060\u200b"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Hello" + MARK + " world.", "Hello world."),
        (MARK + "Hello" + MARK, "Hello"),
        ('{"count":' + MARK + "12}", '{"count":12}'),
        ("`variable" + MARK + "name`", "`variablename`"),
        ("https://example.com/" + MARK + "path", "https://example.com/path"),
        ("👩" + MARK + "\u200d💻", "👩\u200d💻"),
        ("ب" + MARK + "\u200dت", "ب\u200dت"),
        ("אב" + MARK + "גד", "אבגד"),
    ],
)
def test_windows_format_removed_before_context_sensitive_unicode_cleanup(source, expected):
    before = identify_text(source)
    assert "windows_ai_text_watermark" in before.signals
    assert before.platform is None  # A forgeable sequence does not establish attribution.
    result = remove_text_metadata(source)
    assert result.cleaned_text == expected
    assert result.removed_codepoints == source.count(MARK) * len(MARK)
    assert "windows_ai_text_watermark" in result.removed_signals
    assert not result.remaining_markers


def test_windows_format_detection_does_not_strip_literal_escape_sequences():
    source = r"The published marker is \u200b\u034f\u2060\u200b."
    assert "windows_ai_text_watermark" not in identify_text(source).signals
    assert remove_text_metadata(source).cleaned_text == source
