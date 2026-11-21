"""Recover selected glyphs from VSRMark emitted by the existing encoder."""

import pytest

from remove_ai_watermarks.text import (
    decode_stego_payloads,
    encode_unicode_tags,
    encode_vsrmark,
    identify_text,
    remove_text_metadata,
)


@pytest.mark.parametrize("sequence", ["♥\ufe0f", "葛\U000e0100", "4\ufe0f", "≨\ufe00"])
@pytest.mark.parametrize("prefix_length", [0, 5, 11, 12])
def test_vsrmark_removal_preserves_a_registered_variant(sequence: str, prefix_length: int) -> None:
    source = "a" * prefix_length + sequence + "b" * 25
    marked = encode_vsrmark(source, "x")
    tagged = encode_unicode_tags("", "tag") + marked

    assert identify_text(marked).vsrmark_payload == "x"
    assert decode_stego_payloads(marked)["vsrmark"] == "x"
    cleaned = remove_text_metadata(marked)
    assert cleaned.cleaned_text == source
    assert cleaned.removed_codepoints == len(marked) - len(source)
    assert remove_text_metadata(tagged).cleaned_text == source


def test_vsrmark_deinterleaves_several_registered_variants_in_one_frame() -> None:
    source = "♥\ufe0f葛\U000e0100a4\ufe0fb≨\ufe00" + "payload carrier" * 6
    marked = encode_vsrmark(source, "test")

    assert identify_text(marked).vsrmark_payload == "test"
    assert remove_text_metadata(marked).cleaned_text == source


def test_vsrmark_preserves_contextual_unit_spacing() -> None:
    source = "Use 38\u202fmm to check the archive gauge and keep 2\u2060\u00d7\u20603 on the same line."
    marked = encode_vsrmark(source, "shape test")

    report = identify_text(marked)
    assert report.vsrmark_payload == "shape test"
    assert "whitespace_steganography" not in report.signals
    cleaned = remove_text_metadata(marked)
    assert cleaned.cleaned_text == source
    assert cleaned.removed_codepoints == len(marked) - len(source)
