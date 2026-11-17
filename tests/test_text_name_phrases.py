"""Recognized adjacent name words must not be detached during rewriting."""

import pytest

from remove_ai_watermarks import text as lib


@pytest.mark.parametrize(
    ("source", "name"),
    [
        ("A library opened in Willow Creek.", "Willow Creek"),
        ("Жители посетили Нижний Новгород.", "Нижний Новгород"),
    ],
)
def test_recognized_name_phrase_uses_one_fact_marker(source, name):
    masked, replacements = lib._mask_local_rewrite_facts(source)
    assert list(replacements.values()) == [name]
    assert name not in masked
    for marker, literal in replacements.items():
        masked = masked.replace(marker, literal)
    assert masked == source


def test_fact_guard_rejects_detached_parts_of_a_place_name():
    assert not lib.verify_text_rewrite_facts("A village sits in Willow Creek.", "A village named Creek sits in Willow.")


def test_name_phrase_can_change_case_and_whitespace():
    assert lib.verify_text_rewrite_facts("A library serves New York.", "NEW\tYORK is served by a library.")


def test_different_people_separated_by_a_conjunction_keep_separate_markers():
    _masked, replacements = lib._mask_local_rewrite_facts("The volunteers invited John and Mary.")
    assert list(replacements.values()) == ["John", "Mary"]


def test_name_phrase_does_not_match_a_different_longer_word():
    assert not lib.verify_text_rewrite_facts("A library serves New York.", "A library serves New Yorkshire.")
