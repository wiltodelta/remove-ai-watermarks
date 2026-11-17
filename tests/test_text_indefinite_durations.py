"""Selected indefinite English durations bind one to its complete unit phrase."""

import pytest

from remove_ai_watermarks import text as lib


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("The loan lasts a day.", "The loan lasts."),
        ("The loan lasts a day.", "The loan lasts two days."),
        ("The loan lasts a day.", "The loan lasts a week."),
        ("The loan lasts.", "The loan lasts a day."),
        ("The loan lasts at least a day.", "The loan lasts a day."),
        ("The loan lasts at most a day.", "The loan lasts at least a day."),
        ("The loan lasts about a day.", "The loan lasts exactly one day."),
        ("The light comes on when a tool is borrowed for a day.", "The light comes on when a tool is borrowed."),
        ("The waits were an hour and one hour.", "The waits were an hour and two hours."),
    ],
)
def test_lost_changed_or_invented_indefinite_duration_is_rejected(source, candidate):
    assert not lib.verify_text_rewrite_facts(source, candidate)


@pytest.mark.parametrize(
    ("article", "unit"), [("an", "hour"), ("a", "day"), ("a", "week"), ("a", "month"), ("a", "year")]
)
def test_indefinite_duration_matches_a_single_explicit_unit(article, unit):
    assert lib.verify_text_rewrite_facts(f"The loan lasts {article} {unit}.", f"The loan lasts one {unit}.")
    assert lib.verify_text_rewrite_facts(f"The loan lasts 1 {unit}.", f"The loan lasts {article} {unit}.")
    passage = f"The loan lasts {article} {unit}."
    masked, replacements = lib._mask_local_rewrite_facts(passage)
    assert f"{article} {unit}" in replacements.values()
    for token, literal in replacements.items():
        masked = masked.replace(token, literal)
    assert masked == passage


@pytest.mark.parametrize("bound", ["at least", "at most", "about"])
def test_indefinite_duration_keeps_its_numeric_bound(bound):
    assert lib.verify_text_rewrite_facts(f"The loan lasts {bound} a day.", f"The loan lasts {bound} one day.")


def test_frequency_intervals_do_not_become_separate_duration_amounts():
    assert lib._extract_measurements("Volunteers inspect tools once a day.") == ()
    assert lib.verify_text_rewrite_facts("Volunteers inspect tools once a day.", "Volunteers inspect tools daily.")


def test_ambiguous_ordinals_adjectives_and_word_substrings_are_not_indefinite_durations():
    assert lib._extract_measurements("They sought a second opinion about a minute detail in a daily report.") == ()
    assert lib._extract_measurements("A daydream led to an hourly estimate.") == ()


def test_indefinite_durations_do_not_hide_independent_counts():
    assert lib.verify_text_rewrite_facts("Two tools were borrowed for a day.", "2 tools were borrowed for one day.")
    assert not lib.verify_text_rewrite_facts(
        "Two tools were borrowed for a day.", "Three tools were borrowed for one day."
    )
