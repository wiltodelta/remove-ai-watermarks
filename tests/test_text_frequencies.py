"""Selected explicit English schedules must survive rewriting as complete phrases."""

import pytest

from remove_ai_watermarks import text as lib


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("Volunteers inspect tools every other day.", "Volunteers inspect tools periodically."),
        ("Volunteers inspect tools every other day.", "Volunteers inspect tools every day."),
        ("Volunteers inspect tools daily.", "Volunteers inspect tools occasionally."),
        ("Volunteers inspect tools weekly.", "Volunteers inspect tools monthly."),
        ("Volunteers inspect tools twice a day.", "Volunteers inspect tools once a day."),
        ("Volunteers inspect tools twice weekly.", "Volunteers inspect tools weekly."),
        ("Volunteers inspect tools every two days.", "Volunteers inspect tools every two weeks."),
        ("Volunteers inspect tools.", "Volunteers inspect tools every week."),
        (
            "Inspectors clean tools daily. Volunteers inspect tools weekly. Clerks catalogue tools weekly.",
            "Inspectors clean tools daily. Volunteers inspect tools daily. Clerks catalogue tools weekly.",
        ),
        ("Volunteers inspect tools every other month.", "Volunteers inspect tools every month."),
        ("Volunteers inspect tools three times every two days.", "Volunteers inspect tools twice every two days."),
        ("Two volunteers inspect tools every other day.", "Three volunteers inspect tools every two days."),
    ],
)
def test_changed_or_lost_explicit_frequency_is_rejected(source, candidate):
    assert not lib.verify_text_rewrite_facts(source, candidate)


@pytest.mark.parametrize(
    ("frequency", "alias"),
    [
        ("daily", "every day"),
        ("hourly", "once per hour"),
        ("weekly", "once a week"),
        ("monthly", "each month"),
        ("yearly", "annually"),
        ("annually", "once every year"),
        ("every other day", "on alternate days"),
        ("every other day", "every two days"),
        ("every other week", "every second week"),
        ("every other month", "once every two months"),
        ("twice daily", "two times per day"),
        ("thrice weekly", "three times a week"),
        ("three times every two days", "3 times every 2 days"),
        ("every twenty-one days", "every 21 days"),
    ],
)
def test_selected_frequency_aliases_are_equivalent(frequency, alias):
    assert lib.verify_text_rewrite_facts(f"Volunteers inspect tools {frequency}.", f"Volunteers inspect tools {alias}.")


@pytest.mark.parametrize(
    "frequency", ["every other day", "once every two months", "three times every two days", "daily"]
)
def test_local_rewriters_mask_the_whole_frequency(frequency):
    passage = f"Volunteers inspect tools {frequency}."
    masked, replacements = lib._mask_local_rewrite_facts(passage)
    assert frequency in replacements.values()
    restored = masked
    for token, literal in replacements.items():
        restored = restored.replace(token, literal)
    assert restored == passage


def test_schedule_quantities_do_not_hide_unrelated_numerals():
    assert lib.verify_text_rewrite_facts(
        "Two volunteers inspect tools every other day.", "2 volunteers inspect tools every two days."
    )
    assert not lib.verify_text_rewrite_facts(
        "Two volunteers inspect tools every other day.", "3 volunteers inspect tools every two days."
    )


def test_frequency_substrings_and_service_marker_ids_are_not_schedules():
    assert lib._extract_frequencies("Everyday tools and weeklyish notes. [[RAIW_SPAN_5]]") == ()
