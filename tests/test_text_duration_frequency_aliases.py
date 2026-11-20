"""Regressions for observed duration compounds and explicit recurrence aliases."""

from __future__ import annotations

import pytest

from remove_ai_watermarks import text as lib


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("The trial lasted a four-week period.", "The trial lasted four weeks."),
        ("The trial lasted a 4-week period.", "The trial lasted four weeks."),
        ("The trial lasted a twenty-one-day period.", "The trial lasted twenty-one days."),
        ("The trial lasted a two-hour period.", "The trial lasted two hours."),
        ("The trial lasted a five-month period.", "The trial lasted five months."),
        ("The trial lasted a three-year period.", "The trial lasted three years."),
        ("We require at least four-week coverage.", "We require at least four weeks of coverage."),
        ("The garden held its annual meeting.", "The garden held its yearly meeting."),
        ("The garden held its annual meeting.", "The garden held its meeting once every year."),
        ("Staffing costs $500 per week.", "Staffing costs $500 weekly."),
        ("Staffing costs $500 per week.", "Staffing costs $500 each week."),
        ("Staffing costs $500 per month.", "Staffing costs $500 monthly."),
    ],
)
def test_observed_duration_and_frequency_aliases_pass(source: str, candidate: str) -> None:
    assert lib.verify_text_rewrite_facts(source, candidate)
    assert lib.verify_text_rewrite_facts(candidate, source)


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("The trial lasted a four-week period.", "The trial lasted a four-day period."),
        ("The trial lasted a four-week period.", "The trial lasted a five-week period."),
        ("The trial lasted a four-week period.", "The trial lasted a period."),
        ("The trial lasted a period.", "The trial lasted a four-week period."),
        ("The trial lasted at least four weeks.", "The trial lasted at most a four-week period."),
        ("The garden held its annual meeting.", "The garden held its monthly meeting."),
        ("The garden held its annual meeting.", "The garden held its meeting."),
        ("The garden held its meeting.", "The garden held its annual meeting."),
        ("The garden held its annual meeting.", "The garden held its semi-annual meeting."),
        ("The garden held its annual meeting.", "The garden held its biannual meeting."),
        ("The garden held its annual meeting.", "The garden held its biennial meeting."),
        ("The garden held its annual meeting.", "The garden held its non-annual meeting."),
        ("Staffing costs $500 per week.", "Staffing costs $500 per month."),
        ("Staffing costs $500 per week.", "Staffing costs $500."),
        ("Staffing costs $500.", "Staffing costs $500 per week."),
        ("Staffing costs $500 per week.", "Staffing costs $600 weekly."),
        ("Staffing costs $500 per week.", "Staffing costs $500 twice weekly."),
        ("Staffing costs $500 per week.", "Staffing costs $500 every other week."),
        ("Staffing costs $500 per week.", "Staffing costs $500 per two weeks."),
        (
            "Staffing costs $500 weekly and rent costs $500 monthly.",
            "Staffing costs $500 weekly and rent costs $500 weekly.",
        ),
    ],
)
def test_changed_duration_frequency_or_rate_is_rejected(source: str, candidate: str) -> None:
    assert not lib.verify_text_rewrite_facts(source, candidate)


@pytest.mark.parametrize("literal", ["four-week", "4-week", "twenty-one-day", "two-hour", "annual", "per week"])
def test_new_supported_phrases_are_masked_whole(literal: str) -> None:
    passage = f"The note mentions {literal}."
    masked, facts = lib._mask_local_rewrite_facts(passage)
    assert list(facts.values()) == [literal]
    assert masked == "The note mentions [[RAIW_SPAN_0]]."


@pytest.mark.parametrize("literal", ["semi-annual", "bi-annual", "non-annual", "annualized", "biannual", "biennial"])
def test_annual_alias_does_not_match_inside_other_frequency_words(literal: str) -> None:
    assert lib._extract_frequencies(f"The note mentions {literal}.") == ()


def test_duration_compounds_preserve_multiplicity_and_units() -> None:
    assert lib._extract_measurements("A four-week trial and two-hour pickup window.") == (("2", "hour"), ("4", "week"))
    assert lib._extract_numeral_occurrences("A four-week trial and two-hour pickup window.") == ("2", "4")
    assert not lib.verify_text_rewrite_facts(
        "A four-week trial and a four-day trial.", "A four-week trial and a four-week trial."
    )


def test_duration_separator_does_not_claim_general_subtraction_as_measurement() -> None:
    assert lib._extract_measurements("The expressions are 5-m and 4-kg.") == ()
