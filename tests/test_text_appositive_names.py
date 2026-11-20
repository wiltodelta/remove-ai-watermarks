"""Sentence-initial names with explicit appositive descriptions stay whole."""

from __future__ import annotations

import pytest

from remove_ai_watermarks import text as lib


@pytest.mark.parametrize(
    "name",
    ["Robert Johnson", "Alice Parker", "Jean-Luc Martin", "Jane O'Neill", "GreenCycle Delivery Services"],
)
def test_initial_appositive_name_is_one_fact(name: str) -> None:
    source = f"{name}, the project leader, presented the results."
    masked, facts = lib._mask_local_rewrite_facts(source)
    assert list(facts.values()) == [name]
    assert masked == "[[RAIW_SPAN_0]], the project leader, presented the results."


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("Robert Johnson, the curator, led the project.", "Michael Johnson, the curator, led the project."),
        ("Robert Johnson, the curator, led the project.", "Johnson, the curator, led the project."),
        ("Robert Johnson, the curator, led the project.", "Robert, the curator, led the project."),
        ("Alice Parker, a coordinator, led the project.", "Mary Parker, a coordinator, led the project."),
        ("Robert Johnson, the curator, led the project.", "Robert led the Johnson project as curator."),
        ("Robert Johnson, the curator, led the project.", "Robert Johnsonville, the curator, led the project."),
    ],
)
def test_full_initial_name_cannot_change_or_detach(source: str, candidate: str) -> None:
    assert not lib.verify_text_rewrite_facts(source, candidate)


@pytest.mark.parametrize(
    "candidate",
    [
        "The project was led by Robert Johnson, the curator.",
        "The curator Robert Johnson led the project.",
        "ROBERT\tJOHNSON, the curator, headed the project.",
    ],
)
def test_same_name_can_move_and_change_case_or_spacing(candidate: str) -> None:
    assert lib.verify_text_rewrite_facts("Robert Johnson, the curator, led the project.", candidate)


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("During April, the board met.", "In April, the board met."),
        ("In Boston, the board met.", "The board met in Boston."),
        ("Early April, the board met.", "The board met in April, early in the month."),
        ("The Johnson Foundation, a charity, opened.", "Johnson Foundation, a charity, opened."),
        ("After Robert Johnson, the next speaker presented.", "The next speaker presented after Robert Johnson."),
    ],
)
def test_sentence_leading_function_words_are_not_added_to_names(source: str, candidate: str) -> None:
    assert lib.verify_text_rewrite_facts(source, candidate)


def test_sentence_and_paragraph_appositives_are_recognized() -> None:
    source = "The report arrived. Alice Parker, a coordinator, signed it.\n\nRobert Johnson, the curator, checked it."
    _masked, facts = lib._mask_local_rewrite_facts(source)
    assert list(facts.values()) == ["Alice Parker", "Robert Johnson"]


def test_unmarked_single_initial_word_remains_outside_this_extension() -> None:
    _masked, facts = lib._mask_local_rewrite_facts("Robert, the curator, checked the report.")
    assert not facts


def test_appositive_name_guard_is_used_by_native_candidate_acceptance() -> None:
    source = "Robert Johnson, the curator, led the project."
    masked = "Robert [[RAIW_SPAN_0]], the curator, led the project."
    facts = {"[[RAIW_SPAN_0]]": "Johnson"}
    candidate = "Michael [[RAIW_SPAN_0]], the curator, headed the project."
    assert lib._guard_local_rewrite_candidate(source, masked, facts, candidate, None) is None


@pytest.mark.parametrize("ending", ["\n", "\r", "\r\n"])
def test_initial_appositive_after_a_line_boundary(ending: str) -> None:
    source = "Report." + ending + "Robert Johnson, the curator, checked it."
    masked, facts = lib._mask_local_rewrite_facts(source)
    assert "Robert Johnson" in facts.values()
    assert "Robert " not in masked
