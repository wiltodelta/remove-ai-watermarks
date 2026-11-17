"""An honorific period must not split the name protected by local rewriting."""

import pytest

from remove_ai_watermarks import text as lib


@pytest.mark.parametrize(
    ("source", "name"),
    [
        ("The librarian called Mr. Smith.", "Mr. Smith"),
        ("The librarian called Mr.\tSmith.", "Mr.\tSmith"),
        ("Dr. Garcia inspected the kit.", "Dr. Garcia"),
        ("The librarian called Mrs. O'Neill.", "Mrs. O'Neill"),
        ("The librarian called Mrs. O\u2019Neill.", "Mrs. O\u2019Neill"),
        ("The librarian called Ms. Иванова.", "Ms. Иванова"),
        ("The librarian called Dr. Élise Martin.", "Dr. Élise Martin"),
        ("The letter went to Prof. J. R. Tolkien.", "Prof. J. R. Tolkien"),
        ("The letter went to Prof. J.R.R. Tolkien.", "Prof. J.R.R. Tolkien"),
        ("The librarian called MR. SMITH.", "MR. SMITH"),
        ("The letter is Dr. Smith's.", "Dr. Smith"),
        ("The letter is Dr. Smith\u2019s.", "Dr. Smith"),
    ],
)
def test_titled_name_is_one_complete_fact_span(source, name):
    masked, facts = lib._mask_local_rewrite_facts(source)
    assert list(facts.values()) == [name]
    assert name not in masked
    for marker, literal in facts.items():
        masked = masked.replace(marker, literal)
    assert masked == source


def test_each_occurrence_of_a_titled_name_has_its_own_complete_marker():
    source = "The librarian called Mr. Smith, then Mr. Smith checked the kit."
    masked, facts = lib._mask_local_rewrite_facts(source)
    assert list(facts.values()) == ["Mr. Smith", "Mr. Smith"]
    assert all(masked.count(marker) == 1 for marker in facts)


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("Mr. Smith checked the kit.", "Mr. Jones checked the kit."),
        ("Dr. Smith checked the kit.", "Dr. Smithsonian checked the kit."),
        ("Dr. Smith checked the kit.", "Mr. Smith checked the kit."),
        ("The librarian called Mr. Smith.", "The librarian called Mr, Smith."),
        ("The librarian called Mr. Smith.", "The librarian called Mr, who asked Smith."),
        ("The letter went to Prof. J. R. Tolkien.", "The letter went to Prof. J. R. Jones."),
    ],
)
def test_changed_or_detached_titled_name_is_rejected(source, candidate):
    assert not lib.verify_text_rewrite_facts(source, candidate)


def test_titled_name_allows_case_and_whitespace_changes():
    assert lib.verify_text_rewrite_facts("Dr. Élise Martin checked the kit.", "DR.\tÉLISE MARTIN inspected the kit.")


def test_fact_check_does_not_require_every_repeated_title_in_a_pronoun_paraphrase():
    assert lib.verify_text_rewrite_facts(
        "Mr. Smith collected the kit, and Mr. Smith returned it later.",
        "Mr. Smith collected the kit and returned it later.",
    )
