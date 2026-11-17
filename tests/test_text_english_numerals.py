"""Whole English cardinal and signed numeric-literal fidelity regressions."""

import pytest

from remove_ai_watermarks import text as lib


@pytest.mark.parametrize(
    ("words", "value"),
    [
        ("twenty-five", "25"),
        ("twenty five", "25"),
        ("one hundred and five", "105"),
        ("a hundred and two", "102"),
        ("three hundred twenty-six", "326"),
        ("two thousand and five", "2005"),
        ("two million three hundred thousand and five", "2300005"),
        ("minus twenty-five", "-25"),
        ("plus twenty-five", "25"),
        ("one billion two million three thousand and four", "1002003004"),
    ],
)
def test_english_cardinals_share_whole_digit_values(words, value):
    assert lib._extract_normalized_numerals(words) == {value}
    assert lib.verify_text_rewrite_facts(f"They recorded {words} units.", f"They recorded {value} units.")
    assert lib.verify_text_rewrite_facts(f"They recorded {value} units.", f"They recorded {words} units.")


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("They recorded twenty-five units.", "They recorded five units."),
        ("They recorded twenty five units.", "They recorded five twenty units."),
        ("They recorded two thousand five units.", "They recorded five thousand two units."),
        ("They recorded -5 units.", "They recorded 5 units."),
        ("They recorded \u22125 units.", "They recorded 5 units."),
        ("They recorded .5 units.", "They recorded 5 units."),
        ("They recorded minus five units.", "They recorded five units."),
        ("Передали минус пять единиц.", "Передали пять единиц."),
        ("They recorded 1,5 units.", "They recorded 15 units."),
        ("The address was 127.0.0.1.", "The address was 127.1.0.0."),
        ("They recorded 1e3 units.", "They recorded 2e3 units."),
        ("They recorded at least twenty-five units.", "They recorded at least twenty units."),
        ("They recorded at most minus five units.", "They recorded at most five units."),
    ],
)
def test_quantity_changes_cannot_hide_inside_word_parts_or_signs(source, candidate):
    assert not lib.verify_text_rewrite_facts(source, candidate)


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("At least twenty-five units arrived.", "No fewer than 25 units arrived."),
        ("At most one hundred and five units arrived.", "No more than 105 units arrived."),
        ("At least minus five units were recorded.", "At least -5 units were recorded."),
        ("\u041d\u0435 менее минус пяти единиц записали.", "\u041d\u0435 менее -5 единиц записали."),
        ("The selected groups were one and two.", "The selected groups were 1 and 2."),
        ("They selected twenty-one of the best entries.", "They selected 21 of the best entries."),
        ("They selected five and one of the best entries.", "They selected 5 and one of the best entries."),
        ("The groups had two hundred and two hundred units.", "The groups had 200 and 200 units."),
    ],
)
def test_bounds_lists_and_superlatives_remain_usable(source, candidate):
    assert lib.verify_text_rewrite_facts(source, candidate)


@pytest.mark.parametrize(
    ("literal", "value"),
    [
        ("+5", "5"),
        ("\u22125", "-5"),
        ("-.5", "-0.5"),
        (".50", "0.5"),
        ("-1,250.50", "-1250.5"),
        ("1e3", "1000"),
        ("1E-3", "0.001"),
        ("-1e-3", "-0.001"),
    ],
)
def test_signed_decimal_and_exponent_values_are_exact(literal, value):
    assert lib._extract_normalized_numerals(literal) == {value}


def test_decimal_precision_is_not_rounded_to_float_or_context_precision():
    first = "1." + "0" * 80 + "1"
    second = "1." + "0" * 80 + "2"
    assert lib._extract_normalized_numerals(first) != lib._extract_normalized_numerals(second)


def test_ordinary_unsigned_hyphenated_range_is_not_a_negative_endpoint():
    assert lib._extract_normalized_numerals("5-6") == {"5", "6"}


def test_english_number_span_is_frozen_whole_and_restored():
    source = "They received twenty-five books near [[RAIW_SPAN_0]]."
    masked, facts = lib._mask_local_rewrite_facts(source)
    assert list(facts.values()) == ["twenty-five"]
    assert "twenty" not in masked
    candidate = masked.replace("received", "collected")
    assert lib._guard_local_rewrite_candidate(source, masked, facts, candidate, None) == source.replace(
        "received", "collected"
    )


def test_english_superlative_one_is_not_frozen_as_a_quantity():
    source = "It is one of the best tools."
    assert lib._mask_local_rewrite_facts(source) == (source, {})


def test_malformed_number_sequence_is_not_summed_into_a_valid_number():
    assert not lib.verify_text_rewrite_facts("They had two two units.", "They had four units.")


@pytest.mark.parametrize("source", ["They visited One Hundred Years Cafe.", "Мы читали Три Сестры."])
def test_recognized_name_run_takes_priority_over_its_number_words(source):
    _, facts = lib._mask_local_rewrite_facts(source)
    assert list(facts.values()) == ["One Hundred Years Cafe" if source.startswith("They") else "Три Сестры"]


def test_large_exponent_is_canonical_without_expanding_thousands_of_zeroes():
    left = lib._extract_normalized_numerals("1e5000")
    right = lib._extract_normalized_numerals("10e4999")
    assert left == right
    assert len(next(iter(left))) < 100


def test_number_lists_keep_internal_hundred_conjunctions():
    assert lib._extract_normalized_numerals("two hundred and five and six") == {"205", "6"}
    assert lib._extract_normalized_numerals("two and a hundred") == {"2", "100"}


def test_malformed_repeated_word_is_not_silently_discarded():
    assert not lib.verify_text_rewrite_facts("They had two two units.", "They had two units.")


def test_invalid_literal_does_not_change_callers_decimal_flags():
    from decimal import InvalidOperation, localcontext

    with localcontext() as context:
        context.clear_flags()
        lib._normalize_numeral_token("127.0.0.1")
        assert not context.flags[InvalidOperation]


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("The address was 127.0.0.1.", "The address was 0.1.127.0."),
        (
            "At least twenty-five labels and twenty boxes arrived.",
            "At least twenty labels and twenty-five boxes arrived.",
        ),
        (
            "At least minus five degrees and plus five units were recorded.",
            "At least plus five degrees and minus five units were recorded.",
        ),
    ],
)
def test_whole_literals_and_bound_value_binding_prevent_set_collisions(source, candidate):
    assert not lib.verify_text_rewrite_facts(source, candidate)


@pytest.mark.parametrize(("words", "value"), [("минус две тысячи", "-2000"), ("плюс две тысячи", "2000")])
def test_signed_russian_scale_uses_the_same_numeric_value(words, value):
    assert lib._extract_normalized_numerals(words) == {value}


def test_malformed_english_scalar_is_retained_as_unparsed():
    assert lib._normalize_english_integer("two two").startswith("en-unparsed:")


def test_superlative_one_does_not_create_a_factual_count():
    assert lib.verify_text_rewrite_facts("It is one of the best tools.", "It is among the best tools.")


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("At least 1e3 units arrived.", "At least 1000 units arrived."),
        ("At most .5 units were measured.", "At most 0.5 units were measured."),
        ("At least \u22125 units were measured.", "At least -5 units were measured."),
    ],
)
def test_numeric_literal_aliases_share_the_same_bound_value(source, candidate):
    assert lib.verify_text_rewrite_facts(source, candidate)


def test_name_words_elsewhere_do_not_split_a_larger_cardinal_span():
    source = "They visited One Hundred Years Cafe and recorded one hundred and five visits."
    _, facts = lib._mask_local_rewrite_facts(source)
    assert list(facts.values()) == ["One Hundred Years Cafe", "one hundred and five"]


def test_overlap_rescan_keeps_the_unconsumed_name_suffix_protected():
    source = "They used twenty Five Cafe entries."
    masked, facts = lib._mask_local_rewrite_facts(source)
    assert "Cafe" in facts.values()
    restored = masked
    for marker, literal in facts.items():
        restored = restored.replace(marker, literal)
    assert restored == source


@pytest.mark.parametrize(("words", "value"), [("zero thousand", "0"), ("zero million and five", "5")])
def test_explicit_zero_scale_multiplier_is_not_an_implicit_one(words, value):
    assert lib._extract_normalized_numerals(words) == {value}


def test_zero_scale_cannot_be_changed_to_one_scale():
    assert not lib.verify_text_rewrite_facts("They recorded zero thousand units.", "They recorded one thousand units.")
