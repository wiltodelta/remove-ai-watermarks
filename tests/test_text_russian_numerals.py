"""Russian cardinal value and bound regressions for guarded rewriting."""

import pytest

from remove_ai_watermarks import text as lib


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("Передали пять книг.", "Передали шесть книг."),
        ("Передали двадцать пять книг.", "Передали двадцать шесть книг."),
        ("Передали двадцать пять книг.", "Передали пятьдесят две книги."),
        ("Передали сто пять книг.", "Передали сто шесть книг."),
        ("Передали две тысячи пять книг.", "Передали пять тысяч две книги."),
        ("Передали пять книг.", "Передали книги."),
        ("Передали книги.", "Передали пять книг."),
        ("Передали не более двадцати пяти книг.", "Передали более двадцати пяти книг."),
        ("Передали не менее пяти книг.", "Передали пять книг."),
        ("Передали около двухсот пятидесяти книг.", "Передали двести пятьдесят книг."),
        ("Пришёл один из десяти участников.", "Пришли десять участников."),
        ("Пришли тысячи участников.", "Пришли 1000 участников."),
    ],
)
def test_changed_russian_cardinal_or_bound_is_rejected(source, candidate):
    assert not lib.verify_text_rewrite_facts(source, candidate)


@pytest.mark.parametrize(
    ("words", "digits"),
    [
        ("ноль", "0"),
        ("одна", "1"),
        ("две", "2"),
        ("пять", "5"),
        ("двенадцать", "12"),
        ("двадцать пять", "25"),
        ("двадцати пяти", "25"),
        ("ста пяти", "105"),
        ("двести пятьдесят", "250"),
        ("двухсот пятидесяти", "250"),
        ("две тысячи пять", "2005"),
        ("два миллиона три тысячи сорок", "2003040"),
        ("тысяча", "1000"),
    ],
)
def test_cardinal_words_and_digits_share_one_value(words, digits):
    assert lib._extract_normalized_numerals(words) == {digits}
    assert lib._extract_normalized_numerals(digits) == {digits}
    assert lib.verify_text_rewrite_facts(f"Передали {words} книг.", f"Передали {digits} книг.")
    assert lib.verify_text_rewrite_facts(f"Передали {digits} книг.", f"Передали {words} книг.")


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("Передали не более двадцати пяти книг.", "Передали максимум 25 книг."),
        ("Передали не менее пяти книг.", "Передали минимум 5 книг."),
        ("Передали около двухсот пятидесяти книг.", "Передали примерно 250 книг."),
        ("Это один из лучших проектов.", "Это проект в числе лучших."),
        ("Это одна из самых удобных программ.", "Это программа в числе самых удобных."),
    ],
)
def test_equivalent_bounds_and_nonquantity_one_are_allowed(source, candidate):
    assert lib.verify_text_rewrite_facts(source, candidate)


def test_masking_freezes_whole_russian_number_and_restores_it():
    source = "Мы передали двадцать пять книг и оставили [[RAIW_SPAN_0]]."
    masked, facts = lib._mask_local_rewrite_facts(source)
    assert list(facts.values()) == ["двадцать пять"]
    assert "двадцать" not in masked
    assert "[[RAIW_SPAN_0]]" in masked
    assert facts.keys().isdisjoint({"[[RAIW_SPAN_0]]"})
    restored = masked
    for marker, literal in facts.items():
        restored = restored.replace(marker, literal)
    assert restored == source


def test_nonquantity_one_does_not_get_a_factual_marker():
    source = "Это один из лучших проектов."
    assert lib._mask_local_rewrite_facts(source) == (source, {})


@pytest.mark.parametrize(
    ("words", "value"),
    [
        ("двумястами пятьюдесятью тремя", "253"),
        ("трёмстам двадцати пяти", "325"),
        ("двухстах двадцати пяти", "225"),
    ],
)
def test_additional_hundred_case_forms(words, value):
    assert lib._extract_normalized_numerals(words) == {value}


def test_ambiguous_magpie_noun_is_not_forty():
    assert lib._extract_normalized_numerals("Сорока сидела на ветке.") == set()
    assert lib.verify_text_rewrite_facts("Сорока сидела на ветке.", "Птица сидела на ветке.")


@pytest.mark.parametrize("words", ["двадцать двадцать", "один миллион два миллиарда"])
def test_invalid_or_noncanonical_sequences_remain_literal(words):
    values = lib._extract_normalized_numerals(words)
    assert len(values) == 1
    assert next(iter(values)).startswith("ru-unparsed:")


def test_default_candidate_guard_rejects_russian_number_change():
    source = "Мы получили двадцать пять книг."
    changed = "Мы взяли двадцать шесть книг."
    assert lib._guard_local_rewrite_candidate(source, source, {}, changed, None) is None


def test_one_inside_compound_quantity_is_not_a_superlative_idiom():
    source = "Выбрали двадцать один из лучших проектов."
    assert lib._extract_normalized_numerals(source) == {"21"}
    assert lib.verify_text_rewrite_facts(source, "Выбрали 21 из лучших проектов.")
    assert not lib.verify_text_rewrite_facts(source, "Выбрали двадцать из лучших проектов.")


def test_default_guard_accepts_preserved_masked_russian_cardinal():
    source = "Мы передали двадцать пять книг."
    masked, facts = lib._mask_local_rewrite_facts(source)
    candidate = masked.replace("передали", "доставили")
    assert (
        lib._guard_local_rewrite_candidate(source, masked, facts, candidate, None) == "Мы доставили двадцать пять книг."
    )


@pytest.mark.parametrize(
    ("words", "value"),
    [
        ("пятилетняя", "5"),
        ("двадцатипятилетнего", "25"),
        ("двухсотпятидесятилетний", "250"),
        ("тысячелетний", "1000"),
        ("двухчасовая", "2"),
    ],
)
def test_duration_adjectives_keep_explicit_cardinal_value(words, value):
    assert lib._extract_normalized_numerals(words) == {value}


def test_duration_adjective_is_frozen_with_its_unit():
    source = "Они завершили пятилетнюю работу."
    masked, facts = lib._mask_local_rewrite_facts(source)
    assert list(facts.values()) == ["пятилетнюю"]
    assert "пятилетнюю" not in masked


def test_duration_bound_is_preserved():
    assert not lib.verify_text_rewrite_facts("Это не более пятилетняя работа.", "Это пятилетняя работа.")


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("Пришли миллионы участников.", "Пришли миллиарды участников."),
        ("Пришли миллиарды участников.", "Пришли участники."),
    ],
)
def test_indefinite_plural_scale_magnitude_is_not_discarded(source, candidate):
    assert not lib.verify_text_rewrite_facts(source, candidate)
