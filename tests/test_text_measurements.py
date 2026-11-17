"""Selected quantity/unit bindings must survive local text rewriting."""

import pytest

from remove_ai_watermarks import text as lib


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("The distance was five kilometers.", "The distance was five miles."),
        ("The mass was 5 mg.", "The mass was 5 g."),
        ("The sample weighed 5mg.", "The sample weighed 5g."),
        ("The power was 5 mW.", "The power was 5 MW."),
        ("The volume was 5 mL.", "The volume was 5 ML."),
        ("The temperature was 5 °C.", "The temperature was 5 °F."),
        ("The length was 5 m².", "The length was 5 m."),
        ("The speed was 5 km/h.", "The speed was 5 km/s."),
        ("The density was 5 mg/L.", "The density was 5 mg/mL."),
        ("The density was 5 mg/kg/day.", "The density was 5 mg/kg/hour."),
        ("The distance was five kilometers.", "The distance was five."),
        ("The distance was five.", "The distance was five kilometers."),
        ("The distances were 5 km and 10 miles.", "The distances were 10 km and 5 miles."),
        ("The distances were 5 km and 5 miles.", "The distances were 5 km and 5 km."),
        ("The distance was 5 km.", "The distances were 5 km and 5 km."),
        ("The distance was at least 5 km and at most 5 miles.", "The distance was at most 5 km and at least 5 miles."),
        (
            "\u041c\u0430\u0441\u0441\u0430 составила пять килограммов.",
            "\u041c\u0430\u0441\u0441\u0430 составила пять граммов.",
        ),
        ("Расстояние составило двадцать пять километров.", "Расстояние составило двадцать пять метров."),
        ("The mass was -5 mg.", "The mass was 5 mg."),
    ],
)
def test_changed_or_dropped_measurement_bindings_are_rejected(source, candidate):
    assert not lib.verify_text_rewrite_facts(source, candidate)


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("The distance was 5 km.", "The distance was five kilometers."),
        ("The distance was five kilometres.", "The distance was five kilometers."),
        ("The length was 5m.", "The length was 5 meters."),
        ("The sample weighed 5mg.", "The sample weighed five milligrams."),
        ("The mass was -5mg.", "The mass was minus five milligrams."),
        ("The power was 5 MW.", "The power was five megawatts."),
        ("The volume was 5 mL.", "The volume was 5 milliliters."),
        ("The temperature was 5 degrees Celsius.", "The temperature was 5 °C."),
        ("The speed was 5 km / h.", "The speed was five kilometers/hour."),
        ("The mass was 5 μg.", "The mass was five micrograms."),
        ("The mass was 5 µg.", "The mass was five micrograms."),
        ("The length was 5 m².", "The length was 5 m^2."),
        ("The distance was at least 5km.", "The distance was no fewer than five kilometers."),
        ("The fraction was at least 5%.", "The fraction was no fewer than five percent."),
        (
            "\u041c\u0430\u0441\u0441\u0430 составила пять килограммов.",
            "\u041c\u0430\u0441\u0441\u0430 составила 5 кг.",
        ),
        ("Расстояние составило двадцать пять километров.", "Расстояние составило 25 км."),
        ("The length was 5 km and 10 m.", "The length was 10 meters and 5 kilometers."),
    ],
)
def test_selected_measurement_aliases_are_equivalent(source, candidate):
    assert lib.verify_text_rewrite_facts(source, candidate)
    assert lib.verify_text_rewrite_facts(candidate, source)


@pytest.mark.parametrize(
    "literal",
    ["five kilometers", "5mg", "twenty-five miles", "пять килограммов", "5 mW", "5 MW", "5 km/h", "5 m^2"],
)
def test_whole_measurement_is_masked_as_one_fact(literal):
    masked, facts = lib._mask_local_rewrite_facts(f"The measured amount was {literal}.")
    assert list(facts.values()) == [literal]
    assert masked == "The measured amount was [[RAIW_SPAN_0]]."


def test_unknown_units_and_longer_identifiers_are_not_claimed_as_measurements():
    assert lib._extract_measurements("The values were 5 kgood, 5mgood and 5 widgets.") == ()
    assert lib._extract_measurements("The value was 5 km / fortnight.") == ()
    assert lib._extract_measurements("version5mg") == ()


def test_unit_symbols_remain_case_sensitive():
    assert lib._extract_measurements("5 mW") != lib._extract_measurements("5 MW")
    assert lib._extract_measurements("5 mL") != lib._extract_measurements("5 ML")
    assert lib._extract_measurements("5 KM") == ()


def test_service_placeholders_do_not_invent_quantities():
    assert lib._extract_measurements("[[RAIW_SPAN_5]] mg") == ()


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("Началась двухлетняя работа.", "Началась двухдневная работа."),
        ("Началась трёхчасовая работа.", "Началась трёхнедельная работа."),
        ("The task took five hours.", "The task took five days."),
    ],
)
def test_selected_duration_units_cannot_change(source, candidate):
    assert not lib.verify_text_rewrite_facts(source, candidate)


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("Пять лет работы завершились.", "Завершилась пятилетняя работа."),
        ("Работа заняла два дня.", "Завершилась двухдневная работа."),
        ("План требует не менее двух дней.", "План требует как минимум двухдневной работы."),
    ],
)
def test_selected_duration_adjectives_keep_their_units(source, candidate):
    assert lib.verify_text_rewrite_facts(source, candidate)


def test_preposition_in_is_not_claimed_as_inches():
    assert lib._extract_measurements("There were five in total.") == ()
    assert lib.verify_text_rewrite_facts("There were five in total.", "There were five altogether.")


def test_capitalized_unit_names_do_not_become_protected_person_names():
    assert lib.verify_text_rewrite_facts("The temperature was 5 Degrees Celsius.", "The temperature was 5 °C.")
