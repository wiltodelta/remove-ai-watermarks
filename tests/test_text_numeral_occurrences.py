"""Bare numerical facts must preserve occurrence counts as well as values."""

import pytest

from remove_ai_watermarks import text as lib


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("Waiting times at two clinics fell.", "Two clinics two had waiting times fall."),
        (
            "Two boxes are stored at two depots. Three inspectors work at one depot.",
            "Two boxes are stored at three depots. Three inspectors work at one depot.",
        ),
        (
            "There are two red boxes and three blue boxes.",
            "There are two red boxes, two blue boxes and three other boxes.",
        ),
        (
            "Forty-two crates are on forty-two pallets. Seventeen pallets are empty.",
            "Forty-two crates are on seventeen pallets. Seventeen pallets are empty.",
        ),
        (
            "Две коробки стоят \u0443 двух дверей. Три коробки стоят \u0443 одной двери.",
            "Две коробки стоят \u0443 трёх дверей. Три коробки стоят \u0443 одной двери.",
        ),
    ],
)
def test_changed_numeral_multiplicity_is_rejected(source, candidate):
    assert not lib.verify_text_rewrite_facts(source, candidate)


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("Two boxes are stored at two depots.", "2 boxes are stored at 2 depots."),
        ("Forty-two crates are on forty-two pallets.", "42 crates are on 42 pallets."),
        ("Две коробки стоят \u0443 двух дверей.", "2 коробки стоят \u0443 2 дверей."),
        ("The readings were +5, +5 and 1e2.", "The readings were five, five and one hundred."),
        ("The readings were -5 and -5.", "The readings were minus five and minus five."),
        ("The lengths were 5 m and 5 m.", "The lengths were five meters and five meters."),
        ("The loans last a day and an hour.", "The loans last one day and one hour."),
        ("Two volunteers inspect tools every other day.", "2 volunteers inspect tools once every two days."),
        ("One of the most popular books has two editions.", "A book among the most popular has 2 editions."),
    ],
)
def test_aliases_keep_the_same_numerical_occurrences(source, candidate):
    assert lib.verify_text_rewrite_facts(source, candidate)


def test_legacy_value_set_still_deduplicates_for_callers():
    assert lib._extract_normalized_numerals("Two boxes are stored at two depots.") == {"2"}
    assert lib._extract_numeral_occurrences("Two boxes are stored at two depots.") == ("2", "2")


def test_service_marker_indices_and_frequency_counts_are_not_bare_quantities():
    assert lib._extract_numeral_occurrences("[[RAIW_SPAN_5]] twice every two days") == ()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("5 mg and 5 mg", ("5", "5")),
        ("a day and an hour", ("1", "1")),
        ("двухдневный и двухдневный", ("2", "2")),
        ("twenty-one and twenty-one", ("21", "21")),
    ],
)
def test_each_complete_quantity_contributes_exactly_one_occurrence(text, expected):
    assert lib._extract_numeral_occurrences(text) == expected


def test_native_retry_skips_a_duplicate_number_even_when_all_markers_survive(monkeypatch):
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace

    candidates = iter(
        [
            "Visitors reached two clinics [[RAIW_SPAN_0]].",
            "Visitors arrived at [[RAIW_SPAN_0]] clinics.",
        ]
    )
    seeds = []

    class Encoding(dict):
        def to(self, _device):
            return self

    class Tokenizer:
        eos_token_id = 0
        chat_template = "native"

        def apply_chat_template(self, *_args, **_kwargs):
            return "prompt"

        def __call__(self, *_args, **_kwargs):
            return Encoding(input_ids=SimpleNamespace(shape=(1, 3)))

        def decode(self, *_args, **_kwargs):
            return next(candidates)

    model = SimpleNamespace(generate=lambda **_: [[1, 2, 3, 4, 0]])
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(manual_seed=seeds.append, no_grad=nullcontext))
    monkeypatch.setattr(lib, "_load_local_rewriter_runtime", lambda *_: (Tokenizer(), model, "cpu"))
    assert lib._build_local_lm_rewriter()("Visitors reached two clinics.") == "Visitors arrived at two clinics."
    assert seeds == [0, 1]
