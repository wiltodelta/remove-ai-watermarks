"""Preserve explicit seasonal timing without guessing ambiguous spring/fall senses."""

import pytest

from remove_ai_watermarks.text import verify_text_rewrite_facts


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("Families visited in autumn.", "Families visited throughout the year."),
        ("Families visited in autumn.", "Families visited from spring to fall."),
        ("The garden stayed open throughout the season.", "The garden stayed open from spring to fall."),
        ("The workshop runs in winter.", "The workshop runs in summer."),
        ("Visitors arrived in summer.", "Visitors arrived in winter."),
        ("Families visit in springtime.", "Families visit year-round."),
        ("The workshop stays open.", "The workshop stays open in winter."),
        ("Семьи приезжали осенью.", "Семьи приезжали круглый год."),
        ("Посетители приезжают зимой.", "Посетители приезжают летом."),
        ("Посетители приезжают весной.", "Посетители приезжают осенью."),
    ],
)
def test_lost_added_or_changed_explicit_seasons_are_refused(source, candidate):
    assert not verify_text_rewrite_facts(source, candidate)


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("Families visited in autumn.", "Families visited in the fall."),
        ("Families visited in the fall.", "Families visited in autumn."),
        ("The workshop runs in winter.", "The workshop operates during wintertime."),
        ("The workshop operates during wintertime.", "The workshop runs in winter."),
        ("Families visit in springtime.", "Families visit in spring."),
        ("Families visit in spring.", "Families visit in springtime."),
        ("Семьи приезжают осенью.", "Осенью приезжают семьи."),
        ("Посетители приезжают летом.", "Летом приезжают посетители."),
        ("Spring water feeds the pond.", "Water from a natural spring feeds the pond."),
        ("He was hurt in the fall.", "He was injured in that accident."),
        ("The ball may fall.", "The ball may drop."),
        ("The cat can spring forward.", "The cat can leap forward."),
        ("Winters are cold here.", "Winter is cold here."),
        ("Summers are warm here.", "Summer is warm here."),
        ("Autumns are wet here.", "Autumn is wet here."),
        ("Пять лет работы завершились.", "Завершилась пятилетняя работа."),
    ],
)
def test_seasonal_aliases_and_nonseasonal_words_remain_usable(source, candidate):
    assert verify_text_rewrite_facts(source, candidate)
