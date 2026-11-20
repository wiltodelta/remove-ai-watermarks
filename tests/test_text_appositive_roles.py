"""Check selected rewrites of an explicit actor and appositive description."""

import pytest

from remove_ai_watermarks.text import verify_text_rewrite_facts


@pytest.mark.parametrize(
    ("source", "faithful", "changed"),
    [
        (
            "The chief human resources officer, a woman with a clipboard and a permanent frown, greeted her at the "
            "ferry dock, where the deaf ferry engine mechanic was already at work, her hands steady on the controls "
            "despite the wind\u2019s insistence.",
            "The chief human resources officer, a woman with a clipboard and a permanent frown, welcomed her at the "
            "ferry dock, where the deaf ferry engine mechanic was already at work, her hands steady on the controls "
            "despite the wind\u2019s insistence.",
            "A woman with a clipboard and a permanent frown welcomed the chief human resources officer at the ferry "
            "dock, where the deaf ferry engine mechanic was already working, with her hands poised on the steering "
            "wheel, despite the wind's insistence.",
        ),
        (
            "The officer, a woman with a clipboard and a permanent frown, handed her a cup of tea and a map "
            "of the town\u2019s tidal patterns.",
            "The officer, a woman with a clipboard and a permanent frown, handed her a map of the town\u2019s tidal "
            "patterns and a cup of tea.",
            "A woman with a permanent frown and a clipboard was given to her by the officer, along with a map "
            "of the town's tide patterns and a cup of tea.",
        ),
        (
            "The admissions director, a man with a red scarf, interviewed the applicant.",
            "The applicant was interviewed by the admissions director, a man with a red scarf.",
            "A man with a red scarf interviewed the admissions director.",
        ),
        (
            "The shift supervisor, a person with a radio, escorted the visitor to the exit.",
            "The visitor was escorted to the exit by the shift supervisor, a person with a radio.",
            "A person with a radio was escorted by the shift supervisor.",
        ),
        (
            "The museum curator, a woman with a silver badge, handed the intern the keys.",
            "The keys were handed to the intern by the museum curator, a woman with a silver badge.",
            "A woman with a silver badge was handed to the intern by the museum curator.",
        ),
        (
            "The race organizer, an athlete with a blue jacket, thanked the volunteer.",
            "The volunteer was thanked by the race organizer, an athlete with a blue jacket.",
            "An athlete with a blue jacket thanked the race organizer.",
        ),
        (
            "The curator, a woman with a silver badge, handed the intern a map.",
            "The intern was given a map by the curator, a woman with a silver badge.",
            "A woman with a silver badge was given a map by the curator.",
        ),
    ],
)
def test_actor_apposition_is_preserved(source: str, faithful: str, changed: str) -> None:
    assert verify_text_rewrite_facts(source, faithful)
    assert not verify_text_rewrite_facts(source, changed)


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        (
            "The curator greeted a woman with a silver badge.",
            "A woman with a silver badge was greeted by the curator.",
        ),
        (
            "The curator, a woman with a silver badge, greeted the intern.",
            "A woman with a silver badge, the curator, greeted the intern.",
        ),
        (
            "The curator, a woman with a silver badge, greeted the intern.",
            "A woman with a gold badge greeted the curator.",
        ),
        (
            "The curator, a woman with a silver badge, greeted the intern.",
            "A woman with a silver badge worked with the curator.",
        ),
        (
            "The curator, a woman with a silver badge, greeted the intern. "
            "A woman with a silver badge greeted the curator.",
            "A woman with a silver badge, the curator, greeted the intern. "
            "A woman with a silver badge greeted the curator.",
        ),
    ],
)
def test_unmatched_descriptions_and_actions_do_not_trigger_guard(source: str, candidate: str) -> None:
    assert verify_text_rewrite_facts(source, candidate)
