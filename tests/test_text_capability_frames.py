"""Matching technical capability frames need not imply explicit permission."""

import pytest

from remove_ai_watermarks.text import verify_text_rewrite_facts


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        (
            "The new system allowed library staff to record additional information about visitors.",
            "The new system enabled library staff to document extra details about visitors.",
        ),
        (
            "The new system enabled library staff to document extra details about visitors.",
            "The new system allowed library staff to record additional information about visitors.",
        ),
        ("The software allows staff to store data.", "The software enables staff to save data."),
        ("The device allowed technicians to measure pressure.", "The device enabled technicians to measure pressure."),
        ("A calculator allows students to calculate totals.", "A calculator enables students to compute totals."),
        ("The application allows users to log events.", "The application enables users to record events."),
        (
            "The system allowed staff to record data. The owner grants permission for the work.",
            "The system enabled staff to record data. The owner grants permission for the work.",
        ),
    ],
)
def test_matching_technical_capability_is_not_a_permission_loss(source: str, candidate: str) -> None:
    assert verify_text_rewrite_facts(source, candidate)


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("The manager allowed staff to record data.", "The manager enabled staff to record data."),
        ("The policy allowed staff to record data.", "The policy enabled staff to record data."),
        ("The system allowed staff to access records.", "The system enabled staff to access records."),
        ("The system allowed staff to record data.", "The system enabled visitors to record data."),
        ("The system allowed staff to record data.", "The system enabled staff to delete data."),
        ("The system allowed staff to store reports.", "The system enabled staff to display reports."),
        (
            "The system allowed staff to record data about visitors.",
            "The system enabled staff to record data about weather.",
        ),
        ("The old system allowed staff to record data.", "The new system enabled staff to record data."),
        (
            "The system allowed staff to record data, with permission from the owner.",
            "The system enabled staff to record data.",
        ),
        (
            "The system allowed staff to record data, only after approval.",
            "The system enabled staff to record data.",
        ),
        (
            "The system allowed staff to record data. The system allowed staff to record data.",
            "The system enabled staff to record data.",
        ),
        ("The system did not allow staff to record data.", "The system enabled staff to record data."),
        ("Staff were allowed to record data.", "Staff were enabled to record data."),
        ("The system permitted staff to record data.", "The system enabled staff to record data."),
        ("The new system allowed staff to record data.", "The system enabled staff to record data."),
        ("The new system-like panel allowed staff to record data.", "The new system enabled staff to record data."),
        (
            "The system allowed staff to record additional information about visitors.",
            "The system enabled staff to document extra details about visitors for sale.",
        ),
    ],
)
def test_permission_and_unmatched_capability_cues_remain_checked(source: str, candidate: str) -> None:
    assert not verify_text_rewrite_facts(source, candidate)


_FILM_CAPABILITY = (
    "This continuous mechanical advance allows a film camera to capture the same scene multiple times "
    "without manual rewinding."
)
_FILM_PARAPHRASE = (
    "This ongoing mechanical advancement enables a film camera to record the identical scene repeatedly "
    "without requiring manual rewinding."
)


def test_mechanical_film_advance_is_capability_in_both_directions() -> None:
    assert verify_text_rewrite_facts(_FILM_CAPABILITY, _FILM_PARAPHRASE)
    assert verify_text_rewrite_facts(_FILM_PARAPHRASE, _FILM_CAPABILITY)


@pytest.mark.parametrize(
    "candidate",
    [
        _FILM_PARAPHRASE.replace("identical scene", "next scene"),
        _FILM_PARAPHRASE.replace("without requiring manual rewinding", "after manual rewinding"),
        _FILM_PARAPHRASE.replace("film camera", "digital camera"),
        _FILM_PARAPHRASE.replace("mechanical advancement", "policy"),
        _FILM_PARAPHRASE.replace("repeatedly", "once"),
    ],
)
def test_film_capability_keeps_action_and_conditions_bound(candidate: str) -> None:
    assert not verify_text_rewrite_facts(_FILM_CAPABILITY, candidate)


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        (
            "Vapor escapes, acting like a conduit for moisture to evaporate and rise.",
            "Vapor escapes, serving as a channel that allows moisture to evaporate and rise.",
        ),
        (
            "The fabric dries, allowing air circulation throughout the tent.",
            "The fabric dries, enabling air movement throughout the tent.",
        ),
    ],
)
def test_matching_physical_process_cues_preserve_meaning(source: str, candidate: str) -> None:
    assert verify_text_rewrite_facts(source, candidate)
    assert verify_text_rewrite_facts(candidate, source)


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        (
            "The conduit allows moisture to evaporate.",
            "The conduit enables water to pool.",
        ),
        (
            "The fabric dries, allowing air circulation throughout the tent.",
            "The fabric dries, enabling air movement throughout the building.",
        ),
        (
            "The fabric dries, allowing air circulation. The owner grants permission to enter.",
            "The fabric dries, enabling air movement.",
        ),
    ],
)
def test_unmatched_physical_process_and_explicit_permission_remain_checked(source: str, candidate: str) -> None:
    assert not verify_text_rewrite_facts(source, candidate)


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        (
            "A latch is a mechanical device that allows for the effortless opening and closing of a gate.",
            "A latch is a mechanical apparatus that facilitates the smooth opening and closing of a gate.",
        ),
        (
            "Water changes the cellulose fibers, causing them to slide over one another more easily.",
            "Water alters the cellulose fibers, allowing them to slide past each other with greater ease.",
        ),
        (
            "The disengaged latch allows the gate to swing outward freely.",
            "The disengaged latch permits the gate to swing outward without restriction.",
        ),
        (
            "Subjects provide more specific filters, allowing users to narrow their searches.",
            "Subjects offer more precise filtering options, enabling users to refine their searches.",
        ),
    ],
)
def test_matching_physical_actions_preserve_rewrite(source: str, candidate: str) -> None:
    assert verify_text_rewrite_facts(source, candidate)
    assert verify_text_rewrite_facts(candidate, source)


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        (
            "The manager allows for the effortless opening and closing of a gate.",
            "The manager facilitates the smooth opening and closing of a gate.",
        ),
        (
            "The latch is a mechanical device that allows for the effortless opening and closing of a gate.",
            "The latch is a mechanical apparatus that facilitates the smooth opening and closing of a door.",
        ),
        (
            "Water changes the cellulose fibers, causing them to slide over one another more easily.",
            "Water alters the cellulose fibers, allowing them to slide past each other with less ease.",
        ),
        (
            "The disengaged latch allows the gate to swing outward freely.",
            "The disengaged latch permits the gate to swing outward without restriction only after approval.",
        ),
        (
            "The disengaged latch allows the gate to swing outward freely.",
            "The disengaged latch permits the gate to swing inward without restriction.",
        ),
        (
            "The disengaged latch allows the gate to swing outward freely.",
            "The disengaged latch does not permit the gate to swing outward without restriction.",
        ),
        (
            "Water changes the cellulose fibers, causing them to slide over one another. Pressure stays over one bar.",
            "Water changes the cellulose fibers, causing them to slide past each other. Pressure is one bar.",
        ),
        (
            "Subjects provide more specific filters, allowing users to narrow their searches.",
            "Subjects offer more precise filtering options, enabling staff to refine their searches.",
        ),
        (
            "Subjects provide more specific filters, allowing users to narrow their searches. "
            "The librarian grants permission to view the records.",
            "Subjects offer more precise filtering options, enabling users to refine their searches.",
        ),
        (
            "The librarian allows users to narrow their searches.",
            "Subjects offer more precise filtering options, enabling users to refine their searches.",
        ),
    ],
)
def test_changed_actions_and_restrictions_remain_checked(source: str, candidate: str) -> None:
    assert not verify_text_rewrite_facts(source, candidate)
