"""Selected explicit type coverage must survive local prose regeneration."""

import pytest

from remove_ai_watermarks import text as lib


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        (
            "Volunteers provided maintenance checklists for each type of appliance.",
            "Volunteers provided maintenance checklists for various appliance types.",
        ),
        ("The checklist covers every appliance type.", "The checklist covers several appliance types."),
        ("The checklist covers all appliance types.", "The checklist covers appliance types."),
        ("The checklist covers appliance types.", "The checklist covers all appliance types."),
        ("The checklist covers each type.", "The checklist covers a type."),
        (
            "The checklist covers each type. The guide covers each type.",
            "The checklist covers each type. The guide covers various types.",
        ),
    ],
)
def test_lost_or_invented_selected_type_coverage_fails(source: str, candidate: str) -> None:
    assert not lib.verify_text_rewrite_facts(source, candidate)


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        (
            "Volunteers provided maintenance checklists for each type of appliance.",
            "Volunteers provided maintenance checklists for every appliance type.",
        ),
        ("The checklist covers every appliance type.", "The checklist covers all appliance types."),
        ("The checklist covers each appliance type.", "The checklist covers every type of appliance."),
        ("The checklist covers each type.", "The checklist covers all types."),
        ("Neighbors help each other.", "Neighbors help one another."),
        ("The new prototype is ready.", "The new prototype is ready."),
        ("The checklist covers each type-specific issue.", "The checklist covers a type-specific issue."),
        ("The checklist covers each type-specific issue.", "The checklist covers every type-specific issue."),
        ("The checklist covers all typesetting issues.", "The checklist covers typesetting issues."),
    ],
)
def test_selected_aliases_and_unrecognized_phrases_are_not_rejected(source: str, candidate: str) -> None:
    assert lib.verify_text_rewrite_facts(source, candidate)


@pytest.mark.parametrize(
    "phrase", ["each type", "every appliance type", "all appliance types", "ALL TYPES", "each\nsmall appliance type"]
)
def test_whole_selected_coverage_phrase_is_masked(phrase: str) -> None:
    text = "Staff maintain checklists for " + phrase + "."
    masked, facts = lib._mask_local_rewrite_facts(text)
    assert phrase in facts.values()
    assert phrase not in masked
    assert sum(v == phrase for v in facts.values()) == 1
    rebuilt = masked
    for token, literal in facts.items():
        rebuilt = rebuilt.replace(token, literal)
    assert rebuilt == text


@pytest.mark.parametrize(
    "phrase",
    [
        "prototype",
        "all typesetting",
        "each type-specific",
        "all types-based",
        "every type\u2010specific",
        "each type\u2011specific",
    ],
)
def test_word_and_hyphen_boundaries_are_not_coverage_facts(phrase: str) -> None:
    _, facts = lib._mask_local_rewrite_facts("Staff inspect " + phrase + " issues.")
    assert phrase not in facts.values()
    assert not any(v.casefold() in {"each type", "every type", "all types"} for v in facts.values())


def test_known_native_coverage_loss_reaches_shared_local_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    source = (
        "Moreover, volunteers began providing detailed maintenance checklists for each type of "
        "appliance during the initial fix, ensuring clear instructions on how to avoid common "
        "pitfalls. For instance, they advised against overloading washing machines, which was "
        "suspected as a factor contributing to early failure. Implementing these measures has "
        "received positive reception from community members eager to learn more about the best ways "
        "to extend the life of their appliances."
    )
    candidate = (
        "In addition, volunteers started supplying thorough maintenance checklists for various "
        "appliance types during the initial repairs, offering straightforward guidance on steering "
        "clear of typical issues. For example, they cautioned against overstuffing washing "
        "machines, a practice believed to play a role in premature breakdowns. Community members, "
        "keen to understand optimal strategies for prolonging appliance longevity, have responded "
        "favorably to the introduction of these practices."
    )
    # Isolate the actual fact guard; language identification has its own tests.
    monkeypatch.setattr(lib, "_preserves_rewrite_language", lambda *_: True)
    assert lib._guard_local_rewrite_candidate(source, source, {}, candidate, None) is None
