"""Selected literal membership statements must not become critical stances."""

import pytest

from remove_ai_watermarks import text as lib


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("This blog is part of Transforming Manufacturing.", "The blog is critical of Transforming Manufacturing."),
        ("The publication is part of the collection.", "The publication is critical of the collection."),
        ("The blog is critical of the series.", "The blog is part of the series."),
        (
            "Visitors read it.\nThis blog is part of the series.",
            "Visitors read it.\nThe blog is critical of the series.",
        ),
        (
            "The newsletter is part of the network; staff edit it.",
            "The newsletter is critical of the network; staff edit it.",
        ),
        ("The publication was part of the collection.", "The publication was critical of the collection."),
        (
            "I linked it above\nThe blog is part of the series.",
            "I listed it above The blog is critical of the series.",
        ),
    ],
)
def test_matching_literal_membership_and_stance_changes_are_rejected(source: str, candidate: str) -> None:
    assert not lib.verify_text_rewrite_facts(source, candidate)


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("This blog is part of the series.", "The blog is a member of the series."),
        ("The publication is part of the collection.", "The publication is an important part of the collection."),
        ("The blog is critical of the series.", "The blog is critical of this series."),
        ("The blog is critical of the collection.", "The blog is strongly critical of the collection."),
        ("The blog is part of the network.", "The column is critical of the network."),
        ("The blog is part of the network.", "The blog is critical of the committee."),
        ("The blog is part of the series. The blog is critical of the series.", "The blog is critical of the series."),
        ("The blog might be part of the series.", "The blog is critical of the series."),
        ("The blog is not part of the series.", "The blog is not critical of the series."),
    ],
)
def test_preserved_or_unresolved_frames_are_not_rejected(source: str, candidate: str) -> None:
    # Different descriptions, mixed claims and unsupported grammar are unresolved,
    # not a claim that these pairs all preserve meaning.
    assert lib.verify_text_rewrite_facts(source, candidate)


def test_shared_candidate_guard_rejects_membership_replaced_by_stance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lib, "_preserves_rewrite_language", lambda *_args: True)
    source = "This blog is part of Transforming Manufacturing."
    candidate = "The blog is critical of Transforming Manufacturing."
    assert lib._guard_local_rewrite_candidate(source, source, {}, candidate, None) is None
