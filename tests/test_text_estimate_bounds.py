"""A stated numeric estimate must not silently become a measured fact."""

import pytest

from remove_ai_watermarks import text


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("The pilot reduced costs by an estimated 40%.", "The pilot reduced costs by 40%."),
        ("The pilot reduced costs by 40%.", "The pilot reduced costs by an estimated 40%."),
        ("The pilot reduced costs by an estimate of 40%.", "The pilot reduced costs by 40%."),
        ("The duration was estimated at two hours.", "The duration was two hours."),
        ("They saved an estimated 40% and spent 20%.", "They saved 40% and spent an estimated 20%."),
        (
            "Testing took an estimated two hours and two days of waiting.",
            "Testing took two hours and an estimated two days of waiting.",
        ),
    ],
)
def test_estimate_status_cannot_be_added_dropped_or_transferred(source, candidate):
    assert not text.verify_text_rewrite_facts(source, candidate)


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("The pilot reduced costs by an estimated 40%.", "The pilot reduced costs by an estimate of 40%."),
        ("The duration was estimated at two hours.", "The duration was an estimate of 2 hours."),
        ("They estimated forty visitors.", "They estimated 40 visitors."),
        ("They requested an estimate of 40 visitors.", "They requested an estimate of forty visitors."),
        ("The team estimated the costs.", "The team estimated the costs."),
    ],
)
def test_equivalent_explicit_estimate_forms_are_accepted(source, candidate):
    assert text.verify_text_rewrite_facts(source, candidate)


def test_shared_rewriter_guard_rejects_an_estimate_changed_to_a_fact(monkeypatch):
    monkeypatch.setattr(text, "_preserves_rewrite_language", lambda *_args: True)
    source = "The pilot reduced costs by an estimated 40%."
    candidate = "The pilot reduced costs by 40%."
    assert text._guard_local_rewrite_candidate(source, source, {}, candidate, None) is None
