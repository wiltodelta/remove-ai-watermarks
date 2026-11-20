"""Check preservation of selected attempts, intentions, and completed actions."""

import pytest

from remove_ai_watermarks.text import verify_text_rewrite_facts


@pytest.mark.parametrize(
    ("source", "paraphrase", "changed"),
    [
        (
            "As the storm descended, the small grey harbour cat found shelter in the museum\u2019s basement, "
            "where the quiet museum door locksmith lit a candle.",
            "As the storm came down, the small grey harbour cat took refuge in the museum\u2019s basement, "
            "where the quiet museum door locksmith lit a candle.",
            "As the storm came down, the small grey harbour cat sought refuge in the museum\u2019s basement, "
            "where the quiet museum door locksmith lit a candle.",
        ),
        (
            "The hikers sought shelter.",
            "The hikers searched for refuge.",
            "The hikers found refuge.",
        ),
        (
            "The librarian tried to reopen the archive.",
            "The librarian attempted to reopen the archive.",
            "The archive was reopened by the librarian.",
        ),
        (
            "The relay was repaired by the technician.",
            "The technician managed to repair the relay.",
            "The technician tried to repair the relay.",
        ),
        (
            "The council planned to restore the bridge.",
            "The council intended to restore the bridge.",
            "The council attempted to restore the bridge.",
        ),
        (
            "The courier succeeded in delivering the documents.",
            "The courier delivered the documents.",
            "The courier intended to deliver the documents.",
        ),
        (
            "She tried to unlock the spare cabinet.",
            "She attempted to unlock the spare cabinet.",
            "She unlocked the spare cabinet.",
        ),
        (
            "The report was submitted by the coordinator.",
            "The coordinator submitted the report.",
            "The coordinator planned to submit the report.",
        ),
    ],
)
def test_comparable_events_preserve_phase(source: str, paraphrase: str, changed: str) -> None:
    assert verify_text_rewrite_facts(source, paraphrase)
    assert not verify_text_rewrite_facts(source, changed)
    assert not verify_text_rewrite_facts(changed, source)


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("The technician tried to repair the relay.", "The technician tried to fix the relay."),
        ("The technician tried to repair the signal.", "The technician repaired the backup signal."),
        ("The technician tried to repair the relay.", "The assistant repaired the relay."),
        (
            "The technician tried to repair the relay. The technician repaired the relay.",
            "The technician repaired the relay.",
        ),
        (
            "When the ferry arrived, the technician tried to repair the radio.",
            "If the ferry arrived, the technician repaired the radio.",
        ),
        ("The technician tried to repair the relay.", "The technician would repair the relay."),
        ("The hikers sought shelter in the valley.", "The hikers found refuge on the ridge."),
        ("The technician tried to repair the relay.", "The technician repaired the relay?"),
    ],
)
def test_unmatched_or_ambiguous_events_do_not_trigger_phase_check(source: str, candidate: str) -> None:
    assert verify_text_rewrite_facts(source, candidate)
