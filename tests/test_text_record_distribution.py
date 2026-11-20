"""Explicit communal records must not become separate records per item."""

import pytest

from remove_ai_watermarks.text import verify_text_rewrite_facts


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("A communal notebook records the tools.", "Each tool has its own notebook."),
        ("A shared ledger records the tools.", "Every tool receives a separate ledger."),
        ("The communal log records the tools.", "The tools are each accompanied by a personalized notebook."),
        ("A common digital register records the tools.", "Each tool gets a dedicated register."),
        ("Each tool has its own notebook.", "The tools are recorded in a shared notebook."),
        ("Each tool keeps a separate logbook.", "A communal logbook records the tools."),
        ("A communal notebook records the tools.", "A separate notebook accompanies each tool."),
        ("A shared ledger records the tools.", "The tools have a separate ledger per item."),
    ],
)
def test_explicit_record_distribution_changes_fail(source: str, candidate: str) -> None:
    assert not verify_text_rewrite_facts(source, candidate)


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("A communal notebook records the tools.", "The tools are recorded in a shared notebook."),
        ("A shared ledger records the tools.", "A common digital ledger documents the tools."),
        ("Each tool has its own notebook.", "Every tool receives a separate notebook."),
        ("Each volunteer writes in a shared notebook.", "The volunteers use a communal notebook."),
        ("A shared notebook contains a record for each tool.", "Each tool has a record in a shared notebook."),
        ("Each tool has its own page in a shared notebook.", "The shared notebook has a separate page for each tool."),
        ("A communal notebook records the tools.", "Each tool has its own notebook entry."),
        ("A shared ledger records the tools.", "Every tool receives a separate ledger page."),
        ("A common register records the tools.", "Each tool gets a dedicated register code."),
        ("A communal notebook-shaped display holds the tools.", "Each tool has its own notebook."),
        ("Each tool has a dedicated register-sized label.", "A communal register records the tools."),
        ("The shared shelf holds a notebook.", "Each tool has its own notebook."),
        ("A communal notebook records the tools.", "The tools appear in the notebook."),
        ("The communal log records the tools.", "Each tool has a shared notebook entry."),
        ("Each team uses a shared notebook.", "The teams use a communal notebook."),
        ("A communal notebook records the tools. Each tool has its own ledger.", "Each tool has its own notebook."),
    ],
)
def test_equivalent_or_unresolved_record_scope_is_not_rejected(source: str, candidate: str) -> None:
    # Passing these selected heuristics does not certify the unresolved pairs.
    assert verify_text_rewrite_facts(source, candidate)
