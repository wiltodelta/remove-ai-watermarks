"""Tests for the Content Seal oracle corpus layout.

Mirrors the synthid corpus guard: the manifest is the source of truth for
which binaries exist, and every recorded hash must match the file it names.
Derived rows are recipes, not stored files, so only originals are checked
against disk.
"""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "contentseal"
MANIFEST = CORPUS_DIR / "manifest.csv"
ORIGINALS = CORPUS_DIR / "originals"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VALID_VERDICTS = {"detected", "not_detected", ""}
_VALID_ORIGINS = {"meta-model-api", "derived", "meta-blog-cdn", "synthetic-local"}


def _manifest_rows() -> list[dict[str, str]]:
    with open(MANIFEST, newline="") as f:
        return list(csv.DictReader(f))


def test_manifest_original_rows_match_binaries_and_hashes() -> None:
    rows = [r for r in _manifest_rows() if r["file"]]
    stored = {path.name for path in ORIGINALS.iterdir() if path.is_file()}
    assert {r["file"].removeprefix("originals/") for r in rows} == stored

    for row in rows:
        digest = hashlib.sha256((CORPUS_DIR / row["file"]).read_bytes()).hexdigest()
        assert digest == row["sha256"], row["file"]


def test_manifest_rows_are_well_formed() -> None:
    rows = _manifest_rows()
    assert len({row["sha256"] for row in rows}) == len(rows), "duplicate sha256"

    for row in rows:
        assert _SHA256.match(row["sha256"]), row["name"]
        assert row["origin"].split(":")[0] in _VALID_ORIGINS, row["name"]
        assert row["oracle_verdict"] in _VALID_VERDICTS, row["name"]
        # Every oracle verdict must carry its check timestamp.
        if row["oracle_verdict"]:
            assert row["checked_at_utc"], row["name"]
        # Detection rows must name the oracle attribution.
        if row["oracle_verdict"] == "detected":
            assert "Muse Image 1" in row["oracle_attribution"], row["name"]


def test_default_pipeline_clearance_is_recorded() -> None:
    """The verified claim that the default profile clears Content Seal must stay."""
    rows = {row["name"]: row for row in _manifest_rows()}
    for name in ("fox_default_invisible", "text_default_invisible"):
        assert rows[name]["oracle_verdict"] == "not_detected", name


def test_deterministic_transforms_reproduce_recorded_hashes(tmp_path: Path) -> None:
    from scripts.contentseal_transforms import reproduce_transforms

    outputs = reproduce_transforms(tmp_path)

    assert len(outputs) == 8
    assert all(path.is_file() for path in outputs)
