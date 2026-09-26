"""The provider-capture manifest and the files it indexes must agree."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CAPTURES = ROOT / "data" / "captures"
MANIFESTS = sorted(CAPTURES.glob("*/manifest.csv"))


def _rows(manifest: Path) -> list[dict[str, str]]:
    with manifest.open(newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("manifest", MANIFESTS, ids=lambda path: path.parent.name)
def test_every_row_points_at_a_file_with_its_digest(manifest: Path):
    for row in _rows(manifest):
        path = ROOT / row["path"]
        assert path.is_file(), row["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], row["path"]


@pytest.mark.parametrize("manifest", MANIFESTS, ids=lambda path: path.parent.name)
def test_every_stored_capture_is_indexed_once(manifest: Path):
    rows = _rows(manifest)
    indexed = [row["path"] for row in rows]
    assert len(indexed) == len(set(indexed))
    assert len({row["sha256"] for row in rows}) == len(rows)
    on_disk = {
        str(path.relative_to(ROOT))
        for path in manifest.parent.rglob("*")
        if path.is_file() and path.name not in {"manifest.csv", "README.md"}
    }
    assert on_disk == {row["path"] for row in rows if row["stored"] == "new"}


def test_a_manifest_exists():
    assert MANIFESTS
