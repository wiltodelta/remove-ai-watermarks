"""Tests for label-free local SynthID research inventories."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import synthid_research_inventory as inventory


def _write_png(path: Path, color: tuple[int, int, int], *, note: str | None = None) -> None:
    image = Image.new("RGB", (9, 7), color)
    pnginfo = None
    if note is not None:
        pnginfo = PngInfo()
        pnginfo.add_text("note", note)
    image.save(path, format="PNG", pnginfo=pnginfo)


def test_inventory_is_stable_and_contains_no_evidence_labels(tmp_path: Path):
    media = tmp_path / "media"
    media.mkdir()
    _write_png(media / "b.png", (20, 30, 40))
    _write_png(media / "a.png", (10, 20, 30))

    rows = inventory.build_inventory(tmp_path, (Path("media"),))

    assert [row.artifact_path for row in rows] == ["media/a.png", "media/b.png"]
    assert set(inventory.FIELDNAMES).isdisjoint({"target_provider", "synthid_outcome", "verified_via", "split"})
    assert all(row.format == "png" and row.width == 9 and row.height == 7 for row in rows)


def test_inventory_marks_byte_and_pixel_duplicates(tmp_path: Path):
    media = tmp_path / "media"
    media.mkdir()
    first = media / "first.png"
    exact = media / "exact.png"
    metadata_variant = media / "metadata.png"
    _write_png(first, (10, 20, 30), note="one")
    exact.write_bytes(first.read_bytes())
    _write_png(metadata_variant, (10, 20, 30), note="different metadata")

    rows = {row.artifact_path: row for row in inventory.build_inventory(tmp_path, (Path("media"),))}

    assert rows["media/exact.png"].artifact_duplicate_of == ""
    assert rows["media/first.png"].artifact_duplicate_of == "media/exact.png"
    assert rows["media/first.png"].pixel_duplicate_of == "media/exact.png"
    assert rows["media/metadata.png"].artifact_duplicate_of == ""
    assert rows["media/metadata.png"].pixel_duplicate_of == "media/exact.png"
    assert len({row.exact_pixel_group for row in rows.values()}) == 1


def test_discovery_rejects_source_outside_root(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.png"
    _write_png(outside, (10, 20, 30))

    with pytest.raises(ValueError, match="outside inventory root"):
        inventory.discover_images(root, (outside,))


def test_inventory_uses_content_format_not_suffix(tmp_path: Path):
    disguised = tmp_path / "disguised.jpg"
    _write_png(disguised, (10, 20, 30))

    row = inventory.build_inventory(tmp_path, (Path("disguised.jpg"),))[0]

    assert row.format == "png"


def test_write_refuses_to_replace_without_explicit_flag(tmp_path: Path):
    media = tmp_path / "image.png"
    _write_png(media, (10, 20, 30))
    rows = inventory.build_inventory(tmp_path, (Path("image.png"),))
    output = tmp_path / "inventory.csv"
    inventory.write_inventory(output, rows)

    with pytest.raises(FileExistsError, match="--replace"):
        inventory.write_inventory(output, rows)

    inventory.write_inventory(output, rows, replace=True)
    with output.open(newline="", encoding="utf-8") as stream:
        written = list(csv.DictReader(stream))
    assert written[0]["artifact_path"] == "image.png"


def test_summary_reports_only_aggregates(tmp_path: Path):
    _write_png(tmp_path / "image.png", (10, 20, 30))
    rows = inventory.build_inventory(tmp_path, (Path("image.png"),))

    summary = inventory.inventory_summary(rows)

    assert summary == {
        "images": 1,
        "unique_artifacts": 1,
        "unique_pixels": 1,
        "formats": {"png": 1},
        "geometries": {"9x7": 1},
    }
