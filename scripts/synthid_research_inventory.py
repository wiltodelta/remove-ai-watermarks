"""Inventory local SynthID research images without assigning evidence labels.

The inventory is deliberately weaker than the research manifest. It records
artifact hashes, decoded RGB hashes, geometry, format, and exact duplicates,
but contains no provider, SynthID outcome, oracle, or split fields. Promotion
from inventory to manifest therefore remains an explicit evidence decision.

Usage:
    uv run python scripts/synthid_research_inventory.py \
        --root .local-eval/synthid negatives google openai \
        --inventory-out .local-eval/synthid/inventory.csv
"""

from __future__ import annotations

import csv
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import click
from synthid_research_manifest import artifact_sha256, decoded_image_fingerprint

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png", ".webp"}
FIELDNAMES = (
    "artifact_sha256",
    "pixel_sha256",
    "artifact_path",
    "width",
    "height",
    "format",
    "exact_pixel_group",
    "artifact_duplicate_of",
    "pixel_duplicate_of",
)


@dataclass(frozen=True)
class InventoryRow:
    """One decoded local image with exact-duplicate provenance."""

    artifact_sha256: str
    pixel_sha256: str
    artifact_path: str
    width: int
    height: int
    format: str
    exact_pixel_group: str
    artifact_duplicate_of: str
    pixel_duplicate_of: str


def _inside_root(root: Path, path: Path) -> Path:
    """Resolve PATH and reject anything outside ROOT."""
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"source is outside inventory root: {path}") from exc
    return resolved


def discover_images(root: Path, sources: tuple[Path, ...]) -> list[Path]:
    """Return supported images below explicit in-root sources in stable order."""
    resolved_root = root.resolve()
    discovered: set[Path] = set()
    for source in sources:
        candidate = source if source.is_absolute() else resolved_root / source
        candidate = _inside_root(resolved_root, candidate)
        if not candidate.exists():
            raise ValueError(f"inventory source does not exist: {source}")
        if candidate.is_file():
            if candidate.suffix.lower() in IMAGE_SUFFIXES:
                discovered.add(candidate)
            continue
        for path in candidate.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                discovered.add(_inside_root(resolved_root, path))
    return sorted(discovered, key=lambda path: path.relative_to(resolved_root).as_posix())


def build_inventory(root: Path, sources: tuple[Path, ...]) -> list[InventoryRow]:
    """Hash and decode all selected images without inferring any labels."""
    resolved_root = root.resolve()
    first_artifact: dict[str, str] = {}
    first_pixels: dict[str, str] = {}
    rows: list[InventoryRow] = []
    for path in discover_images(resolved_root, sources):
        relative = path.relative_to(resolved_root).as_posix()
        artifact_digest = artifact_sha256(path)
        pixel_digest, width, height, image_format = decoded_image_fingerprint(path)
        rows.append(
            InventoryRow(
                artifact_sha256=artifact_digest,
                pixel_sha256=pixel_digest,
                artifact_path=relative,
                width=width,
                height=height,
                format=image_format,
                exact_pixel_group=f"pixel-{pixel_digest[:16]}",
                artifact_duplicate_of=first_artifact.get(artifact_digest, ""),
                pixel_duplicate_of=first_pixels.get(pixel_digest, ""),
            )
        )
        first_artifact.setdefault(artifact_digest, relative)
        first_pixels.setdefault(pixel_digest, relative)
    return rows


def write_inventory(path: Path, rows: list[InventoryRow], *, replace: bool = False) -> None:
    """Write a complete inventory atomically enough to avoid partial decode results."""
    if path.exists() and not replace:
        raise FileExistsError(f"inventory already exists: {path}; pass --replace to overwrite it")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(asdict(row) for row in rows)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def inventory_summary(rows: list[InventoryRow]) -> dict[str, object]:
    """Return aggregate coverage without exposing media paths."""
    formats = Counter(row.format for row in rows)
    geometries = Counter(f"{row.width}x{row.height}" for row in rows)
    return {
        "images": len(rows),
        "unique_artifacts": len({row.artifact_sha256 for row in rows}),
        "unique_pixels": len({row.pixel_sha256 for row in rows}),
        "formats": dict(sorted(formats.items())),
        "geometries": dict(sorted(geometries.items())),
    }


@click.command()
@click.option("--root", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("sources", nargs=-1, required=True, type=click.Path(path_type=Path))
@click.option("--inventory-out", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--replace", is_flag=True, help="Replace an existing generated inventory.")
def main(root: Path, sources: tuple[Path, ...], inventory_out: Path, replace: bool) -> None:
    """Inventory image SOURCES below ROOT without assigning SynthID labels."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        rows = build_inventory(root, sources)
        if not rows:
            raise ValueError("selected sources contain no supported images")
        write_inventory(inventory_out, rows, replace=replace)
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    summary = inventory_summary(rows)
    log.info(
        "Wrote inventory: %s images=%s unique_artifacts=%s unique_pixels=%s",
        inventory_out,
        summary["images"],
        summary["unique_artifacts"],
        summary["unique_pixels"],
    )


if __name__ == "__main__":
    main()
