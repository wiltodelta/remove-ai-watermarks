"""Audit a local image corpus against the library's own ``identify`` detector.

Two jobs in one pass:

1. **Report** -- run ``identify`` over every image and write one CSV row per file
   (verdict, platform, confidence, watermarks, signals, raw metadata markers,
   candidate classes, integrity clashes, and errors).
2. **Gap audit** -- for every ``unknown``-verdict file, scan only its *metadata
   region* (PNG text/eXIf chunks, JPEG APPn segments before SOS, or the file
   head for other containers) for known provenance markers. A marker found there
   on a file the detector calls ``unknown`` is a concrete lib gap: a serialization
   or generator we do not yet parse. Scanning the metadata region -- not the whole
   file -- is deliberate: short tokens collide randomly inside compressed PNG
   ``IDAT`` / JPEG scan data, which produced false "xAI/Flux/AIGC" hits when the
   first audit naively scanned the first megabyte.

This is how new detector gaps get found (it is what surfaced the JPEG-EXIF
``{"AIGC":{...}}`` form). Re-run after collecting a fresh evaluation batch.

Usage:
    uv run python scripts/corpus_gap_scan.py --corpus .local-eval/originals
    uv run python scripts/corpus_gap_scan.py --corpus .local-eval/originals \\
        --report .local-eval/detector-report.csv
    uv run python scripts/corpus_gap_scan.py --corpus .local-eval/originals \\
        --since 2026-09-01 --report .local-eval/detector-report-weekly.csv
"""

from __future__ import annotations

import csv
import logging
from collections import Counter
from datetime import date
from importlib.metadata import version
from pathlib import Path
from typing import Any

import click
from _plain_console import Console, Table

from remove_ai_watermarks.identify import _metadata_region as identify_metadata_region
from remove_ai_watermarks.identify import identify
from remove_ai_watermarks.metadata import scan_head

log = logging.getLogger(__name__)
console = Console()

# Distinctive, multi-byte provenance markers worth flagging when they appear in a
# file the detector calls `unknown`. Kept long enough that a random collision in a
# (non-scanned) compressed stream is implausible; the metadata-region restriction
# below is the primary guard, this list is the second. Group: C2PA/JUMBF infra,
# AI source-type / labeling schemes, and distinctive generator name strings.
MARKERS: tuple[bytes, ...] = (
    # C2PA / JUMBF infrastructure and AI source-type / labeling schemes.
    b"c2pa",
    b"jumbf",
    b"contentauth",
    b"trainedAlgorithmicMedia",
    b"digitalSourceType",
    b'"AIGC"',
    b"<TC260:AIGC>",
    b"TC260:AIGC",
    b"tc260.org.cn",
    b"AISystemUsed",
    b"SynthID",
    b"hf-job-id",
    b"genAIType",
    b"PhotoEditor_Re_Edit",
    b"Signature:",
    # Distinctive multi-word generator strings only. Bare single words (Luma,
    # Gemini, Sora, ...) are omitted: they collide with unrelated metadata prose
    # (e.g. "Luma" in Lightroom's EnhanceDenoiseLumaAmount), defeating precision.
    b"Midjourney",
    b"Stable Diffusion",
    b"StableDiffusion",
    b"ComfyUI",
    b"Automatic1111",
    b"DALL-E",
    b"Ideogram AI",
    b"Adobe Firefly",
    b"Black Forest",
    b"volcengine",
    b"Doubao",
    b"\xe8\xb1\x86\xe5\x8c\x85",
    b"Nano Banana",
    b"Stability AI",
    b"Samsung Galaxy",
)

REPORT_FIELDS: tuple[str, ...] = (
    "path",
    "suffix",
    "lib_version",
    "is_ai",
    "platform",
    "confidence",
    "watermarks",
    "signals",
    "integrity_clashes",
    "markers",
    "candidate_classes",
    "error",
)


def _base_row(path: str, suffix: str, lib_version: str) -> dict[str, str]:
    row = dict.fromkeys(REPORT_FIELDS, "")
    row.update(path=path, suffix=suffix, lib_version=lib_version)
    return row


def _row(rep, *, path: str, suffix: str, lib_version: str) -> dict[str, str]:  # noqa: ANN001
    return {
        **_base_row(path, suffix, lib_version),
        "is_ai": str(rep.is_ai_generated),
        "platform": rep.platform or "",
        "confidence": rep.confidence,
        "watermarks": "|".join(rep.watermarks),
        "signals": "|".join(s.name for s in rep.signals),
        "integrity_clashes": "|".join(rep.integrity_clashes),
    }


def _marker_hits(region: bytes) -> list[str]:
    """Return marker labels found case-insensitively in one metadata region."""
    folded = region.lower()
    return sorted({marker.decode("latin-1", "replace") for marker in MARKERS if marker.lower() in folded})


def _marker_hits_for_path(path: Path) -> list[str]:
    """Return bounded metadata hits without turning an unreadable row into a fatal scan."""
    try:
        return _marker_hits(identify_metadata_region(scan_head(path)))
    except OSError:
        return []


def _candidate_classes(rep: Any, hits: list[str]) -> list[str]:
    """Classify review-worthy outcomes without treating them as proven bugs."""
    classes: list[str] = []
    if hits and not rep.is_ai_generated and not rep.signals:
        classes.append("blind_marker")
    if rep.is_ai_generated and not rep.platform:
        classes.append("unattributed_ai")
    elif rep.signals and not rep.platform:
        classes.append("unattributed_signal")
    if rep.integrity_clashes:
        classes.append("integrity_clash")
    return classes


def _day_of(path: Path, corpus: Path) -> date | None:
    """Read the leading YYYY-MM-DD corpus segment, if this layout has one."""
    try:
        segment = path.relative_to(corpus).parts[0]
        return date.fromisoformat(segment)
    except (ValueError, IndexError):
        return None


def _files(corpus: Path, since: date | None) -> list[Path]:
    """Return the deterministic corpus walk, optionally bounded by its date segment."""
    if since is None:
        return sorted(path for path in corpus.rglob("*") if path.is_file())
    roots = [
        path
        for path in corpus.iterdir()
        if path.is_dir() and (day := _day_of(path, corpus)) is not None and day >= since
    ]
    return sorted(path for root in roots for path in root.rglob("*") if path.is_file())


@click.command()
@click.option(
    "--corpus",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path(".local-eval/originals"),
    show_default=True,
    help="Directory of images to scan (recursively).",
)
@click.option(
    "--report",
    type=click.Path(path_type=Path),
    default=None,
    help="Write the per-file CSV here (default: <corpus>/../detector_report.csv).",
)
@click.option(
    "--since",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Only scan files under YYYY-MM-DD corpus directories on or after this date.",
)
@click.option("--limit", type=int, default=0, help="Scan at most N files (0 = all).")
def main(corpus: Path, report: Path | None, since, limit: int) -> None:  # noqa: ANN001
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    report = report or corpus.parent / "detector_report.csv"

    since_date = since.date() if since is not None else None
    files = _files(corpus, since_date)
    if limit:
        files = files[:limit]
    console.print(f"Scanning [bold]{len(files)}[/bold] files under {corpus} ...")

    verdicts: Counter[str] = Counter()
    platforms: Counter[str] = Counter()
    shapes: Counter[str] = Counter()
    rows: list[dict[str, str]] = []
    errors = 0
    lib_version = version("remove-ai-watermarks")

    with click.progressbar(files, label="identify") as bar:
        for p in bar:
            rel = str(p.relative_to(corpus))
            shapes[p.suffix.lower() or "(none)"] += 1
            try:
                rep = identify(p)
            except Exception as exc:
                log.warning("identify failed on %s: %s", rel, exc)
                errors += 1
                row = _base_row(rel, p.suffix.lower(), lib_version)
                row["candidate_classes"] = "identify_error"
                row["error"] = f"{type(exc).__name__}: {exc}"[:300]
                hits = _marker_hits_for_path(p)
                row["markers"] = "|".join(hits)
                rows.append(row)
                continue
            row = _row(rep, path=rel, suffix=p.suffix.lower(), lib_version=lib_version)
            # identify() has already populated scan_head's bounded cache, so this
            # reuses the exact metadata region rather than reading the file again.
            hits = _marker_hits_for_path(p)
            classes = _candidate_classes(rep, hits)
            row["markers"] = "|".join(hits)
            row["candidate_classes"] = "|".join(classes)
            rows.append(row)
            if rep.is_ai_generated:
                verdicts["ai"] += 1
                platforms[rep.platform or "?"] += 1
                continue
            verdicts["unknown"] += 1

    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    console.print(f"\nWrote [bold]{len(rows)}[/bold] rows -> {report}")

    console.print(f"\n[bold]Verdicts:[/bold] AI {verdicts['ai']} | unknown {verdicts['unknown']} | errors {errors}")
    console.print("[bold]Input shapes:[/bold] " + " | ".join(f"{name} {n}" for name, n in shapes.most_common()))
    plat = Table(title="AI platforms", show_header=False)
    for name, n in platforms.most_common():
        plat.add_row(str(n), name)
    console.print(plat)

    candidates = [row for row in rows if row["candidate_classes"]]
    if candidates:
        candidate_counts = Counter(candidate for row in candidates for candidate in row["candidate_classes"].split("|"))
        gap_tokens = Counter(marker for row in rows for marker in row["markers"].split("|") if marker)
        console.print(
            f"\n[bold red]Review candidates[/bold red]: {len(candidates)} file(s) "
            f"({', '.join(f'{name}={n}' for name, n in candidate_counts.most_common())})"
        )
        tok = Table(title="metadata markers seen across the run")
        tok.add_column("count", justify="right")
        tok.add_column("marker")
        for name, n in gap_tokens.most_common():
            tok.add_row(str(n), name)
        console.print(tok)
        for row in candidates:
            detail = f"; markers={row['markers'].replace('|', ', ')}" if row["markers"] else ""
            console.print(f"  {row['path']}  ->  {row['candidate_classes'].replace('|', ', ')}{detail}")
    else:
        console.print("\n[green]No review candidates in this run.[/green]")


if __name__ == "__main__":
    main()
