"""Audit a local image corpus against the library's own ``identify`` detector.

Two jobs in one pass:

1. **Report** -- run metadata-only ``identify`` over every file and write one CSV
   row per file (verdict, platform, confidence, watermarks, signals, raw metadata
   markers, candidate classes, integrity clashes, and errors). Visible and
   invisible pixel detectors belong to their own corpus audits.
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
        --workers 8 --report .local-eval/detector-report.csv
    uv run python scripts/corpus_gap_scan.py --corpus .local-eval/originals \\
        --since 2026-09-01 --report .local-eval/detector-report-weekly.csv

Rows stream into ``<report>.progress.jsonl`` and resume by relative path after an
interruption. A schema version prevents an older checkpoint from silently
producing blank fields after the report grows. Use a new report name for a new
code/corpus snapshot, or pass ``--restart`` when intentionally replacing the
checkpoint. The final CSV is written atomically after every selected row is
present. C2PA candidates include normalized signer, claim-generator, action,
source-type, and validation fields for aggregate review without reopening every
manifest.
"""

from __future__ import annotations

import csv
import itertools
import json
import logging
import os
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import date

import click
from _corpus_scan import corpus_day, corpus_files, repair_jsonl_tail
from _plain_console import Console, Table

from remove_ai_watermarks._internal.c2pa import C2PA_STRUCTURAL_MARKERS
from remove_ai_watermarks.identify import _metadata_region as identify_metadata_region
from remove_ai_watermarks.identify import extract_provenance_evidence, identify_from_evidence
from remove_ai_watermarks.metadata import (
    IPTC_AI_FIELD_MARKERS,
    IPTC_AI_MARKERS,
    KLING_SEI_TAG,
    aigc_label_from_metadata,
    app_generator_from_metadata,
    c2pa_marker_in,
    samsung_genai_in,
    scan_head,
    xai_signature_in_metadata,
)

log = logging.getLogger(__name__)
console = Console()

# Distinctive, multi-byte provenance markers worth flagging when they appear in a
# file the detector calls `unknown`. Kept long enough that a random collision in a
# (non-scanned) compressed stream is implausible; the metadata-region restriction
# below is the primary guard, this list is the second. Group: C2PA/JUMBF infra,
# AI source-type / labeling schemes, and distinctive generator name strings.
MARKERS: tuple[bytes, ...] = (
    # AI source-type / labeling schemes that are meaningful by presence.
    b"trainedAlgorithmicMedia",
    b"AISystemUsed",
    b"c2pa.ai-disclosure",
    b"hf-job-id",
    # Distinctive multi-word or domain-scoped generator strings only. Bare single
    # words (Luma, Gemini, Sora, ...) are omitted: they collide with unrelated
    # metadata prose (e.g. "Luma" in Lightroom's EnhanceDenoiseLumaAmount),
    # defeating precision.
    b"Midjourney",
    b"Stable Diffusion",
    b"StableDiffusion",
    b"ComfyUI",
    b"Automatic1111",
    b"InvokeAI",
    b"DALL-E",
    b"gpt-image",
    b"GPT Image",
    b"Ideogram AI",
    b"Adobe Firefly",
    b"Black Forest",
    b"volcengine",
    b"Byteplus",
    b"Nano Banana",
    b"Stability AI",
    b"DreamStudio",
    b"NightCafe",
    b"NovelAI",
    b"reve.com",
    b"Luma AI",
    b"lumalabs",
    b"Aphrodite AI",
    b"Apple Image Playground",
    b"Apple Photos Generative Edit",
    b"Apple Photos Clean Up",
    b"Grok Imagine",
    b"Amazon Bedrock",
    b"Higgsfield AI",
    b"recraft.ai",
    b"fal-ai",
    b"Producer.ai",
    b"Synchronicity Labs",
    b"Runway AI",
    b"Pruna AI",
    b"Autoenhance.ai",
    b"ks-c2pa-service",
    b"Kling AI C2PA",
    b"PhotoDirector",
    b"Topaz Labs",
    b"pixelbin.io",
    b"Pixelcut",
    b"vivo Albums",
    b"Raw profile type app11",
    b"C2PATXT",
    b"BEGIN C2PA MANIFEST",
)
REPORT_FIELDS: tuple[str, ...] = (
    "report_schema_version",
    "path",
    "suffix",
    "lib_version",
    "is_ai",
    "platform",
    "confidence",
    "watermarks",
    "signals",
    "c2pa_issuer",
    "c2pa_claim_generator",
    "c2pa_actions",
    "c2pa_source_type",
    "c2pa_validation",
    "integrity_clashes",
    "markers",
    "candidate_classes",
    "error",
)
_REPORT_SCHEMA_VERSION = "2"
_DISPLAY_CANDIDATE_LIMIT = 50
_WORKER_BATCH_SIZE = 16


def _base_row(path: str, suffix: str, lib_version: str) -> dict[str, str]:
    row = dict.fromkeys(REPORT_FIELDS, "")
    row.update(report_schema_version=_REPORT_SCHEMA_VERSION, path=path, suffix=suffix, lib_version=lib_version)
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


_LOWERED_MARKERS: tuple[tuple[str, bytes], ...] = tuple(
    (marker.decode("latin-1", "replace"), marker.lower()) for marker in MARKERS
)
_STRUCTURAL_MEDIA_METADATA_KEYS: frozenset[str] = C2PA_STRUCTURAL_MARKERS | {"source_type"}


def _c2pa_fields(mapping: Mapping[str, object]) -> dict[str, str]:
    return {
        "c2pa_issuer": str(mapping.get("issuer", "")),
        "c2pa_claim_generator": str(mapping.get("claim_generator", "")),
        "c2pa_actions": str(mapping.get("actions", "")),
        "c2pa_source_type": str(mapping.get("source_type", "")),
        "c2pa_validation": str(mapping.get("c2pa_validation_state", "")),
    }


def _marker_hits(region: bytes) -> list[str]:
    """Return high-precision marker labels found in one metadata region."""
    folded = region.lower()
    hits = {label for label, lowered in _LOWERED_MARKERS if lowered in folded}
    if c2pa_marker_in(region):
        hits.add("C2PA/JUMBF")
    if aigc_label_from_metadata(region) is not None:
        hits.add("TC260 AIGC")
    if app_generator_from_metadata(region) is not None:
        hits.add("App AIGC disclosure")
    if any(marker in region for marker in IPTC_AI_MARKERS + IPTC_AI_FIELD_MARKERS):
        hits.add("IPTC AI disclosure")
    if samsung_genai_in(region) is not None:
        hits.add("Samsung genAIType")
    if xai_signature_in_metadata(region):
        hits.add("xAI signature pair")
    if KLING_SEI_TAG in region:
        hits.add("Kling SEI tag")
    return sorted(hits)


def _marker_hits_for_path(path: Path) -> list[str]:
    """Return bounded metadata hits without turning an unreadable row into a fatal scan."""
    try:
        return _marker_hits(identify_metadata_region(scan_head(path)))
    except OSError:
        return []


def _candidate_classes(rep: Any, hits: Sequence[str]) -> list[str]:
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
    return corpus_day(path, corpus)


def _files(corpus: Path, since: date | None) -> list[Path]:
    """Return the deterministic corpus walk, optionally bounded by its date segment."""
    return corpus_files(corpus, since)


def _audio_platform(markers: dict[str, str]) -> str:
    from remove_ai_watermarks.video import _platform_from_video_metadata

    if platform := _platform_from_video_metadata(markers):
        return platform
    for key in ("audio_generator", "exif_generator"):
        if value := markers.get(key, "").strip():
            return value
    return ""


def _media_marker_row(
    *,
    path: Path,
    relative_path: str,
    suffix: str,
    lib_version: str,
    is_ai: bool | None,
    platform: str | None,
    confidence: str,
    metadata_markers: Mapping[str, object],
) -> dict[str, str]:
    from types import SimpleNamespace

    markers_str = "|".join(metadata_markers.keys())
    row = {
        **_base_row(relative_path, suffix, lib_version),
        "is_ai": str(is_ai),
        "platform": platform or "",
        "confidence": confidence,
        "watermarks": markers_str,
        "signals": markers_str,
        **_c2pa_fields(metadata_markers),
    }
    hits = _marker_hits_for_path(path)
    ai_signals = sorted(set(metadata_markers.keys()) - _STRUCTURAL_MEDIA_METADATA_KEYS)
    summary = SimpleNamespace(
        is_ai_generated=is_ai,
        platform=platform or None,
        signals=ai_signals,
        integrity_clashes=(),
    )
    row["markers"] = "|".join(hits)
    row["candidate_classes"] = "|".join(_candidate_classes(summary, hits))
    return row


def _scan_one(args: tuple[str, str, str]) -> dict[str, str]:
    """Scan one path in a worker process and always return a report row."""
    from types import SimpleNamespace

    from remove_ai_watermarks._internal.c2pa import markers_claim_ai
    from remove_ai_watermarks._internal.constants import AUDIO_EXTENSIONS
    from remove_ai_watermarks.audio import identify_audio
    from remove_ai_watermarks.text import TEXT_EXTENSIONS, identify_text
    from remove_ai_watermarks.video import VIDEO_EXTENSIONS, identify_video

    path_str, relative_path, lib_version = args
    path = Path(path_str)
    suffix = path.suffix.lower()
    try:
        if suffix in VIDEO_EXTENSIONS:
            vrep = identify_video(path, check_visible=False)
            return _media_marker_row(
                path=path,
                relative_path=relative_path,
                suffix=suffix,
                lib_version=lib_version,
                is_ai=vrep.is_ai_generated,
                platform=vrep.platform,
                confidence=vrep.confidence,
                metadata_markers=vrep.metadata_markers,
            )

        if suffix in AUDIO_EXTENSIONS:
            arep = identify_audio(path, check_invisible=False, scheme="auto")
            is_ai = True if markers_claim_ai(arep.metadata_markers) else None
            platform = _audio_platform(arep.metadata_markers)
            return _media_marker_row(
                path=path,
                relative_path=relative_path,
                suffix=suffix,
                lib_version=lib_version,
                is_ai=is_ai,
                platform=platform,
                confidence="high" if is_ai else "none",
                metadata_markers=arep.metadata_markers,
            )

        if suffix in TEXT_EXTENSIONS:
            trep = identify_text(path)
            signals_str = "|".join(trep.signals)
            markers_str = "|".join(trep.metadata_markers.keys())
            conf = "none" if trep.confidence == "unknown" else trep.confidence
            row = {
                **_base_row(relative_path, suffix, lib_version),
                "is_ai": str(trep.is_ai_generated),
                "platform": trep.platform or "",
                "confidence": conf,
                "watermarks": markers_str or signals_str,
                "signals": signals_str,
                **_c2pa_fields(trep.c2pa_info),
            }
            try:
                with path.open("rb") as handle:
                    hits = _marker_hits(handle.read(1024 * 1024))
            except OSError:
                hits = []
            text_summary = SimpleNamespace(
                is_ai_generated=trep.is_ai_generated,
                platform=trep.platform or ("text-provenance" if trep.signals else None),
                signals=trep.signals,
                integrity_clashes=(),
            )
            row["markers"] = "|".join(hits)
            row["candidate_classes"] = "|".join(_candidate_classes(text_summary, hits))
            return row

        evidence = extract_provenance_evidence(path)
        rep = identify_from_evidence(evidence)
    except Exception as exc:
        log.warning("identify failed on %s: %s", relative_path, exc)
        row = _base_row(relative_path, suffix, lib_version)
        row["candidate_classes"] = "identify_error"
        row["error"] = f"{type(exc).__name__}: {exc}"[:300].replace("\n", " ")
        row["markers"] = "|".join(_marker_hits_for_path(path))
        return row

    row = _row(rep, path=relative_path, suffix=suffix, lib_version=lib_version)
    row.update(_c2pa_fields(evidence.c2pa_info))
    hits = _marker_hits(identify_metadata_region(evidence.scan))
    row["markers"] = "|".join(hits)
    row["candidate_classes"] = "|".join(_candidate_classes(rep, hits))
    return row


def _scan_batch(args: tuple[tuple[str, str, str], ...]) -> list[dict[str, str]]:
    """Scan a small path batch to amortize process-pool dispatch overhead."""
    return [_scan_one(item) for item in args]


def _read_checkpoint(path: Path) -> dict[str, dict[str, str]]:
    """Read complete JSONL rows, tolerating an interrupted final write."""
    rows: dict[str, dict[str, str]] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if (
                isinstance(row, dict)
                and row.get("report_schema_version") == _REPORT_SCHEMA_VERSION
                and isinstance(row.get("path"), str)
            ):
                rows[row["path"]] = {field: str(row.get(field, "")) for field in REPORT_FIELDS}
    return rows


def _repair_checkpoint(path: Path) -> None:
    """Remove an interrupted final JSONL fragment before another append."""
    repair_jsonl_tail(path)


def _summarize(rows: list[dict[str, str]]) -> None:
    """Print bounded aggregate output for a completed or resumed report."""
    verdicts: Counter[str] = Counter()
    platforms: Counter[str] = Counter()
    shapes: Counter[str] = Counter()
    errors = 0
    for row in rows:
        shapes[row["suffix"] or "(none)"] += 1
        if row["error"]:
            errors += 1
        elif row["is_ai"] == "True":
            verdicts["ai"] += 1
            platforms[row["platform"] or "?"] += 1
        else:
            verdicts["unknown"] += 1

    console.print(f"\n[bold]Verdicts:[/bold] AI {verdicts['ai']} | unknown {verdicts['unknown']} | errors {errors}")
    console.print("[bold]Input shapes:[/bold] " + " | ".join(f"{name} {n}" for name, n in shapes.most_common()))
    plat = Table(title="AI platforms", show_header=False)
    for name, count in platforms.most_common():
        plat.add_row(str(count), name)
    console.print(plat)

    candidates = [row for row in rows if row["candidate_classes"]]
    if not candidates:
        console.print("\n[green]No review candidates in this run.[/green]")
        return
    candidate_counts = Counter(candidate for row in candidates for candidate in row["candidate_classes"].split("|"))
    gap_tokens = Counter(marker for row in rows for marker in row["markers"].split("|") if marker)
    console.print(
        f"\n[bold red]Review candidates[/bold red]: {len(candidates)} file(s) "
        f"({', '.join(f'{name}={n}' for name, n in candidate_counts.most_common())})"
    )
    tok = Table(title="metadata markers seen across the run")
    tok.add_column("count", justify="right")
    tok.add_column("marker")
    for name, count in gap_tokens.most_common():
        tok.add_row(str(count), name)
    console.print(tok)
    for row in candidates[:_DISPLAY_CANDIDATE_LIMIT]:
        detail = f"; markers={row['markers'].replace('|', ', ')}" if row["markers"] else ""
        console.print(f"  {row['path']}  ->  {row['candidate_classes'].replace('|', ', ')}{detail}")
    if (hidden := len(candidates) - _DISPLAY_CANDIDATE_LIMIT) > 0:
        console.print(f"  ... {hidden} more candidate row(s); inspect the CSV for the complete set")


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
@click.option("--workers", type=click.IntRange(min=1), default=max(1, (os.cpu_count() or 4) - 2), show_default=True)
@click.option("--restart", is_flag=True, help="Discard this report's checkpoint and scan every selected file again.")
def main(corpus: Path, report: Path | None, since, limit: int, workers: int, restart: bool) -> None:  # noqa: ANN001
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    report = report or corpus.parent / "detector_report.csv"

    since_date = since.date() if since is not None else None
    files = _files(corpus, since_date)
    if limit:
        files = files[:limit]
    checkpoint = report.with_suffix(report.suffix + ".progress.jsonl")
    if restart:
        checkpoint.unlink(missing_ok=True)
        report.unlink(missing_ok=True)
    _repair_checkpoint(checkpoint)
    lib_version = version("remove-ai-watermarks")
    selected = [(path, str(path.relative_to(corpus))) for path in files]
    selected_paths = {relative for _, relative in selected}
    completed = {relative: row for relative, row in _read_checkpoint(checkpoint).items() if relative in selected_paths}
    todo = [(path, relative) for path, relative in selected if relative not in completed]
    console.print(
        f"Scanning [bold]{len(files)}[/bold] files under {corpus}: "
        f"[bold]{len(completed)}[/bold] checkpointed, [bold]{len(todo)}[/bold] remaining, workers {workers}"
    )

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint.open("a", encoding="utf-8") as stream, ProcessPoolExecutor(max_workers=workers) as executor:
        work = iter((str(path), relative, lib_version) for path, relative in todo)

        def submit_batch() -> Future[list[dict[str, str]]] | None:
            batch = tuple(itertools.islice(work, _WORKER_BATCH_SIZE))
            return executor.submit(_scan_batch, batch) if batch else None

        pending = {future for _ in range(workers * 2) if (future := submit_batch()) is not None}
        processed = 0
        next_progress = 500
        while pending:
            ready, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in ready:
                for row in future.result():
                    completed[row["path"]] = row
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    processed += 1
                if replacement := submit_batch():
                    pending.add(replacement)
            stream.flush()
            if processed >= next_progress:
                console.print(
                    f"  {processed}/{len(todo)} new; {len(completed)}/{len(files)} total",
                    highlight=False,
                )
                next_progress += 500

    rows = [completed[relative] for _, relative in selected]
    report.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = report.with_suffix(report.suffix + ".tmp")
    with temporary_report.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary_report.replace(report)
    console.print(f"\nWrote [bold]{len(rows)}[/bold] rows -> {report}")
    _summarize(rows)


if __name__ == "__main__":
    main()
