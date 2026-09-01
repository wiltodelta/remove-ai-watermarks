"""Build and verify an immutable confirmatory batch for a SynthID oracle.

The batch contains a referenced untouched source plus four lossless PNG views:
an exact-pixel re-encode, aligned periodic-tile subtraction, a cyclically
shifted tile, and an orthogonal random tile. Building the batch performs no
network requests. Oracle results remain empty until a separately authorized
submission records them.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import click
from PIL import Image
from synthid_periodic_tile_ablation import (
    candidate_quality,
    control_templates,
    phase_score_pixels,
    tile_score_pixels,
)
from synthid_periodic_tile_probe import PeriodicTileModel
from synthid_periodic_tile_probe import load_model as load_tile_model
from synthid_phase_carrier import PhaseCarrierModel
from synthid_phase_carrier import load_model as load_phase_model
from synthid_pixel_attack import load_rgb
from synthid_research_manifest import artifact_sha256, pixel_fingerprint
from synthid_tile_attack import subtract_tiled_template

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)

ROLE_ORDER = ("source", "reencode_control", "aligned", "shifted", "orthogonal_random")
DERIVATIVE_ROLES = ROLE_ORDER[1:]
FORMAT_VERSION = 1
SYNTHID_RESULTS = {"detected", "not_detected", "indeterminate", "refused"}
C2PA_RESULTS = {"present", "absent", "indeterminate", "unavailable"}


def _inside(path: Path, parent: Path) -> bool:
    """Return whether resolved PATH is inside resolved PARENT."""
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _json_float(value: float) -> float | None:
    """Return finite VALUE or None for JSON interoperability."""
    return value if math.isfinite(value) else None


def _write_png(path: Path, pixels: np.ndarray) -> None:
    """Write exact RGB PIXELS as a deterministic lossless PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f"refusing to overwrite oracle artifact: {path}")
    Image.fromarray(pixels, mode="RGB").save(path, format="PNG", compress_level=9)


def _score_row(
    pixels: np.ndarray,
    *,
    source: np.ndarray,
    tile_model: PeriodicTileModel,
    phase_model: PhaseCarrierModel,
    tile_threshold: float,
    phase_threshold: float,
    active_threshold: float,
) -> dict[str, object]:
    """Return local frozen scores and paired fidelity for PIXELS."""
    tile_score = tile_score_pixels(pixels, tile_model)
    phase_score, active_support = phase_score_pixels(pixels, phase_model)
    quality = candidate_quality(source, pixels)
    return {
        "tile_score": tile_score,
        "tile_accepted": tile_score >= tile_threshold,
        "phase_score": phase_score,
        "active_support": active_support,
        "phase_accepted": phase_score >= phase_threshold and active_support >= active_threshold,
        "residual_rms": quality["residual_rms"],
        "psnr_db": _json_float(quality["psnr_db"]),
        "ssim": quality["ssim"],
        "changed_pixel_fraction": quality["changed_pixel_fraction"],
    }


def _artifact_row(
    path: Path,
    *,
    role: str,
    source_id: str,
    in_batch: bool,
    batch_root: Path,
    scores: dict[str, object],
) -> dict[str, object]:
    """Return one hash-frozen manifest row for PATH."""
    pixel_sha256, width, height = pixel_fingerprint(path)
    return {
        "source_id": source_id,
        "role": role,
        "path": str(path.relative_to(batch_root) if in_batch else path.resolve()),
        "in_batch": in_batch,
        "artifact_sha256": artifact_sha256(path),
        "pixel_sha256": pixel_sha256,
        "width": width,
        "height": height,
        "synthid_result": None,
        "c2pa_result": None,
        "submitted_at": None,
        **scores,
    }


def build_batch(
    sources: list[Path],
    *,
    output_dir: Path,
    tile_model_path: Path,
    phase_model_path: Path,
    tile_threshold: float,
    phase_threshold: float,
    active_threshold: float,
    strength: float,
    seed: int,
    provider: str,
    repository_root: Path,
) -> Path:
    """Build a preregistered, unsubmitted oracle batch and return its manifest."""
    if not sources:
        raise ValueError("at least one source is required")
    if strength <= 0.0 or not math.isfinite(strength):
        raise ValueError("strength must be finite and positive")
    if provider not in {"google", "openai"}:
        raise ValueError("provider must be google or openai")
    if _inside(output_dir, repository_root):
        raise ValueError("oracle batches must be written outside the repository")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output directory must not already contain files")
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_model = load_tile_model(tile_model_path)
    phase_model = load_phase_model(phase_model_path)
    if (tile_model.height, tile_model.width) != (phase_model.height, phase_model.width):
        raise ValueError("tile and phase model geometries differ")
    templates = control_templates(tile_model.template, seed=seed)

    rows: list[dict[str, object]] = []
    seen_source_hashes: set[str] = set()
    for source_path in sources:
        source_hash = artifact_sha256(source_path)
        if source_hash in seen_source_hashes:
            raise ValueError("duplicate source artifact")
        seen_source_hashes.add(source_hash)
        source = load_rgb(source_path)
        if source.shape != (tile_model.height, tile_model.width, 3):
            raise ValueError(f"{source_path}: geometry does not match the frozen models")
        source_id = source_hash[:16]
        score_args = {
            "source": source,
            "tile_model": tile_model,
            "phase_model": phase_model,
            "tile_threshold": tile_threshold,
            "phase_threshold": phase_threshold,
            "active_threshold": active_threshold,
        }
        source_scores = _score_row(source, **score_args)
        rows.append(
            _artifact_row(
                source_path,
                role="source",
                source_id=source_id,
                in_batch=False,
                batch_root=output_dir,
                scores=source_scores,
            )
        )
        for role in DERIVATIVE_ROLES:
            variant = (
                source
                if role == "reencode_control"
                else subtract_tiled_template(
                    source,
                    templates[role] * tile_model.expected_norm,
                    strength=strength,
                )
            )
            output_path = output_dir / source_id / f"{role}.png"
            _write_png(output_path, variant)
            rows.append(
                _artifact_row(
                    output_path,
                    role=role,
                    source_id=source_id,
                    in_batch=True,
                    batch_root=output_dir,
                    scores=(source_scores if role == "reencode_control" else _score_row(variant, **score_args)),
                )
            )

    manifest = {
        "format_version": FORMAT_VERSION,
        "status": "preregistered_unsubmitted",
        "source_count": len(sources),
        "request_count": len(rows),
        "request_order": ROLE_ORDER,
        "provider": provider,
        "decision_rule": (
            "Count causal success only when source, reencode_control, shifted, and "
            "orthogonal_random are detected in the matching provider SynthID oracle, "
            "aligned is not_detected, and indeterminate is never treated as negative."
        ),
        "strength": strength,
        "seed": seed,
        "tile_threshold": tile_threshold,
        "phase_threshold": phase_threshold,
        "active_threshold": active_threshold,
        "tile_model": str(tile_model_path.resolve()),
        "tile_model_sha256": artifact_sha256(tile_model_path),
        "phase_model": str(phase_model_path.resolve()),
        "phase_model_sha256": artifact_sha256(phase_model_path),
        "rows": rows,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    manifest_hash = artifact_sha256(manifest_path)
    (output_dir / "manifest.sha256").write_text(manifest_hash + "  manifest.json\n", encoding="utf-8")
    results_template = {
        "format_version": FORMAT_VERSION,
        "manifest_sha256": manifest_hash,
        "provider": provider,
        "rows": [
            {
                "source_id": row["source_id"],
                "role": row["role"],
                "artifact_sha256": row["artifact_sha256"],
                "synthid_result": None,
                "c2pa_result": None,
                "raw_response": None,
                "submitted_at": None,
            }
            for row in rows
        ],
    }
    (output_dir / "results-template.json").write_text(
        json.dumps(results_template, indent=2) + "\n",
        encoding="utf-8",
    )
    verify_batch(manifest_path, repository_root=repository_root)
    return manifest_path


def verify_batch(manifest_path: Path, *, repository_root: Path) -> dict[str, object]:
    """Verify manifest, model, source, and derivative hashes without mutation."""
    batch_root = manifest_path.parent
    if _inside(batch_root, repository_root):
        raise ValueError("oracle batches must remain outside the repository")
    digest_path = batch_root / "manifest.sha256"
    expected_manifest_hash = digest_path.read_text(encoding="utf-8").split()[0]
    if artifact_sha256(manifest_path) != expected_manifest_hash:
        raise ValueError("manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported oracle-batch manifest version")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != manifest.get("request_count"):
        raise ValueError("manifest request count mismatch")
    if artifact_sha256(Path(manifest["tile_model"])) != manifest["tile_model_sha256"]:
        raise ValueError("tile model hash mismatch")
    if artifact_sha256(Path(manifest["phase_model"])) != manifest["phase_model_sha256"]:
        raise ValueError("phase model hash mismatch")

    groups: dict[str, list[str]] = {}
    artifact_hashes: set[str] = set()
    for row in rows:
        source_id = str(row["source_id"])
        groups.setdefault(source_id, []).append(str(row["role"]))
        path = batch_root / str(row["path"]) if row["in_batch"] else Path(str(row["path"]))
        if artifact_sha256(path) != row["artifact_sha256"]:
            raise ValueError(f"artifact hash mismatch for {source_id}/{row['role']}")
        pixel_sha256, width, height = pixel_fingerprint(path)
        if (pixel_sha256, width, height) != (row["pixel_sha256"], row["width"], row["height"]):
            raise ValueError(f"pixel fingerprint mismatch for {source_id}/{row['role']}")
        artifact_hashes.add(str(row["artifact_sha256"]))
    if any(tuple(roles) != ROLE_ORDER for roles in groups.values()):
        raise ValueError("each source must have the fixed request-role order")
    if len(groups) != manifest.get("source_count"):
        raise ValueError("manifest source count mismatch")
    if len(artifact_hashes) < len(rows) - len(groups):
        raise ValueError("unexpected duplicate derivative artifacts")
    template = json.loads((batch_root / "results-template.json").read_text(encoding="utf-8"))
    expected_identities = [(row["source_id"], row["role"], row["artifact_sha256"]) for row in rows]
    template_identities = [(row["source_id"], row["role"], row["artifact_sha256"]) for row in template.get("rows", [])]
    if template.get("manifest_sha256") != expected_manifest_hash or template.get("provider") != manifest["provider"]:
        raise ValueError("results template does not identify the preregistered batch")
    if template_identities != expected_identities:
        raise ValueError("results template row identities differ from the manifest")
    return manifest


def _parse_submitted_at(value: object) -> None:
    """Reject timestamps that are absent or not timezone-aware ISO-8601."""
    if not isinstance(value, str):
        raise ValueError("submitted_at must be a timezone-aware ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("submitted_at must be a timezone-aware ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError("submitted_at must be a timezone-aware ISO-8601 timestamp")


def evaluate_results(
    manifest_path: Path,
    results_path: Path,
    *,
    repository_root: Path,
) -> dict[str, object]:
    """Validate a complete result file and apply the preregistered decision rule."""
    manifest = verify_batch(manifest_path, repository_root=repository_root)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    manifest_hash = artifact_sha256(manifest_path)
    if results.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported oracle-results version")
    if results.get("manifest_sha256") != manifest_hash or results.get("provider") != manifest["provider"]:
        raise ValueError("results do not identify the preregistered batch")
    manifest_rows = manifest["rows"]
    result_rows = results.get("rows")
    if not isinstance(result_rows, list) or len(result_rows) != len(manifest_rows):
        raise ValueError("oracle results must cover every preregistered request")

    grouped: dict[str, dict[str, str]] = {}
    for expected, result in zip(manifest_rows, result_rows, strict=True):
        identity = (result.get("source_id"), result.get("role"), result.get("artifact_sha256"))
        expected_identity = (expected["source_id"], expected["role"], expected["artifact_sha256"])
        if identity != expected_identity:
            raise ValueError("oracle result order or artifact identity differs from the manifest")
        synthid_result = result.get("synthid_result")
        c2pa_result = result.get("c2pa_result")
        if synthid_result not in SYNTHID_RESULTS:
            raise ValueError("invalid or missing SynthID result")
        if c2pa_result not in C2PA_RESULTS:
            raise ValueError("invalid or missing C2PA result")
        if not isinstance(result.get("raw_response"), str) or not result["raw_response"].strip():
            raise ValueError("raw_response must preserve the nonempty verbatim oracle result")
        _parse_submitted_at(result.get("submitted_at"))
        grouped.setdefault(str(expected["source_id"]), {})[str(expected["role"])] = str(synthid_result)

    source_results: list[dict[str, str]] = []
    for source_id, roles in grouped.items():
        if tuple(roles) != ROLE_ORDER:
            raise ValueError("oracle results do not preserve the fixed role order")
        values = set(roles.values())
        if values & {"indeterminate", "refused"}:
            verdict = "indeterminate"
        elif (
            roles["source"] == "detected"
            and roles["reencode_control"] == "detected"
            and roles["shifted"] == "detected"
            and roles["orthogonal_random"] == "detected"
            and roles["aligned"] == "not_detected"
        ):
            verdict = "causal_success"
        elif any(roles[role] != "detected" for role in ("source", "reencode_control", "shifted", "orthogonal_random")):
            verdict = "control_failed"
        else:
            verdict = "aligned_still_detected"
        source_results.append({"source_id": source_id, "verdict": verdict})
    counts = {
        verdict: sum(row["verdict"] == verdict for row in source_results)
        for verdict in ("causal_success", "aligned_still_detected", "control_failed", "indeterminate")
    }
    return {
        "format_version": FORMAT_VERSION,
        "manifest_sha256": manifest_hash,
        "provider": manifest["provider"],
        "source_count": manifest["source_count"],
        "counts": counts,
        "sources": source_results,
    }


@click.group()
def main() -> None:
    """Build and verify an immutable confirmatory oracle batch."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


@main.command("build")
@click.argument("tile_model_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("phase_model_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("sources", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--tile-threshold", type=float, required=True)
@click.option("--phase-threshold", type=float, required=True)
@click.option("--active-threshold", type=float, required=True)
@click.option("--strength", type=click.FloatRange(min=0.0, min_open=True), default=2.0, show_default=True)
@click.option("--seed", type=int, default=20260810, show_default=True)
@click.option("--provider", type=click.Choice(["google", "openai"]), required=True)
def build_command(
    tile_model_path: Path,
    phase_model_path: Path,
    sources: tuple[Path, ...],
    output_dir: Path,
    tile_threshold: float,
    phase_threshold: float,
    active_threshold: float,
    strength: float,
    seed: int,
    provider: str,
) -> None:
    """Build a frozen batch from exact-geometry SOURCES."""
    manifest = build_batch(
        list(sources),
        output_dir=output_dir,
        tile_model_path=tile_model_path,
        phase_model_path=phase_model_path,
        tile_threshold=tile_threshold,
        phase_threshold=phase_threshold,
        active_threshold=active_threshold,
        strength=strength,
        seed=seed,
        provider=provider,
        repository_root=Path(__file__).resolve().parent.parent,
    )
    log.info("Wrote preregistered oracle batch: %s", manifest)


@main.command("verify")
@click.argument("manifest_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def verify_command(manifest_path: Path) -> None:
    """Verify an existing batch without changing it."""
    manifest = verify_batch(manifest_path, repository_root=Path(__file__).resolve().parent.parent)
    log.info("Verified %d immutable oracle requests", manifest["request_count"])


@main.command("evaluate")
@click.argument("manifest_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("results_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--report-out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def evaluate_command(manifest_path: Path, results_path: Path, report_out: Path) -> None:
    """Validate RESULTS_PATH and write the preregistered batch verdict."""
    repository_root = Path(__file__).resolve().parent.parent
    if _inside(report_out, repository_root):
        raise click.BadParameter("oracle result reports must be written outside the repository")
    report = evaluate_results(
        manifest_path,
        results_path,
        repository_root=repository_root,
    )
    report_out.parent.mkdir(parents=True, exist_ok=True)
    if report_out.exists():
        raise click.BadParameter("refusing to overwrite an oracle result report")
    report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log.info("Wrote oracle batch verdict: %s", report_out)


if __name__ == "__main__":
    main()
