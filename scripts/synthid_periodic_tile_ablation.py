"""Measure whether a frozen periodic tile causally controls local carrier scores.

This harness compares subtraction of the learned tile with cyclically shifted
and orthogonal random tiles of the same norm. It measures local research
detectors only. A score reversal is not evidence that a provider oracle would
stop detecting SynthID.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import click
import numpy as np
from synthid_periodic_tile import unit_tile
from synthid_periodic_tile_probe import PeriodicTileModel
from synthid_periodic_tile_probe import load_model as load_tile_model
from synthid_periodic_tile_probe import score_pixels as score_tile_pixels
from synthid_phase_carrier import PhaseCarrierModel, score_pixels
from synthid_phase_carrier import load_model as load_phase_model
from synthid_pixel_attack import load_rgb, measure
from synthid_research_manifest import artifact_sha256
from synthid_tile_attack import parse_positive_floats, subtract_tiled_template

log = logging.getLogger(__name__)


def exact_sign_test(negative: int, positive: int) -> float:
    """Return an exact two-sided sign-test p-value after excluding ties."""
    count = negative + positive
    if count == 0:
        return 1.0
    tail = sum(math.comb(count, index) for index in range(min(negative, positive) + 1)) / 2**count
    return min(1.0, 2.0 * tail)


def control_templates(template: np.ndarray, *, seed: int) -> dict[str, np.ndarray]:
    """Return aligned, shifted, and norm-matched orthogonal control tiles."""
    rng = np.random.default_rng(seed)
    random_tile = rng.normal(size=template.shape)
    random_tile -= np.mean(random_tile, axis=(0, 1), keepdims=True)
    random_tile -= np.sum(random_tile * template) * template
    random_tile, norm = unit_tile(random_tile)
    if norm == 0.0 or abs(float(np.sum(random_tile * template))) > 1e-12:
        raise ValueError("could not construct an orthogonal random control")
    return {
        "aligned": template,
        "shifted": np.roll(template, shift=(1, 1), axis=(0, 1)),
        "orthogonal_random": random_tile,
    }


def phase_score_pixels(pixels: np.ndarray, model: PhaseCarrierModel) -> tuple[float, float]:
    """Return the unregistered phase score and active support for PIXELS."""
    result = score_pixels(pixels, model)
    return result.score, result.active_weight_fraction


def tile_score_pixels(pixels: np.ndarray, model: PeriodicTileModel) -> float:
    """Return the fixed-phase periodic-tile score for PIXELS."""
    return score_tile_pixels(pixels, model).score


def summarize(values: list[float]) -> dict[str, float]:
    """Return bounded descriptive statistics for VALUES."""
    return {
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "maximum": float(np.max(values)),
    }


def direction_summary(values: list[float]) -> dict[str, float | int]:
    """Return direction counts and a two-sided sign test for VALUES."""
    negative = sum(value < 0.0 for value in values)
    positive = sum(value > 0.0 for value in values)
    return {
        "negative": negative,
        "positive": positive,
        "ties": len(values) - negative - positive,
        "two_sided_sign_p": exact_sign_test(negative, positive),
    }


def candidate_quality(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Return paired fidelity metrics for one equal-geometry candidate."""
    measurement = measure(reference, candidate, name="candidate", path=Path("<memory>"))
    return {
        "residual_rms": measurement.residual_rms,
        "psnr_db": measurement.psnr_db,
        "ssim": measurement.ssim,
        "changed_pixel_fraction": measurement.changed_pixel_fraction,
    }


def run_ablation(
    sources: list[Path],
    *,
    tile_model: PeriodicTileModel,
    phase_model: PhaseCarrierModel,
    tile_threshold: float,
    phase_threshold: float,
    active_threshold: float,
    strengths: tuple[float, ...],
    phase_strength: float,
    seed: int,
) -> dict[str, object]:
    """Evaluate aligned subtraction and controls on exact-geometry SOURCES."""
    if not sources:
        raise ValueError("at least one source is required")
    if phase_strength not in strengths:
        raise ValueError("phase strength must be one of the swept strengths")
    if (tile_model.height, tile_model.width) != (phase_model.height, phase_model.width):
        raise ValueError("tile and phase model geometries differ")
    if not all(np.isfinite(value) for value in (tile_threshold, phase_threshold, active_threshold)):
        raise ValueError("thresholds must be finite")

    templates = control_templates(tile_model.template, seed=seed)
    rows: list[dict[str, object]] = []
    for source_path in sources:
        source = load_rgb(source_path)
        if source.shape != (tile_model.height, tile_model.width, 3):
            raise ValueError(f"{source_path}: geometry does not match the models")
        source_hash = artifact_sha256(source_path)
        original_tile = tile_score_pixels(source, tile_model)
        original_phase, original_support = phase_score_pixels(source, phase_model)
        for strength in strengths:
            for control, template in templates.items():
                candidate = subtract_tiled_template(
                    source,
                    template * tile_model.expected_norm,
                    strength=strength,
                )
                tile_score = tile_score_pixels(candidate, tile_model)
                row: dict[str, object] = {
                    "path": str(source_path),
                    "artifact_sha256": source_hash,
                    "control": control,
                    "strength": strength,
                    "original_tile_score": original_tile,
                    "tile_score": tile_score,
                    "tile_delta": tile_score - original_tile,
                    "tile_accepted": tile_score >= tile_threshold,
                    "original_phase_score": original_phase,
                    "original_active_support": original_support,
                }
                if strength == phase_strength:
                    phase_score, active_support = phase_score_pixels(candidate, phase_model)
                    row.update(
                        {
                            **candidate_quality(source, candidate),
                            "phase_score": phase_score,
                            "active_support": active_support,
                            "phase_delta": phase_score - original_phase,
                            "phase_accepted": phase_score >= phase_threshold and active_support >= active_threshold,
                        }
                    )
                rows.append(row)

    grouped: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["control"]), float(row["strength"]))].append(row)
    tile_summaries: list[dict[str, object]] = []
    for (control, strength), group in sorted(grouped.items()):
        deltas = [float(row["tile_delta"]) for row in group]
        tile_summaries.append(
            {
                "control": control,
                "strength": strength,
                "accepted": sum(bool(row["tile_accepted"]) for row in group),
                "delta": summarize(deltas),
                "direction": direction_summary(deltas),
            }
        )

    selected = [row for row in rows if float(row["strength"]) == phase_strength]
    phase_summaries: list[dict[str, object]] = []
    for control in templates:
        group = [row for row in selected if row["control"] == control]
        phase_summaries.append(
            {
                "control": control,
                "accepted": sum(bool(row["phase_accepted"]) for row in group),
                "delta": summarize([float(row["phase_delta"]) for row in group]),
                "active_support": summarize([float(row["active_support"]) for row in group]),
                "psnr_db": summarize([float(row["psnr_db"]) for row in group]),
                "ssim": summarize([float(row["ssim"]) for row in group]),
                "changed_pixel_fraction": summarize([float(row["changed_pixel_fraction"]) for row in group]),
            }
        )

    paired_comparisons: list[dict[str, object]] = []
    for control in ("shifted", "orthogonal_random"):
        for metric in ("tile_delta", "phase_delta"):
            aligned = {
                str(row["artifact_sha256"]): float(row[metric]) for row in selected if row["control"] == "aligned"
            }
            comparison = {
                str(row["artifact_sha256"]): float(row[metric]) for row in selected if row["control"] == control
            }
            differences = [aligned[key] - comparison[key] for key in sorted(aligned)]
            paired_comparisons.append(
                {
                    "aligned_minus": control,
                    "metric": metric,
                    "difference": summarize(differences),
                    "direction": direction_summary(differences),
                }
            )

    return {
        "source_count": len(sources),
        "tile_threshold": tile_threshold,
        "phase_threshold": phase_threshold,
        "active_threshold": active_threshold,
        "strengths": strengths,
        "phase_strength": phase_strength,
        "seed": seed,
        "original": {
            "tile_accepted": sum(
                float(row["original_tile_score"]) >= tile_threshold for row in selected if row["control"] == "aligned"
            ),
            "phase_accepted": sum(
                float(row["original_phase_score"]) >= phase_threshold
                and float(row["original_active_support"]) >= active_threshold
                for row in selected
                if row["control"] == "aligned"
            ),
        },
        "tile_summaries": tile_summaries,
        "phase_summaries": phase_summaries,
        "paired_comparisons": paired_comparisons,
        "items": rows,
    }


@click.command()
@click.argument("tile_model_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("phase_model_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("sources", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--tile-threshold", type=float, required=True)
@click.option("--phase-threshold", type=float, required=True)
@click.option("--active-threshold", type=float, required=True)
@click.option("--strengths", default="1,1.5,2,3,4", show_default=True)
@click.option("--phase-strength", type=click.FloatRange(min=0.0, min_open=True), default=2.0, show_default=True)
@click.option("--seed", type=int, default=20260810, show_default=True)
@click.option("--report-out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(
    tile_model_path: Path,
    phase_model_path: Path,
    sources: tuple[Path, ...],
    tile_threshold: float,
    phase_threshold: float,
    active_threshold: float,
    strengths: str,
    phase_strength: float,
    seed: int,
    report_out: Path,
) -> None:
    """Run a fixed periodic-tile causal ablation on exact-geometry SOURCES."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    strength_values = parse_positive_floats(strengths, option_name="strengths")
    report = run_ablation(
        list(sources),
        tile_model=load_tile_model(tile_model_path),
        phase_model=load_phase_model(phase_model_path),
        tile_threshold=tile_threshold,
        phase_threshold=phase_threshold,
        active_threshold=active_threshold,
        strengths=strength_values,
        phase_strength=phase_strength,
        seed=seed,
    )
    report["tile_model"] = str(tile_model_path)
    report["tile_model_sha256"] = artifact_sha256(tile_model_path)
    report["phase_model"] = str(phase_model_path)
    report["phase_model_sha256"] = artifact_sha256(phase_model_path)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log.info("Wrote periodic-tile causal ablation: %s", report_out)


if __name__ == "__main__":
    main()
