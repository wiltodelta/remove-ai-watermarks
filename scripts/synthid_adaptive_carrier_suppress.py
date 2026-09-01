"""Suppress the recovered periodic carrier without image regeneration.

This research tool controls the project's local fixed-template score. A local
score reversal is not evidence that a provider SynthID verifier will stop
detecting the image.
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import click
from PIL import Image
from synthid_pixel_attack import load_rgb, measure  # pyright: ignore[reportUnknownVariableType]
from synthid_research_manifest import artifact_sha256
from synthid_runtime.synthid_detector import (
    TILE_THRESHOLD,
    _geometry_supported,  # pyright: ignore[reportPrivateUsage]
    _load_template,  # pyright: ignore[reportPrivateUsage]
    folded_template_score,
)
from synthid_tile_attack import subtract_tiled_template

if TYPE_CHECKING:
    from numpy.typing import NDArray

log = logging.getLogger(__name__)


def apply_template(pixels: NDArray[Any], template: NDArray[Any], *, amplitude: float) -> NDArray[Any]:
    """Subtract AMPLITUDE times periodic TEMPLATE from arbitrary RGB PIXELS."""
    return subtract_tiled_template(pixels, template, strength=amplitude)


def carrier_score(pixels: NDArray[Any], template: NDArray[Any], sigma: float) -> float:
    """Return the local fixed-template carrier score for PIXELS."""
    score, _folded = folded_template_score(pixels, template, sigma)
    return score


def find_minimum_amplitude(
    pixels: NDArray[Any],
    template: NDArray[Any],
    sigma: float,
    *,
    target_score: float,
    maximum_amplitude: float,
    iterations: int,
) -> tuple[float, NDArray[Any], float]:
    """Return the smallest searched amplitude whose score reaches TARGET_SCORE."""
    if not math.isfinite(target_score):
        raise ValueError("target score must be finite")
    if not math.isfinite(maximum_amplitude) or maximum_amplitude <= 0.0:
        raise ValueError("maximum amplitude must be finite and positive")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    maximum_pixels = apply_template(pixels, template, amplitude=maximum_amplitude)
    maximum_score = carrier_score(maximum_pixels, template, sigma)
    if maximum_score > target_score:
        raise ValueError(
            f"maximum amplitude {maximum_amplitude:g} reached score {maximum_score:.6f}, "
            f"above target {target_score:.6f}"
        )

    low = 0.0
    high = maximum_amplitude
    best_pixels = maximum_pixels
    best_score = maximum_score
    for _iteration in range(iterations):
        middle = (low + high) / 2.0
        candidate = apply_template(pixels, template, amplitude=middle)
        candidate_score = carrier_score(candidate, template, sigma)
        if candidate_score <= target_score:
            high = middle
            best_pixels = candidate
            best_score = candidate_score
        else:
            low = middle
    return high, best_pixels, best_score


def suppress_carrier(
    pixels: NDArray[Any],
    *,
    target_score: float = -0.25,
    maximum_amplitude: float = 40.0,
    iterations: int = 8,
) -> tuple[NDArray[Any], dict[str, float | int | str]]:
    """Suppress a locally detected carrier and return pixels plus measurements."""
    height, width = pixels.shape[:2]
    if not _geometry_supported(width, height):
        raise ValueError(f"unsupported decoded geometry: {width}x{height}")
    if target_score >= TILE_THRESHOLD:
        raise ValueError(f"target score must be below the detector threshold {TILE_THRESHOLD:.6f}")
    template, sigma, _model_height, _model_width, tile_height, tile_width = _load_template()
    original_score = carrier_score(pixels, template, sigma)
    if original_score < TILE_THRESHOLD:
        raise ValueError(
            f"local carrier is not detected: score {original_score:.6f} is below threshold {TILE_THRESHOLD:.6f}"
        )

    started = time.perf_counter()
    amplitude, candidate, candidate_score = find_minimum_amplitude(
        pixels,
        template,
        sigma,
        target_score=target_score,
        maximum_amplitude=maximum_amplitude,
        iterations=iterations,
    )
    quality = measure(pixels, candidate, name="adaptive-carrier", path=Path("<memory>"))
    return candidate, {
        "status": "local_carrier_suppressed",
        "detector_scope": "local fixed-template carrier, not provider-verified SynthID removal",
        "width": width,
        "height": height,
        "tile_height": tile_height,
        "tile_width": tile_width,
        "threshold": TILE_THRESHOLD,
        "target_score": target_score,
        "original_score": original_score,
        "candidate_score": candidate_score,
        "amplitude": amplitude,
        "maximum_amplitude": maximum_amplitude,
        "iterations": iterations,
        "residual_rms": quality.residual_rms,
        "psnr_db": quality.psnr_db,
        "ssim": quality.ssim,
        "changed_pixel_fraction": quality.changed_pixel_fraction,
        "elapsed_seconds": time.perf_counter() - started,
    }


@click.command()
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--target-score", type=float, default=-0.25, show_default=True)
@click.option("--maximum-amplitude", type=click.FloatRange(min=0.0, min_open=True), default=40.0, show_default=True)
@click.option("--iterations", type=click.IntRange(min=1), default=8, show_default=True)
@click.option("--report-out", type=click.Path(dir_okay=False, path_type=Path))
def main(
    source: Path,
    output: Path,
    target_score: float,
    maximum_amplitude: float,
    iterations: int,
    report_out: Path | None,
) -> None:
    """Write a lossless PNG with the recovered local carrier suppressed."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if output.suffix.lower() != ".png":
        raise click.BadParameter("output must use the .png extension", param_hint="output")
    report_path = report_out or output.with_suffix(".json")
    for path in (output, report_path):
        if path.exists():
            raise click.ClickException(f"refusing to overwrite existing file: {path}")
    try:
        candidate, report = suppress_carrier(
            load_rgb(source),  # pyright: ignore[reportUnknownArgumentType]
            target_score=target_score,
            maximum_amplitude=maximum_amplitude,
            iterations=iterations,
        )
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(candidate, mode="RGB").save(output, format="PNG", compress_level=9)
    report.update(
        {
            "source": str(source.resolve()),
            "source_sha256": artifact_sha256(source),
            "output": str(output.resolve()),
            "output_sha256": artifact_sha256(output),
        }
    )
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    log.info(
        "Suppressed local carrier %.6f -> %.6f at %.2f dB PSNR; wrote %s",
        report["original_score"],
        report["candidate_score"],
        report["psnr_db"],
        output,
    )
    log.info("Research caveat: this is not provider-verified SynthID removal")


if __name__ == "__main__":
    main()
