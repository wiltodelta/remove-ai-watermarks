"""Build non-generative spatial-fragmentation SynthID attack candidates.

The candidates combine deterministic smooth local warps with mild global
resampling, color changes, and codec round-trips. These are pixel transforms,
not semantic reconstruction or generative inpainting. Provider-oracle results
must be evaluated in a frozen batch with a source-positive control.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import click
import cv2
import numpy as np
from PIL import Image
from synthid_pixel_attack import (
    jpeg_round_trip,
    load_rgb,
    measure,
    norm_matched_noise,
    resize_squeeze,
    smooth_warp,
)
from synthid_v3_codebook_probe import load_v3_model, score_image

log = logging.getLogger(__name__)


def bounded_smooth_warp(
    pixels: np.ndarray,
    *,
    max_displacement: float,
    sigma: float,
    seed: int,
) -> np.ndarray:
    """Apply a smooth warp whose per-axis displacement is absolutely bounded."""
    if max_displacement < 0.0 or sigma <= 0.0:
        raise ValueError("max_displacement must be nonnegative and sigma positive")
    height, width = pixels.shape[:2]
    rng = np.random.default_rng(seed)
    fields: list[np.ndarray] = []
    for _ in range(2):
        noise = rng.normal(size=(height, width)).astype(np.float32)
        field = cv2.GaussianBlur(noise, (0, 0), sigmaX=sigma, sigmaY=sigma)
        maximum = float(np.max(np.abs(field)))
        fields.append(np.zeros_like(field) if maximum == 0.0 else field * (max_displacement / maximum))
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    return cv2.remap(
        pixels,
        xx + fields[0],
        yy + fields[1],
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def affine_combo(pixels: np.ndarray, *, rotation_degrees: float, zoom: float) -> np.ndarray:
    """Apply one centered rotation-and-zoom resampling operation."""
    if zoom < 0.0:
        raise ValueError("zoom must be nonnegative")
    height, width = pixels.shape[:2]
    matrix = cv2.getRotationMatrix2D(
        center=((width - 1) / 2.0, (height - 1) / 2.0),
        angle=rotation_degrees,
        scale=1.0 + zoom,
    )
    return cv2.warpAffine(
        pixels,
        matrix,
        (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def color_nudge(
    pixels: np.ndarray,
    *,
    brightness: float,
    contrast: float,
    saturation: float,
    hue_degrees: float,
) -> np.ndarray:
    """Apply bounded global RGB contrast and HSV saturation/hue changes."""
    rgb = pixels.astype(np.float32) / 255.0
    rgb = np.clip((rgb - 0.5) * (1.0 + contrast) + 0.5 + brightness, 0.0, 1.0)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hsv[:, :, 0] = np.mod(hsv[:, :, 0] + hue_degrees, 360.0)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + saturation), 0.0, 1.0)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return np.clip(np.rint(result * 255.0), 0, 255).astype(np.uint8)


def jpeg_chain(pixels: np.ndarray, qualities: tuple[int, ...]) -> np.ndarray:
    """Apply sequential JPEG round-trips at QUALITIES."""
    result = pixels
    for quality in qualities:
        result = jpeg_round_trip(result, quality)
    return result


def build_candidates(source: np.ndarray) -> dict[str, np.ndarray]:
    """Build the frozen spatial-fragmentation ladder for SOURCE."""
    candidates: dict[str, np.ndarray] = {
        "control": source.copy(),
        "elastic-075": smooth_warp(source, amplitude=0.75, sigma=56.0, seed=20260812),
        "elastic-125": smooth_warp(source, amplitude=1.25, sigma=52.0, seed=20260813),
        "bounded-100": bounded_smooth_warp(
            source,
            max_displacement=1.0,
            sigma=56.0,
            seed=20260817,
        ),
        "bounded-180": bounded_smooth_warp(
            source,
            max_displacement=1.8,
            sigma=56.0,
            seed=20260818,
        ),
        "bounded-280": bounded_smooth_warp(
            source,
            max_displacement=2.8,
            sigma=44.0,
            seed=20260819,
        ),
    }

    balanced = smooth_warp(source, amplitude=0.75, sigma=56.0, seed=20260814)
    balanced = affine_combo(balanced, rotation_degrees=0.2, zoom=0.004)
    balanced = resize_squeeze(balanced, 0.94)
    balanced = color_nudge(
        balanced,
        brightness=0.004,
        contrast=0.006,
        saturation=-0.005,
        hue_degrees=0.15,
    )
    balanced = jpeg_chain(balanced, (94, 90))
    candidates["fragment-balanced"] = balanced

    strong = smooth_warp(source, amplitude=1.5, sigma=48.0, seed=20260815)
    strong = affine_combo(strong, rotation_degrees=0.4, zoom=0.01)
    strong = resize_squeeze(strong, 0.88)
    strong = color_nudge(
        strong,
        brightness=0.008,
        contrast=0.012,
        saturation=-0.01,
        hue_degrees=0.3,
    )
    strong = jpeg_chain(strong, (92, 88))
    candidates["fragment-strong"] = strong
    candidates["sham-strong-rms"] = norm_matched_noise(source, strong, seed=20260816)

    bounded_balanced = bounded_smooth_warp(
        source,
        max_displacement=1.8,
        sigma=56.0,
        seed=20260820,
    )
    bounded_balanced = resize_squeeze(bounded_balanced, 0.98)
    bounded_balanced = color_nudge(
        bounded_balanced,
        brightness=0.002,
        contrast=0.003,
        saturation=-0.003,
        hue_degrees=0.1,
    )
    bounded_balanced = jpeg_chain(bounded_balanced, (96,))
    candidates["bounded-fragment-balanced"] = bounded_balanced

    bounded_strong = bounded_smooth_warp(
        source,
        max_displacement=2.8,
        sigma=44.0,
        seed=20260821,
    )
    bounded_strong = affine_combo(bounded_strong, rotation_degrees=0.2, zoom=0.004)
    bounded_strong = resize_squeeze(bounded_strong, 0.94)
    bounded_strong = color_nudge(
        bounded_strong,
        brightness=0.004,
        contrast=0.006,
        saturation=-0.005,
        hue_degrees=0.15,
    )
    bounded_strong = jpeg_chain(bounded_strong, (94, 90))
    candidates["bounded-fragment-strong"] = bounded_strong
    candidates["sham-bounded-strong-rms"] = norm_matched_noise(source, bounded_strong, seed=20260822)
    return candidates


@click.command()
@click.argument("codebook", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option("--height", type=click.IntRange(min=64), required=True)
@click.option("--width", type=click.IntRange(min=64), required=True)
@click.option("--peak-count", type=click.IntRange(min=1), default=256, show_default=True)
def main(codebook: Path, source: Path, output_dir: Path, height: int, width: int, peak_count: int) -> None:
    """Write a frozen spatial-fragmentation batch for SOURCE."""
    reference = load_rgb(source)
    if reference.shape != (height, width, 3):
        raise click.BadParameter("source geometry does not match --height and --width")
    model = load_v3_model(codebook, height=height, width=width, peak_count=peak_count)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for name, pixels in build_candidates(reference).items():
        path = output_dir / f"{name}.png"
        Image.fromarray(pixels, mode="RGB").save(path)
        rows.append(
            {
                **asdict(measure(reference, pixels, name=name, path=path)),
                **asdict(score_image(path, model)),
            }
        )
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "source": str(source),
                "codebook": str(codebook),
                "height": height,
                "width": width,
                "peak_count": peak_count,
                "variants": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log.info("Wrote %d frozen fragmentation candidates: %s", len(rows), report_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
