"""Probe complex cross-spectral coupling at preregistered carrier shifts."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import cv2
import numpy as np
from synthid_affine_lattice_probe import (
    _canonical_pixels,
    _opponent_channels,
    _patch_origins,
    template_harmonics,
)
from synthid_pixel_attack import load_rgb

if TYPE_CHECKING:
    from numpy.typing import NDArray

log = logging.getLogger(__name__)

_OFF_CARRIER_OFFSETS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1))


@dataclass(frozen=True)
class CyclostationaryScore:
    """Complex carrier-versus-neighbor contrast on disjoint patch groups."""

    selection_carrier: float
    selection_off_carrier_median: float
    selection_contrast: float
    confirmation_carrier: float
    confirmation_off_carrier_median: float
    confirmation_contrast: float
    joint_contrast: float
    harmonic_count: int
    selection_patches: int
    confirmation_patches: int


def _patch_cyclic_matrices(
    pixels: NDArray[Any],
    harmonics: NDArray[Any],
    *,
    tile_size: int,
    denoise_sigma: float,
    band_min: float,
    band_max: float,
) -> NDArray[Any]:
    """Return normalized complex cross-channel matrices for one patch."""
    patch_size = pixels.shape[0]
    if pixels.shape[:2] != (patch_size, patch_size) or patch_size % tile_size:
        raise ValueError("cyclostationary patches must be square multiples of the tile size")
    channels = _opponent_channels(np.asarray(pixels, dtype=np.float64))
    for channel in range(channels.shape[2]):
        residual = channels[:, :, channel]
        channels[:, :, channel] = residual - cv2.GaussianBlur(
            residual,
            (0, 0),
            sigmaX=denoise_sigma,
            sigmaY=denoise_sigma,
            borderType=cv2.BORDER_REFLECT_101,
        )
    window_1d = np.hanning(patch_size)
    window = window_1d[:, None] * window_1d[None, :]
    spectrum = np.fft.fft2(channels * window[:, :, None], axes=(0, 1))
    frequency_y = np.fft.fftfreq(patch_size)[:, None]
    frequency_x = np.fft.fftfreq(patch_size)[None, :]
    radius = np.sqrt(frequency_y * frequency_y + frequency_x * frequency_x)
    base_mask = (radius >= band_min) & (radius <= band_max)

    offset_values = ((0, 0), *_OFF_CARRIER_OFFSETS)
    matrices = np.empty((len(harmonics), len(offset_values), 3, 3), dtype=np.complex128)
    for harmonic_index, (signed_row_value, signed_column_value) in enumerate(harmonics):
        alpha_y = round(float(signed_row_value) * patch_size / tile_size)
        alpha_x = round(float(signed_column_value) * patch_size / tile_size)
        for offset_index, (offset_y, offset_x) in enumerate(offset_values):
            shift_y = alpha_y + offset_y
            shift_x = alpha_x + offset_x
            shifted = np.roll(spectrum, shift=(-shift_y, -shift_x), axis=(0, 1))
            mask = base_mask & np.roll(base_mask, shift=(-shift_y, -shift_x), axis=(0, 1))
            base_values = spectrum[mask]
            shifted_values = shifted[mask]
            normalizer = math.sqrt(float(np.sum(np.abs(base_values) ** 2)) * float(np.sum(np.abs(shifted_values) ** 2)))
            if normalizer <= 1e-12:
                matrices[harmonic_index, offset_index] = 0.0
            else:
                matrices[harmonic_index, offset_index] = shifted_values.T @ np.conj(base_values) / normalizer
    return matrices


def _group_score(values: list[NDArray[Any]], harmonic_weights: NDArray[Any]) -> tuple[float, float, float]:
    if not values:
        raise ValueError("cyclostationary score needs at least one patch")
    mean_matrices = np.mean(np.stack(values), axis=0)
    coherence = np.linalg.norm(mean_matrices, axis=(2, 3))
    carrier = float(np.sum(coherence[:, 0] * harmonic_weights))
    off_scores = [
        float(np.sum(coherence[:, offset_index] * harmonic_weights)) for offset_index in range(1, coherence.shape[1])
    ]
    off_median = float(np.median(off_scores))
    return carrier, off_median, carrier - off_median


def score_cyclostationary(
    pixels: NDArray[Any],
    template: NDArray[Any],
    *,
    period: float,
    patch_size: int = 256,
    grid_size: int = 4,
    harmonic_count: int = 8,
    denoise_sigma: float = 1.0,
    band_min: float = 0.05,
    band_max: float = 0.35,
) -> CyclostationaryScore:
    """Measure split-confirmed complex spectral coupling at one period."""
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("pixels must have shape (height, width, 3)")
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError("period must be finite and positive")
    if not (0.0 < band_min < band_max < 0.5):
        raise ValueError("frequency band must satisfy 0 < min < max < 0.5")
    if not math.isfinite(denoise_sigma) or denoise_sigma <= 0.0:
        raise ValueError("denoise sigma must be finite and positive")

    canonical = _canonical_pixels(pixels, template, period)
    tile_size = template.shape[0]
    harmonics, channel_weights, _coefficient_units = template_harmonics(template, harmonic_count)
    harmonic_weights = np.linalg.norm(channel_weights, axis=1)
    harmonic_weights /= np.sum(harmonic_weights)
    grouped_values: dict[int, list[NDArray[Any]]] = {0: [], 1: []}
    for origin_y, origin_x, group in _patch_origins(*canonical.shape[:2], patch_size, grid_size):
        aligned_y = (origin_y // tile_size) * tile_size
        aligned_x = (origin_x // tile_size) * tile_size
        patch = canonical[aligned_y : aligned_y + patch_size, aligned_x : aligned_x + patch_size]
        grouped_values[group].append(
            _patch_cyclic_matrices(
                patch,
                harmonics,
                tile_size=tile_size,
                denoise_sigma=denoise_sigma,
                band_min=band_min,
                band_max=band_max,
            )
        )
    selection_carrier, selection_null, selection_contrast = _group_score(grouped_values[0], harmonic_weights)
    confirmation_carrier, confirmation_null, confirmation_contrast = _group_score(grouped_values[1], harmonic_weights)
    return CyclostationaryScore(
        selection_carrier=selection_carrier,
        selection_off_carrier_median=selection_null,
        selection_contrast=selection_contrast,
        confirmation_carrier=confirmation_carrier,
        confirmation_off_carrier_median=confirmation_null,
        confirmation_contrast=confirmation_contrast,
        joint_contrast=min(selection_contrast, confirmation_contrast),
        harmonic_count=len(harmonics),
        selection_patches=len(grouped_values[0]),
        confirmation_patches=len(grouped_values[1]),
    )


def _load_template(path: Path) -> NDArray[Any]:
    with np.load(path, allow_pickle=False) as artifact:
        template = np.asarray(artifact["template"], dtype=np.float64)
    if template.shape != (16, 16, 3) or not np.all(np.isfinite(template)):
        raise ValueError("template artifact does not contain a finite 16x16 RGB template")
    return template


@click.command()
@click.argument("template_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("images", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--period", type=click.FloatRange(min=1.0), required=True)
@click.option("--input-scale", type=click.FloatRange(min=0.01), default=1.0, show_default=True)
@click.option("--report-out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(
    template_path: Path,
    images: tuple[Path, ...],
    period: float,
    input_scale: float,
    report_out: Path,
) -> None:
    """Score IMAGES for complex cross-spectral carrier coupling."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    template = _load_template(template_path)
    rows = []
    for path in images:
        try:
            pixels = load_rgb(path)
            if input_scale != 1.0:
                width = max(1, round(pixels.shape[1] * input_scale))
                height = max(1, round(pixels.shape[0] * input_scale))
                interpolation = cv2.INTER_AREA if input_scale < 1.0 else cv2.INTER_CUBIC
                pixels = cv2.resize(pixels, (width, height), interpolation=interpolation)
            score = score_cyclostationary(pixels, template, period=period)
        except ValueError as error:
            rows.append({"path": str(path), "status": "unsupported", "reason": str(error)})
            log.warning("%s: unsupported: %s", path, error)
            continue
        rows.append({"path": str(path), "status": "scored", "score": asdict(score)})
        log.info("%s: joint contrast=%.6f", path, score.joint_contrast)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "template": str(template_path),
                "period": period,
                "input_scale": input_scale,
                "records": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log.info("Wrote %d cyclostationary records: %s", len(rows), report_out)


if __name__ == "__main__":
    main()
