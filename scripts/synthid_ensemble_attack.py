"""Build pixel-only alternating-projection candidates for the ensemble detector.

The attack removes only the positive complex-spectrum projection onto the
learned RGB phases and HSV saturation/value phases. It preserves geometry and
does not invoke a generative model. Clearing the local research detector is not
proof that a provider oracle will clear SynthID.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import click
import cv2
import numpy as np
from PIL import Image
from synthid_ensemble_detector import EnsembleConfig, detect_image, load_config, load_models
from synthid_pixel_attack import load_rgb, measure, norm_matched_noise

if TYPE_CHECKING:
    from synthid_color_space_probe import ColorPhaseModel

log = logging.getLogger(__name__)


def _remove_positive_projection(
    channel: np.ndarray,
    *,
    rows: np.ndarray,
    columns: np.ndarray,
    phases: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Remove STRENGTH of each positive phase projection from CHANNEL."""
    if strength < 0.0 or strength > 1.0:
        raise ValueError("strength must be between zero and one")
    height, width = channel.shape
    spectrum = np.fft.fft2(channel.astype(np.float64))
    for row, column, phase in zip(rows, columns, phases, strict=True):
        row_index = int(row)
        column_index = int(column)
        direction = np.exp(1j * float(phase))
        value = spectrum[row_index, column_index]
        projection = max(0.0, float(np.real(value * np.conj(direction))))
        delta = strength * projection * direction
        conjugate_row = (-row_index) % height
        conjugate_column = (-column_index) % width
        spectrum[row_index, column_index] -= delta
        if (conjugate_row, conjugate_column) == (row_index, column_index):
            spectrum[row_index, column_index] = complex(spectrum[row_index, column_index].real, 0.0)
        else:
            spectrum[conjugate_row, conjugate_column] -= np.conj(delta)
    return np.fft.ifft2(spectrum).real


def _project_model_channels(
    pixels: np.ndarray,
    model: ColorPhaseModel,
    *,
    included_channels: frozenset[int],
    strength: float,
) -> np.ndarray:
    """Apply positive-projection removal to selected MODEL channels."""
    result = pixels.astype(np.float64, copy=True)
    for channel in included_channels:
        positions = np.flatnonzero(model.channels == channel)
        if len(positions) == 0:
            continue
        result[:, :, channel] = _remove_positive_projection(
            result[:, :, channel],
            rows=model.rows[positions],
            columns=model.columns[positions],
            phases=model.phases[positions],
            strength=strength,
        )
    return result


def alternating_projection(
    pixels: np.ndarray,
    rgb_model: ColorPhaseModel,
    hsv_model: ColorPhaseModel,
    *,
    strength: float,
    iterations: int,
) -> np.ndarray:
    """Alternate RGB and HSV S/V phase projections without regeneration."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    expected_shape = (rgb_model.height, rgb_model.width, 3)
    if pixels.shape != expected_shape or pixels.shape != (hsv_model.height, hsv_model.width, 3):
        raise ValueError("pixel and model geometries do not match")
    result = pixels.astype(np.float64)
    for _ in range(iterations):
        result = _project_model_channels(
            result,
            rgb_model,
            included_channels=frozenset({0, 1, 2}),
            strength=strength,
        )
        rgb_unit = np.clip(result / 255.0, 0.0, 1.0).astype(np.float32)
        hsv = cv2.cvtColor(rgb_unit, cv2.COLOR_RGB2HSV).astype(np.float64)
        hsv = _project_model_channels(
            hsv,
            hsv_model,
            included_channels=frozenset({1, 2}),
            strength=strength,
        )
        hsv[:, :, 0] = np.mod(hsv[:, :, 0], 360.0)
        hsv[:, :, 1:] = np.clip(hsv[:, :, 1:], 0.0, 1.0)
        result = cv2.cvtColor(hsv.astype(np.float32), cv2.COLOR_HSV2RGB).astype(np.float64) * 255.0
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def parse_positive_floats(value: str) -> tuple[float, ...]:
    """Parse strictly increasing strengths in the interval (0, 1]."""
    try:
        values = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise click.BadParameter("strengths must be comma-separated numbers") from error
    if not values or any(not np.isfinite(item) or item <= 0.0 or item > 1.0 for item in values):
        raise click.BadParameter("strengths must be finite and in the interval (0, 1]")
    if tuple(sorted(set(values))) != values:
        raise click.BadParameter("strengths must be unique and strictly increasing")
    return values


def parse_positive_integers(value: str) -> tuple[int, ...]:
    """Parse strictly increasing positive iteration counts."""
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise click.BadParameter("iterations must be comma-separated integers") from error
    if not values or any(item < 1 for item in values):
        raise click.BadParameter("iterations must be positive")
    if tuple(sorted(set(values))) != values:
        raise click.BadParameter("iterations must be unique and strictly increasing")
    return values


def _load_source(path: Path, config: EnsembleConfig) -> np.ndarray:
    """Load an exact-geometry RGB source."""
    pixels = load_rgb(path)
    if pixels.shape != (config.height, config.width, 3):
        raise ValueError("source geometry does not match detector config")
    return pixels


@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option("--strengths", default="0.25,0.5,0.75,1", show_default=True)
@click.option("--iterations", default="1,2,4", show_default=True)
def main(config_path: Path, source: Path, output_dir: Path, strengths: str, iterations: str) -> None:
    """Write a frozen pixel-only alternating-projection batch for SOURCE."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = load_config(config_path)
    rgb_model, hsv_model = load_models(config)
    reference = _load_source(source, config)
    strength_values = parse_positive_floats(strengths)
    iteration_values = parse_positive_integers(iterations)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants: list[dict[str, object]] = []
    strongest = reference
    for iteration_count in iteration_values:
        for strength in strength_values:
            pixels = alternating_projection(
                reference,
                rgb_model,
                hsv_model,
                strength=strength,
                iterations=iteration_count,
            )
            strength_name = f"{strength:g}".replace(".", "p")
            name = f"project-s{strength_name}-i{iteration_count}"
            path = output_dir / f"{name}.png"
            Image.fromarray(pixels, mode="RGB").save(path)
            variants.append(
                {
                    **asdict(measure(reference, pixels, name=name, path=path)),
                    **asdict(detect_image(path, config, rgb_model, hsv_model)),
                    "strength": strength,
                    "iterations": iteration_count,
                }
            )
            strongest = pixels

    sham = norm_matched_noise(reference, strongest, seed=20260823)
    sham_path = output_dir / "sham-strongest-rms.png"
    Image.fromarray(sham, mode="RGB").save(sham_path)
    variants.append(
        {
            **asdict(measure(reference, sham, name="sham-strongest-rms", path=sham_path)),
            **asdict(detect_image(sham_path, config, rgb_model, hsv_model)),
            "strength": None,
            "iterations": None,
        }
    )

    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "source": str(source),
                "config": str(config_path),
                "variants": variants,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log.info("Wrote %d alternating-projection candidates: %s", len(variants), report_path)


if __name__ == "__main__":
    main()
