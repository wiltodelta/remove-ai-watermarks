"""Build pixel-only V3 carrier-subtraction candidates and matched controls.

The command uses a frozen numeric frequency profile as a local research
surrogate. It subtracts a sparse Hermitian spectrum, preserves image geometry,
and never invokes a generative model. A lower local score is not evidence that
the provider's SynthID verifier will change its decision.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import click
import numpy as np
from PIL import Image
from synthid_pixel_attack import load_rgb, measure, norm_matched_noise
from synthid_v3_codebook_probe import V3CarrierModel, load_v3_model, score_image

log = logging.getLogger(__name__)


def load_exact_rgb(path: Path, model: V3CarrierModel) -> np.ndarray:
    """Load PATH as RGB and reject geometry that differs from MODEL."""
    pixels = load_rgb(path)
    if pixels.shape != (model.height, model.width, 3):
        height, width = pixels.shape[:2]
        raise ValueError(f"image geometry {width}x{height} does not match profile {model.width}x{model.height}")
    return pixels


def subtract_carrier(pixels: np.ndarray, model: V3CarrierModel, *, strength: float) -> np.ndarray:
    """Subtract STRENGTH times MODEL's sparse complex carrier from PIXELS."""
    if strength < 0.0:
        raise ValueError("strength must be nonnegative")
    expected_shape = (model.height, model.width, 3)
    if pixels.shape != expected_shape:
        raise ValueError(f"pixel shape {pixels.shape} does not match {expected_shape}")

    result = np.empty_like(pixels, dtype=np.float64)
    for channel in range(3):
        spectrum = np.fft.fft2(pixels[:, :, channel].astype(np.float64))
        positions = np.flatnonzero(model.channels == channel)
        deltas: dict[tuple[int, int], complex] = {}
        for position in positions:
            row = int(model.rows[position])
            column = int(model.columns[position])
            delta = strength * model.expected_magnitudes[position] * np.exp(1j * model.phases[position])
            key = (row, column)
            conjugate_key = ((-row) % model.height, (-column) % model.width)
            deltas[key] = deltas.get(key, 0.0j) + delta
            if conjugate_key == key:
                deltas[key] = complex(deltas[key].real, 0.0)
            else:
                deltas[conjugate_key] = deltas.get(conjugate_key, 0.0j) + np.conj(delta)
        for (row, column), delta in deltas.items():
            spectrum[row, column] -= delta
        result[:, :, channel] = np.fft.ifft2(spectrum).real
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def parse_strengths(value: str) -> tuple[float, ...]:
    """Parse a comma-separated, strictly increasing nonnegative sweep."""
    try:
        strengths = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise click.BadParameter("strengths must be comma-separated numbers") from error
    if not strengths or any(not np.isfinite(item) or item < 0.0 for item in strengths):
        raise click.BadParameter("strengths must be finite and nonnegative")
    if tuple(sorted(set(strengths))) != strengths:
        raise click.BadParameter("strengths must be unique and strictly increasing")
    return strengths


@click.command()
@click.argument("codebook", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option("--height", type=click.IntRange(min=64), required=True)
@click.option("--width", type=click.IntRange(min=64), required=True)
@click.option("--peak-count", type=click.IntRange(min=1), default=256, show_default=True)
@click.option("--strengths", default="0.25,0.5,1,1.5,2,4", show_default=True)
def main(
    codebook: Path,
    source: Path,
    output_dir: Path,
    height: int,
    width: int,
    peak_count: int,
    strengths: str,
) -> None:
    """Write a frozen analytical carrier-subtraction batch for SOURCE."""
    model = load_v3_model(codebook, height=height, width=width, peak_count=peak_count)
    reference = load_exact_rgb(source, model)
    sweep = parse_strengths(strengths)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants: list[dict[str, object]] = []
    strongest = reference
    for strength in sweep:
        pixels = subtract_carrier(reference, model, strength=strength)
        name = f"subtract-{strength:g}".replace(".", "p")
        path = output_dir / f"{name}.png"
        Image.fromarray(pixels, mode="RGB").save(path)
        variants.append(
            {
                **asdict(measure(reference, pixels, name=name, path=path)),
                **asdict(score_image(path, model)),
                "strength": strength,
            }
        )
        strongest = pixels

    sham = norm_matched_noise(reference, strongest, seed=20260809)
    sham_path = output_dir / "sham-strongest-rms.png"
    Image.fromarray(sham, mode="RGB").save(sham_path)
    variants.append(
        {
            **asdict(measure(reference, sham, name="sham-strongest-rms", path=sham_path)),
            **asdict(score_image(sham_path, model)),
            "strength": None,
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
                "variants": variants,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log.info("Wrote %d frozen carrier candidates: %s", len(variants), report_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
