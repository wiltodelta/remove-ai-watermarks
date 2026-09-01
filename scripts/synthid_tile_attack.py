"""Estimate and subtract a periodic SynthID residual tile without regeneration.

At 1536x2816, the dominant carrier bins lie on an FFT lattice spaced by 96
rows and 88 columns, corresponding to a 16x32 spatial tile. Folding a
high-pass residual modulo that tile averages over 8448 repetitions and
suppresses non-periodic image content.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import click
import numpy as np
from PIL import Image
from synthid_ensemble_detector import detect_image, load_config, load_models
from synthid_periodic_tile import fold_residual_template
from synthid_pixel_attack import load_rgb, measure

log = logging.getLogger(__name__)


def subtract_tiled_template(pixels: np.ndarray, template: np.ndarray, *, strength: float) -> np.ndarray:
    """Subtract STRENGTH times TEMPLATE repeated over PIXELS."""
    if not np.isfinite(strength) or strength < 0.0:
        raise ValueError("strength must be finite and nonnegative")
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("pixels must have shape (height, width, 3)")
    if template.ndim != 3 or template.shape[2] != 3:
        raise ValueError("template must have shape (tile height, tile width, 3)")
    height, width = pixels.shape[:2]
    tile_height, tile_width = template.shape[:2]
    if tile_height == 0 or tile_width == 0:
        raise ValueError("template dimensions must be positive")

    result = np.empty_like(pixels, dtype=np.uint8)
    repeats_x = (width + tile_width - 1) // tile_width
    for top in range(0, height, 256):
        bottom = min(top + 256, height)
        template_rows = template[np.arange(top, bottom) % tile_height]
        repeated = np.tile(template_rows, (1, repeats_x, 1))[:, :width]
        stripe = pixels[top:bottom].astype(np.float64) - strength * repeated
        result[top:bottom] = np.clip(np.rint(stripe), 0, 255).astype(np.uint8)
    return result


def parse_positive_floats(value: str, *, option_name: str) -> tuple[float, ...]:
    """Parse a strictly increasing comma-separated positive-float sweep."""
    try:
        values = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise click.BadParameter(f"{option_name} must be comma-separated numbers") from error
    if not values or any(not np.isfinite(item) or item <= 0.0 for item in values):
        raise click.BadParameter(f"{option_name} must be finite and positive")
    if tuple(sorted(set(values))) != values:
        raise click.BadParameter(f"{option_name} must be unique and strictly increasing")
    return values


@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option("--tile-height", type=click.IntRange(min=1), default=16, show_default=True)
@click.option("--tile-width", type=click.IntRange(min=1), default=32, show_default=True)
@click.option("--denoise-sigmas", default="0.6,1,1.5", show_default=True)
@click.option("--strengths", default="0.5,1,1.5,2", show_default=True)
def main(
    config_path: Path,
    source: Path,
    output_dir: Path,
    tile_height: int,
    tile_width: int,
    denoise_sigmas: str,
    strengths: str,
) -> None:
    """Write a frozen periodic-tile subtraction sweep for SOURCE."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = load_config(config_path)
    rgb_model, hsv_model = load_models(config)
    reference = load_rgb(source)
    if reference.shape != (config.height, config.width, 3):
        raise click.BadParameter("source geometry does not match detector config")
    sigma_values = parse_positive_floats(denoise_sigmas, option_name="denoise sigmas")
    strength_values = parse_positive_floats(strengths, option_name="strengths")
    output_dir.mkdir(parents=True, exist_ok=True)

    variants: list[dict[str, object]] = []
    templates: list[dict[str, object]] = []
    for sigma in sigma_values:
        template = fold_residual_template(
            reference,
            tile_height=tile_height,
            tile_width=tile_width,
            denoise_sigma=sigma,
        )
        sigma_name = f"{sigma:g}".replace(".", "p")
        templates.append(
            {
                "denoise_sigma": sigma,
                "template_rms": float(np.sqrt(np.mean(np.square(template)))),
                "template_max_abs": float(np.max(np.abs(template))),
            }
        )
        shifted = np.roll(template, shift=(1, 1), axis=(0, 1))
        for strength in strength_values:
            strength_name = f"{strength:g}".replace(".", "p")
            for control_name, selected_template in (("aligned", template), ("shifted", shifted)):
                name = f"tile-{control_name}-sigma{sigma_name}-s{strength_name}"
                pixels = subtract_tiled_template(reference, selected_template, strength=strength)
                path = output_dir / f"{name}.png"
                Image.fromarray(pixels, mode="RGB").save(path)
                variants.append(
                    {
                        **asdict(measure(reference, pixels, name=name, path=path)),
                        **asdict(detect_image(path, config, rgb_model, hsv_model)),
                        "denoise_sigma": sigma,
                        "strength": strength,
                        "template_alignment": control_name,
                    }
                )

    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "source": str(source),
                "config": str(config_path),
                "tile_height": tile_height,
                "tile_width": tile_width,
                "repeat_count": (config.height // tile_height) * (config.width // tile_width),
                "templates": templates,
                "variants": variants,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log.info("Wrote %d periodic-tile candidates: %s", len(variants), report_path)


if __name__ == "__main__":
    main()
