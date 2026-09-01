"""Build non-generative hybrid phase-projection and fragmentation candidates.

The matrix combines two independently measured mechanisms: sparse RGB/HSV
phase projection and spatially varying subpixel displacement. It is intended
for frozen provider-oracle batches with a visible-mark-removed source control.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import click
from PIL import Image
from synthid_ensemble_attack import alternating_projection
from synthid_ensemble_detector import detect_image, load_config, load_models
from synthid_fragment_attack import bounded_smooth_warp, color_nudge, jpeg_chain
from synthid_pixel_attack import load_rgb, measure, norm_matched_noise, resize_squeeze, smooth_warp

if TYPE_CHECKING:
    import numpy as np
    from synthid_color_space_probe import ColorPhaseModel

log = logging.getLogger(__name__)


def build_candidates(
    source: np.ndarray,
    rgb_model: ColorPhaseModel,
    hsv_model: ColorPhaseModel,
) -> dict[str, np.ndarray]:
    """Return a frozen mechanism matrix derived from SOURCE."""
    projected_075 = alternating_projection(
        source,
        rgb_model,
        hsv_model,
        strength=0.75,
        iterations=1,
    )
    projected_100 = alternating_projection(
        source,
        rgb_model,
        hsv_model,
        strength=1.0,
        iterations=1,
    )
    bounded_100 = bounded_smooth_warp(
        source,
        max_displacement=1.0,
        sigma=56.0,
        seed=20260824,
    )
    projected_bounded_100 = bounded_smooth_warp(
        projected_075,
        max_displacement=1.0,
        sigma=56.0,
        seed=20260824,
    )

    bounded_polish = resize_squeeze(projected_bounded_100, 0.98)
    bounded_polish = color_nudge(
        bounded_polish,
        brightness=0.002,
        contrast=0.003,
        saturation=-0.003,
        hue_degrees=0.1,
    )
    bounded_polish = jpeg_chain(bounded_polish, (96,))

    elastic_combo = smooth_warp(projected_100, amplitude=0.75, sigma=56.0, seed=20260825)
    elastic_combo = resize_squeeze(elastic_combo, 0.98)
    elastic_combo = jpeg_chain(elastic_combo, (96,))

    return {
        "projection-075": projected_075,
        "bounded-100": bounded_100,
        "projection-075-bounded-100": projected_bounded_100,
        "projection-075-bounded-polish": bounded_polish,
        "projection-100-elastic-075": elastic_combo,
    }


@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
def main(config_path: Path, source: Path, output_dir: Path) -> None:
    """Write a frozen non-generative hybrid attack matrix for SOURCE."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = load_config(config_path)
    rgb_model, hsv_model = load_models(config)
    reference = load_rgb(source)
    if reference.shape != (config.height, config.width, 3):
        raise click.BadParameter("source geometry does not match detector config")
    output_dir.mkdir(parents=True, exist_ok=True)

    variants: list[dict[str, object]] = []
    candidates = build_candidates(reference, rgb_model, hsv_model)
    for name, pixels in candidates.items():
        path = output_dir / f"{name}.png"
        Image.fromarray(pixels, mode="RGB").save(path)
        variants.append(
            {
                **asdict(measure(reference, pixels, name=name, path=path)),
                **asdict(detect_image(path, config, rgb_model, hsv_model)),
            }
        )

    selected = candidates["projection-075-bounded-100"]
    sham = norm_matched_noise(reference, selected, seed=20260826)
    sham_path = output_dir / "sham-projection-bounded-rms.png"
    Image.fromarray(sham, mode="RGB").save(sham_path)
    variants.append(
        {
            **asdict(measure(reference, sham, name="sham-projection-bounded-rms", path=sham_path)),
            **asdict(detect_image(sham_path, config, rgb_model, hsv_model)),
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
    log.info("Wrote %d frozen hybrid candidates: %s", len(variants), report_path)


if __name__ == "__main__":
    main()
