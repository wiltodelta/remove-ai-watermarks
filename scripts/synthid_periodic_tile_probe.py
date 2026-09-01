"""Discover and evaluate an exact-geometry periodic residual carrier.

The model folds a high-pass residual modulo a fixed tile, averaging thousands
of spatial repetitions before normalized correlation. It is a positive-only
research signal, not a universal or certified SynthID decoder.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import click
import numpy as np
from synthid_periodic_tile import cyclic_tile_correlations, fold_residual_template, unit_tile
from synthid_pixel_attack import load_rgb
from synthid_research_manifest import artifact_sha256

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PeriodicTileModel:
    """One exact-geometry normalized periodic-residual template."""

    height: int
    width: int
    tile_height: int
    tile_width: int
    denoise_sigma: float
    template: np.ndarray
    expected_norm: float


@dataclass(frozen=True)
class PeriodicTileScore:
    """Normalized tile correlation and support for one image."""

    path: str
    score: float
    active_support: float
    row_shift: int
    column_shift: int
    repeat_count: int


def _load_rgb(path: Path, *, height: int, width: int) -> np.ndarray:
    """Load PATH as exact-geometry uint8 RGB."""
    pixels = load_rgb(path)
    if pixels.shape != (height, width, 3):
        raise ValueError(f"{path}: geometry {pixels.shape[1]}x{pixels.shape[0]} does not match {width}x{height}")
    return pixels


def discover_model(
    paths: list[Path],
    *,
    tile_height: int,
    tile_width: int,
    denoise_sigma: float = 1.0,
) -> PeriodicTileModel:
    """Learn a normalized periodic template from positive PATHS."""
    if len(paths) < 3:
        raise ValueError("at least three positive images are required")
    first_pixels = load_rgb(paths[0])
    height, width = first_pixels.shape[:2]
    unit_sum = np.zeros((tile_height, tile_width, 3), dtype=np.float64)
    norms: list[float] = []
    for index, path in enumerate(paths):
        folded = fold_residual_template(
            first_pixels if index == 0 else _load_rgb(path, height=height, width=width),
            tile_height=tile_height,
            tile_width=tile_width,
            denoise_sigma=denoise_sigma,
        )
        unit, norm = unit_tile(folded)
        unit_sum += unit
        norms.append(norm)
    template, template_norm = unit_tile(unit_sum / len(paths))
    if template_norm == 0.0:
        raise ValueError("positive images expose no periodic residual template")
    return PeriodicTileModel(
        height=height,
        width=width,
        tile_height=tile_height,
        tile_width=tile_width,
        denoise_sigma=denoise_sigma,
        template=template,
        expected_norm=float(np.median(norms)),
    )


def score_pixels(
    pixels: np.ndarray,
    model: PeriodicTileModel,
    *,
    register: bool = False,
    path: str = "<array>",
) -> PeriodicTileScore:
    """Score exact-geometry RGB PIXELS, optionally searching cyclic tile shifts."""
    if pixels.shape != (model.height, model.width, 3):
        raise ValueError("pixel geometry does not match periodic-tile model")
    folded = fold_residual_template(
        pixels,
        tile_height=model.tile_height,
        tile_width=model.tile_width,
        denoise_sigma=model.denoise_sigma,
    )
    unit, norm = unit_tile(folded)
    if register:
        correlations = cyclic_tile_correlations(model.template, unit)
        row_shift, column_shift = np.unravel_index(int(np.argmax(correlations)), correlations.shape)
        score = float(correlations[row_shift, column_shift])
    else:
        score = float(np.sum(model.template * unit))
        row_shift = column_shift = 0
    return PeriodicTileScore(
        path=path,
        score=score,
        active_support=min(norm / (model.expected_norm + 1e-12), 1.0),
        row_shift=row_shift,
        column_shift=column_shift,
        repeat_count=(model.height // model.tile_height) * (model.width // model.tile_width),
    )


def score_image(path: Path, model: PeriodicTileModel, *, register: bool = False) -> PeriodicTileScore:
    """Score PATH against MODEL, optionally searching cyclic tile shifts."""
    pixels = _load_rgb(path, height=model.height, width=model.width)
    return score_pixels(pixels, model, register=register, path=str(path))


def calibrate_threshold(paths: list[Path], model: PeriodicTileModel, *, register: bool = False) -> float:
    """Return the first float above every negative score in PATHS."""
    if not paths:
        raise ValueError("at least one calibration negative is required")
    maximum = max(score_image(path, model, register=register).score for path in paths)
    return float(np.nextafter(maximum, np.inf))


def save_model(path: Path, model: PeriodicTileModel) -> None:
    """Save MODEL as a pickle-free numeric artifact without precision loss."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        format_version=np.asarray(1, dtype=np.int32),
        height=np.asarray(model.height, dtype=np.int32),
        width=np.asarray(model.width, dtype=np.int32),
        tile_height=np.asarray(model.tile_height, dtype=np.int32),
        tile_width=np.asarray(model.tile_width, dtype=np.int32),
        denoise_sigma=np.asarray(model.denoise_sigma, dtype=np.float64),
        template=model.template.astype(np.float64),
        expected_norm=np.asarray(model.expected_norm, dtype=np.float64),
    )


def load_model(path: Path) -> PeriodicTileModel:
    """Load and validate one numeric periodic-tile model."""
    with np.load(path, allow_pickle=False) as artifact:
        if int(artifact["format_version"]) != 1:
            raise ValueError("unsupported periodic-tile model format version")
        model = PeriodicTileModel(
            height=int(artifact["height"]),
            width=int(artifact["width"]),
            tile_height=int(artifact["tile_height"]),
            tile_width=int(artifact["tile_width"]),
            denoise_sigma=float(artifact["denoise_sigma"]),
            template=np.asarray(artifact["template"], dtype=np.float64),
            expected_norm=float(artifact["expected_norm"]),
        )
    if model.height < 1 or model.width < 1 or model.tile_height < 1 or model.tile_width < 1:
        raise ValueError("invalid periodic-tile geometry")
    if model.height % model.tile_height or model.width % model.tile_width:
        raise ValueError("image geometry is not divisible by periodic-tile geometry")
    if model.template.shape != (model.tile_height, model.tile_width, 3):
        raise ValueError("invalid periodic-tile template shape")
    if not np.all(np.isfinite(model.template)) or not np.isclose(np.linalg.norm(model.template), 1.0):
        raise ValueError("invalid periodic-tile template")
    if not np.isfinite(model.denoise_sigma) or model.denoise_sigma <= 0.0:
        raise ValueError("invalid periodic-tile denoise sigma")
    if not np.isfinite(model.expected_norm) or model.expected_norm <= 0.0:
        raise ValueError("invalid periodic-tile expected norm")
    return model


@click.group()
def main() -> None:
    """Discover and evaluate an exact-geometry periodic carrier."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


@main.command()
@click.argument("positives", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--tile-height", type=click.IntRange(min=1), required=True)
@click.option("--tile-width", type=click.IntRange(min=1), required=True)
@click.option("--denoise-sigma", type=click.FloatRange(min=0.0, min_open=True), default=1.0, show_default=True)
@click.option("--model-out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def discover(
    positives: tuple[Path, ...],
    tile_height: int,
    tile_width: int,
    denoise_sigma: float,
    model_out: Path,
) -> None:
    """Learn a periodic tile from exact-geometry POSITIVES."""
    save_model(
        model_out,
        discover_model(
            list(positives),
            tile_height=tile_height,
            tile_width=tile_width,
            denoise_sigma=denoise_sigma,
        ),
    )
    log.info("Wrote periodic-tile model: %s", model_out)


@main.command()
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("negatives", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--register", is_flag=True, help="Search every cyclic shift within the learned tile.")
@click.option("--report-out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def calibrate(model_path: Path, negatives: tuple[Path, ...], register: bool, report_out: Path) -> None:
    """Calibrate a zero-observed-error threshold on NEGATIVES."""
    model = load_model(model_path)
    threshold = calibrate_threshold(list(negatives), model, register=register)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(
        json.dumps(
            {
                "model": str(model_path),
                "model_sha256": artifact_sha256(model_path),
                "register": register,
                "negative_count": len(negatives),
                "threshold": threshold,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log.info("Wrote periodic-tile calibration report: %s", report_out)


@main.command()
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("images", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--register", is_flag=True, help="Search every cyclic shift within the learned tile.")
@click.option("--report-out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def score(model_path: Path, images: tuple[Path, ...], register: bool, report_out: Path) -> None:
    """Score exact-geometry IMAGES with MODEL_PATH."""
    model = load_model(model_path)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(
        json.dumps(
            {
                "model": str(model_path),
                "model_sha256": artifact_sha256(model_path),
                "register": register,
                "scores": [asdict(score_image(image, model, register=register)) for image in images],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log.info("Wrote periodic-tile score report: %s", report_out)


if __name__ == "__main__":
    main()
