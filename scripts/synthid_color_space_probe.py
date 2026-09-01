"""Compare an exact-geometry phase carrier across color spaces.

Every model uses the same spatial-frequency candidates. Phases, expected
magnitudes, selected channels, and weights are learned independently from the
supplied positive images. The resulting scores are experimental evidence, not
a proprietary SynthID decoder or a certified production detector.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import click
import cv2
import numpy as np
from PIL import Image
from synthid_phase_carrier import _leave_one_out_coherence
from synthid_phase_registration import extract_frequency_values
from synthid_v3_codebook_probe import load_v3_model

log = logging.getLogger(__name__)

COLOR_SPACES = ("rgb", "ycbcr", "ycocg", "opponent", "lab", "hsv")
CHANNEL_NAMES = {
    "rgb": ("R", "G", "B"),
    "ycbcr": ("Y", "Cb", "Cr"),
    "ycocg": ("Y", "Co", "Cg"),
    "opponent": ("L", "R-G", "R+G-2B"),
    "lab": ("L*", "a*", "b*"),
    "hsv": ("H", "S", "V"),
}


@dataclass(frozen=True)
class ColorPhaseModel:
    """Sparse phase carrier learned in one color space."""

    color_space: str
    height: int
    width: int
    rows: np.ndarray
    columns: np.ndarray
    channels: np.ndarray
    phases: np.ndarray
    weights: np.ndarray
    expected_magnitudes: np.ndarray


@dataclass(frozen=True)
class ColorPhaseScore:
    """Phase-carrier evidence and channel contributions for one image."""

    path: str
    color_space: str
    phase_score: float
    active_weight_fraction: float
    evidence_score: float
    channel_evidence: tuple[float, float, float]
    selected_peak_counts: tuple[int, int, int]
    peak_count: int


def transform_color_space(rgb: np.ndarray, color_space: str) -> np.ndarray:
    """Transform float RGB values in [0, 255] into COLOR_SPACE."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape (height, width, 3)")
    pixels = np.asarray(rgb, dtype=np.float64)
    red, green, blue = np.moveaxis(pixels, 2, 0)
    if color_space == "rgb":
        transformed = pixels
    elif color_space == "ycbcr":
        transformed = np.stack(
            (
                0.299 * red + 0.587 * green + 0.114 * blue,
                128.0 - 0.168736 * red - 0.331264 * green + 0.5 * blue,
                128.0 + 0.5 * red - 0.418688 * green - 0.081312 * blue,
            ),
            axis=2,
        )
    elif color_space == "ycocg":
        transformed = np.stack(
            (
                0.25 * red + 0.5 * green + 0.25 * blue,
                0.5 * red - 0.5 * blue,
                -0.25 * red + 0.5 * green - 0.25 * blue,
            ),
            axis=2,
        )
    elif color_space == "opponent":
        transformed = np.stack(
            (
                (red + green + blue) / np.sqrt(3.0),
                (red - green) / np.sqrt(2.0),
                (red + green - 2.0 * blue) / np.sqrt(6.0),
            ),
            axis=2,
        )
    elif color_space == "lab":
        transformed = cv2.cvtColor((pixels / 255.0).astype(np.float32), cv2.COLOR_RGB2LAB).astype(np.float64)
    elif color_space == "hsv":
        transformed = cv2.cvtColor((pixels / 255.0).astype(np.float32), cv2.COLOR_RGB2HSV).astype(np.float64)
    else:
        raise ValueError(f"unsupported color space: {color_space}")
    if not np.all(np.isfinite(transformed)):
        raise ValueError(f"{color_space} transform produced non-finite values")
    return transformed


def candidate_bins_from_codebook(
    codebook_path: Path,
    *,
    height: int,
    width: int,
    source_peak_count: int = 256,
) -> np.ndarray:
    """Expand the codebook's unique spatial coordinates over three channels."""
    prior = load_v3_model(
        codebook_path,
        height=height,
        width=width,
        peak_count=source_peak_count,
    )
    spatial = np.unique(np.column_stack((prior.rows, prior.columns)), axis=0)
    return np.asarray(
        [(int(row), int(column), channel) for row, column in spatial for channel in range(3)],
        dtype=np.int32,
    )


def _load_rgb(path: Path, *, height: int, width: int) -> np.ndarray:
    """Load an exact-geometry image as float64 RGB."""
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if rgb.size != (width, height):
            raise ValueError(f"{path}: geometry {rgb.width}x{rgb.height} does not match {width}x{height}")
        return np.asarray(rgb, dtype=np.float64)


def discover_model(
    paths: list[Path],
    *,
    color_space: str,
    candidate_bins: np.ndarray,
    peak_count: int = 256,
) -> ColorPhaseModel:
    """Learn one color-space phase carrier from exact-geometry PATHS."""
    if len(paths) < 3:
        raise ValueError("at least three positive images are required")
    if color_space not in COLOR_SPACES:
        raise ValueError(f"unsupported color space: {color_space}")
    with Image.open(paths[0]) as first:
        width, height = first.size
    bins = np.asarray(candidate_bins, dtype=np.int32)
    if bins.ndim != 2 or bins.shape[1] != 3:
        raise ValueError("candidate_bins must have shape (count, 3)")
    if len(bins) < peak_count:
        raise ValueError(f"only {len(bins)} candidate bins for {peak_count} peaks")
    if (
        np.any(bins[:, 0] < 0)
        or np.any(bins[:, 0] >= height)
        or np.any(bins[:, 1] <= 0)
        or np.any(bins[:, 1] > width // 2)
        or np.any(bins[:, 2] < 0)
        or np.any(bins[:, 2] > 2)
    ):
        raise ValueError("candidate_bins contain out-of-range coordinates")

    image_values = np.empty((len(paths), len(bins)), dtype=np.complex128)
    for index, path in enumerate(paths):
        rgb = _load_rgb(path, height=height, width=width)
        image_values[index] = extract_frequency_values(
            transform_color_space(rgb, color_space),
            bins[:, 0],
            bins[:, 1],
            bins[:, 2],
        )
    magnitudes = np.abs(image_values)
    units = np.divide(image_values, magnitudes, out=np.zeros_like(image_values), where=magnitudes != 0.0)
    unit_sum = np.sum(units, axis=0)
    count = float(len(paths))
    minimum_loo = np.ones(len(bins), dtype=np.float64)
    for unit in units:
        minimum_loo = np.minimum(minimum_loo, _leave_one_out_coherence(unit_sum, unit, count))
    expected_magnitudes = np.mean(magnitudes, axis=0)
    selection = np.power(minimum_loo, 4.0) * np.log1p(expected_magnitudes)
    chosen = np.argpartition(selection, -peak_count)[-peak_count:]
    chosen = chosen[np.argsort(selection[chosen])[::-1]]
    raw_weights = selection[chosen]
    if np.sum(raw_weights) <= 0.0:
        raise ValueError("candidate bins have no usable phase consensus")
    selected = bins[chosen]
    return ColorPhaseModel(
        color_space=color_space,
        height=height,
        width=width,
        rows=selected[:, 0].astype(np.int32),
        columns=selected[:, 1].astype(np.int32),
        channels=selected[:, 2].astype(np.int8),
        phases=np.angle(unit_sum[chosen] / count).astype(np.float64),
        weights=(raw_weights / np.sum(raw_weights)).astype(np.float64),
        expected_magnitudes=expected_magnitudes[chosen].astype(np.float64),
    )


def score_image(path: Path, model: ColorPhaseModel) -> ColorPhaseScore:
    """Score PATH against MODEL and expose additive channel evidence."""
    rgb = _load_rgb(path, height=model.height, width=model.width)
    bins = np.column_stack((model.rows, model.columns, model.channels))
    values = extract_frequency_values(
        transform_color_space(rgb, model.color_space),
        bins[:, 0],
        bins[:, 1],
        bins[:, 2],
    )
    magnitude_gate = np.minimum(np.abs(values) / (model.expected_magnitudes + 1e-12), 1.0)
    contributions = model.weights * magnitude_gate * np.cos(np.angle(values) - model.phases)
    active_weights = model.weights * magnitude_gate
    active_weight = float(np.sum(active_weights))
    evidence = float(np.sum(contributions))
    phase_score = 0.0 if active_weight == 0.0 else evidence / active_weight
    channel_evidence = tuple(float(np.sum(contributions[model.channels == channel])) for channel in range(3))
    selected_peak_counts = tuple(int(np.sum(model.channels == channel)) for channel in range(3))
    return ColorPhaseScore(
        path=str(path),
        color_space=model.color_space,
        phase_score=phase_score,
        active_weight_fraction=active_weight,
        evidence_score=evidence,
        channel_evidence=channel_evidence,
        selected_peak_counts=selected_peak_counts,
        peak_count=len(model.rows),
    )


def save_model(path: Path, model: ColorPhaseModel) -> None:
    """Save MODEL as a pickle-free numeric NPZ."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        format_version=np.asarray(1, dtype=np.int32),
        color_space=np.asarray(model.color_space),
        height=np.asarray(model.height, dtype=np.int32),
        width=np.asarray(model.width, dtype=np.int32),
        rows=model.rows.astype(np.int32),
        columns=model.columns.astype(np.int32),
        channels=model.channels.astype(np.int8),
        phases=model.phases.astype(np.float32),
        weights=model.weights.astype(np.float32),
        expected_magnitudes=model.expected_magnitudes.astype(np.float64),
    )


def load_model(path: Path) -> ColorPhaseModel:
    """Load and validate one color-space phase-carrier artifact."""
    with np.load(path, allow_pickle=False) as artifact:
        if int(artifact["format_version"]) != 1:
            raise ValueError("unsupported color-phase model format version")
        model = ColorPhaseModel(
            color_space=str(artifact["color_space"]),
            height=int(artifact["height"]),
            width=int(artifact["width"]),
            rows=np.asarray(artifact["rows"], dtype=np.int32),
            columns=np.asarray(artifact["columns"], dtype=np.int32),
            channels=np.asarray(artifact["channels"], dtype=np.int8),
            phases=np.asarray(artifact["phases"], dtype=np.float64),
            weights=np.asarray(artifact["weights"], dtype=np.float64),
            expected_magnitudes=np.asarray(artifact["expected_magnitudes"], dtype=np.float64),
        )
    count = len(model.rows)
    arrays = (model.columns, model.channels, model.phases, model.weights, model.expected_magnitudes)
    if model.color_space not in COLOR_SPACES:
        raise ValueError("invalid model color space")
    if model.height < 64 or model.width < 64 or count == 0 or any(array.shape != (count,) for array in arrays):
        raise ValueError("invalid color-phase model shapes")
    if np.any(model.rows < 0) or np.any(model.rows >= model.height):
        raise ValueError("invalid color-phase row indices")
    if np.any(model.columns <= 0) or np.any(model.columns > model.width // 2):
        raise ValueError("invalid color-phase column indices")
    if np.any(model.channels < 0) or np.any(model.channels > 2):
        raise ValueError("invalid color-phase channel indices")
    if not np.isclose(np.sum(model.weights), 1.0, atol=1e-5) or np.any(model.weights < 0.0):
        raise ValueError("invalid color-phase weights")
    if np.any(model.expected_magnitudes < 0.0):
        raise ValueError("invalid expected magnitudes")
    return model


@click.group()
def main() -> None:
    """Discover and score phase carriers in multiple color spaces."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


@main.command()
@click.argument("codebook", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("positives", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--color-space", type=click.Choice(COLOR_SPACES), required=True)
@click.option("--source-peak-count", type=click.IntRange(min=1), default=256, show_default=True)
@click.option("--peak-count", type=click.IntRange(min=1), default=256, show_default=True)
@click.option("--model-out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def discover(
    codebook: Path,
    positives: tuple[Path, ...],
    color_space: str,
    source_peak_count: int,
    peak_count: int,
    model_out: Path,
) -> None:
    """Learn a color-space carrier from exact-geometry POSITIVES."""
    with Image.open(positives[0]) as first:
        width, height = first.size
    candidates = candidate_bins_from_codebook(
        codebook,
        height=height,
        width=width,
        source_peak_count=source_peak_count,
    )
    model = discover_model(
        list(positives),
        color_space=color_space,
        candidate_bins=candidates,
        peak_count=peak_count,
    )
    save_model(model_out, model)
    log.info("Wrote %s color-phase model with %s candidates: %s", color_space, len(candidates), model_out)


@main.command()
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("images", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--report-out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def score(model_path: Path, images: tuple[Path, ...], report_out: Path) -> None:
    """Score exact-geometry IMAGES with MODEL_PATH."""
    model = load_model(model_path)
    payload = {
        "model": str(model_path),
        "color_space": model.color_space,
        "channel_names": CHANNEL_NAMES[model.color_space],
        "height": model.height,
        "width": model.width,
        "peak_count": len(model.rows),
        "scores": [asdict(score_image(image, model)) for image in images],
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log.info("Wrote %s color-phase score report: %s", model.color_space, report_out)


if __name__ == "__main__":
    main()
