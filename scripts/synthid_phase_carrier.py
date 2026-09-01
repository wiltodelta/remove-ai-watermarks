"""Discover and evaluate an exact-geometry phase-carrier hypothesis.

The model is learned only from supplied images and stored as numeric arrays in
a pickle-free NPZ. It is a research baseline, not a proprietary SynthID
decoder. A valid detector claim still requires provider labels, same-provider
hard negatives, group-aware splits, and a locked operating point.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import click
import numpy as np
from PIL import Image
from synthid_phase_registration import MAX_TRANSLATION_SHIFT, extract_frequency_values, register_phase_translations

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PhaseCarrierModel:
    """Sparse exact-geometry phase carrier learned from positive images."""

    height: int
    width: int
    rows: np.ndarray
    columns: np.ndarray
    channels: np.ndarray
    phases: np.ndarray
    weights: np.ndarray
    expected_magnitudes: np.ndarray


@dataclass(frozen=True)
class PhaseCarrierScore:
    """Alignment of one exact-geometry image with a phase-carrier model."""

    path: str
    score: float
    active_weight_fraction: float
    peak_count: int


@dataclass(frozen=True)
class RegisteredPhaseCarrierScore:
    """Best phase-carrier score across a bounded translation search."""

    path: str
    score: float
    active_weight_fraction: float
    peak_count: int
    row_shift: int
    column_shift: int


def _load_rgb(path: Path, *, height: int, width: int, canonicalize_geometry: bool = False) -> np.ndarray:
    """Load PATH as float64 RGB, optionally resizing to model geometry."""
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if rgb.size != (width, height):
            if not canonicalize_geometry:
                raise ValueError(f"{path}: geometry {rgb.width}x{rgb.height} does not match {width}x{height}")
            rgb = rgb.resize((width, height), Image.Resampling.LANCZOS)
        return np.asarray(rgb, dtype=np.float64)


def _valid_frequency_mask(height: int, width: int, min_radius: float) -> np.ndarray:
    """Return eligible non-DC bins in an rFFT half-plane."""
    rows = np.arange(height)
    signed_rows = np.where(rows > height // 2, rows - height, rows)
    columns = np.arange(width // 2 + 1)
    radius = np.sqrt(np.square(signed_rows[:, None]) + np.square(columns[None, :]))
    return (radius >= min_radius) & (columns[None, :] > 0)


def _leave_one_out_coherence(unit_sum: np.ndarray, held_out_unit: np.ndarray, count: float) -> np.ndarray:
    """Return phase coherence after removing HELD_OUT_UNIT from UNIT_SUM."""
    if count <= 1.0:
        raise ValueError("leave-one-out coherence requires at least two samples")
    return np.abs((unit_sum - held_out_unit) / (count - 1.0))


def discover_model(
    paths: list[Path],
    *,
    peak_count: int = 256,
    min_radius: float = 15.0,
    candidate_bins: np.ndarray | None = None,
) -> PhaseCarrierModel:
    """Learn a sparse phase-consensus model from exact-geometry PATHS."""
    if len(paths) < 3:
        raise ValueError("at least three positive images are required")
    with Image.open(paths[0]) as first:
        width, height = first.size
    if min(height, width) < 64:
        raise ValueError("images must be at least 64 pixels per side")

    half_width = width // 2 + 1
    unit_sum = np.zeros((height, half_width, 3), dtype=np.complex64)
    magnitude_sum = np.zeros((height, half_width, 3), dtype=np.float64)
    for path in paths:
        pixels = _load_rgb(path, height=height, width=width)
        for channel in range(3):
            spectrum = np.fft.rfft2(pixels[:, :, channel])
            magnitude = np.abs(spectrum)
            unit_sum[:, :, channel] += np.divide(
                spectrum,
                magnitude,
                out=np.zeros_like(spectrum),
                where=magnitude != 0.0,
            ).astype(np.complex64)
            magnitude_sum[:, :, channel] += magnitude

    count = float(len(paths))
    mean_unit = unit_sum / count
    minimum_loo_coherence = np.ones((height, half_width, 3), dtype=np.float32)
    for path in paths:
        pixels = _load_rgb(path, height=height, width=width)
        for channel in range(3):
            spectrum = np.fft.rfft2(pixels[:, :, channel])
            magnitude = np.abs(spectrum)
            unit = np.divide(
                spectrum,
                magnitude,
                out=np.zeros_like(spectrum),
                where=magnitude != 0.0,
            )
            loo_coherence = _leave_one_out_coherence(unit_sum[:, :, channel], unit, count)
            np.minimum(minimum_loo_coherence[:, :, channel], loo_coherence, out=minimum_loo_coherence[:, :, channel])
    expected_magnitude = magnitude_sum / count
    selection = np.power(minimum_loo_coherence.astype(np.float64), 4.0) * np.log1p(expected_magnitude)
    selection *= _valid_frequency_mask(height, width, min_radius)[:, :, None]
    if candidate_bins is None:
        candidate_indices = np.flatnonzero(selection)
    else:
        bins = np.asarray(candidate_bins, dtype=np.int64)
        if bins.ndim != 2 or bins.shape[1] != 3:
            raise ValueError("candidate_bins must have shape (count, 3)")
        if (
            np.any(bins[:, 0] < 0)
            or np.any(bins[:, 0] >= height)
            or np.any(bins[:, 1] <= 0)
            or np.any(bins[:, 1] > width // 2)
            or np.any(bins[:, 2] < 0)
            or np.any(bins[:, 2] > 2)
        ):
            raise ValueError("candidate_bins contain out-of-range coordinates")
        candidate_indices = np.unique(np.ravel_multi_index(bins.T, selection.shape))
        candidate_indices = candidate_indices[selection.ravel()[candidate_indices] > 0.0]
    candidate_count = len(candidate_indices)
    if candidate_count < peak_count:
        raise ValueError(f"only {candidate_count} eligible bins for {peak_count} peaks")

    flat = selection.ravel()
    candidate_scores = flat[candidate_indices]
    chosen = np.argpartition(candidate_scores, -peak_count)[-peak_count:]
    indices = candidate_indices[chosen]
    indices = indices[np.argsort(flat[indices])[::-1]]
    rows, columns, channels = np.unravel_index(indices, selection.shape)
    raw_weights = selection[rows, columns, channels]
    return PhaseCarrierModel(
        height=height,
        width=width,
        rows=rows.astype(np.int32),
        columns=columns.astype(np.int32),
        channels=channels.astype(np.int8),
        phases=np.angle(mean_unit[rows, columns, channels]).astype(np.float64),
        weights=(raw_weights / np.sum(raw_weights)).astype(np.float64),
        expected_magnitudes=expected_magnitude[rows, columns, channels].astype(np.float64),
    )


def _frequency_values(pixels: np.ndarray, model: PhaseCarrierModel) -> np.ndarray:
    """Return MODEL's selected complex coefficients from PIXELS."""
    return extract_frequency_values(pixels, model.rows, model.columns, model.channels)


def score_pixels(pixels: np.ndarray, model: PhaseCarrierModel, *, path: str = "<array>") -> PhaseCarrierScore:
    """Score exact-geometry RGB PIXELS against MODEL."""
    if pixels.shape != (model.height, model.width, 3):
        raise ValueError("pixel geometry does not match phase-carrier model")
    values = _frequency_values(np.asarray(pixels, dtype=np.float64), model)
    magnitude_gate = np.minimum(np.abs(values) / (model.expected_magnitudes + 1e-12), 1.0)
    active_weights = model.weights * magnitude_gate
    active_weight = float(np.sum(active_weights))
    score = (
        0.0
        if active_weight == 0.0
        else float(np.sum(active_weights * np.cos(np.angle(values) - model.phases)) / active_weight)
    )
    return PhaseCarrierScore(
        path=path,
        score=score,
        active_weight_fraction=active_weight,
        peak_count=len(model.rows),
    )


def score_image(
    path: Path,
    model: PhaseCarrierModel,
    *,
    canonicalize_geometry: bool = False,
) -> PhaseCarrierScore:
    """Score PATH against MODEL, with optional geometry canonicalization."""
    pixels = _load_rgb(
        path,
        height=model.height,
        width=model.width,
        canonicalize_geometry=canonicalize_geometry,
    )
    return score_pixels(pixels, model, path=str(path))


def score_translations(
    path: Path,
    model: PhaseCarrierModel,
    *,
    max_shift: int = 4,
    canonicalize_geometry: bool = False,
) -> RegisteredPhaseCarrierScore:
    """Return the best score after compensating bounded integer translations."""
    pixels = _load_rgb(
        path,
        height=model.height,
        width=model.width,
        canonicalize_geometry=canonicalize_geometry,
    )
    registration = register_phase_translations(
        _frequency_values(pixels, model),
        phases=model.phases,
        weights=model.weights,
        expected_magnitudes=model.expected_magnitudes,
        rows=model.rows,
        columns=model.columns,
        height=model.height,
        width=model.width,
        max_shift=max_shift,
    )
    return RegisteredPhaseCarrierScore(
        path=str(path),
        score=registration.score,
        active_weight_fraction=registration.active_weight_fraction,
        peak_count=len(model.rows),
        row_shift=registration.row_shift,
        column_shift=registration.column_shift,
    )


def save_model(path: Path, model: PhaseCarrierModel) -> None:
    """Save MODEL as a validated numeric NPZ artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        format_version=np.asarray(1, dtype=np.int32),
        height=np.asarray(model.height, dtype=np.int32),
        width=np.asarray(model.width, dtype=np.int32),
        rows=model.rows.astype(np.int32),
        columns=model.columns.astype(np.int32),
        channels=model.channels.astype(np.int8),
        phases=model.phases.astype(np.float32),
        weights=model.weights.astype(np.float32),
        expected_magnitudes=model.expected_magnitudes.astype(np.float64),
    )


def load_model(path: Path) -> PhaseCarrierModel:
    """Load and validate a numeric phase-carrier artifact."""
    with np.load(path, allow_pickle=False) as artifact:
        if int(artifact["format_version"]) != 1:
            raise ValueError("unsupported phase-carrier format version")
        model = PhaseCarrierModel(
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
    if model.height < 64 or model.width < 64 or any(array.shape != (count,) for array in arrays):
        raise ValueError("invalid phase-carrier model shapes")
    if count == 0 or np.any(model.rows < 0) or np.any(model.rows >= model.height):
        raise ValueError("invalid phase-carrier row indices")
    if np.any(model.columns <= 0) or np.any(model.columns > model.width // 2):
        raise ValueError("invalid phase-carrier column indices")
    if np.any(model.channels < 0) or np.any(model.channels > 2):
        raise ValueError("invalid phase-carrier channel indices")
    if not np.isclose(np.sum(model.weights), 1.0, atol=1e-5) or np.any(model.weights < 0.0):
        raise ValueError("invalid phase-carrier weights")
    return model


@click.group()
def main() -> None:
    """Discover and evaluate an exact-geometry phase carrier."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


@main.command()
@click.argument("positives", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--peak-count", type=click.IntRange(min=1), default=256, show_default=True)
@click.option("--candidate-codebook", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--candidate-count", type=click.IntRange(min=1), default=16384, show_default=True)
@click.option("--model-out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def discover(
    positives: tuple[Path, ...],
    peak_count: int,
    candidate_codebook: Path | None,
    candidate_count: int,
    model_out: Path,
) -> None:
    """Learn a phase carrier from exact-geometry POSITIVES."""
    candidate_bins: np.ndarray | None = None
    if candidate_codebook is not None:
        from synthid_v3_codebook_probe import load_v3_model

        with Image.open(positives[0]) as first:
            width, height = first.size
        prior = load_v3_model(
            candidate_codebook,
            height=height,
            width=width,
            peak_count=candidate_count,
        )
        candidate_bins = np.column_stack((prior.rows, prior.columns, prior.channels))
    model = discover_model(list(positives), peak_count=peak_count, candidate_bins=candidate_bins)
    save_model(model_out, model)
    log.info("Wrote phase-carrier model: %s", model_out)


@main.command()
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("images", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--report-out", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--canonicalize-geometry", is_flag=True, help="Resize inputs to the model geometry before scoring.")
@click.option("--max-shift", type=click.IntRange(min=0, max=MAX_TRANSLATION_SHIFT), default=0, show_default=True)
def score(
    model_path: Path,
    images: tuple[Path, ...],
    report_out: Path,
    canonicalize_geometry: bool,
    max_shift: int,
) -> None:
    """Score IMAGES with MODEL_PATH."""
    model = load_model(model_path)
    scores = (
        [asdict(score_image(image, model, canonicalize_geometry=canonicalize_geometry)) for image in images]
        if max_shift == 0
        else [
            asdict(
                score_translations(
                    image,
                    model,
                    max_shift=max_shift,
                    canonicalize_geometry=canonicalize_geometry,
                )
            )
            for image in images
        ]
    )
    payload = {
        "model": str(model_path),
        "height": model.height,
        "width": model.width,
        "peak_count": len(model.rows),
        "canonicalize_geometry": canonicalize_geometry,
        "max_shift": max_shift,
        "scores": scores,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log.info("Wrote phase-carrier score report: %s", report_out)


if __name__ == "__main__":
    main()
