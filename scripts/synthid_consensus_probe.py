"""Discover a polarity-invariant spectral carrier from low-texture groups.

This research harness is intentionally separate from the shipping detector. It
uses repeated, independently generated low-texture images to find Fourier bins
whose phase is stable within a content group and whose phase axis is stable
across groups. Treat its output as a carrier hypothesis until it passes the
provider-oracle and hard-negative gates in the SynthID research plan.

Usage:
    uv run --extra pixels python scripts/synthid_consensus_probe.py discover \
        refs/black refs/white refs/red --limit 5 \
        --model-out .local-eval/synthid/consensus.npz

    uv run --extra pixels python scripts/synthid_consensus_probe.py score \
        .local-eval/synthid/consensus.npz images/*.png
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import click
import numpy as np
from PIL import Image, ImageFilter

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png", ".webp"}


@dataclass(frozen=True)
class ConsensusScore:
    """One image's alignment with a frozen carrier hypothesis."""

    path: str
    score: float
    active_weight_fraction: float
    peak_count: int


@dataclass(frozen=True)
class ConsensusModel:
    """A compact, pickle-free carrier hypothesis."""

    size: int
    peaks: np.ndarray
    axial_phase: np.ndarray
    weights: np.ndarray
    expected_magnitude: np.ndarray


def _image_paths(directory: Path, limit: int | None) -> list[Path]:
    """Return a deterministic list of supported images in DIRECTORY."""
    paths = sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise ValueError(f"no supported images in {directory}")
    return paths


def _high_pass_rgb(path: Path, size: int, blur_radius: float) -> np.ndarray:
    """Decode PATH, canonicalize its geometry, and remove local image content."""
    with Image.open(path) as source:
        image = source.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.float64)
    blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=blur_radius)), dtype=np.float64)
    return pixels - blurred


def _spectrum(path: Path, size: int, blur_radius: float) -> np.ndarray:
    """Return a centered channel-wise spectrum for PATH."""
    residual = _high_pass_rgb(path, size, blur_radius)
    return np.fft.fftshift(np.fft.fft2(residual, axes=(0, 1)), axes=(0, 1))


def _group_statistics(paths: list[Path], size: int, blur_radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Return phase coherence and mean magnitude for one reference group."""
    spectra = [_spectrum(path, size, blur_radius) for path in paths]
    units = [spectrum / (np.abs(spectrum) + 1e-12) for spectrum in spectra]
    mean_unit = np.mean(units, axis=0)
    coherence = np.abs(mean_unit)
    mean_magnitude = np.mean([np.abs(spectrum) for spectrum in spectra], axis=0)
    phase = np.angle(mean_unit)
    return coherence * np.exp(1j * phase), mean_magnitude


def _valid_half_plane(size: int, min_radius: float, max_radius_fraction: float) -> np.ndarray:
    """Return the nonredundant Fourier region allowed for carrier selection."""
    center = size // 2
    yy, xx = np.ogrid[:size, :size]
    dy = yy - center
    dx = xx - center
    radius = np.sqrt(np.square(dy) + np.square(dx))
    half_plane = (dy > 0) | ((dy == 0) & (dx > 0))
    off_axis = (dy != 0) & (dx != 0)
    return half_plane & off_axis & (radius >= min_radius) & (radius <= size * max_radius_fraction)


def discover_model(
    groups: list[list[Path]],
    *,
    size: int = 512,
    blur_radius: float = 2.0,
    peak_count: int = 256,
    min_radius: float = 8.0,
    max_radius_fraction: float = 0.4,
) -> ConsensusModel:
    """Discover a polarity-invariant carrier from independent image GROUPS."""
    if len(groups) < 2:
        raise ValueError("at least two reference groups are required")
    if any(len(group) < 2 for group in groups):
        raise ValueError("each reference group requires at least two images")

    group_units: list[np.ndarray] = []
    group_magnitudes: list[np.ndarray] = []
    for paths in groups:
        unit, magnitude = _group_statistics(paths, size, blur_radius)
        group_units.append(unit)
        group_magnitudes.append(magnitude)

    stacked = np.stack(group_units, axis=0)
    within_coherence = np.abs(stacked)
    group_phase = np.angle(stacked)
    axial_mean = np.mean(np.exp(2j * group_phase) * within_coherence, axis=0)
    axial_coherence = np.abs(axial_mean) / (np.mean(within_coherence, axis=0) + 1e-12)
    mean_within = np.mean(within_coherence, axis=0)
    expected_magnitude = np.mean(group_magnitudes, axis=0)

    magnitude_scale = np.median(expected_magnitude, axis=(0, 1), keepdims=True) + 1e-12
    magnitude_score = np.log1p(expected_magnitude / magnitude_scale)
    selection_score = np.square(mean_within) * np.square(axial_coherence) * magnitude_score
    selection_score *= _valid_half_plane(size, min_radius, max_radius_fraction)[:, :, None]

    candidate_count = int(np.count_nonzero(selection_score))
    if candidate_count < peak_count:
        raise ValueError(f"only {candidate_count} valid carrier candidates for {peak_count} peaks")
    flat = selection_score.ravel()
    indices = np.argpartition(flat, -peak_count)[-peak_count:]
    indices = indices[np.argsort(flat[indices])[::-1]]
    rows, columns, channels = np.unravel_index(indices, selection_score.shape)
    peaks = np.column_stack((rows - size // 2, columns - size // 2, channels)).astype(np.int32)

    axial_phase = 0.5 * np.angle(axial_mean[rows, columns, channels])
    weights = selection_score[rows, columns, channels]
    weights /= np.sum(weights)
    magnitudes = expected_magnitude[rows, columns, channels]
    return ConsensusModel(
        size=size,
        peaks=peaks,
        axial_phase=axial_phase.astype(np.float64),
        weights=weights.astype(np.float64),
        expected_magnitude=magnitudes.astype(np.float64),
    )


def score_image(path: Path, model: ConsensusModel, *, blur_radius: float = 2.0) -> ConsensusScore:
    """Score PATH against a frozen polarity-invariant carrier model."""
    spectrum = _spectrum(path, model.size, blur_radius)
    center = model.size // 2
    rows = center + model.peaks[:, 0]
    columns = center + model.peaks[:, 1]
    channels = model.peaks[:, 2]
    values = spectrum[rows, columns, channels]
    phase_alignment = np.cos(2.0 * (np.angle(values) - model.axial_phase))
    magnitude_gate = np.minimum(np.abs(values) / (model.expected_magnitude + 1e-12), 1.0)
    active_weights = model.weights * magnitude_gate
    active_weight = float(np.sum(active_weights))
    score = 0.0 if active_weight == 0.0 else float(np.sum(active_weights * phase_alignment) / active_weight)
    return ConsensusScore(
        path=str(path),
        score=score,
        active_weight_fraction=active_weight,
        peak_count=len(model.peaks),
    )


def save_model(path: Path, model: ConsensusModel) -> None:
    """Save MODEL as a pickle-free NPZ artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        size=np.asarray(model.size, dtype=np.int32),
        peaks=model.peaks.astype(np.int32),
        axial_phase=model.axial_phase.astype(np.float32),
        weights=model.weights.astype(np.float32),
        expected_magnitude=model.expected_magnitude.astype(np.float32),
    )


def load_model(path: Path) -> ConsensusModel:
    """Load and validate a pickle-free carrier model."""
    with np.load(path, allow_pickle=False) as artifact:
        model = ConsensusModel(
            size=int(artifact["size"]),
            peaks=np.asarray(artifact["peaks"], dtype=np.int32),
            axial_phase=np.asarray(artifact["axial_phase"], dtype=np.float64),
            weights=np.asarray(artifact["weights"], dtype=np.float64),
            expected_magnitude=np.asarray(artifact["expected_magnitude"], dtype=np.float64),
        )
    count = len(model.peaks)
    if model.peaks.ndim != 2 or model.peaks.shape[1] != 3:
        raise ValueError("invalid peak shape")
    if any(array.shape != (count,) for array in (model.axial_phase, model.weights, model.expected_magnitude)):
        raise ValueError("model arrays do not match peak count")
    if model.size < 64 or np.any(np.abs(model.peaks[:, :2]) >= model.size // 2):
        raise ValueError("invalid canonical size or peak coordinates")
    if np.any(model.peaks[:, 2] < 0) or np.any(model.peaks[:, 2] > 2):
        raise ValueError("invalid channel index")
    if not np.isclose(np.sum(model.weights), 1.0, atol=1e-5):
        raise ValueError("model weights must sum to one")
    return model


@click.group()
def main() -> None:
    """Run low-texture carrier discovery and scoring experiments."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


@main.command()
@click.argument("group_dirs", nargs=-1, required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--limit", type=click.IntRange(min=2))
@click.option("--size", type=click.IntRange(min=64), default=512, show_default=True)
@click.option("--peak-count", type=click.IntRange(min=1), default=256, show_default=True)
@click.option("--model-out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def discover(group_dirs: tuple[Path, ...], limit: int | None, size: int, peak_count: int, model_out: Path) -> None:
    """Discover a carrier from the images in each GROUP_DIRS directory."""
    groups = [_image_paths(directory, limit) for directory in group_dirs]
    model = discover_model(groups, size=size, peak_count=peak_count)
    save_model(model_out, model)
    log.info("Wrote consensus model: %s", model_out)


@main.command()
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("images", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--report-out", type=click.Path(dir_okay=False, path_type=Path))
def score(model_path: Path, images: tuple[Path, ...], report_out: Path | None) -> None:
    """Score IMAGES against MODEL_PATH."""
    model = load_model(model_path)
    payload = {
        "model": str(model_path),
        "scores": [asdict(score_image(image, model)) for image in images],
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    if report_out is None:
        log.info("%s", rendered.rstrip())
        return
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(rendered, encoding="utf-8")
    log.info("Wrote score report: %s", report_out)


if __name__ == "__main__":
    main()
