"""Independently evaluate a numeric reverse-SynthID V3 NPZ codebook.

The loader accepts only the documented dense or sparse numeric format-v2 arrays
and disables pickle. It does not import or execute third-party code. Scores are
exploratory: the external reference provenance and labels still require
independent oracle validation before this can support a SynthID detector claim.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import click
import numpy as np
from PIL import Image
from synthid_phase_registration import (
    MAX_TRANSLATION_SHIFT,
    extract_frequency_values,
    phase_adjustment,
    register_phase_translations,
)

log = logging.getLogger(__name__)
LOG_2 = np.log(2.0)


@dataclass(frozen=True)
class V3CarrierModel:
    """Selected numeric bins from one exact-resolution V3 profile."""

    height: int
    width: int
    rows: np.ndarray
    columns: np.ndarray
    channels: np.ndarray
    phases: np.ndarray
    weights: np.ndarray
    expected_magnitudes: np.ndarray


@dataclass(frozen=True)
class V3Score:
    """Phase-alignment scores for one image."""

    path: str
    phase_score: float
    axial_phase_score: float
    active_weight_fraction: float
    peak_count: int


@dataclass(frozen=True)
class V3RegisteredScore:
    """Best phase-alignment score across a bounded translation search."""

    path: str
    phase_score: float
    axial_phase_score: float
    active_weight_fraction: float
    peak_count: int
    row_shift: int
    column_shift: int


@dataclass(frozen=True)
class _CarrierCandidates:
    """Selected numeric carrier arrays from one V3 profile."""

    weights: np.ndarray
    rows: np.ndarray
    columns: np.ndarray
    channels: np.ndarray
    phases: np.ndarray
    magnitudes: np.ndarray


def _load_sparse_channel(artifact: np.lib.npyio.NpzFile, prefix: str, channel: int) -> tuple[np.ndarray, ...]:
    """Load one sparse channel without reconstructing full image-sized arrays."""
    indices = np.asarray(artifact[f"{prefix}idx_{channel}"], dtype=np.uint32)
    log_magnitudes = np.asarray(artifact[f"{prefix}mag_{channel}"])
    phases = np.asarray(artifact[f"{prefix}phase_{channel}"])
    coherence = np.asarray(artifact[f"{prefix}cons_{channel}"], dtype=np.float64) / 255.0
    if not (indices.shape == log_magnitudes.shape == phases.shape == coherence.shape):
        raise ValueError("sparse profile arrays have inconsistent shapes")
    return indices, log_magnitudes, phases, coherence


def _top_positions(selection: np.ndarray, tie_breaker: np.ndarray, peak_count: int) -> np.ndarray:
    """Return deterministic descending positions for the strongest candidates."""
    if len(selection) < peak_count:
        raise ValueError(f"profile exposes only {len(selection)} eligible bins")
    cutoff = np.partition(selection, -peak_count)[-peak_count]
    stronger = np.flatnonzero(selection > cutoff)
    tied = np.flatnonzero(selection == cutoff)
    remaining = peak_count - len(stronger)
    if remaining < len(tied):
        tied = tied[np.argpartition(tie_breaker[tied], -remaining)[-remaining:]]
    positions = np.concatenate((stronger, tied))
    order = np.lexsort((-tie_breaker[positions], -selection[positions]))
    return positions[order]


def _dense_candidates(
    artifact: np.lib.npyio.NpzFile,
    prefix: str,
    *,
    height: int,
    width: int,
    min_radius: float,
    peak_count: int,
) -> _CarrierCandidates:
    """Select candidate arrays from one dense numeric profile."""
    shape = (height, width // 2 + 1, 3)
    log_magnitudes = np.asarray(artifact[f"{prefix}mag"])
    phases = np.asarray(artifact[f"{prefix}phase"])
    coherence = np.asarray(artifact[f"{prefix}cons"], dtype=np.float64) / 255.0
    if not (log_magnitudes.shape == phases.shape == coherence.shape == shape):
        raise ValueError("dense profile arrays have inconsistent shapes")

    rows = np.arange(height)
    signed_rows = np.where(rows > height // 2, rows - height, rows)
    columns = np.arange(shape[1])
    radius = np.sqrt(np.square(signed_rows[:, None]) + np.square(columns[None, :]))
    valid_spatial = (radius >= min_radius) & (columns[None, :] > 0)
    valid = np.broadcast_to(valid_spatial[:, :, None], shape)
    flat_valid = np.flatnonzero(valid)
    selection = (np.square(coherence) * log_magnitudes * LOG_2).ravel()[flat_valid]
    positions = _top_positions(selection, flat_valid, peak_count)
    selected = flat_valid[positions]
    selected_rows, selected_columns, selected_channels = np.unravel_index(selected, shape)
    return _CarrierCandidates(
        weights=selection[positions],
        rows=selected_rows,
        columns=selected_columns,
        channels=selected_channels,
        phases=np.asarray(phases.ravel()[selected], dtype=np.float64),
        magnitudes=np.exp2(np.asarray(log_magnitudes.ravel()[selected], dtype=np.float64)) - 1.0,
    )


def _sparse_candidates(
    artifact: np.lib.npyio.NpzFile,
    prefix: str,
    *,
    height: int,
    width: int,
    min_radius: float,
    peak_count: int,
) -> _CarrierCandidates:
    """Select candidate arrays from one sparse numeric profile."""
    half_width = width // 2 + 1
    selections: list[np.ndarray] = []
    candidate_rows: list[np.ndarray] = []
    candidate_columns: list[np.ndarray] = []
    candidate_channels: list[np.ndarray] = []
    candidate_phases: list[np.ndarray] = []
    candidate_log_magnitudes: list[np.ndarray] = []
    for channel in range(3):
        indices, log_magnitudes, phases, coherence = _load_sparse_channel(artifact, prefix, channel)
        rows, columns = np.unravel_index(indices, (height, half_width))
        signed_rows = np.where(rows > height // 2, rows - height, rows)
        radius = np.sqrt(np.square(signed_rows) + np.square(columns))
        valid = (radius >= min_radius) & (columns > 0)
        selections.append((np.square(coherence) * log_magnitudes * LOG_2)[valid])
        candidate_rows.append(rows[valid])
        candidate_columns.append(columns[valid])
        candidate_channels.append(np.full(np.count_nonzero(valid), channel, dtype=np.int8))
        candidate_phases.append(phases[valid])
        candidate_log_magnitudes.append(log_magnitudes[valid])

    selection = np.concatenate(selections)
    rows = np.concatenate(candidate_rows)
    columns = np.concatenate(candidate_columns)
    channels = np.concatenate(candidate_channels)
    tie_breaker = np.ravel_multi_index((rows, columns, channels), (height, half_width, 3))
    selected = _top_positions(selection, tie_breaker, peak_count)
    return _CarrierCandidates(
        weights=selection[selected],
        rows=rows[selected],
        columns=columns[selected],
        channels=channels[selected],
        phases=np.asarray(np.concatenate(candidate_phases)[selected], dtype=np.float64),
        magnitudes=np.exp2(np.asarray(np.concatenate(candidate_log_magnitudes)[selected], dtype=np.float64)) - 1.0,
    )


def load_v3_model(
    path: Path,
    *,
    height: int,
    width: int,
    peak_count: int = 256,
    min_radius: float = 15.0,
) -> V3CarrierModel:
    """Load top phase-consistent bins from a numeric V3 codebook profile."""
    prefix = f"{height}x{width}/"
    with np.load(path, allow_pickle=False) as artifact:
        if int(artifact["format_version"]) != 2:
            raise ValueError("only numeric V3 format version 2 is supported")
        loader = _sparse_candidates if bool(int(artifact[f"{prefix}sparse"])) else _dense_candidates
        candidates = loader(
            artifact,
            prefix,
            height=height,
            width=width,
            min_radius=min_radius,
            peak_count=peak_count,
        )
    return V3CarrierModel(
        height=height,
        width=width,
        rows=np.asarray(candidates.rows, dtype=np.int32),
        columns=np.asarray(candidates.columns, dtype=np.int32),
        channels=np.asarray(candidates.channels, dtype=np.int8),
        phases=candidates.phases,
        weights=candidates.weights / np.sum(candidates.weights),
        expected_magnitudes=candidates.magnitudes,
    )


def _load_profile_rgb(path: Path, model: V3CarrierModel) -> np.ndarray:
    """Load PATH and resize only when it does not match the profile geometry."""
    with Image.open(path) as source:
        image = source.convert("RGB")
        if image.size != (model.width, model.height):
            image = image.resize((model.width, model.height), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float64)


def _frequency_values(path: Path, model: V3CarrierModel) -> np.ndarray:
    """Return the selected complex coefficients from PATH."""
    pixels = _load_profile_rgb(path, model)
    return extract_frequency_values(pixels, model.rows, model.columns, model.channels)


def _score_values(
    values: np.ndarray,
    model: V3CarrierModel,
    phase_offsets: np.ndarray | float = 0.0,
) -> tuple[float, float, float]:
    """Return phase, axial-phase, and active-weight scores for VALUES."""
    phase_difference = np.angle(values) - model.phases
    magnitude_gate = np.minimum(np.abs(values) / (model.expected_magnitudes + 1e-12), 1.0)
    active_weights = model.weights * magnitude_gate
    active_weight = float(np.sum(active_weights))
    if active_weight == 0.0:
        phase_score = 0.0
        axial_score = 0.0
    else:
        adjusted = phase_difference + phase_offsets
        phase_score = float(np.sum(active_weights * np.cos(adjusted)) / active_weight)
        axial_score = float(np.sum(active_weights * np.cos(2.0 * adjusted)) / active_weight)
    return phase_score, axial_score, active_weight


def score_image(path: Path, model: V3CarrierModel) -> V3Score:
    """Score PATH against selected V3 phase bins."""
    phase_score, axial_score, active_weight = _score_values(_frequency_values(path, model), model)
    return V3Score(
        path=str(path),
        phase_score=phase_score,
        axial_phase_score=axial_score,
        active_weight_fraction=active_weight,
        peak_count=len(model.rows),
    )


def score_translations(path: Path, model: V3CarrierModel, *, max_shift: int = 4) -> V3RegisteredScore:
    """Return the best score after compensating bounded integer translations."""
    values = _frequency_values(path, model)
    registration = register_phase_translations(
        values,
        phases=model.phases,
        weights=model.weights,
        expected_magnitudes=model.expected_magnitudes,
        rows=model.rows,
        columns=model.columns,
        height=model.height,
        width=model.width,
        max_shift=max_shift,
    )
    selected_adjustment = phase_adjustment(
        model.rows,
        model.columns,
        height=model.height,
        width=model.width,
        row_shift=registration.row_shift,
        column_shift=registration.column_shift,
    )
    phase_score, axial_score, active_weight = _score_values(values, model, selected_adjustment)
    return V3RegisteredScore(
        path=str(path),
        phase_score=phase_score,
        axial_phase_score=axial_score,
        active_weight_fraction=active_weight,
        peak_count=len(model.rows),
        row_shift=registration.row_shift,
        column_shift=registration.column_shift,
    )


@click.command()
@click.argument("codebook", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("images", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--height", type=click.IntRange(min=64), required=True)
@click.option("--width", type=click.IntRange(min=64), required=True)
@click.option("--peak-count", type=click.IntRange(min=1), default=256, show_default=True)
@click.option("--max-shift", type=click.IntRange(min=0, max=MAX_TRANSLATION_SHIFT), default=0, show_default=True)
@click.option("--report-out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(
    codebook: Path,
    images: tuple[Path, ...],
    height: int,
    width: int,
    peak_count: int,
    max_shift: int,
    report_out: Path,
) -> None:
    """Score IMAGES against one exact-resolution profile from CODEBOOK."""
    model = load_v3_model(codebook, height=height, width=width, peak_count=peak_count)
    scores = (
        [asdict(score_image(image, model)) for image in images]
        if max_shift == 0
        else [asdict(score_translations(image, model, max_shift=max_shift)) for image in images]
    )
    payload = {
        "codebook": str(codebook),
        "height": height,
        "width": width,
        "peak_count": peak_count,
        "max_shift": max_shift,
        "scores": scores,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log.info("Wrote V3 score report: %s", report_out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
