"""Probe a periodic carrier lattice and codeword with split confirmation.

The synchronization parameters are selected on one checkerboard of image
patches and scored on the disjoint checkerboard. The statistics measure complex
phase coherence and a content-whitened multichannel template match after
correcting every patch for its global origin. They do not use filenames,
metadata, or scene-class features.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import click
import cv2
import numpy as np
from synthid_pixel_attack import jpeg_round_trip, load_rgb
from synthid_runtime._synthid_confirmation import (
    registered_confirmation_passes as runtime_registered_confirmation_passes,
)
from synthid_runtime._synthid_registered import RegisteredComponents, registered_components
from synthid_runtime.synthid_detector import fold_residual_template, folded_template_score, unit_tile

if TYPE_CHECKING:
    from numpy.typing import NDArray

log = logging.getLogger(__name__)

FIXED_CANDIDATE_THRESHOLD = 0.28
PATCH_SHIFT_MIN_MARGIN = 0.45
PATCH_SHIFT_STRONG_MARGIN = 1.0
PATCH_SHIFT_MIN_Z = 2.5
OPPONENT_REGISTERED_FIXED_MIN = 0.16
OPPONENT_REGISTERED_RED_GREEN_MIN = 0.60
OPPONENT_REGISTERED_BLUE_YELLOW_MIN = 0.55
SAME_IMAGE_NULL_OFFSETS = (
    -2.0,
    -1.75,
    -1.5,
    -1.25,
    -1.0,
    -0.75,
    -0.5,
    -0.35,
    0.35,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
)


def _webp_round_trip(pixels: NDArray[Any], quality: int) -> NDArray[Any]:
    """Apply one in-memory WebP encode/decode while returning RGB pixels."""
    if quality < 1 or quality > 101:
        raise ValueError("WebP quality must be between 1 and 101")
    success, encoded = cv2.imencode(
        ".webp",
        cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_WEBP_QUALITY, quality],
    )
    if not success:
        raise RuntimeError("WebP encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("WebP decoding failed")
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


@dataclass(frozen=True)
class LatticeScore:
    """One select-confirm reciprocal-lattice observation."""

    selected_period: float
    selected_rotation_degrees: float
    selected_orientation_degrees: int
    selected_horizontal_reflection: bool
    selected_deskew_degrees: float
    deskew_candidate_count: int
    deskew_selection_coherence: float
    deskew_direct_selection_match: float
    deskew_direct_confirmation_match: float
    deskew_direct_joint_match: float
    peak_period: float
    peak_rotation_degrees: float
    peak_selection_coherence: float
    selection_coherence: float
    selection_candidate_p95: float
    selection_candidate_p99: float
    selection_excess_p95: float
    selection_excess_p99: float
    confirmation_coherence: float
    confirmation_candidate_p95: float
    confirmation_candidate_p99: float
    confirmation_excess_p95: float
    confirmation_excess_p99: float
    joint_coherence: float
    joint_excess_p99: float
    selected_shift_y: int
    selected_shift_x: int
    selection_codeword: float
    confirmation_codeword: float
    confirmation_codeword_shift_p95: float
    confirmation_codeword_shift_p99: float
    joint_codeword: float
    unknown_codeword_shift_y: int
    unknown_codeword_shift_x: int
    unknown_codeword_selection: float
    unknown_codeword_confirmation: float
    unknown_codeword_fixed_confirmation: float
    unknown_codeword_fixed_all: float
    unknown_codeword_confirmation_p95: float
    unknown_codeword_confirmation_p99: float
    unknown_codeword_excess_p99: float
    amplitude_shift_y: int
    amplitude_shift_x: int
    selection_amplitude: float
    confirmation_amplitude: float
    joint_amplitude: float
    selection_whitened_match: float
    confirmation_whitened_match: float
    joint_whitened_match: float
    amplitude_candidate_count: int
    amplitude_rerank_count: int
    canonical_template_score: float
    canonical_registered_template_score: float
    selection_patches: int
    confirmation_patches: int


@dataclass(frozen=True)
class _AmplitudeScore:
    """Amplitude-aware select-confirm score for one period candidate."""

    period: float
    selection: float
    confirmation: float
    shift_y: int
    shift_x: int
    unaligned_full: float
    aligned_full: float


@dataclass(frozen=True)
class SameImageNullScore:
    """Target-period coherence relative to neighboring periods in one image."""

    target_period: float
    target_selection_coherence: float
    target_confirmation_coherence: float
    joint_coherence: float
    selection_off_period_max: float
    confirmation_off_period_max: float
    joint_excess: float
    off_period_count: int


@dataclass(frozen=True)
class PatchShiftConsensusScore:
    """Robust per-patch agreement on one cyclic carrier phase."""

    period: float
    selected_shift_y: int
    selected_shift_x: int
    selection_trimmed_z: float
    confirmation_trimmed_z: float
    joint_trimmed_z: float
    selection_support_fraction: float
    confirmation_support_fraction: float
    joint_support_fraction: float
    selection_patches: int
    confirmation_patches: int


@dataclass(frozen=True)
class OpponentRegisteredScore:
    """Scale-registered opponent-color carrier observation."""

    selected_period: float
    spectral_period: float
    spectral_score: float
    fixed_score: float
    red_green_spatial: float
    blue_yellow_spatial: float
    candidate_count: int

    @property
    def decision_score(self) -> float:
        """Return the minimum normalized research gate margin."""
        return min(
            self.fixed_score / OPPONENT_REGISTERED_FIXED_MIN,
            self.red_green_spatial / OPPONENT_REGISTERED_RED_GREEN_MIN,
            self.blue_yellow_spatial / OPPONENT_REGISTERED_BLUE_YELLOW_MIN,
        )


def fixed_candidate_passes(score: float) -> bool:
    """Return the frozen precision-first fixed-period candidate verdict."""
    return math.isfinite(score) and score >= FIXED_CANDIDATE_THRESHOLD


def registered_confirmation_passes(score: LatticeScore) -> bool:
    """Return the frozen period-aware split-confirmation verdict."""
    return runtime_registered_confirmation_passes(
        score.selected_period,
        score.joint_coherence,
        score.joint_amplitude,
        score.unknown_codeword_fixed_confirmation,
    )


def patch_shift_recovery_passes(
    *,
    amplitude_margin: float,
    high_band_margin: float,
    periods_agree: bool,
    confirmation_passes: bool,
    joint_trimmed_z: float,
) -> bool:
    """Apply the frozen research-only content-adaptive recovery rule."""
    margins = (amplitude_margin, high_band_margin)
    return (
        all(math.isfinite(value) for value in (*margins, joint_trimmed_z))
        and periods_agree
        and confirmation_passes
        and min(margins) >= PATCH_SHIFT_MIN_MARGIN
        and max(margins) >= PATCH_SHIFT_STRONG_MARGIN
        and joint_trimmed_z >= PATCH_SHIFT_MIN_Z
    )


def _opponent_channels(values: NDArray[Any]) -> NDArray[Any]:
    """Return Green, Red-minus-Green, and Blue-minus-Yellow channels."""
    red = values[:, :, 0]
    green = values[:, :, 1]
    blue = values[:, :, 2]
    return np.stack((green, red - green, blue - 0.5 * (red + green)), axis=2)


def _opponent_color_pair(values: NDArray[Any]) -> NDArray[Any]:
    """Return the large-carrier Red-Green and Blue-Yellow planes."""
    red = values[:, :, 0]
    green = values[:, :, 1]
    blue = values[:, :, 2]
    return np.stack((red - green, blue - 0.5 * (red + green)), axis=2)


def template_harmonics(template: NDArray[Any], count: int = 16) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Return strong unique half-plane harmonics and their channel weights."""
    if template.ndim != 3 or template.shape[2] != 3:
        raise ValueError("template must have shape (height, width, 3)")
    if count < 1:
        raise ValueError("count must be positive")
    opponent = _opponent_channels(np.asarray(template, dtype=np.float64))
    spectrum = np.fft.fft2(opponent, axes=(0, 1))
    height, width = template.shape[:2]
    candidates: list[tuple[float, int, int, int, int]] = []
    for row in range(height):
        signed_row = row if row <= height // 2 else row - height
        for column in range(width):
            signed_column = column if column <= width // 2 else column - width
            if signed_row < 0 or (signed_row == 0 and signed_column <= 0):
                continue
            power = float(np.sum(np.abs(spectrum[row, column]) ** 2))
            candidates.append((power, signed_row, signed_column, row, column))
    candidates.sort(reverse=True)
    selected = candidates[:count]
    harmonics = np.asarray([(row, column) for _power, row, column, _y, _x in selected], dtype=np.float64)
    coefficients = np.asarray([spectrum[y, x] for _power, _row, _column, y, x in selected])
    weights = np.abs(coefficients)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        raise ValueError("template has no nonzero periodic harmonics")
    coefficient_units = np.divide(
        coefficients,
        np.abs(coefficients),
        out=np.zeros_like(coefficients),
        where=np.abs(coefficients) > 1e-12,
    )
    return harmonics, weights / weight_sum, coefficient_units


def _patch_origins(height: int, width: int, patch_size: int, grid_size: int) -> list[tuple[int, int, int]]:
    if patch_size < 32 or height < patch_size or width < patch_size:
        raise ValueError("image must contain at least one patch of the requested size")
    if grid_size < 2:
        raise ValueError("grid size must be at least two")
    y_values = np.linspace(0, height - patch_size, min(grid_size, height // patch_size), dtype=np.int64)
    x_values = np.linspace(0, width - patch_size, min(grid_size, width // patch_size), dtype=np.int64)
    origins = []
    for y_index, y in enumerate(np.unique(y_values)):
        for x_index, x in enumerate(np.unique(x_values)):
            origins.append((int(y), int(x), (y_index + x_index) % 2))
    if {group for _y, _x, group in origins} != {0, 1}:
        raise ValueError("image geometry does not provide two patch groups")
    return origins


def _bilinear_sample(
    spectrum: NDArray[Any],
    y: NDArray[Any],
    x: NDArray[Any],
) -> NDArray[Any]:
    height, width = spectrum.shape
    y_floor = np.floor(y)
    x_floor = np.floor(x)
    y0 = y_floor.astype(np.int64) % height
    x0 = x_floor.astype(np.int64) % width
    y1 = (y0 + 1) % height
    x1 = (x0 + 1) % width
    dy = y - y_floor
    dx = x - x_floor
    return (
        spectrum[y0, x0] * (1.0 - dy) * (1.0 - dx)
        + spectrum[y1, x0] * dy * (1.0 - dx)
        + spectrum[y0, x1] * (1.0 - dy) * dx
        + spectrum[y1, x1] * dy * dx
    )


def _candidate_frequencies(
    periods: NDArray[Any],
    rotations_degrees: NDArray[Any],
    harmonics: NDArray[Any],
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    period_grid, rotation_grid = np.meshgrid(periods, rotations_degrees, indexing="ij")
    flat_periods = period_grid.ravel()
    flat_rotations = rotation_grid.ravel()
    angles = np.deg2rad(flat_rotations)
    cosine = np.cos(angles)[:, None]
    sine = np.sin(angles)[:, None]
    base_y = harmonics[None, :, 0] / flat_periods[:, None]
    base_x = harmonics[None, :, 1] / flat_periods[:, None]
    frequencies_y = sine * base_x + cosine * base_y
    frequencies_x = cosine * base_x - sine * base_y
    return flat_periods, flat_rotations, frequencies_y, frequencies_x


def _patch_unit_values(
    pixels: NDArray[Any],
    origin_y: int,
    origin_x: int,
    patch_size: int,
    frequencies_y: NDArray[Any],
    frequencies_x: NDArray[Any],
) -> NDArray[Any]:
    patch = np.asarray(
        pixels[origin_y : origin_y + patch_size, origin_x : origin_x + patch_size],
        dtype=np.float32,
    )
    channels = _opponent_channels(patch)
    window_1d = np.hanning(patch_size).astype(np.float32)
    window = window_1d[:, None] * window_1d[None, :]
    sampled = np.empty((*frequencies_y.shape, 3), dtype=np.complex128)
    sample_y = frequencies_y * patch_size
    sample_x = frequencies_x * patch_size
    for channel in range(3):
        residual = channels[:, :, channel]
        residual = residual - cv2.GaussianBlur(
            residual,
            (0, 0),
            sigmaX=1.0,
            sigmaY=1.0,
            borderType=cv2.BORDER_REFLECT_101,
        )
        spectrum = np.fft.fft2(residual * window)
        sampled[:, :, channel] = _bilinear_sample(spectrum, sample_y, sample_x)
    phase_correction = np.exp(-2j * math.pi * (frequencies_y * origin_y + frequencies_x * origin_x))
    sampled *= phase_correction[:, :, None]
    magnitudes = np.abs(sampled)
    return np.divide(sampled, magnitudes, out=np.zeros_like(sampled), where=magnitudes > 1e-12)


def _coherence(unit_values: list[NDArray[Any]], weights: NDArray[Any]) -> NDArray[Any]:
    if not unit_values:
        raise ValueError("coherence needs at least one patch")
    values = np.stack(unit_values)
    coherence = np.abs(np.mean(values, axis=0))
    return np.sum(coherence * weights[None, :, :], axis=(1, 2))


def score_same_image_period_null(
    pixels: NDArray[Any],
    template: NDArray[Any],
    target_period: float,
    *,
    period_min: float = 7.5,
    period_max: float = 24.5,
    patch_size: int = 256,
    grid_size: int = 4,
    harmonic_count: int = 16,
) -> SameImageNullScore:
    """Compare a fixed target period with preregistered neighboring periods."""
    if not math.isfinite(target_period) or not period_min <= target_period <= period_max:
        raise ValueError("target period must be finite and inside the null range")
    off_periods = np.asarray(
        [
            target_period + offset
            for offset in SAME_IMAGE_NULL_OFFSETS
            if period_min <= target_period + offset <= period_max
        ],
        dtype=np.float64,
    )
    if not len(off_periods):
        raise ValueError("same-image period null needs at least one neighboring period")
    periods = np.concatenate((np.asarray([target_period]), off_periods))
    harmonics, weights, _coefficient_units = template_harmonics(template, harmonic_count)
    _periods, _rotations, frequencies_y, frequencies_x = _candidate_frequencies(
        periods,
        np.asarray([0.0]),
        harmonics,
    )
    grouped_values: dict[int, list[NDArray[Any]]] = {0: [], 1: []}
    for origin_y, origin_x, group in _patch_origins(*pixels.shape[:2], patch_size, grid_size):
        grouped_values[group].append(
            _patch_unit_values(
                pixels,
                origin_y,
                origin_x,
                patch_size,
                frequencies_y,
                frequencies_x,
            )
        )
    selection = _coherence(grouped_values[0], weights)
    confirmation = _coherence(grouped_values[1], weights)
    selection_off_max = float(np.max(selection[1:]))
    confirmation_off_max = float(np.max(confirmation[1:]))
    target_selection = float(selection[0])
    target_confirmation = float(confirmation[0])
    return SameImageNullScore(
        target_period=target_period,
        target_selection_coherence=target_selection,
        target_confirmation_coherence=target_confirmation,
        joint_coherence=min(target_selection, target_confirmation),
        selection_off_period_max=selection_off_max,
        confirmation_off_period_max=confirmation_off_max,
        joint_excess=min(
            target_selection - selection_off_max,
            target_confirmation - confirmation_off_max,
        ),
        off_period_count=len(off_periods),
    )


def _codeword_scores(
    selection_values: list[NDArray[Any]],
    confirmation_values: list[NDArray[Any]],
    selected_index: int,
    harmonics: NDArray[Any],
    coefficient_units: NDArray[Any],
    weights: NDArray[Any],
    tile_size: int,
) -> tuple[int, int, float, float, float, float]:
    selection_mean = np.mean(np.stack(selection_values), axis=0)[selected_index]
    confirmation_mean = np.mean(np.stack(confirmation_values), axis=0)[selected_index]
    shift_y, shift_x = np.meshgrid(np.arange(tile_size), np.arange(tile_size), indexing="ij")
    flat_y = shift_y.ravel()
    flat_x = shift_x.ravel()
    phase = np.exp(
        -2j * math.pi * (flat_y[:, None] * harmonics[None, :, 0] + flat_x[:, None] * harmonics[None, :, 1]) / tile_size
    )
    shifted_codewords = coefficient_units[None, :, :] * phase[:, :, None]

    def scores(values: NDArray[Any]) -> NDArray[Any]:
        agreement = np.real(values[None, :, :] * np.conj(shifted_codewords))
        return np.sum(agreement * weights[None, :, :], axis=(1, 2))

    selection_scores = scores(selection_mean)
    confirmation_scores = scores(confirmation_mean)
    selected_shift = int(np.argmax(selection_scores))
    return (
        int(flat_y[selected_shift]),
        int(flat_x[selected_shift]),
        float(selection_scores[selected_shift]),
        float(confirmation_scores[selected_shift]),
        float(np.quantile(confirmation_scores, 0.95)),
        float(np.quantile(confirmation_scores, 0.99)),
    )


def _unknown_codeword_scores(
    selection_values: list[NDArray[Any]],
    confirmation_values: list[NDArray[Any]],
    selected_index: int,
    harmonics: NDArray[Any],
    weights: NDArray[Any],
    tile_size: int,
) -> tuple[int, int, float, float, float, float, float, float]:
    """Fit an unknown patch codeword and confirm it on held-out harmonics."""
    if len(harmonics) < 4:
        raise ValueError("unknown-codeword confirmation needs at least four harmonics")
    selection_mean = np.mean(np.stack(selection_values), axis=0)[selected_index]
    confirmation_mean = np.mean(np.stack(confirmation_values), axis=0)[selected_index]
    cross_codeword = confirmation_mean * np.conj(selection_mean)

    shift_y, shift_x = np.meshgrid(np.arange(tile_size), np.arange(tile_size), indexing="ij")
    flat_y = shift_y.ravel()
    flat_x = shift_x.ravel()
    phase = np.exp(
        -2j * math.pi * (flat_y[:, None] * harmonics[None, :, 0] + flat_x[:, None] * harmonics[None, :, 1]) / tile_size
    )
    shifted_cross = cross_codeword[None, :, :] * phase[:, :, None]
    selection_mask = np.arange(len(harmonics)) % 2 == 0
    confirmation_mask = ~selection_mask

    def scores(mask: NDArray[Any]) -> NDArray[Any]:
        masked_weights = weights[mask]
        normalizer = float(np.sum(masked_weights))
        if normalizer <= 0.0:
            raise ValueError("unknown-codeword harmonic split has no template weight")
        return np.abs(np.sum(shifted_cross[:, mask, :] * masked_weights[None, :, :], axis=(1, 2))) / normalizer

    selection_scores = scores(selection_mask)
    confirmation_scores = scores(confirmation_mask)
    selected_shift = int(np.argmax(selection_scores))
    fixed_all = float(np.abs(np.sum(cross_codeword * weights)))
    return (
        int(flat_y[selected_shift]),
        int(flat_x[selected_shift]),
        float(selection_scores[selected_shift]),
        float(confirmation_scores[selected_shift]),
        float(confirmation_scores[0]),
        fixed_all,
        float(np.quantile(confirmation_scores, 0.95)),
        float(np.quantile(confirmation_scores, 0.99)),
    )


def _period_candidate_indices(
    periods: NDArray[Any],
    rotations_degrees: NDArray[Any],
    selection_scores: NDArray[Any],
    count: int,
) -> list[int]:
    """Return separated zero-rotation candidates ranked on selection patches."""
    candidates: list[int] = []
    eligible = np.flatnonzero(np.isclose(rotations_degrees, 0.0, atol=1e-12))
    for index in eligible[np.argsort(selection_scores[eligible])[::-1]]:
        if any(abs(float(periods[index] - periods[prior])) < 0.5 for prior in candidates):
            continue
        candidates.append(int(index))
        if len(candidates) == count:
            break
    if not candidates:
        raise ValueError("amplitude confirmation requires a zero-rotation candidate")
    return candidates


def _period_alias_candidate_indices(
    periods: NDArray[Any],
    rotations_degrees: NDArray[Any],
    base_indices: list[int],
) -> list[int]:
    """Expand coarse periods with octave aliases and adjacent grid bins."""
    eligible = np.flatnonzero(np.isclose(rotations_degrees, 0.0, atol=1e-12))
    eligible_periods = periods[eligible]
    candidates: list[int] = []
    seen: set[int] = set()
    for base_index in base_indices:
        for ratio in (1.0, 0.5, 2.0):
            target = float(periods[base_index] * ratio)
            if target < float(np.min(eligible_periods)) or target > float(np.max(eligible_periods)):
                continue
            center = int(np.argmin(np.abs(eligible_periods - target)))
            for position in range(max(0, center - 1), min(len(eligible), center + 2)):
                index = int(eligible[position])
                if index not in seen:
                    seen.add(index)
                    candidates.append(index)
    return candidates


def _cyclic_correlations(template: NDArray[Any], tile: NDArray[Any]) -> NDArray[Any]:
    template_spectrum = np.fft.fft2(template, axes=(0, 1))
    tile_spectrum = np.fft.fft2(tile, axes=(0, 1))
    return np.fft.ifft2(np.sum(template_spectrum * np.conj(tile_spectrum), axis=2)).real


def _canonical_pixels(pixels: NDArray[Any], template: NDArray[Any], period: float) -> NDArray[Any]:
    width = max(template.shape[1], round(pixels.shape[1] * template.shape[1] / period))
    height = max(template.shape[0], round(pixels.shape[0] * template.shape[0] / period))
    if (height, width) == pixels.shape[:2]:
        return pixels
    interpolation = cv2.INTER_AREA if width < pixels.shape[1] else cv2.INTER_CUBIC
    return np.asarray(cv2.resize(pixels, (width, height), interpolation=interpolation))


def _real_correlation(left: NDArray[Any], right: NDArray[Any]) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.real(np.vdot(right, left)) / denominator) if denominator > 0.0 else 0.0


def _opponent_period_curve(
    pixels: NDArray[Any],
    template: NDArray[Any],
    periods: NDArray[Any],
    *,
    harmonic_count: int,
) -> NDArray[Any]:
    """Return signed opponent-color template coherence over PERIODS."""
    template_opponent = _opponent_color_pair(np.asarray(template, dtype=np.float64))
    template_spectrum = np.fft.fft2(template_opponent, axes=(0, 1))
    power = np.sum(np.abs(template_spectrum) ** 2, axis=2)
    power[0, 0] = 0.0
    indices = np.argsort(power.ravel())[::-1][:harmonic_count]
    rows, columns = np.unravel_index(indices, power.shape)
    height, width = template.shape[:2]
    signed_rows = np.where(rows <= height // 2, rows, rows - height)
    signed_columns = np.where(columns <= width // 2, columns, columns - width)
    harmonics = np.column_stack((signed_rows, signed_columns)).astype(np.float64)
    coefficients = template_spectrum[rows, columns]

    image_height, image_width = pixels.shape[:2]
    sample_y = periods[:, None] ** -1 * harmonics[None, :, 0] * image_height
    sample_x = periods[:, None] ** -1 * harmonics[None, :, 1] * image_width
    sampled = np.empty((len(periods), len(harmonics), 2), dtype=np.complex128)
    image_opponent = _opponent_color_pair(np.asarray(pixels, dtype=np.float32))
    for channel in range(2):
        residual = image_opponent[:, :, channel]
        residual -= cv2.GaussianBlur(
            residual,
            (0, 0),
            sigmaX=1.0,
            sigmaY=1.0,
            borderType=cv2.BORDER_REFLECT_101,
        )
        sampled[:, :, channel] = _bilinear_sample(
            np.fft.fft2(residual),
            sample_y % image_height,
            sample_x % image_width,
        )
    numerator = np.real(np.sum(np.conj(coefficients)[None, :, :] * sampled, axis=(1, 2)))
    denominator = np.linalg.norm(coefficients) * np.linalg.norm(sampled, axis=(1, 2))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0.0)


def score_opponent_registered(
    pixels: NDArray[Any],
    template: NDArray[Any],
    *,
    periods: NDArray[Any],
    harmonic_count: int = 30,
    candidate_count: int = 5,
    denoise_sigma: float = 1.0,
) -> OpponentRegisteredScore:
    """Search scale using the large-carrier opponent-color representation."""
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("pixels must have shape (height, width, 3)")
    if periods.ndim != 1 or not len(periods) or np.any(~np.isfinite(periods)) or np.any(periods <= 0.0):
        raise ValueError("periods must be a nonempty positive finite vector")
    if harmonic_count < 1 or candidate_count < 1:
        raise ValueError("harmonic and candidate counts must be positive")
    curve = _opponent_period_curve(
        pixels,
        template,
        periods,
        harmonic_count=harmonic_count,
    )
    rotations = np.zeros_like(periods)
    candidate_indices = _period_candidate_indices(periods, rotations, curve, candidate_count)
    observations: list[OpponentRegisteredScore] = []
    for index in candidate_indices:
        period = float(periods[index])
        canonical = _canonical_pixels(pixels, template, period)
        fixed_score, folded = folded_template_score(canonical, template, denoise_sigma)
        folded_opponent = _opponent_color_pair(folded)
        template_opponent = _opponent_color_pair(template)
        observations.append(
            OpponentRegisteredScore(
                selected_period=period,
                spectral_period=float(periods[int(np.argmax(curve))]),
                spectral_score=float(curve[index]),
                fixed_score=fixed_score,
                red_green_spatial=_real_correlation(folded_opponent[:, :, 0], template_opponent[:, :, 0]),
                blue_yellow_spatial=_real_correlation(folded_opponent[:, :, 1], template_opponent[:, :, 1]),
                candidate_count=len(candidate_indices),
            )
        )
    return max(observations, key=lambda score: score.decision_score)


def _content_whitened_patch_score(
    pixels: NDArray[Any],
    template: NDArray[Any],
    harmonics: NDArray[Any],
    *,
    denoise_sigma: float,
    noise_radius: int = 4,
    guard_radius: int = 1,
    ridge: float = 0.1,
) -> float:
    """Return a complex matched-filter cosine after local color whitening."""
    patch_height, patch_width = pixels.shape[:2]
    tile_height, tile_width = template.shape[:2]
    if patch_height % tile_height or patch_width % tile_width:
        raise ValueError("whitened patch dimensions must be multiples of the template dimensions")
    if noise_radius <= guard_radius or guard_radius < 0:
        raise ValueError("noise radius must exceed the nonnegative guard radius")
    if not math.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("whitening ridge must be finite and positive")

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
    spectrum = np.fft.fft2(channels, axes=(0, 1))
    template_spectrum = np.fft.fft2(_opponent_channels(np.asarray(template, dtype=np.float64)), axes=(0, 1))

    numerator = 0.0
    template_energy = 0.0
    observation_energy = 0.0
    for signed_row_value, signed_column_value in harmonics:
        signed_row = round(float(signed_row_value))
        signed_column = round(float(signed_column_value))
        bin_y = (signed_row * patch_height // tile_height) % patch_height
        bin_x = (signed_column * patch_width // tile_width) % patch_width
        noise = np.asarray(
            [
                spectrum[(bin_y + offset_y) % patch_height, (bin_x + offset_x) % patch_width]
                for offset_y in range(-noise_radius, noise_radius + 1)
                for offset_x in range(-noise_radius, noise_radius + 1)
                if max(abs(offset_y), abs(offset_x)) > guard_radius
            ]
        )
        covariance = noise.T @ np.conj(noise) / len(noise)
        local_power = float(np.trace(covariance).real / covariance.shape[0])
        covariance += np.eye(covariance.shape[0]) * (ridge * local_power + 1e-12)
        template_value = template_spectrum[signed_row % tile_height, signed_column % tile_width]
        observation = spectrum[bin_y, bin_x]
        whitened_template = np.linalg.solve(covariance, template_value)
        whitened_observation = np.linalg.solve(covariance, observation)
        numerator += float(np.vdot(template_value, whitened_observation).real)
        template_energy += float(np.vdot(template_value, whitened_template).real)
        observation_energy += float(np.vdot(observation, whitened_observation).real)
    denominator = math.sqrt(max(0.0, template_energy * observation_energy))
    return numerator / denominator if denominator > 1e-12 else 0.0


def _content_whitened_score(
    pixels: NDArray[Any],
    template: NDArray[Any],
    harmonics: NDArray[Any],
    *,
    period: float,
    patch_size: int,
    grid_size: int,
    denoise_sigma: float,
) -> tuple[float, float]:
    """Score a selected period on disjoint patch groups after local whitening."""
    canonical = _canonical_pixels(pixels, template, period)
    tile_height, tile_width = template.shape[:2]
    grouped_scores: dict[int, list[float]] = {0: [], 1: []}
    for origin_y, origin_x, group in _patch_origins(*canonical.shape[:2], patch_size, grid_size):
        aligned_y = (origin_y // tile_height) * tile_height
        aligned_x = (origin_x // tile_width) * tile_width
        patch = canonical[aligned_y : aligned_y + patch_size, aligned_x : aligned_x + patch_size]
        grouped_scores[group].append(
            _content_whitened_patch_score(
                patch,
                template,
                harmonics,
                denoise_sigma=denoise_sigma,
            )
        )
    return float(np.mean(grouped_scores[0])), float(np.mean(grouped_scores[1]))


def _amplitude_score(
    pixels: NDArray[Any],
    template: NDArray[Any],
    *,
    period: float,
    patch_size: int,
    grid_size: int,
    denoise_sigma: float,
) -> _AmplitudeScore:
    """Select cyclic phase on one patch group and confirm it on the other."""
    canonical = _canonical_pixels(pixels, template, period)
    grouped_units: dict[int, list[NDArray[Any]]] = {0: [], 1: []}
    tile_height, tile_width = template.shape[:2]
    for origin_y, origin_x, group in _patch_origins(*canonical.shape[:2], patch_size, grid_size):
        aligned_y = (origin_y // tile_height) * tile_height
        aligned_x = (origin_x // tile_width) * tile_width
        folded = fold_residual_template(
            canonical[aligned_y : aligned_y + patch_size, aligned_x : aligned_x + patch_size],
            tile_height=tile_height,
            tile_width=tile_width,
            denoise_sigma=denoise_sigma,
        )
        unit, _norm = unit_tile(folded)
        grouped_units[group].append(unit)
    selection_tile, _selection_norm = unit_tile(np.mean(grouped_units[0], axis=0))
    confirmation_tile, _confirmation_norm = unit_tile(np.mean(grouped_units[1], axis=0))
    selection_correlations = _cyclic_correlations(template, selection_tile)
    confirmation_correlations = _cyclic_correlations(template, confirmation_tile)
    selected_shift = int(np.argmax(selection_correlations))
    shift_y, shift_x = np.unravel_index(selected_shift, selection_correlations.shape)
    unaligned_full, full_folded = folded_template_score(canonical, template, denoise_sigma)
    aligned_full, _aligned_norm = unit_tile(np.roll(full_folded, shift=(shift_y, shift_x), axis=(0, 1)))
    return _AmplitudeScore(
        period=period,
        selection=float(selection_correlations[shift_y, shift_x]),
        confirmation=float(confirmation_correlations[shift_y, shift_x]),
        shift_y=int(shift_y),
        shift_x=int(shift_x),
        unaligned_full=unaligned_full,
        aligned_full=float(np.sum(template * aligned_full)),
    )


def _robust_shift_z(correlations: NDArray[Any]) -> NDArray[Any]:
    """Standardize each patch against its own cyclic-shift null."""
    flattened = correlations.reshape(correlations.shape[0], -1)
    center = np.median(flattened, axis=1, keepdims=True)
    mad = np.median(np.abs(flattened - center), axis=1, keepdims=True)
    scale = 1.4826 * mad
    fallback = np.std(flattened, axis=1, keepdims=True)
    scale = np.where(scale > 1e-12, scale, fallback)
    standardized = np.divide(
        flattened - center,
        scale,
        out=np.zeros_like(flattened),
        where=scale > 1e-12,
    )
    return standardized.reshape(correlations.shape)


def _top_half_mean(values: NDArray[Any], *, axis: int) -> NDArray[Any]:
    """Average the strongest half without letting one patch dominate."""
    count = values.shape[axis]
    retained = max(1, (count + 1) // 2)
    partitioned = np.partition(values, count - retained, axis=axis)
    indices = np.arange(count - retained, count)
    return np.mean(np.take(partitioned, indices, axis=axis), axis=axis)


def _shift_support_fraction(correlations: NDArray[Any], shift_y: int, shift_x: int) -> float:
    """Return the patch fraction placing one shift in its upper five percent."""
    selected = correlations[:, shift_y, shift_x]
    flattened = correlations.reshape(correlations.shape[0], -1)
    less = np.sum(flattened < selected[:, None], axis=1)
    equal = np.sum(flattened == selected[:, None], axis=1)
    percentile = (less + 0.5 * equal) / flattened.shape[1]
    return float(np.mean(percentile >= 0.95))


def score_patch_shift_consensus(
    pixels: NDArray[Any],
    template: NDArray[Any],
    period: float,
    *,
    patch_size: int = 256,
    grid_size: int = 4,
    denoise_sigma: float = 1.0,
) -> PatchShiftConsensusScore:
    """Select a robust cyclic phase and confirm it on disjoint patches.

    This probe targets a content-adaptive encoder that may place the shared
    detection carrier strongly in only part of an image. Every patch is
    normalized against its own 2-D cyclic-shift distribution before the
    strongest half are pooled. It is a research statistic, not a detector
    threshold.
    """
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError("period must be finite and positive")
    canonical = _canonical_pixels(pixels, template, period)
    tile_height, tile_width = template.shape[:2]
    grouped_correlations: dict[int, list[NDArray[Any]]] = {0: [], 1: []}
    for origin_y, origin_x, group in _patch_origins(*canonical.shape[:2], patch_size, grid_size):
        aligned_y = (origin_y // tile_height) * tile_height
        aligned_x = (origin_x // tile_width) * tile_width
        folded = fold_residual_template(
            canonical[aligned_y : aligned_y + patch_size, aligned_x : aligned_x + patch_size],
            tile_height=tile_height,
            tile_width=tile_width,
            denoise_sigma=denoise_sigma,
        )
        unit, _norm = unit_tile(folded)
        grouped_correlations[group].append(_cyclic_correlations(template, unit))

    selection = np.stack(grouped_correlations[0])
    confirmation = np.stack(grouped_correlations[1])
    selection_z = _robust_shift_z(selection)
    confirmation_z = _robust_shift_z(confirmation)
    selection_trimmed = _top_half_mean(selection_z, axis=0)
    selected_shift = int(np.argmax(selection_trimmed))
    shift_y, shift_x = np.unravel_index(selected_shift, selection_trimmed.shape)
    selection_score = float(selection_trimmed[shift_y, shift_x])
    confirmation_score = float(_top_half_mean(confirmation_z[:, shift_y, shift_x], axis=0))
    selection_support = _shift_support_fraction(selection, int(shift_y), int(shift_x))
    confirmation_support = _shift_support_fraction(confirmation, int(shift_y), int(shift_x))
    return PatchShiftConsensusScore(
        period=period,
        selected_shift_y=int(shift_y),
        selected_shift_x=int(shift_x),
        selection_trimmed_z=selection_score,
        confirmation_trimmed_z=confirmation_score,
        joint_trimmed_z=min(selection_score, confirmation_score),
        selection_support_fraction=selection_support,
        confirmation_support_fraction=confirmation_support,
        joint_support_fraction=min(selection_support, confirmation_support),
        selection_patches=len(selection),
        confirmation_patches=len(confirmation),
    )


def score_lattice(
    pixels: NDArray[Any],
    template: NDArray[Any],
    *,
    periods: NDArray[Any],
    rotations_degrees: NDArray[Any],
    patch_size: int = 256,
    grid_size: int = 4,
    harmonic_count: int = 16,
    amplitude_candidate_count: int = 5,
    denoise_sigma: float = 1.0,
) -> LatticeScore:
    """Select a lattice on one patch group and confirm it on the other."""
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("pixels must have shape (height, width, 3)")
    if periods.ndim != 1 or not len(periods) or np.any(~np.isfinite(periods)) or np.any(periods <= 0.0):
        raise ValueError("periods must be a nonempty positive finite vector")
    if rotations_degrees.ndim != 1 or not len(rotations_degrees) or np.any(~np.isfinite(rotations_degrees)):
        raise ValueError("rotations must be a nonempty finite vector")
    if not math.isfinite(denoise_sigma) or denoise_sigma <= 0.0:
        raise ValueError("denoise sigma must be finite and positive")
    if amplitude_candidate_count < 1:
        raise ValueError("amplitude candidate count must be positive")
    harmonics, weights, coefficient_units = template_harmonics(template, harmonic_count)
    flat_periods, flat_rotations, frequencies_y, frequencies_x = _candidate_frequencies(
        periods,
        rotations_degrees,
        harmonics,
    )
    grouped_values: dict[int, list[NDArray[Any]]] = {0: [], 1: []}
    for origin_y, origin_x, group in _patch_origins(*pixels.shape[:2], patch_size, grid_size):
        grouped_values[group].append(
            _patch_unit_values(
                pixels,
                origin_y,
                origin_x,
                patch_size,
                frequencies_y,
                frequencies_x,
            )
        )
    selection_scores = _coherence(grouped_values[0], weights)
    confirmation_scores = _coherence(grouped_values[1], weights)
    peak_index = int(np.argmax(selection_scores))
    coarse_candidate_indices = _period_candidate_indices(
        flat_periods,
        flat_rotations,
        selection_scores,
        amplitude_candidate_count,
    )
    candidate_indices = _period_alias_candidate_indices(
        flat_periods,
        flat_rotations,
        coarse_candidate_indices,
    )
    whitened_scores = [
        _content_whitened_score(
            pixels,
            template,
            harmonics,
            period=float(flat_periods[index]),
            patch_size=patch_size,
            grid_size=grid_size,
            denoise_sigma=denoise_sigma,
        )
        for index in candidate_indices
    ]
    whitened_order = np.argsort([selection for selection, _confirmation in whitened_scores])[::-1]
    rerank_candidates = [int(index) for index in whitened_order[:3]]
    rerank_amplitudes = [
        _amplitude_score(
            pixels,
            template,
            period=float(flat_periods[candidate_indices[index]]),
            patch_size=patch_size,
            grid_size=grid_size,
            denoise_sigma=denoise_sigma,
        )
        for index in rerank_candidates
    ]
    rerank_scores = [
        min(*whitened_scores[index], amplitude.selection, amplitude.confirmation)
        for index, amplitude in zip(rerank_candidates, rerank_amplitudes, strict=True)
    ]
    rerank_winner = int(np.argmax(rerank_scores))
    selected_candidate = rerank_candidates[rerank_winner]
    selected_index = candidate_indices[selected_candidate]
    amplitude = rerank_amplitudes[rerank_winner]
    selection_whitened, confirmation_whitened = whitened_scores[selected_candidate]
    selection_p95 = float(np.quantile(selection_scores, 0.95))
    selection_p99 = float(np.quantile(selection_scores, 0.99))
    confirmation_p95 = float(np.quantile(confirmation_scores, 0.95))
    confirmation_p99 = float(np.quantile(confirmation_scores, 0.99))
    confirmation = float(confirmation_scores[selected_index])
    selection = float(selection_scores[selected_index])
    shift_y, shift_x, selection_codeword, confirmation_codeword, codeword_p95, codeword_p99 = _codeword_scores(
        grouped_values[0],
        grouped_values[1],
        selected_index,
        harmonics,
        coefficient_units,
        weights,
        template.shape[0],
    )
    (
        unknown_shift_y,
        unknown_shift_x,
        unknown_selection,
        unknown_confirmation,
        unknown_fixed_confirmation,
        unknown_fixed_all,
        unknown_p95,
        unknown_p99,
    ) = _unknown_codeword_scores(
        grouped_values[0],
        grouped_values[1],
        selected_index,
        harmonics,
        weights,
        template.shape[0],
    )
    return LatticeScore(
        selected_period=float(flat_periods[selected_index]),
        selected_rotation_degrees=float(flat_rotations[selected_index]),
        selected_orientation_degrees=0,
        selected_horizontal_reflection=False,
        selected_deskew_degrees=0.0,
        deskew_candidate_count=0,
        deskew_selection_coherence=0.0,
        deskew_direct_selection_match=0.0,
        deskew_direct_confirmation_match=0.0,
        deskew_direct_joint_match=0.0,
        peak_period=float(flat_periods[peak_index]),
        peak_rotation_degrees=float(flat_rotations[peak_index]),
        peak_selection_coherence=float(selection_scores[peak_index]),
        selection_coherence=selection,
        selection_candidate_p95=selection_p95,
        selection_candidate_p99=selection_p99,
        selection_excess_p95=selection - selection_p95,
        selection_excess_p99=selection - selection_p99,
        confirmation_coherence=confirmation,
        confirmation_candidate_p95=confirmation_p95,
        confirmation_candidate_p99=confirmation_p99,
        confirmation_excess_p95=confirmation - confirmation_p95,
        confirmation_excess_p99=confirmation - confirmation_p99,
        joint_coherence=min(selection, confirmation),
        joint_excess_p99=min(selection - selection_p99, confirmation - confirmation_p99),
        selected_shift_y=shift_y,
        selected_shift_x=shift_x,
        selection_codeword=selection_codeword,
        confirmation_codeword=confirmation_codeword,
        confirmation_codeword_shift_p95=codeword_p95,
        confirmation_codeword_shift_p99=codeword_p99,
        joint_codeword=min(selection_codeword, confirmation_codeword),
        unknown_codeword_shift_y=unknown_shift_y,
        unknown_codeword_shift_x=unknown_shift_x,
        unknown_codeword_selection=unknown_selection,
        unknown_codeword_confirmation=unknown_confirmation,
        unknown_codeword_fixed_confirmation=unknown_fixed_confirmation,
        unknown_codeword_fixed_all=unknown_fixed_all,
        unknown_codeword_confirmation_p95=unknown_p95,
        unknown_codeword_confirmation_p99=unknown_p99,
        unknown_codeword_excess_p99=unknown_confirmation - unknown_p99,
        amplitude_shift_y=amplitude.shift_y,
        amplitude_shift_x=amplitude.shift_x,
        selection_amplitude=amplitude.selection,
        confirmation_amplitude=amplitude.confirmation,
        joint_amplitude=min(amplitude.selection, amplitude.confirmation),
        selection_whitened_match=selection_whitened,
        confirmation_whitened_match=confirmation_whitened,
        joint_whitened_match=min(selection_whitened, confirmation_whitened),
        amplitude_candidate_count=len(candidate_indices),
        amplitude_rerank_count=len(rerank_candidates),
        canonical_template_score=amplitude.unaligned_full,
        canonical_registered_template_score=amplitude.aligned_full,
        selection_patches=len(grouped_values[0]),
        confirmation_patches=len(grouped_values[1]),
    )


def score_orientation_bank(
    pixels: NDArray[Any],
    template: NDArray[Any],
    *,
    periods: NDArray[Any],
    rotations_degrees: NDArray[Any],
    patch_size: int = 256,
    grid_size: int = 4,
    harmonic_count: int = 16,
    amplitude_candidate_count: int = 5,
    denoise_sigma: float = 1.0,
) -> LatticeScore:
    """Select a right-angle orientation on selection-patch whitened match."""
    candidates = [
        score_lattice(
            np.rot90(pixels, k=quarter_turns),
            template,
            periods=periods,
            rotations_degrees=rotations_degrees,
            patch_size=patch_size,
            grid_size=grid_size,
            harmonic_count=harmonic_count,
            amplitude_candidate_count=amplitude_candidate_count,
            denoise_sigma=denoise_sigma,
        )
        for quarter_turns in range(4)
    ]
    selected_index = int(np.argmax([candidate.selection_whitened_match for candidate in candidates]))
    return replace(candidates[selected_index], selected_orientation_degrees=selected_index * 90)


def score_dihedral_bank(
    pixels: NDArray[Any],
    template: NDArray[Any],
    *,
    periods: NDArray[Any],
    rotations_degrees: NDArray[Any],
    patch_size: int = 256,
    grid_size: int = 4,
    harmonic_count: int = 16,
    amplitude_candidate_count: int = 5,
    denoise_sigma: float = 1.0,
) -> LatticeScore:
    """Select a rotation and optional reflection on selection patches."""
    candidates: list[LatticeScore] = []
    transforms: list[tuple[int, bool]] = []
    for reflected in (False, True):
        reflected_pixels = np.fliplr(pixels) if reflected else pixels
        for quarter_turns in range(4):
            candidates.append(
                score_lattice(
                    np.rot90(reflected_pixels, k=quarter_turns),
                    template,
                    periods=periods,
                    rotations_degrees=rotations_degrees,
                    patch_size=patch_size,
                    grid_size=grid_size,
                    harmonic_count=harmonic_count,
                    amplitude_candidate_count=amplitude_candidate_count,
                    denoise_sigma=denoise_sigma,
                )
            )
            transforms.append((quarter_turns * 90, reflected))
    selected_index = int(np.argmax([candidate.selection_whitened_match for candidate in candidates]))
    selected_orientation, selected_reflection = transforms[selected_index]
    return replace(
        candidates[selected_index],
        selected_orientation_degrees=selected_orientation,
        selected_horizontal_reflection=selected_reflection,
    )


def _rotate_fixed_canvas(pixels: NDArray[Any], angle_degrees: float) -> NDArray[Any]:
    """Rotate RGB pixels around the image center without changing the canvas."""
    height, width = pixels.shape[:2]
    transform = cv2.getRotationMatrix2D(((width - 1) / 2.0, (height - 1) / 2.0), angle_degrees, 1.0)
    return np.asarray(
        cv2.warpAffine(
            pixels,
            transform,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT_101,
        )
    )


def _select_deskew_angle(
    pixels: NDArray[Any],
    template: NDArray[Any],
    periods: NDArray[Any],
    deskew_degrees: NDArray[Any],
    *,
    patch_size: int,
    grid_size: int,
    harmonic_count: int,
) -> tuple[float, float, float]:
    """Select one deskew angle by carrier coherence on selection patches."""
    harmonics, weights, _coefficient_units = template_harmonics(template, harmonic_count)
    flat_periods, flat_rotations, frequencies_y, frequencies_x = _candidate_frequencies(
        periods,
        deskew_degrees,
        harmonics,
    )
    selection_values = [
        _patch_unit_values(
            pixels,
            origin_y,
            origin_x,
            patch_size,
            frequencies_y,
            frequencies_x,
        )
        for origin_y, origin_x, group in _patch_origins(*pixels.shape[:2], patch_size, grid_size)
        if group == 0
    ]
    selection_scores = _coherence(selection_values, weights)
    selected_index = int(np.argmax(selection_scores))
    return (
        float(flat_periods[selected_index]),
        float(flat_rotations[selected_index]),
        float(selection_scores[selected_index]),
    )


def _affine_whitened_patch_score(
    pixels: NDArray[Any],
    expected_template: NDArray[Any],
    frequencies_y: NDArray[Any],
    frequencies_x: NDArray[Any],
    *,
    origin_y: int,
    origin_x: int,
    denoise_sigma: float,
    noise_radius: int = 4,
    guard_radius: int = 1,
    ridge: float = 0.1,
) -> float:
    """Match an affine carrier directly, without resampling the image."""
    patch_size = pixels.shape[0]
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
    spectrum = np.fft.fft2(channels * (window_1d[:, None] * window_1d[None, :])[:, :, None], axes=(0, 1))
    offset_y, offset_x = zip(
        *(
            (y, x)
            for y in range(-noise_radius, noise_radius + 1)
            for x in range(-noise_radius, noise_radius + 1)
            if max(abs(y), abs(x)) > guard_radius
        ),
        strict=True,
    )
    offset_y_values = np.asarray(offset_y, dtype=np.float64)
    offset_x_values = np.asarray(offset_x, dtype=np.float64)
    sample_y = frequencies_y * patch_size
    sample_x = frequencies_x * patch_size
    phase_correction = np.exp(-2j * math.pi * (frequencies_y * origin_y + frequencies_x * origin_x))
    numerator = 0.0
    template_energy = 0.0
    observation_energy = 0.0
    for harmonic_index in range(len(frequencies_y)):
        y = np.asarray([sample_y[harmonic_index]])
        x = np.asarray([sample_x[harmonic_index]])
        observation = np.asarray(
            [_bilinear_sample(spectrum[:, :, channel], y, x)[0] for channel in range(channels.shape[2])]
        )
        observation *= phase_correction[harmonic_index]
        noise = np.stack(
            [
                _bilinear_sample(
                    spectrum[:, :, channel],
                    y + offset_y_values,
                    x + offset_x_values,
                )
                for channel in range(channels.shape[2])
            ],
            axis=1,
        )
        covariance = noise.T @ np.conj(noise) / len(noise)
        local_power = float(np.trace(covariance).real / covariance.shape[0])
        covariance += np.eye(covariance.shape[0]) * (ridge * local_power + 1e-12)
        template_value = expected_template[harmonic_index]
        whitened_template = np.linalg.solve(covariance, template_value)
        whitened_observation = np.linalg.solve(covariance, observation)
        numerator += float(np.vdot(template_value, whitened_observation).real)
        template_energy += float(np.vdot(template_value, whitened_template).real)
        observation_energy += float(np.vdot(observation, whitened_observation).real)
    denominator = math.sqrt(max(0.0, template_energy * observation_energy))
    return numerator / denominator if denominator > 1e-12 else 0.0


def _affine_whitened_score(
    pixels: NDArray[Any],
    template: NDArray[Any],
    harmonics: NDArray[Any],
    *,
    period: float,
    deskew_degrees: float,
    patch_size: int,
    grid_size: int,
    denoise_sigma: float,
) -> tuple[float, float]:
    """Score a rotated carrier in the original pixels on disjoint patches."""
    _periods, _rotations, frequencies_y, frequencies_x = _candidate_frequencies(
        np.asarray([period]),
        np.asarray([deskew_degrees]),
        harmonics,
    )
    height, width = pixels.shape[:2]
    input_rotation = -deskew_degrees
    forward = cv2.getRotationMatrix2D(
        ((width - 1) / 2.0, (height - 1) / 2.0),
        input_rotation,
        1.0,
    )
    inverse = cv2.invertAffineTransform(forward)
    origin_x, origin_y = inverse[:, 2]
    phase = np.exp(2j * math.pi * (harmonics[:, 0] * origin_y / period + harmonics[:, 1] * origin_x / period))
    template_spectrum = np.fft.fft2(_opponent_channels(np.asarray(template, dtype=np.float64)), axes=(0, 1))
    expected_template = (
        np.asarray(
            [
                template_spectrum[round(float(row)) % template.shape[0], round(float(column)) % template.shape[1]]
                for row, column in harmonics
            ]
        )
        * phase[:, None]
    )
    grouped_scores: dict[int, list[float]] = {0: [], 1: []}
    for patch_y, patch_x, group in _patch_origins(height, width, patch_size, grid_size):
        patch = pixels[patch_y : patch_y + patch_size, patch_x : patch_x + patch_size]
        grouped_scores[group].append(
            _affine_whitened_patch_score(
                patch,
                expected_template,
                frequencies_y[0],
                frequencies_x[0],
                origin_y=patch_y,
                origin_x=patch_x,
                denoise_sigma=denoise_sigma,
            )
        )
    return float(np.mean(grouped_scores[0])), float(np.mean(grouped_scores[1]))


def score_deskew_bank(
    pixels: NDArray[Any],
    template: NDArray[Any],
    *,
    periods: NDArray[Any],
    deskew_degrees: NDArray[Any],
    patch_size: int = 256,
    grid_size: int = 4,
    harmonic_count: int = 16,
    amplitude_candidate_count: int = 5,
    denoise_sigma: float = 1.0,
) -> LatticeScore:
    """Select a small-angle deskew operation on selection patches."""
    if deskew_degrees.ndim != 1 or not len(deskew_degrees) or np.any(~np.isfinite(deskew_degrees)):
        raise ValueError("deskew angles must be a nonempty finite vector")
    harmonics, _weights, _coefficient_units = template_harmonics(template, harmonic_count)
    selected_period, selected_angle, selection_coherence = _select_deskew_angle(
        pixels,
        template,
        periods,
        deskew_degrees,
        patch_size=patch_size,
        grid_size=grid_size,
        harmonic_count=harmonic_count,
    )
    direct_selection, direct_confirmation = _affine_whitened_score(
        pixels,
        template,
        harmonics,
        period=selected_period,
        deskew_degrees=selected_angle,
        patch_size=patch_size,
        grid_size=grid_size,
        denoise_sigma=denoise_sigma,
    )
    score = score_lattice(
        _rotate_fixed_canvas(pixels, selected_angle),
        template,
        periods=periods,
        rotations_degrees=np.asarray([0.0]),
        patch_size=patch_size,
        grid_size=grid_size,
        harmonic_count=harmonic_count,
        amplitude_candidate_count=amplitude_candidate_count,
        denoise_sigma=denoise_sigma,
    )
    return replace(
        score,
        selected_deskew_degrees=selected_angle,
        deskew_candidate_count=len(periods) * len(deskew_degrees),
        deskew_selection_coherence=selection_coherence,
        deskew_direct_selection_match=direct_selection,
        deskew_direct_confirmation_match=direct_confirmation,
        deskew_direct_joint_match=min(direct_selection, direct_confirmation),
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
@click.option("--period-min", type=click.FloatRange(min=1.0), default=7.5, show_default=True)
@click.option("--period-max", type=click.FloatRange(min=1.0), default=24.5, show_default=True)
@click.option("--period-step", type=click.FloatRange(min=0.01), default=0.1, show_default=True)
@click.option("--rotation-min", type=float, default=-3.0, show_default=True)
@click.option("--rotation-max", type=float, default=3.0, show_default=True)
@click.option("--rotation-step", type=click.FloatRange(min=0.01), default=0.25, show_default=True)
@click.option("--input-angle", type=float, default=0.0, show_default=True)
@click.option("--deskew-search/--no-deskew-search", default=False, show_default=True)
@click.option("--input-scale", type=click.FloatRange(min=0.01), default=1.0, show_default=True)
@click.option("--crop-fraction", type=click.FloatRange(min=0.0, max=0.49), default=0.0, show_default=True)
@click.option("--jpeg-quality", type=click.IntRange(min=1, max=100), default=None)
@click.option("--webp-quality", type=click.IntRange(min=1, max=101), default=None)
@click.option("--input-rotation", type=click.Choice(("0", "90", "180", "270")), default="0", show_default=True)
@click.option(
    "--input-flip",
    type=click.Choice(("none", "horizontal", "vertical", "both")),
    default="none",
    show_default=True,
)
@click.option("--orientation-search/--no-orientation-search", default=False, show_default=True)
@click.option("--dihedral-search/--no-dihedral-search", default=False, show_default=True)
@click.option("--registered-period/--no-registered-period", default=False, show_default=True)
@click.option("--same-image-null/--no-same-image-null", default=False, show_default=True)
@click.option("--patch-shift-consensus/--no-patch-shift-consensus", default=False, show_default=True)
@click.option("--opponent-registered/--no-opponent-registered", default=False, show_default=True)
@click.option("--amplitude-candidate-count", type=click.IntRange(min=1), default=5, show_default=True)
@click.option("--report-out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(
    template_path: Path,
    images: tuple[Path, ...],
    period_min: float,
    period_max: float,
    period_step: float,
    rotation_min: float,
    rotation_max: float,
    rotation_step: float,
    input_angle: float,
    deskew_search: bool,
    input_scale: float,
    crop_fraction: float,
    jpeg_quality: int | None,
    webp_quality: int | None,
    input_rotation: str,
    input_flip: str,
    orientation_search: bool,
    dihedral_search: bool,
    registered_period: bool,
    same_image_null: bool,
    patch_shift_consensus: bool,
    opponent_registered: bool,
    amplitude_candidate_count: int,
    report_out: Path,
) -> None:
    """Score IMAGES with split-confirmed reciprocal-lattice coherence."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if period_max < period_min or rotation_max < rotation_min:
        raise click.BadParameter("search maximum must be at least its minimum")
    if jpeg_quality is not None and webp_quality is not None:
        raise click.UsageError("JPEG and WebP transforms are mutually exclusive")
    enabled_geometric_searches = sum((orientation_search, dihedral_search, deskew_search))
    if enabled_geometric_searches > 1:
        raise click.UsageError("orientation, dihedral, and deskew search are mutually exclusive")
    if registered_period and enabled_geometric_searches:
        raise click.UsageError("registered-period scoring cannot be combined with a geometric search")
    if same_image_null and not registered_period:
        raise click.UsageError("same-image-null scoring requires registered-period scoring")
    if patch_shift_consensus and not registered_period:
        raise click.UsageError("patch-shift consensus requires registered-period scoring")
    periods = np.arange(period_min, period_max + period_step / 2.0, period_step)
    rotations = np.arange(rotation_min, rotation_max + rotation_step / 2.0, rotation_step)
    template = _load_template(template_path)
    rows = []
    for path in images:
        registered: RegisteredComponents | None = None
        period_null: SameImageNullScore | None = None
        patch_consensus: PatchShiftConsensusScore | None = None
        opponent_score: OpponentRegisteredScore | None = None
        try:
            pixels = load_rgb(path)
            if input_scale != 1.0:
                scaled_width = max(1, round(pixels.shape[1] * input_scale))
                scaled_height = max(1, round(pixels.shape[0] * input_scale))
                interpolation = cv2.INTER_AREA if input_scale < 1.0 else cv2.INTER_CUBIC
                pixels = cv2.resize(pixels, (scaled_width, scaled_height), interpolation=interpolation)
            if crop_fraction:
                crop_y = round(pixels.shape[0] * crop_fraction)
                crop_x = round(pixels.shape[1] * crop_fraction)
                pixels = pixels[crop_y:, crop_x:]
            if jpeg_quality is not None:
                pixels = jpeg_round_trip(pixels, jpeg_quality)
            if webp_quality is not None:
                pixels = _webp_round_trip(pixels, webp_quality)
            if input_angle:
                pixels = _rotate_fixed_canvas(pixels, input_angle)
            quarter_turns = int(input_rotation) // 90
            if quarter_turns:
                pixels = np.rot90(pixels, k=-quarter_turns)
            if input_flip in {"horizontal", "both"}:
                pixels = np.fliplr(pixels)
            if input_flip in {"vertical", "both"}:
                pixels = np.flipud(pixels)
            image_periods = periods
            image_rotations = rotations
            if registered_period:
                registered = registered_components(pixels, template, 1.0)
                image_periods = np.asarray([registered.selected_period])
                image_rotations = np.asarray([0.0])
                if same_image_null:
                    period_null = score_same_image_period_null(
                        pixels,
                        template,
                        registered.selected_period,
                    )
                if patch_shift_consensus:
                    patch_consensus = score_patch_shift_consensus(
                        pixels,
                        template,
                        registered.selected_period,
                    )
            if opponent_registered:
                opponent_score = score_opponent_registered(
                    pixels,
                    template,
                    periods=periods,
                )
            if dihedral_search:
                scorer = score_dihedral_bank
            elif orientation_search:
                scorer = score_orientation_bank
            elif deskew_search:
                score = score_deskew_bank(
                    pixels,
                    template,
                    periods=periods,
                    deskew_degrees=rotations,
                    amplitude_candidate_count=amplitude_candidate_count,
                )
                scorer = None
            else:
                scorer = score_lattice
            if scorer is not None:
                score = scorer(
                    pixels,
                    template,
                    periods=image_periods,
                    rotations_degrees=image_rotations,
                    amplitude_candidate_count=amplitude_candidate_count,
                )
        except (OSError, ValueError) as error:
            rows.append({"path": str(path), "status": "unsupported", "reason": str(error)})
            log.warning("%s: unsupported: %s", path, error)
            continue
        row: dict[str, Any] = {"path": str(path), "status": "scored", "score": asdict(score)}
        if registered is not None:
            row["registered"] = {**asdict(registered), "decision_score": registered.decision_score}
            row["registered_confirmation_passes"] = registered_confirmation_passes(score)
        if period_null is not None:
            row["same_image_null"] = asdict(period_null)
        if patch_consensus is not None:
            row["patch_shift_consensus"] = asdict(patch_consensus)
        if opponent_score is not None:
            row["opponent_registered"] = {
                **asdict(opponent_score),
                "decision_score": opponent_score.decision_score,
            }
        rows.append(row)
        log.info(
            "%s: period=%.3f rotation=%.3f lattice=%.6f whitened=%.6f template=%.6f",
            path,
            score.selected_period,
            score.selected_rotation_degrees,
            score.joint_coherence,
            score.joint_whitened_match,
            score.canonical_template_score,
        )
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(
        json.dumps(
            {
                "schema_version": 13,
                "template": str(template_path),
                "periods": [float(value) for value in periods],
                "rotations_degrees": [float(value) for value in rotations],
                "input_angle": input_angle,
                "deskew_search": deskew_search,
                "input_scale": input_scale,
                "crop_fraction": crop_fraction,
                "jpeg_quality": jpeg_quality,
                "webp_quality": webp_quality,
                "input_rotation": int(input_rotation),
                "input_flip": input_flip,
                "orientation_search": orientation_search,
                "dihedral_search": dihedral_search,
                "registered_period": registered_period,
                "same_image_null": same_image_null,
                "patch_shift_consensus": patch_shift_consensus,
                "opponent_registered": opponent_registered,
                "amplitude_candidate_count": amplitude_candidate_count,
                "records": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log.info("Wrote %d lattice records: %s", len(rows), report_out)


if __name__ == "__main__":
    main()
