"""Independent split-patch confirmation for the registered SynthID carrier."""

# The optional numeric libraries do not provide complete types for this path.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from synthid_runtime.synthid_detector import fold_residual_template, unit_tile

if TYPE_CHECKING:
    from numpy.typing import NDArray

MIN_PERIOD = 10.0
MIN_COHERENCE = 0.30
MIN_AMPLITUDE = 0.0
H5_PERIOD = (18.0, 18.6)
H5_MIN = 0.13
STRONG_COHERENCE_PERIOD = (18.6, 20.0)
STRONG_COHERENCE_MIN = 0.40
WEAK_H5_PERIOD = (20.0, 22.0)
WEAK_H5_MIN = 0.02
PATCH_SIZE = 256
GRID_SIZE = 4
HARMONIC_COUNT = 16


@dataclass(frozen=True)
class RegisteredConfirmationComponents:
    """Auditable split-patch confirmation components for one fixed period."""

    period: float
    joint_coherence: float
    joint_amplitude: float
    unknown_codeword_fixed_confirmation: float
    selection_patches: int
    confirmation_patches: int

    @property
    def passes(self) -> bool:
        """Whether every frozen period-aware confirmation gate passes."""
        return registered_confirmation_passes(
            self.period,
            self.joint_coherence,
            self.joint_amplitude,
            self.unknown_codeword_fixed_confirmation,
        )


def registered_confirmation_passes(
    period: float,
    joint_coherence: float,
    joint_amplitude: float,
    unknown_codeword_fixed_confirmation: float,
) -> bool:
    """Apply the single frozen registered-carrier confirmation rule."""
    if period < MIN_PERIOD:
        return False
    if joint_coherence < MIN_COHERENCE or joint_amplitude < MIN_AMPLITUDE:
        return False
    if H5_PERIOD[0] <= period < H5_PERIOD[1]:
        return unknown_codeword_fixed_confirmation >= H5_MIN
    if STRONG_COHERENCE_PERIOD[0] <= period < STRONG_COHERENCE_PERIOD[1]:
        return joint_coherence >= STRONG_COHERENCE_MIN
    if WEAK_H5_PERIOD[0] <= period < WEAK_H5_PERIOD[1]:
        return unknown_codeword_fixed_confirmation >= WEAK_H5_MIN
    return True


def _opponent_channels(values: NDArray[Any]) -> NDArray[Any]:
    red = values[:, :, 0]
    green = values[:, :, 1]
    blue = values[:, :, 2]
    return np.stack((green, red - green, blue - 0.5 * (red + green)), axis=2)


def _template_harmonics(template: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
    opponent = _opponent_channels(np.asarray(template, dtype=np.float64))
    spectrum = np.fft.fft2(opponent, axes=(0, 1))
    height, width = template.shape[:2]
    candidates: list[tuple[float, int, int]] = []
    for row in range(height):
        signed_row = row if row <= height // 2 else row - height
        for column in range(width):
            signed_column = column if column <= width // 2 else column - width
            if signed_row < 0 or (signed_row == 0 and signed_column <= 0):
                continue
            power = float(np.sum(np.abs(spectrum[row, column]) ** 2))
            candidates.append((power, signed_row, signed_column))
    candidates.sort(reverse=True)
    selected = candidates[:HARMONIC_COUNT]
    harmonics = np.asarray([(row, column) for _power, row, column in selected], dtype=np.float64)
    coefficients = np.asarray([spectrum[int(row) % height, int(column) % width] for row, column in harmonics])
    weights = np.abs(coefficients)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        raise ValueError("template has no nonzero periodic harmonics")
    return harmonics, weights / weight_sum


def _patch_origins(height: int, width: int) -> list[tuple[int, int, int]]:
    if height < PATCH_SIZE or width < PATCH_SIZE:
        raise ValueError("registered confirmation needs both image sides to be at least 256 pixels")
    y_values = np.linspace(0, height - PATCH_SIZE, min(GRID_SIZE, height // PATCH_SIZE), dtype=np.int64)
    x_values = np.linspace(0, width - PATCH_SIZE, min(GRID_SIZE, width // PATCH_SIZE), dtype=np.int64)
    origins = [
        (int(y), int(x), (y_index + x_index) % 2)
        for y_index, y in enumerate(np.unique(y_values))
        for x_index, x in enumerate(np.unique(x_values))
    ]
    if {group for _y, _x, group in origins} != {0, 1}:
        raise ValueError("registered confirmation needs two independent patch groups")
    return origins


def _bilinear_sample(spectrum: NDArray[Any], y: NDArray[Any], x: NDArray[Any]) -> NDArray[Any]:
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


def _patch_unit_values(
    pixels: NDArray[Any],
    origin_y: int,
    origin_x: int,
    period: float,
    harmonics: NDArray[Any],
    denoise_sigma: float,
) -> NDArray[Any]:
    patch = np.asarray(
        pixels[origin_y : origin_y + PATCH_SIZE, origin_x : origin_x + PATCH_SIZE],
        dtype=np.float32,
    )
    channels = _opponent_channels(patch)
    window_1d = np.hanning(PATCH_SIZE).astype(np.float32)
    window = window_1d[:, None] * window_1d[None, :]
    frequencies_y = harmonics[:, 0] / period
    frequencies_x = harmonics[:, 1] / period
    sample_y = frequencies_y * PATCH_SIZE
    sample_x = frequencies_x * PATCH_SIZE
    sampled = np.empty((len(harmonics), 3), dtype=np.complex128)
    for channel in range(3):
        residual = channels[:, :, channel]
        residual -= cv2.GaussianBlur(
            residual,
            (0, 0),
            sigmaX=denoise_sigma,
            sigmaY=denoise_sigma,
            borderType=cv2.BORDER_REFLECT_101,
        )
        sampled[:, channel] = _bilinear_sample(np.fft.fft2(residual * window), sample_y, sample_x)
    sampled *= np.exp(-2j * math.pi * (frequencies_y * origin_y + frequencies_x * origin_x))[:, None]
    magnitudes = np.abs(sampled)
    return np.divide(sampled, magnitudes, out=np.zeros_like(sampled), where=magnitudes > 1e-12)


def _coherence(values: list[NDArray[Any]], weights: NDArray[Any]) -> float:
    coherence = np.abs(np.mean(np.stack(values), axis=0))
    return float(np.sum(coherence * weights))


def _unknown_codeword_fixed_confirmation(
    selection_values: list[NDArray[Any]],
    confirmation_values: list[NDArray[Any]],
    weights: NDArray[Any],
) -> float:
    cross_codeword = np.mean(np.stack(confirmation_values), axis=0) * np.conj(
        np.mean(np.stack(selection_values), axis=0)
    )
    confirmation_mask = np.arange(len(weights)) % 2 == 1
    masked_weights = weights[confirmation_mask]
    return float(np.abs(np.sum(cross_codeword[confirmation_mask] * masked_weights)) / np.sum(masked_weights))


def _canonical_pixels(pixels: NDArray[Any], template: NDArray[Any], period: float) -> NDArray[Any]:
    width = max(template.shape[1], round(pixels.shape[1] * template.shape[1] / period))
    height = max(template.shape[0], round(pixels.shape[0] * template.shape[0] / period))
    if (height, width) == pixels.shape[:2]:
        return pixels
    interpolation = cv2.INTER_AREA if width < pixels.shape[1] else cv2.INTER_CUBIC
    return np.asarray(cv2.resize(pixels, (width, height), interpolation=interpolation))


def _cyclic_correlations(template: NDArray[Any], tile: NDArray[Any]) -> NDArray[Any]:
    template_spectrum = np.fft.fft2(template, axes=(0, 1))
    tile_spectrum = np.fft.fft2(tile, axes=(0, 1))
    return np.fft.ifft2(np.sum(template_spectrum * np.conj(tile_spectrum), axis=2)).real


def _joint_amplitude(
    pixels: NDArray[Any],
    template: NDArray[Any],
    period: float,
    denoise_sigma: float,
) -> tuple[float, int, int]:
    canonical = _canonical_pixels(pixels, template, period)
    tile_height, tile_width = template.shape[:2]
    grouped_units: dict[int, list[NDArray[Any]]] = {0: [], 1: []}
    origins = _patch_origins(*canonical.shape[:2])
    for origin_y, origin_x, group in origins:
        aligned_y = (origin_y // tile_height) * tile_height
        aligned_x = (origin_x // tile_width) * tile_width
        folded = fold_residual_template(
            canonical[aligned_y : aligned_y + PATCH_SIZE, aligned_x : aligned_x + PATCH_SIZE],
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
    shift_y, shift_x = np.unravel_index(int(np.argmax(selection_correlations)), selection_correlations.shape)
    return (
        min(
            float(selection_correlations[shift_y, shift_x]),
            float(confirmation_correlations[shift_y, shift_x]),
        ),
        len(grouped_units[0]),
        len(grouped_units[1]),
    )


def registered_confirmation_components(
    pixels: NDArray[Any],
    template: NDArray[Any],
    period: float,
    denoise_sigma: float,
) -> RegisteredConfirmationComponents:
    """Measure the frozen split-patch gates at one registered carrier period."""
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("pixels must have shape (height, width, 3)")
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError("registered period must be finite and positive")
    harmonics, weights = _template_harmonics(template)
    grouped_values: dict[int, list[NDArray[Any]]] = {0: [], 1: []}
    for origin_y, origin_x, group in _patch_origins(*pixels.shape[:2]):
        grouped_values[group].append(
            _patch_unit_values(
                pixels,
                origin_y,
                origin_x,
                period,
                harmonics,
                denoise_sigma,
            )
        )
    amplitude, selection_patches, confirmation_patches = _joint_amplitude(
        pixels,
        template,
        period,
        denoise_sigma,
    )
    return RegisteredConfirmationComponents(
        period=period,
        joint_coherence=min(
            _coherence(grouped_values[0], weights),
            _coherence(grouped_values[1], weights),
        ),
        joint_amplitude=amplitude,
        unknown_codeword_fixed_confirmation=_unknown_codeword_fixed_confirmation(
            grouped_values[0],
            grouped_values[1],
            weights,
        ),
        selection_patches=selection_patches,
        confirmation_patches=confirmation_patches,
    )
