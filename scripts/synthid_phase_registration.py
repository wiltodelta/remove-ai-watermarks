"""Shared bounded translation registration for phase-carrier probes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_TRANSLATION_SHIFT = 32


@dataclass(frozen=True)
class TranslationRegistration:
    """Best phase score and offset from one bounded translation search."""

    score: float
    active_weight_fraction: float
    row_shift: int
    column_shift: int


def phase_adjustment(
    rows: np.ndarray,
    columns: np.ndarray,
    *,
    height: int,
    width: int,
    row_shift: int | np.ndarray,
    column_shift: int | np.ndarray,
) -> np.ndarray:
    """Return the Fourier phase adjustment for one integer translation."""
    signed_rows = np.where(rows > height // 2, rows - height, rows)
    return 2.0 * np.pi * (signed_rows * row_shift / height + columns * column_shift / width)


def extract_frequency_values(
    pixels: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    channels: np.ndarray,
) -> np.ndarray:
    """Extract sparse three-channel rFFT coefficients from PIXELS."""
    values = np.empty(len(rows), dtype=np.complex128)
    for channel in range(3):
        positions = np.flatnonzero(channels == channel)
        if len(positions) == 0:
            continue
        spectrum = np.fft.rfft2(pixels[:, :, channel])
        values[positions] = spectrum[rows[positions], columns[positions]]
    return values


def register_phase_translations(
    values: np.ndarray,
    *,
    phases: np.ndarray,
    weights: np.ndarray,
    expected_magnitudes: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    height: int,
    width: int,
    max_shift: int,
) -> TranslationRegistration:
    """Find the strongest phase alignment over bounded integer translations."""
    if not 0 <= max_shift <= MAX_TRANSLATION_SHIFT:
        raise ValueError(f"max_shift must be between 0 and {MAX_TRANSLATION_SHIFT}")
    magnitude_gate = np.minimum(np.abs(values) / (expected_magnitudes + 1e-12), 1.0)
    active_weights = weights * magnitude_gate
    active_weight = float(np.sum(active_weights))
    if active_weight == 0.0:
        return TranslationRegistration(0.0, 0.0, 0, 0)

    shifts = np.arange(-max_shift, max_shift + 1)
    adjustment = phase_adjustment(
        rows[:, None, None],
        columns[:, None, None],
        height=height,
        width=width,
        row_shift=shifts[None, :, None],
        column_shift=shifts[None, None, :],
    )
    difference = np.angle(values) - phases
    scores = np.sum(active_weights[:, None, None] * np.cos(difference[:, None, None] + adjustment), axis=0)
    scores /= active_weight
    row_index, column_index = np.unravel_index(int(np.argmax(scores)), scores.shape)
    return TranslationRegistration(
        score=float(scores[row_index, column_index]),
        active_weight_fraction=active_weight,
        row_shift=int(shifts[row_index]),
        column_shift=int(shifts[column_index]),
    )
