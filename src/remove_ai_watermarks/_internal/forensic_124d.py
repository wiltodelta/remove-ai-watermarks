"""124-d patch residual features for the frozen provider heads.

Lifted from the 2026-08-31 freeze extractor. The vector is ratios on
lattice-aligned 256 px patches: FFT band energy, comb contrast, and residual
autocovariance. Images smaller than the patch return None.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

PATCH = 256
MAX_PATCHES = 12
BLUR_SIGMA = 1.0
BANDS = 8
LAGS = (8, 16)
CONTROL_LAGS = (11, 13)
REACH = 3
FEATURE_WIDTH = 124


def _lag_grid() -> list[tuple[int, int]]:
    lags = [(0, dx) for dx in range(1, REACH + 1)]
    lags += [(dy, dx) for dy in range(1, REACH + 1) for dx in range(-REACH, REACH + 1)]
    return lags


LAG_GRID = _lag_grid()


def _opponent(pixels: NDArray[np.uint8]) -> NDArray[np.float64]:
    values = pixels.astype(np.float64)
    red, green, blue = values[:, :, 0], values[:, :, 1], values[:, :, 2]
    return np.stack(
        [
            0.299 * red + 0.587 * green + 0.114 * blue,
            red - green,
            blue - 0.5 * (red + green),
        ],
        axis=-1,
    )


def _residual(planes: NDArray[np.float64]) -> NDArray[np.float64]:
    blurred = cv2.GaussianBlur(planes, (0, 0), sigmaX=BLUR_SIGMA, sigmaY=BLUR_SIGMA, borderType=cv2.BORDER_REFLECT_101)
    return planes - blurred


def _patch_origins(height: int, width: int) -> list[tuple[int, int]]:
    if height < PATCH or width < PATCH:
        return []
    rows = list(range(0, height - PATCH + 1, PATCH))
    columns = list(range(0, width - PATCH + 1, PATCH))
    origins = [(row, column) for row in rows for column in columns]
    if len(origins) <= MAX_PATCHES:
        return origins
    step = len(origins) / MAX_PATCHES
    return [origins[int(index * step)] for index in range(MAX_PATCHES)]


def _masks() -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_], list[NDArray[np.bool_]]]:
    frequency_rows = np.fft.fftfreq(PATCH)[:, None]
    frequency_columns = np.fft.fftfreq(PATCH)[None, :]
    radius = np.hypot(frequency_rows, frequency_columns)
    index_rows = np.arange(PATCH)[:, None]
    index_columns = np.arange(PATCH)[None, :]
    period16 = (index_rows % (PATCH // 16) == 0) & (index_columns % (PATCH // 16) == 0)
    period8 = (index_rows % (PATCH // 8) == 0) & (index_columns % (PATCH // 8) == 0)
    comb16_only = period16 & ~period8
    background = ~period16
    edges = np.linspace(0.0, 0.5, BANDS + 1)
    bands = [(radius > edges[i]) & (radius <= edges[i + 1]) for i in range(BANDS)]
    return comb16_only, period8, background, bands


COMB16, COMB8, BACKGROUND, BAND_MASKS = _masks()


def _contrast(power: NDArray[np.float64], comb: NDArray[np.bool_]) -> float:
    reach = comb & (BAND_MASKS[1] | BAND_MASKS[2] | BAND_MASKS[3] | BAND_MASKS[4])
    floor = BACKGROUND & (BAND_MASKS[1] | BAND_MASKS[2] | BAND_MASKS[3] | BAND_MASKS[4])
    if not reach.any() or not floor.any():
        return 0.0
    background_power = float(power[floor].mean())
    if background_power <= 0.0:
        return 0.0
    return float(power[reach].mean() / background_power)


def _autocorrelation(patch: NDArray[np.float64], lag: int) -> float:
    centred = patch - patch.mean()
    energy = float((centred**2).sum())
    if energy <= 0.0:
        return 0.0
    rows = float((centred[:-lag, :] * centred[lag:, :]).sum())
    columns = float((centred[:, :-lag] * centred[:, lag:]).sum())
    return (rows + columns) / (2.0 * energy)


def image_features(pixels: NDArray[np.uint8]) -> NDArray[np.float64] | None:
    """Return the 124-d vector for RGB uint8 pixels, or None when geometry fails."""
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        return None
    residual = _residual(_opponent(pixels))
    origins = _patch_origins(*pixels.shape[:2])
    if not origins:
        return None
    per_patch: list[list[float]] = []
    spread_slice: slice | None = None
    for row, column in origins:
        window = residual[row : row + PATCH, column : column + PATCH]
        values: list[float] = []
        for channel in range(3):
            plane = window[:, :, channel]
            power = np.asarray(np.abs(np.fft.fft2(plane)) ** 2, dtype=np.float64)
            total = float(power[1:, 1:].sum())
            if total <= 0.0:
                return None
            values.extend(float(power[mask].sum() / total) for mask in BAND_MASKS)
            values.append(_contrast(power, COMB16))
            values.append(_contrast(power, COMB8))
            if spread_slice is None:
                spread_slice = slice(0, len(values))
            values.extend(_autocorrelation(plane, lag) for lag in LAGS)
            values.extend(_autocorrelation(plane, lag) for lag in CONTROL_LAGS)
            correlation = np.fft.ifft2(power).real
            centre = float(correlation[0, 0])
            if centre <= 0.0:
                return None
            values.extend(float(correlation[dy, dx] / centre) for dy, dx in LAG_GRID)
        per_patch.append(values)
    stack = np.asarray(per_patch, dtype=np.float64)
    if spread_slice is None:
        return None
    vector = np.concatenate([np.median(stack, axis=0), stack[:, spread_slice].std(axis=0)])
    if vector.shape != (FEATURE_WIDTH,):
        return None
    return vector
