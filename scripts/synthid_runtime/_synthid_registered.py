"""Opt-in scale registration for the measured periodic SynthID carrier."""

# The optional numeric libraries do not provide complete types for this path.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from synthid_runtime._synthid_confirmation import (
    RegisteredConfirmationComponents,
    registered_confirmation_components,
)
from synthid_runtime.synthid_detector import folded_template_score

if TYPE_CHECKING:
    from numpy.typing import NDArray

_PYRAMID_SCALES = (0.75, 1.0, 1.25)
_SEARCH_PERIODS = np.linspace(5.0, 32.0, 541, dtype=np.float64)
_CANONICAL_PERIODS = np.linspace(7.5, 24.5, 1701, dtype=np.float64)
_OPPONENT_SEARCH_PERIODS = np.linspace(7.5, 14.5, 141, dtype=np.float64)
_FINE_OPPONENT_COARSE_PERIODS = np.linspace(7.5, 9.0, 31, dtype=np.float64)
_FINE_OPPONENT_PROBE_SIZE = 384
_PERIOD_THRESHOLDS = (
    (7.5, 8.5, 0.3770629524888979),
    (8.5, 10.0, 0.25174716660523494),
    (10.0, 12.0, 0.284692023502354),
    (12.0, 14.0, 0.19794247706938645),
    (14.0, 16.0, 0.33930082812296375),
    (16.0, 18.0, 0.28915284982686323),
    (18.0, 20.0, 0.22885510746595789),
    (20.0, 22.0, 0.24570317032768269),
    (22.0, 24.5, 0.3142958338390489),
)
REGISTERED_HIGH_BAND_THRESHOLD = 0.075
OPPONENT_REGISTERED_MIN_PERIOD = 7.9
OPPONENT_REGISTERED_MAX_PERIOD = 12.0
OPPONENT_REGISTERED_CODEC_VETO_MAX_PERIOD = 8.1
OPPONENT_REGISTERED_MAX_P8_EDGE_RATIO = 1.05
FINE_OPPONENT_REGISTERED_MIN_PERIOD = 7.5
FINE_OPPONENT_REGISTERED_MAX_PERIOD = 9.0
OPPONENT_REGISTERED_FIXED_MIN = 0.16
OPPONENT_REGISTERED_RED_GREEN_MIN = 0.60
OPPONENT_REGISTERED_BLUE_YELLOW_MIN = 0.55


@dataclass(frozen=True)
class RegisteredComponents:
    """Calibrated components of one scale-registered decision."""

    raw_score: float
    amplitude_threshold: float
    selected_period: float
    spectral_period: float
    high_band_score: float
    confirmation: RegisteredConfirmationComponents | None = None

    @property
    def base_decision_score(self) -> float:
        """Return the unchanged registered-v2 decision statistic."""
        if self.selected_period != self.spectral_period:
            return 0.0
        return min(
            self.raw_score / self.amplitude_threshold,
            self.high_band_score / REGISTERED_HIGH_BAND_THRESHOLD,
        )

    @property
    def decision_score(self) -> float:
        """Return the base score only after split confirmation passes."""
        base_score = self.base_decision_score
        if base_score < 1.0:
            return base_score
        if self.confirmation is None or not self.confirmation.passes:
            return 0.0
        return base_score


@dataclass(frozen=True)
class OpponentRegisteredComponents:
    """Auditable margins for the bounded opponent-color fallback."""

    selected_period: float
    spectral_period: float
    spectral_score: float
    fixed_score: float
    red_green_spatial: float
    blue_yellow_spatial: float
    candidate_count: int
    red_green_p8_edge_ratio: float | None
    blue_yellow_p8_edge_ratio: float | None

    @property
    def base_decision_score(self) -> float:
        """Return the minimum normalized color-carrier margin."""
        return min(
            self.fixed_score / OPPONENT_REGISTERED_FIXED_MIN,
            self.red_green_spatial / OPPONENT_REGISTERED_RED_GREEN_MIN,
            self.blue_yellow_spatial / OPPONENT_REGISTERED_BLUE_YELLOW_MIN,
        )

    @property
    def decision_score(self) -> float:
        """Return the margin only inside the independently challenged period band."""
        if not OPPONENT_REGISTERED_MIN_PERIOD <= self.selected_period <= OPPONENT_REGISTERED_MAX_PERIOD:
            return 0.0
        if self.selected_period <= OPPONENT_REGISTERED_CODEC_VETO_MAX_PERIOD:
            ratios = (self.red_green_p8_edge_ratio, self.blue_yellow_p8_edge_ratio)
            if any(value is None or value > OPPONENT_REGISTERED_MAX_P8_EDGE_RATIO for value in ratios):
                return 0.0
        return self.base_decision_score

    @property
    def fine_decision_score(self) -> float:
        """Return the margin for the separately calibrated fine-period expert."""
        if not FINE_OPPONENT_REGISTERED_MIN_PERIOD <= self.selected_period <= FINE_OPPONENT_REGISTERED_MAX_PERIOD:
            return 0.0
        if self.selected_period <= OPPONENT_REGISTERED_CODEC_VETO_MAX_PERIOD:
            ratios = (self.red_green_p8_edge_ratio, self.blue_yellow_p8_edge_ratio)
            if any(value is None or value > OPPONENT_REGISTERED_MAX_P8_EDGE_RATIO for value in ratios):
                return 0.0
        return self.base_decision_score


def _resize(pixels: NDArray[Any], width: int, height: int) -> NDArray[Any]:
    interpolation = cv2.INTER_AREA if width < pixels.shape[1] else cv2.INTER_CUBIC
    return np.asarray(cv2.resize(pixels, (width, height), interpolation=interpolation))


def _template_frequency_features(
    template: NDArray[Any],
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    spectrum = np.fft.fft2(template, axes=(0, 1))
    power = np.sum(np.abs(spectrum) ** 2, axis=2)
    power[0, 0] = 0.0
    indices = np.argsort(power.ravel())[::-1][:30]
    rows, columns = np.unravel_index(indices, power.shape)
    height, width = template.shape[:2]
    signed_rows = np.where(rows <= height // 2, rows, rows - height)
    signed_columns = np.where(columns <= width // 2, columns, columns - width)
    harmonics = np.column_stack((signed_rows, signed_columns)).astype(np.float64)
    return harmonics, spectrum[rows, columns], spectrum


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


def _spectral_curve(
    pixels: NDArray[Any],
    periods: NDArray[Any],
    harmonics: NDArray[Any],
    coefficients: NDArray[Any],
) -> NDArray[Any]:
    height, width = pixels.shape[:2]
    y = (periods[:, None] ** -1) * harmonics[None, :, 0] * height
    x = (periods[:, None] ** -1) * harmonics[None, :, 1] * width
    sampled = np.empty((len(periods), len(harmonics), 3), dtype=np.complex128)
    for channel in range(3):
        residual = pixels[:, :, channel].astype(np.float32)
        residual -= cv2.GaussianBlur(
            residual,
            (0, 0),
            sigmaX=1.0,
            sigmaY=1.0,
            borderType=cv2.BORDER_REFLECT_101,
        )
        spectrum = np.fft.fft2(residual)
        sampled[:, :, channel] = _bilinear_sample(spectrum, y % height, x % width)
    numerator = np.real(np.sum(np.conj(coefficients)[None, :, :] * sampled, axis=(1, 2)))
    denominator = np.linalg.norm(coefficients) * np.linalg.norm(sampled, axis=(1, 2))
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )


def _period_candidates(
    periods: NDArray[Any],
    scores: NDArray[Any],
    count: int = 3,
) -> list[float]:
    candidates: list[float] = []
    for index in np.argsort(scores)[::-1]:
        period = float(periods[index])
        if any(abs(period - existing_period) < 0.25 for existing_period in candidates):
            continue
        candidates.append(period)
        if len(candidates) == count:
            break
    return candidates


def _period_threshold(period: float) -> float:
    for index, (lower, upper, threshold) in enumerate(_PERIOD_THRESHOLDS):
        if lower <= period < upper or (index == len(_PERIOD_THRESHOLDS) - 1 and period == upper):
            return threshold
    raise ValueError(f"registered period {period} is outside the calibrated range")


def _high_band_score(
    folded: NDArray[Any],
    template_spectrum: NDArray[Any],
) -> float:
    folded_spectrum = np.fft.fft2(folded, axes=(0, 1))
    tile_height, tile_width = template_spectrum.shape[:2]
    y_coordinates = np.minimum(np.arange(tile_height), tile_height - np.arange(tile_height))
    x_coordinates = np.minimum(np.arange(tile_width), tile_width - np.arange(tile_width))
    radius = np.sqrt(y_coordinates[:, None] ** 2 + x_coordinates[None, :] ** 2)
    correlations = []
    for lower, upper in ((4.5, 6.5), (6.5, 12.0)):
        mask = (radius >= lower) & (radius < upper)
        selected_folded = folded_spectrum[mask]
        selected_template = template_spectrum[mask]
        denominator = np.linalg.norm(selected_folded) * np.linalg.norm(selected_template)
        correlations.append(
            float(np.real(np.vdot(selected_template, selected_folded)) / denominator) if denominator > 0.0 else 0.0
        )
    return min(correlations)


def _best_canonical(
    pixels: NDArray[Any],
    periods: list[float],
    template: NDArray[Any],
    sigma: float,
) -> tuple[float, NDArray[Any], NDArray[Any], float]:
    best_score = -math.inf
    best_canonical: NDArray[Any] | None = None
    best_folded: NDArray[Any] | None = None
    best_period: float | None = None
    for period in periods:
        predicted_width = round(pixels.shape[1] * template.shape[1] / period)
        for delta in range(-4, 5):
            width = predicted_width + delta
            height = round(pixels.shape[0] * width / pixels.shape[1])
            canonical = _resize(pixels, width, height)
            score, folded = folded_template_score(canonical, template, sigma)
            if score > best_score:
                best_score = score
                best_canonical = canonical
                best_folded = folded
                best_period = period
    if best_canonical is None or best_folded is None or best_period is None:
        raise RuntimeError("scale registration produced no canonical view")
    return float(best_score), best_canonical, best_folded, best_period


def _quadrant_median(
    canonical: NDArray[Any],
    template: NDArray[Any],
    sigma: float,
) -> float:
    tile_height, tile_width = template.shape[:2]
    split_y = max(tile_height, (canonical.shape[0] // (2 * tile_height)) * tile_height)
    split_x = max(tile_width, (canonical.shape[1] // (2 * tile_width)) * tile_width)
    scores = []
    for region in (
        canonical[:split_y, :split_x],
        canonical[:split_y, split_x:],
        canonical[split_y:, :split_x],
        canonical[split_y:, split_x:],
    ):
        score, _folded = folded_template_score(region, template, sigma)
        scores.append(score)
    return float(np.median(scores))


def _pyramid_locked_mean(
    pixels: NDArray[Any],
    harmonics: NDArray[Any],
    coefficients: NDArray[Any],
    base_curve: NDArray[Any],
) -> float:
    curves = []
    candidates = []
    for scale in _PYRAMID_SCALES:
        if scale == 1.0:
            curve = base_curve
        else:
            level = _resize(
                pixels,
                max(16, round(pixels.shape[1] * scale)),
                max(16, round(pixels.shape[0] * scale)),
            )
            curve = _spectral_curve(level, _SEARCH_PERIODS, harmonics, coefficients)
        curves.append(curve)
        candidates.append(_period_candidates(_SEARCH_PERIODS, curve))
    combinations = itertools.product(*candidates)

    def spread(combination: tuple[float, ...]) -> float:
        normalized_periods = [
            candidate / scale
            for candidate, scale in zip(
                combination,
                _PYRAMID_SCALES,
                strict=True,
            )
        ]
        return float(np.std(np.log(normalized_periods)))

    best = min(
        combinations,
        key=spread,
    )
    base_period = float(np.median([candidate / scale for candidate, scale in zip(best, _PYRAMID_SCALES, strict=True)]))
    locked = [
        float(np.interp(base_period * scale, _SEARCH_PERIODS, curve))
        for curve, scale in zip(curves, _PYRAMID_SCALES, strict=True)
    ]
    return float(np.mean(locked))


def _opponent_pair(values: NDArray[Any]) -> NDArray[Any]:
    """Return Red-minus-Green and Blue-minus-Yellow color planes."""
    red = values[:, :, 0]
    green = values[:, :, 1]
    blue = values[:, :, 2]
    return np.stack((red - green, blue - 0.5 * (red + green)), axis=2)


def _opponent_period_curve(
    pixels: NDArray[Any],
    template: NDArray[Any],
    periods: NDArray[Any] = _OPPONENT_SEARCH_PERIODS,
) -> NDArray[Any]:
    """Return signed opponent-color coherence across the frozen search grid."""
    template_opponent = _opponent_pair(np.asarray(template, dtype=np.float64))
    template_spectrum = np.fft.fft2(template_opponent, axes=(0, 1))
    power = np.sum(np.abs(template_spectrum) ** 2, axis=2)
    power[0, 0] = 0.0
    indices = np.argsort(power.ravel())[::-1][:30]
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
    image_opponent = _opponent_pair(np.asarray(pixels, dtype=np.float32))
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


def _opponent_period_candidates(scores: NDArray[Any], count: int = 3) -> list[int]:
    """Return separated period indices in descending spectral-score order."""
    candidates: list[int] = []
    for index in np.argsort(scores)[::-1]:
        period = float(_OPPONENT_SEARCH_PERIODS[index])
        if any(abs(period - float(_OPPONENT_SEARCH_PERIODS[prior])) < 0.5 for prior in candidates):
            continue
        candidates.append(int(index))
        if len(candidates) == count:
            break
    return candidates


def _canonical_at_period(
    pixels: NDArray[Any],
    template: NDArray[Any],
    period: float,
) -> NDArray[Any]:
    """Resample PIXELS so PERIOD maps to the frozen template period."""
    width = max(template.shape[1], round(pixels.shape[1] * template.shape[1] / period))
    height = max(template.shape[0], round(pixels.shape[0] * template.shape[0] / period))
    if (height, width) == pixels.shape[:2]:
        return pixels
    return _resize(pixels, width, height)


def _correlation(left: NDArray[Any], right: NDArray[Any]) -> float:
    """Return the signed real cosine between equal-shaped arrays."""
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.real(np.vdot(right, left)) / denominator) if denominator > 0.0 else 0.0


def _period8_edge_ratio(values: NDArray[Any]) -> float:
    """Measure native 8-pixel block edges relative to non-block phases."""
    phase_values = np.zeros(8, dtype=np.float64)
    for axis in (0, 1):
        differences = np.abs(np.diff(values, axis=axis))
        indices = np.arange(differences.shape[axis])
        for phase in range(8):
            selected = indices[(indices + 1) % 8 == phase]
            phase_values[phase] += 0.5 * float(np.take(differences, selected, axis=axis).mean())
    baseline = float(np.median(phase_values[[1, 2, 3, 5, 6, 7]]))
    return float(phase_values[0] / baseline) if baseline > 1e-9 else math.inf


def _period8_opponent_edge_ratios(pixels: NDArray[Any]) -> tuple[float, float]:
    """Return codec-grid ratios for the two opponent-color planes."""
    opponent = _opponent_pair(np.asarray(pixels, dtype=np.float32))
    return _period8_edge_ratio(opponent[:, :, 0]), _period8_edge_ratio(opponent[:, :, 1])


def _opponent_components_at_period(
    pixels: NDArray[Any],
    template: NDArray[Any],
    sigma: float,
    period: float,
    *,
    spectral_period: float,
    spectral_score: float,
    candidate_count: int,
    period8_edge_ratios: tuple[float, float] | None = None,
) -> OpponentRegisteredComponents:
    """Measure one period without selecting it from the image being scored."""
    canonical = _canonical_at_period(pixels, template, period)
    fixed_score, folded = folded_template_score(canonical, template, sigma)
    folded_opponent = _opponent_pair(folded)
    template_opponent = _opponent_pair(template)
    red_green_p8_edge_ratio, blue_yellow_p8_edge_ratio = period8_edge_ratios or (None, None)
    return OpponentRegisteredComponents(
        selected_period=period,
        spectral_period=spectral_period,
        spectral_score=spectral_score,
        fixed_score=fixed_score,
        red_green_spatial=_correlation(folded_opponent[:, :, 0], template_opponent[:, :, 0]),
        blue_yellow_spatial=_correlation(folded_opponent[:, :, 1], template_opponent[:, :, 1]),
        candidate_count=candidate_count,
        red_green_p8_edge_ratio=red_green_p8_edge_ratio,
        blue_yellow_p8_edge_ratio=blue_yellow_p8_edge_ratio,
    )


def opponent_registered_components(
    pixels: NDArray[Any],
    template: NDArray[Any],
    sigma: float,
) -> OpponentRegisteredComponents:
    """Measure the bounded lossless-resize carrier in opponent-color space."""
    curve = _opponent_period_curve(pixels, template)
    candidate_indices = _opponent_period_candidates(curve)
    observations: list[OpponentRegisteredComponents] = []
    period8_edge_ratios: tuple[float, float] | None = None
    for index in candidate_indices:
        period = float(_OPPONENT_SEARCH_PERIODS[index])
        if period <= OPPONENT_REGISTERED_CODEC_VETO_MAX_PERIOD and period8_edge_ratios is None:
            period8_edge_ratios = _period8_opponent_edge_ratios(pixels)
        observations.append(
            _opponent_components_at_period(
                pixels,
                template,
                sigma,
                period,
                spectral_period=float(_OPPONENT_SEARCH_PERIODS[int(np.argmax(curve))]),
                spectral_score=float(curve[index]),
                candidate_count=len(candidate_indices),
                period8_edge_ratios=period8_edge_ratios,
            )
        )
    if not observations:
        raise RuntimeError("opponent-color registration produced no candidates")
    return max(observations, key=lambda observation: observation.base_decision_score)


def _fine_opponent_period_groups(curve: NDArray[Any]) -> list[list[float]]:
    """Return fine period grids around separated absolute spectral peaks."""
    centers: list[float] = []
    for index in np.argsort(np.abs(curve))[::-1]:
        period = float(_FINE_OPPONENT_COARSE_PERIODS[index])
        if any(abs(period - existing) < 0.2 for existing in centers):
            continue
        centers.append(period)
        if len(centers) == 3:
            break
    return [
        sorted(
            {
                round(float(period), 2)
                for period in np.arange(center - 0.36, center + 0.361, 0.01)
                if FINE_OPPONENT_REGISTERED_MIN_PERIOD <= period <= FINE_OPPONENT_REGISTERED_MAX_PERIOD
            }
        )
        for center in centers
    ]


def fine_opponent_registered_components(
    pixels: NDArray[Any],
    template: NDArray[Any],
    sigma: float,
) -> OpponentRegisteredComponents:
    """Select and score the calibrated fine-period lossless-resize expert."""
    curve = _opponent_period_curve(pixels, template, _FINE_OPPONENT_COARSE_PERIODS)
    spectral_index = int(np.argmax(np.abs(curve)))
    spectral_period = float(_FINE_OPPONENT_COARSE_PERIODS[spectral_index])
    period_groups = _fine_opponent_period_groups(curve)
    probe = pixels[
        : min(_FINE_OPPONENT_PROBE_SIZE, pixels.shape[0]),
        : min(_FINE_OPPONENT_PROBE_SIZE, pixels.shape[1]),
    ]
    candidate_count = sum(len(group) for group in period_groups)
    unique_periods = sorted({period for group in period_groups for period in group})
    probe_by_period = {
        period: _opponent_components_at_period(
            probe,
            template,
            sigma,
            period,
            spectral_period=spectral_period,
            spectral_score=float(np.interp(period, _FINE_OPPONENT_COARSE_PERIODS, curve)),
            candidate_count=candidate_count,
        )
        for period in unique_periods
    }
    probe_groups = [[probe_by_period[period] for period in group] for group in period_groups]
    probe_observations = [observation for group in probe_groups for observation in group]
    finalist_periods = {
        observation.selected_period
        for group in probe_groups
        for observation in sorted(group, key=lambda value: value.base_decision_score, reverse=True)[:2]
    }
    finalist_periods.update(
        observation.selected_period
        for observation in sorted(
            probe_observations,
            key=lambda value: value.base_decision_score,
            reverse=True,
        )[:5]
    )
    period8_edge_ratios = (
        _period8_opponent_edge_ratios(pixels)
        if any(period <= OPPONENT_REGISTERED_CODEC_VETO_MAX_PERIOD for period in finalist_periods)
        else None
    )
    observations = [
        _opponent_components_at_period(
            pixels,
            template,
            sigma,
            period,
            spectral_period=spectral_period,
            spectral_score=float(np.interp(period, _FINE_OPPONENT_COARSE_PERIODS, curve)),
            candidate_count=len(probe_observations),
            period8_edge_ratios=period8_edge_ratios,
        )
        for period in sorted(finalist_periods)
    ]
    if not observations:
        raise RuntimeError("fine opponent-color registration produced no candidates")
    return max(observations, key=lambda observation: observation.base_decision_score)


def registered_components(
    pixels: NDArray[Any],
    template: NDArray[Any],
    sigma: float,
) -> RegisteredComponents:
    """Measure a carrier after bounded scale registration."""
    harmonics, coefficients, template_spectrum = _template_frequency_features(template)
    combined_periods = np.concatenate((_SEARCH_PERIODS, _CANONICAL_PERIODS))
    combined_curve = _spectral_curve(pixels, combined_periods, harmonics, coefficients)
    base_curve = combined_curve[: len(_SEARCH_PERIODS)]
    canonical_curve = combined_curve[len(_SEARCH_PERIODS) :]
    candidates = _period_candidates(_CANONICAL_PERIODS, canonical_curve)
    baseline, canonical, folded, selected_period = _best_canonical(pixels, candidates, template, sigma)
    quadrant = _quadrant_median(canonical, template, sigma)
    pyramid = _pyramid_locked_mean(
        pixels,
        harmonics,
        coefficients,
        base_curve,
    )
    raw_score = float((baseline + quadrant + pyramid) / 3.0)
    components = RegisteredComponents(
        raw_score=raw_score,
        amplitude_threshold=_period_threshold(selected_period),
        selected_period=selected_period,
        spectral_period=candidates[0],
        high_band_score=_high_band_score(folded, template_spectrum),
    )
    if components.base_decision_score < 1.0:
        return components
    try:
        confirmation = registered_confirmation_components(
            pixels,
            template,
            selected_period,
            sigma,
        )
    except ValueError:
        return components
    return RegisteredComponents(
        raw_score=components.raw_score,
        amplitude_threshold=components.amplitude_threshold,
        selected_period=components.selected_period,
        spectral_period=components.spectral_period,
        high_band_score=components.high_band_score,
        confirmation=confirmation,
    )


def registered_score(
    pixels: NDArray[Any],
    template: NDArray[Any],
    sigma: float,
) -> float:
    """Return the calibrated registered decision statistic."""
    return registered_components(pixels, template, sigma).decision_score


def opponent_registered_score(
    pixels: NDArray[Any],
    template: NDArray[Any],
    sigma: float,
) -> float:
    """Return the bounded opponent-color fallback decision statistic."""
    return opponent_registered_components(pixels, template, sigma).decision_score


def fine_opponent_registered_score(
    pixels: NDArray[Any],
    template: NDArray[Any],
    sigma: float,
) -> float:
    """Return the separately calibrated fine-period decision statistic."""
    return fine_opponent_registered_components(pixels, template, sigma).fine_decision_score
