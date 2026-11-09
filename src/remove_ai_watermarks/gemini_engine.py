"""Locate the visible Gemini sparkle and build a mask for shared inpainting."""

# OpenCV and NumPy expose incomplete types at this array-processing boundary.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportAttributeAccessIssue=false, reportPrivateImportUsage=false, reportPrivateUsage=false, reportInvalidTypeForm=false, reportConstantRedefinition=false, reportUnnecessaryComparison=false
from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from remove_ai_watermarks import image_io

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class WatermarkSize(Enum):
    """Provider size tier selected from the source dimensions."""

    SMALL = "small"
    LARGE = "large"


@dataclass
class DetectionResult:
    """Detection decision and its component scores."""

    detected: bool = False
    confidence: float = 0.0
    region: tuple[int, int, int, int] = (0, 0, 0, 0)
    size: WatermarkSize = WatermarkSize.SMALL
    spatial_score: float = 0.0
    gradient_score: float = 0.0
    variance_score: float = 0.0


@dataclass(frozen=True, slots=True)
class WatermarkPosition:
    """Expected provider margins and logo size."""

    margin_right: int
    margin_bottom: int
    logo_size: int

    def get_position(self, image_width: int, image_height: int) -> tuple[int, int]:
        return image_width - self.margin_right - self.logo_size, image_height - self.margin_bottom - self.logo_size


@dataclass(frozen=True, slots=True)
class _Candidate:
    scale: int
    x: int
    y: int
    spatial: float
    gradient: float = 0.0
    variance: float = 0.0

    @property
    def fused(self) -> float:
        if self.spatial < 0.25:
            return max(0.0, self.spatial * 0.5)
        return self.spatial * 0.50 + self.gradient * 0.30 + self.variance * 0.20


@dataclass(frozen=True, slots=True)
class _SparkleScan:
    """The provenance-BLIND half of sparkle detection, reusable across trust levels.

    ``source`` is the BGR-normalized image the false-positive gate re-reads, ``best``
    the winning candidate, and ``base`` a template result carrying everything the scan
    already resolved (``size`` and, when a candidate won, ``region`` and the component
    scores). ``best is None`` covers both no-candidate cases; ``base`` distinguishes
    them, since an empty image never resolves a ``size`` and a candidate-less one does.

    Per-CALL only, never cached on the engine: ``remove_auto_marks`` re-invokes each
    engine on a progressively cleaned frame within one process, so a memo on ``self``
    would hand back a pre-fill scan of a different image.
    """

    source: NDArray[Any] | None
    best: _Candidate | None
    base: DetectionResult


def get_watermark_size(width: int, height: int) -> WatermarkSize:
    """Return the provider's large tier only when both axes exceed 1024."""
    return WatermarkSize.LARGE if width > 1024 and height > 1024 else WatermarkSize.SMALL


def get_watermark_config(width: int, height: int) -> WatermarkPosition:
    """Return the observed standard placement for the selected size tier."""
    if get_watermark_size(width, height) is WatermarkSize.LARGE:
        return WatermarkPosition(64, 64, 96)
    return WatermarkPosition(32, 32, 48)


def _calculate_alpha_map(background_capture: NDArray[Any]) -> NDArray[Any]:
    """Convert a black-background sparkle capture to a normalized opacity map."""
    if background_capture.ndim == 2:
        intensity = background_capture
    elif background_capture.shape[2] >= 3:
        intensity = background_capture[:, :, :3].max(axis=2)
    else:
        intensity = background_capture[:, :, 0]
    return intensity.astype(np.float32) / 255.0


def _load_capture(filename: str, expected_side: int) -> NDArray[Any]:
    capture = image_io.imread(Path(__file__).parent / "assets" / filename, cv2.IMREAD_COLOR)
    if capture is None:
        raise RuntimeError(f"Failed to decode embedded asset: {filename}")
    if capture.shape[:2] != (expected_side, expected_side):
        capture = cv2.resize(capture, (expected_side, expected_side), interpolation=cv2.INTER_AREA)
    return capture


def _gray_float(image: NDArray[Any]) -> NDArray[Any]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 and image.shape[2] >= 3 else image
    return gray.astype(np.float32) / 255.0


def _overlaps(candidate: _Candidate, selected: _Candidate) -> bool:
    radius = 0.5 * max(candidate.scale, selected.scale)
    return abs(candidate.x - selected.x) < radius and abs(candidate.y - selected.y) < radius


_TEMPLATE_SCALES = tuple(range(16, 120, 2))


class GeminiEngine:
    """Project-native detector and mask builder for the white Gemini sparkle."""

    _CORE_ALPHA_FRAC = 0.8
    _SPARKLE_FP_CONF = 0.65
    _SPARKLE_FP_MARGIN = 5.0
    _SPARKLE_FP_GRAD = 0.55
    _SPARKLE_KEEP_CONF = 0.52
    _SPARKLE_WHITE_SAT = 0.20
    _FP_MAX_CORNER_AXIS_FRAC = 0.5
    _FP_OFF_CORNER_MARGIN_MULT = 2
    _FP_OFF_CORNER_MIN_SCALE_FRAC = 0.6
    _CORNER_PROMOTE_NCC = 0.85
    _CORNER_PROMOTE_FRAC = 0.20
    _CORNER_PROMOTE_MIN = 96
    _CORNER_PROMOTE_MAX = 384
    _SELECT_TOPK = 3
    _MASK_ALPHA = 0.04
    _MASK_DILATE_FRAC = 0.18

    def __init__(self, logo_value: float = 255.0) -> None:
        self.logo_value = logo_value
        self._alpha_small = _calculate_alpha_map(_load_capture("gemini_bg_48.png", 48))
        self._alpha_large = _calculate_alpha_map(_load_capture("gemini_bg_96.png", 96))
        self._tmpl_cache: dict[int, NDArray[Any]] = {
            side: cv2.resize(self._alpha_large, (side, side), interpolation=cv2.INTER_AREA) for side in _TEMPLATE_SCALES
        }

    def get_alpha_map(self, size: WatermarkSize) -> NDArray[Any]:
        return self._alpha_small if size is WatermarkSize.SMALL else self._alpha_large

    def get_interpolated_alpha(self, size_px: int) -> NDArray[Any]:
        if size_px == self._alpha_large.shape[1]:
            return self._alpha_large.copy()
        method = cv2.INTER_LINEAR if size_px > self._alpha_large.shape[1] else cv2.INTER_AREA
        return cv2.resize(self._alpha_large, (size_px, size_px), interpolation=method)

    def _scan_scales(self, gray: NDArray[Any]) -> Iterator[tuple[int, float, tuple[int, int]]]:
        """Yield the strongest normalized template match at every usable scale."""
        height, width = gray.shape[:2]
        for side, template in self._tmpl_cache.items():
            if side > height or side > width:
                continue
            response = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            _minimum, maximum, _min_location, max_location = cv2.minMaxLoc(response)
            yield side, float(maximum), max_location

    def _global_candidates(self, image: NDArray[Any]) -> list[_Candidate]:
        height, width = image.shape[:2]
        search_side = min(height, width, 512)
        origin_x, origin_y = width - search_side, height - search_side
        gray = _gray_float(image[origin_y:height, origin_x:width])
        ranked = sorted(
            (
                (
                    score * min(1.0, (side / 96.0) ** 0.5),
                    _Candidate(side, origin_x + location[0], origin_y + location[1], score),
                )
                for side, score, location in self._scan_scales(gray)
            ),
            key=lambda item: (item[0], item[1].scale, item[1].spatial, item[1].x, item[1].y),
            reverse=True,
        )
        selected: list[_Candidate] = []
        for _weighted, candidate in ranked:
            if any(_overlaps(candidate, prior) for prior in selected):
                continue
            selected.append(candidate)
            if len(selected) == self._SELECT_TOPK:
                break
        return selected

    def _score_candidate(self, image: NDArray[Any], candidate: _Candidate) -> _Candidate:
        if candidate.spatial < 0.25:
            return candidate
        gradient, variance = self._grad_var_scores(image, candidate.scale, candidate.x, candidate.y)
        return _Candidate(candidate.scale, candidate.x, candidate.y, candidate.spatial, gradient, variance)

    def detect_watermark(
        self,
        image: NDArray[Any],
        force_size: WatermarkSize | None = None,
        *,
        trust_provenance: bool = False,
    ) -> DetectionResult:
        """Return the strongest sparkle-shaped bottom-right candidate."""
        scan = self._sparkle_scan(image, force_size)
        return self._verdict(scan, trust_provenance=trust_provenance)

    def detect_watermark_both(
        self, image: NDArray[Any], force_size: WatermarkSize | None = None
    ) -> tuple[DetectionResult, DetectionResult]:
        """``(strict, relaxed)`` from ONE scan of the image.

        The scan -- global candidate search, corner promotion and fused scoring -- is
        provenance-blind; ``trust_provenance`` only decides whether the false-positive
        gate demotes the confidence afterwards. Two calls therefore repeated the whole
        sweep to reach two verdicts.

        The two results are DISTINCT objects with genuinely different confidences (the
        gate rewrites one of them), and callers mutate them.
        """
        scan = self._sparkle_scan(image, force_size)
        return (
            self._verdict(scan, trust_provenance=False),
            self._verdict(scan, trust_provenance=True),
        )

    def _sparkle_scan(self, image: NDArray[Any], force_size: WatermarkSize | None) -> _SparkleScan:
        """Everything in detection that does not depend on the trust level."""
        if image is None or image.size == 0:
            return _SparkleScan(None, None, DetectionResult())

        source = image_io.to_bgr(image)
        height, width = source.shape[:2]
        size = force_size or get_watermark_size(width, height)
        candidates = self._global_candidates(source)
        promoted = self._corner_promote(source, candidates[0].spatial if candidates else -1.0)
        if promoted is not None:
            candidates.append(_Candidate(promoted[0], promoted[1], promoted[2], promoted[3]))
        # The no-candidate result is NOT the empty-image one: `size` is already resolved
        # here, and it is a public field the caller can force.
        base = DetectionResult(size=size)
        if not candidates:
            return _SparkleScan(None, None, base)

        best = max((self._score_candidate(source, candidate) for candidate in candidates), key=lambda item: item.fused)
        base.region = (best.x, best.y, best.scale, best.scale)
        base.spatial_score = float(best.spatial)
        base.gradient_score = float(best.gradient)
        base.variance_score = float(best.variance)
        return _SparkleScan(source, best, base)

    def _verdict(self, scan: _SparkleScan, *, trust_provenance: bool) -> DetectionResult:
        """Apply the trust-level-dependent tail to a scan, as a fresh result object."""
        result = replace(scan.base)
        if scan.best is None or scan.source is None:
            return result
        best = scan.best
        confidence = best.fused
        if best.spatial >= 0.25 and confidence < self._SPARKLE_FP_CONF and not trust_provenance:
            confidence = self._apply_false_positive_gate(scan.source, best, confidence)
        result.confidence = float(np.clip(confidence, 0.0, 1.0))
        result.detected = result.confidence >= 0.35
        return result

    def _has_plausible_corner_geometry(self, width: int, height: int, candidate: _Candidate) -> bool:
        config = get_watermark_config(width, height)
        margin_right = width - candidate.x - candidate.scale
        margin_bottom = height - candidate.y - candidate.scale
        max_margin = max(margin_right, margin_bottom)
        if max_margin > min(width, height) * self._FP_MAX_CORNER_AXIS_FRAC:
            return False
        off_corner = max_margin > config.margin_right * self._FP_OFF_CORNER_MARGIN_MULT
        undersized = candidate.scale < config.logo_size * self._FP_OFF_CORNER_MIN_SCALE_FRAC
        return not (off_corner and undersized)

    def _apply_false_positive_gate(self, image: NDArray[Any], candidate: _Candidate, confidence: float) -> float:
        height, width = image.shape[:2]
        if not self._has_plausible_corner_geometry(width, height, candidate):
            logger.debug(
                "Sparkle candidate demoted on corner geometry: confidence=%.3f, region=(%d, %d, %d, %d), image=%dx%d",
                confidence,
                candidate.x,
                candidate.y,
                candidate.scale,
                candidate.scale,
                width,
                height,
            )
            return min(confidence, 0.30)
        alpha = self.get_interpolated_alpha(candidate.scale)
        position = (candidate.x, candidate.y)
        margin = self._core_ring_margin(image, alpha, position)
        low_margin = margin is not None and margin < self._SPARKLE_FP_MARGIN
        low_gradient = candidate.gradient < self._SPARKLE_FP_GRAD
        if not low_margin and not low_gradient:
            # A bright core AND a sharp shape match clear the gate without further
            # scrutiny -- including for a decorative sparkle-shaped glyph, not just a
            # genuine sparkle. See "Gemini sparkle" in docs/module-internals.md
            # (incident 2026-08-27) for why margin/gradient/saturation can't tell
            # them apart, and TestDecorativeGlyphFalsePositive for the regression.
            # Logged so a future fix can be calibrated against real occurrences.
            logger.debug(
                "Sparkle candidate cleared FP gate without scrutiny: confidence=%.3f, margin=%s, gradient=%.3f",
                confidence,
                margin,
                candidate.gradient,
            )
            return confidence
        saturation = self._core_saturation(image, alpha, position)
        neutral_core = not low_margin and saturation is not None and saturation <= self._SPARKLE_WHITE_SAT
        if confidence >= self._SPARKLE_KEEP_CONF and neutral_core:
            return confidence
        logger.debug(
            "Sparkle candidate demoted: confidence=%.3f, margin=%s, gradient=%.3f, saturation=%s",
            confidence,
            margin,
            candidate.gradient,
            saturation,
        )
        return min(confidence, 0.30)

    def _grad_var_scores(self, image: NDArray[Any], scale: int, pos_x: int, pos_y: int) -> tuple[float, float]:
        height, width = image.shape[:2]
        x2, y2 = min(width, pos_x + scale), min(height, pos_y + scale)
        region = image[pos_y:y2, pos_x:x2]
        gray = _gray_float(region)
        alpha = self.get_interpolated_alpha(scale)[: y2 - pos_y, : x2 - pos_x]

        image_edges = cv2.magnitude(
            cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
        )
        alpha_edges = cv2.magnitude(
            cv2.Sobel(alpha, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(alpha, cv2.CV_32F, 0, 1, ksize=3),
        )
        response = cv2.matchTemplate(image_edges, alpha_edges, cv2.TM_CCOEFF_NORMED)
        _minimum, gradient, _min_location, _max_location = cv2.minMaxLoc(response)

        variance = 0.0
        reference_height = min(pos_y, scale)
        if reference_height > 8:
            reference = image[pos_y - reference_height : pos_y, pos_x:x2]
            reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) if reference.ndim == 3 else reference
            _mean, region_std = cv2.meanStdDev((gray * 255.0).astype(np.uint8))
            _reference_mean, reference_std = cv2.meanStdDev(reference_gray)
            if reference_std[0][0] > 5.0:
                variance = float(np.clip(1.0 - region_std[0][0] / reference_std[0][0], 0.0, 1.0))
        return float(gradient), variance

    def _corner_promote(self, image: NDArray[Any], current_raw_ncc: float) -> tuple[int, int, int, float] | None:
        height, width = image.shape[:2]
        desired = round(min(width, height) * self._CORNER_PROMOTE_FRAC)
        side = min(min(width, height), max(self._CORNER_PROMOTE_MIN, min(self._CORNER_PROMOTE_MAX, desired)))
        origin_x, origin_y = width - side, height - side
        matches = self._scan_scales(_gray_float(image[origin_y:height, origin_x:width]))
        best = max(matches, key=lambda item: item[1], default=None)
        if best is None or best[1] < self._CORNER_PROMOTE_NCC or best[1] <= current_raw_ncc:
            return None
        return best[0], origin_x + best[2][0], origin_y + best[2][1], float(best[1])

    def footprint_mask(
        self,
        image: NDArray[Any],
        *,
        force: bool = False,
        dilate: int | None = None,
        region: tuple[int, int, int, int] | None = None,
    ) -> NDArray[Any] | None:
        """Build a full-frame mask from a resolved or newly detected sparkle."""
        if image is None or image.size == 0:
            return None
        source = image_io.to_bgr(image)
        height, width = source.shape[:2]
        if region is not None:
            x, y, scale = region[:3]
        else:
            detection = self.detect_watermark(source)
            if detection.detected:
                x, y, scale = detection.region[:3]
            elif force:
                config = get_watermark_config(width, height)
                x, y = config.get_position(width, height)
                scale = config.logo_size
            else:
                return None

        placed = self._footprint_indices(self.get_interpolated_alpha(scale), (x, y), source.shape)
        if placed is None:
            return None
        alpha, (y1, y2, x1, x2) = placed
        silhouette = (alpha > self._MASK_ALPHA).astype(np.uint8) * 255
        if not silhouette.any():
            return None
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y1:y2, x1:x2] = silhouette
        radius = dilate if dilate is not None else max(13, int(scale * self._MASK_DILATE_FRAC))
        if radius > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            mask = cv2.dilate(mask, kernel)
        return mask

    def _footprint_indices(
        self,
        alpha_map: NDArray[Any],
        position: tuple[int, int],
        image_shape: tuple[int, ...],
    ) -> tuple[NDArray[Any], tuple[int, int, int, int]] | None:
        x, y = position
        alpha_height, alpha_width = alpha_map.shape[:2]
        image_height, image_width = image_shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(image_width, x + alpha_width), min(image_height, y + alpha_height)
        if x1 >= x2 or y1 >= y2:
            return None
        alpha_x, alpha_y = x1 - x, y1 - y
        clipped = alpha_map[alpha_y : alpha_y + y2 - y1, alpha_x : alpha_x + x2 - x1]
        return clipped, (y1, y2, x1, x2)

    def _core_mask_and_box(
        self,
        image: NDArray[Any],
        alpha_map: NDArray[Any],
        position: tuple[int, int],
    ) -> tuple[NDArray[Any], NDArray[Any], tuple[int, int, int, int], float] | None:
        placed = self._footprint_indices(alpha_map, position, image.shape)
        if placed is None:
            return None
        alpha, bounds = placed
        peak = float(alpha.max())
        if peak < 0.2:
            return None
        core = alpha >= peak * self._CORE_ALPHA_FRAC
        if not core.any():
            return None
        y1, y2, x1, x2 = bounds
        return core, image[y1:y2, x1:x2], bounds, peak

    def _core_and_bg(
        self,
        image: NDArray[Any],
        alpha_map: NDArray[Any],
        position: tuple[int, int],
    ) -> tuple[float, float, float] | None:
        sample = self._core_mask_and_box(image, alpha_map, position)
        if sample is None:
            return None
        core, _box, (y1, y2, x1, x2), peak = sample
        height, width = image.shape[:2]
        padding = int((x2 - x1) * 0.7)
        ry1, ry2 = max(0, y1 - padding), min(height, y2 + padding)
        rx1, rx2 = max(0, x1 - padding), min(width, x2 + padding)
        luminance = image[ry1:ry2, rx1:rx2].astype(np.float32).mean(axis=2)
        fy1, fy2, fx1, fx2 = y1 - ry1, y2 - ry1, x1 - rx1, x2 - rx1
        core_value = float(np.percentile(luminance[fy1:fy2, fx1:fx2][core], 75))
        background = np.ones(luminance.shape, dtype=bool)
        background[fy1:fy2, fx1:fx2] = False
        if background.sum() < 10:
            return None
        return core_value, float(np.median(luminance[background])), peak

    def _core_ring_margin(
        self,
        image: NDArray[Any],
        alpha_map: NDArray[Any],
        position: tuple[int, int],
    ) -> float | None:
        sample = self._core_and_bg(image, alpha_map, position)
        return None if sample is None else sample[0] - sample[1]

    def _core_saturation(
        self,
        image: NDArray[Any],
        alpha_map: NDArray[Any],
        position: tuple[int, int],
    ) -> float | None:
        sample = self._core_mask_and_box(image, alpha_map, position)
        if sample is None:
            return None
        core, box, _bounds, _peak = sample
        pixels = box[core].astype(np.float32)
        brightest = pixels.max(axis=1)
        darkest = pixels.min(axis=1)
        return float(np.median((brightest - darkest) / (brightest + 1.0)))


@functools.lru_cache(maxsize=1)
def _shared_engine() -> GeminiEngine:
    return GeminiEngine()


def detect_sparkle_confidence(image_path: Path, *, image: NDArray[Any] | None = None) -> float | None:
    """Return the local sparkle confidence, or None when decoding fails."""
    decoded = image if image is not None else image_io.imread(image_path)
    if decoded is None:
        return None
    return float(_shared_engine().detect_watermark(decoded).confidence)
