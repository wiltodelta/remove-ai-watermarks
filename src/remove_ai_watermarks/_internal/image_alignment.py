"""Bounded geometric alignment of already regenerated RGB images.

Source luminance supplies displacement geometry and coarse scalar weights.
All output colors come from regenerated samples. Every exact output still
requires independent validation; geometry is not watermark evidence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class _DenseFlow(Protocol):
    """Narrow OpenCV's omitted optional initial-flow annotation."""

    def calc(self, first: NDArray[Any], second: NDArray[Any], initial: NDArray[Any] | None) -> NDArray[Any]: ...


def _integer_coordinate(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def align_regenerated(
    reference: NDArray[Any],
    donor: NDArray[Any],
    protected_boxes: tuple[tuple[int, int, int, int], ...] = (),
    *,
    resampling: str = "linear",
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Align a regenerated donor with bounded, smooth, nonfolding displacements.

    Fixed exploratory policy: max-768 grayscale input, sigma-2 prefilter,
    DIS medium, sigma-12 full-resolution flow, 16-pixel maximum displacement,
    16-pixel protected-region transition, and a 0.35 gradient-norm bound.
    """
    if (
        reference.shape != donor.shape
        or reference.ndim != 3
        or reference.shape[2] != 3
        or reference.dtype != np.uint8
        or donor.dtype != np.uint8
        or min(reference.shape[:2]) < 64
    ):
        raise ValueError("Expected matching uint8 RGB images with sides at least 64 pixels")
    modes = {"linear": cv2.INTER_LINEAR, "lanczos": cv2.INTER_LANCZOS4}
    if resampling not in modes:
        raise ValueError("Unknown alignment resampling mode")
    height, width = reference.shape[:2]
    for left, top, right, bottom in protected_boxes:
        if not all(_integer_coordinate(v) for v in (left, top, right, bottom)):
            raise ValueError("Protected boxes require integer coordinates")
        if not 0 <= left < right <= width or not 0 <= top < bottom <= height:
            raise ValueError("Protected box outside image")
    if np.array_equal(reference, donor):
        return donor.copy(), np.zeros((height, width, 2), np.float32)
    scale = min(1.0, 768 / max(height, width))
    size = (max(64, round(width * scale)), max(64, round(height * scale)))
    grays = [
        cv2.GaussianBlur(
            cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), size, interpolation=cv2.INTER_AREA),
            (0, 0),
            2,
        )
        for image in (reference, donor)
    ]
    estimator = cast("_DenseFlow", cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM))
    flow = estimator.calc(grays[0], grays[1], None)
    flow = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
    flow *= np.array([width / size[0], height / size[1]], np.float32)
    flow = cv2.GaussianBlur(flow, (0, 0), 12)
    if not np.all(np.isfinite(flow)):
        raise ValueError("Nonfinite alignment flow")
    magnitude = np.linalg.norm(flow, axis=2)
    flow *= np.minimum(1, 16 / np.maximum(magnitude, 1e-6))[..., None]
    if protected_boxes:
        allowed = np.ones((height, width), np.uint8)
        for left, top, right, bottom in protected_boxes:
            allowed[top:bottom, left:right] = 0
        distance = cv2.distanceTransform(allowed, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        weight = 0.5 - 0.5 * np.cos(np.pi * np.minimum(distance / 16, 1))
        flow *= weight[..., None]
    derivatives = [np.gradient(flow[..., axis]) for axis in range(2)]
    norm = np.sqrt(sum(component**2 for pair in derivatives for component in pair))
    flow *= min(1.0, 0.35 / max(float(np.max(norm)), 1e-6))
    yy, xx = np.indices((height, width), dtype=np.float32)
    output = cv2.remap(
        donor, xx + flow[..., 0], yy + flow[..., 1], modes[resampling], borderMode=cv2.BORDER_REFLECT_101
    )
    return output, flow


def attenuate_regenerated(
    donor: NDArray[Any],
    flow: NDArray[Any],
    weights: NDArray[Any],
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Apply coarse error weights to displacement before one Lanczos resampling.

    Flow must come from the bounded alignment stage. Zero displacement remains
    zero, including protected regions. This avoids mixing shifted RGB contours
    but does not guarantee either local fidelity or watermark removal.
    """
    if donor.ndim != 3 or donor.shape[2] != 3 or donor.dtype != np.uint8 or min(donor.shape[:2]) < 64:
        raise ValueError("Expected uint8 RGB donor with sides at least 64 pixels")
    height, width = donor.shape[:2]
    if flow.shape != (height, width, 2) or weights.shape != (height, width):
        raise ValueError("Flow and weight dimensions must match donor")
    if not np.issubdtype(flow.dtype, np.floating) or not np.issubdtype(weights.dtype, np.floating):
        raise ValueError("Flow and weights must be floating point")
    if not np.all(np.isfinite(flow)) or not np.all(np.isfinite(weights)):
        raise ValueError("Nonfinite displacement or weights")
    if np.min(weights) < 0 or np.max(weights) > 1:
        raise ValueError("Weights must be in [0, 1]")
    if np.max(np.linalg.norm(flow, axis=2)) > 16.0001:
        raise ValueError("Input displacement exceeds alignment bound")
    weighted = flow.astype(np.float32) * weights.astype(np.float32)[..., None]
    derivatives = [np.gradient(weighted[..., axis]) for axis in range(2)]
    norm = np.sqrt(sum(component**2 for pair in derivatives for component in pair))
    weighted *= min(1.0, 0.35 / max(float(np.max(norm)), 1e-6))
    yy, xx = np.indices((height, width), dtype=np.float32)
    output = cv2.remap(
        donor, xx + weighted[..., 0], yy + weighted[..., 1], cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101
    )
    return output, weighted


def local_error_weights(
    reference: NDArray[Any],
    baseline: NDArray[Any],
    candidate: NDArray[Any],
    *,
    sigma: float = 64.0,
) -> NDArray[Any]:
    """Estimate bounded coarse error weights without copying source colors."""
    if reference.ndim != 3 or reference.shape[2] != 3 or 0 in reference.shape:
        raise ValueError("Inputs must have nonempty RGB dimensions")
    if baseline.shape != reference.shape or candidate.shape != reference.shape:
        raise ValueError("Input dimensions must match")
    if any(image.dtype != np.uint8 for image in (reference, baseline, candidate)):
        raise ValueError("Inputs must be RGB8")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Sigma must be positive and finite")

    height, width = reference.shape[:2]
    ratio = min(1.0, 256.0 / max(height, width))
    size = (max(1, round(width * ratio)), max(1, round(height * ratio)))
    delta = candidate.astype(np.float32) - baseline
    residual = reference.astype(np.float32) - baseline
    numerator = cast("NDArray[Any]", np.mean(delta * residual, axis=2))
    denominator = cast("NDArray[Any]", np.mean(delta * delta, axis=2))
    numerator = cv2.resize(numerator, size, interpolation=cv2.INTER_AREA)
    denominator = cv2.resize(denominator, size, interpolation=cv2.INTER_AREA)

    def smooth(values: NDArray[Any]) -> NDArray[Any]:
        return cv2.GaussianBlur(values, (0, 0), sigma * size[0] / width, sigmaY=sigma * size[1] / height)

    weight = np.clip(smooth(numerator) / np.maximum(smooth(denominator), 1.0), 0.0, 1.0)
    return np.clip(cv2.resize(smooth(weight), (width, height), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)
