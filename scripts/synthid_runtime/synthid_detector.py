"""Detect the confirmed periodic SynthID image carrier at calibrated image sizes.

This is a positive-only detector for one measured carrier epoch, not Google's
private payload decoder. A positive result is strong local evidence for the
carrier. An indeterminate result means only that the selected detector did not
find it; image sizes outside that mode's calibrated range are reported separately.

Research runtime. Not exported by ``remove_ai_watermarks``. Needs numpy and OpenCV.
"""

# The optional numeric libraries do not provide complete types for this path.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from numpy.typing import NDArray

SynthIDDetectionStatus = Literal["detected", "indeterminate", "unsupported"]

DETECTOR_ID = "synthid-periodic-tile-v2"
REGISTERED_DETECTOR_ID = "synthid-periodic-tile-registered-v3"
OPPONENT_REGISTERED_DETECTOR_ID = "synthid-periodic-tile-opponent-registered-v1"
FINE_OPPONENT_REGISTERED_DETECTOR_ID = "synthid-periodic-tile-opponent-fine-registered-v1"
LARGE_DETECTOR_ID = "synthid-periodic-tile-large-v1"
MODEL_FILENAME = "synthid_periodic_tile_2048_v1.npz"
# The template remains frozen at this model geometry. Runtime images are never
# resized. The supported pixel-count interval is the separately challenged domain:
# below it too few repetitions make the positive-only statistic unreliable, and
# above it resource use and specificity have not been calibrated.
MODEL_WIDTH = 2048
MODEL_HEIGHT = 2048
MIN_SUPPORTED_PIXELS = 1_000_000
MAX_SUPPORTED_PIXELS = 18_000_000
TILE_THRESHOLD = 0.17357069773071196
REGISTERED_MIN_SUPPORTED_PIXELS = 250_000
REGISTERED_MAX_SUPPORTED_PIXELS = 10_000_000
# Registered-v3 can confirm a positive only when both disjoint checkerboard
# groups contain a complete frozen 256-pixel patch. Narrower geometries need a
# separately calibrated adaptive-patch expert and must not masquerade as misses.
REGISTERED_MIN_SIDE = 256
# The registered score preserves the minimum normalized v2 margin only after
# independent split-patch phase, amplitude, and held-out codeword confirmation.
REGISTERED_THRESHOLD = 1.0
# The opponent-color fallback is a narrower precision-first route for lossless
# scale changes. Smaller rasters retained a natural period-10 false positive.
OPPONENT_REGISTERED_THRESHOLD = 1.0
OPPONENT_REGISTERED_MIN_PIXELS = 1_000_000
OPPONENT_REGISTERED_MIN_SIDE = 768
# Fine-period registration is separately frozen for the dense 0.47-0.55
# lossless-resize challenge. Its more expensive selector is bounded to the
# geometry range covered by the locked and reserve negative sets.
FINE_OPPONENT_REGISTERED_THRESHOLD = 1.05
FINE_OPPONENT_REGISTERED_MIN_PIXELS = 1_000_000
FINE_OPPONENT_REGISTERED_MAX_PIXELS = 5_000_000
FINE_OPPONENT_REGISTERED_MIN_SIDE = 768
# The large-image score combines all-window fixed and spatial opponent gates
# with an any-window signed opponent mid-band gate. The one vulnerable portrait
# geometry has an additional Green mid-band upper gate.
LARGE_THRESHOLD = 1.0
LARGE_MIN_PIXELS = 10_000_000
LARGE_MAX_PIXELS = 18_000_000
LARGE_WINDOW = 2_048
LARGE_PHASE = 16
LARGE_FIXED_SCORE_MIN = 0.14
LARGE_RED_GREEN_SPATIAL_MIN = 0.90
LARGE_BLUE_YELLOW_SPATIAL_MIN = 0.70
LARGE_BLUE_YELLOW_MID_BAND_MAX = -0.15
LARGE_PORTRAIT_GEOMETRY = (3_072, 5_504)
LARGE_PORTRAIT_GREEN_MID_BAND_MAX = 0.06
INSTALL_HINT = "install numpy and opencv-python-headless"


@dataclass(frozen=True)
class SynthIDDetection:
    """One local periodic-lattice verdict.

    The family this reports is NOT the watermark, and the field names say so. The
    statistic is destroyed by a crop of seven pixels, while SynthID's published
    evaluation retains 99.97% TPR under aggressive crop and resize, so what
    crosses the threshold is a generation-pipeline lattice anchored at the image
    origin. It identifies the pipeline, not the mark. The measurement is in
    ``docs/synthid-detector-research.md`` and ``docs/synthid-classifiers.md``.
    """

    status: SynthIDDetectionStatus
    width: int
    height: int
    score: float | None
    threshold: float
    detector: str = DETECTOR_ID
    reason: str | None = None
    signal_family: str = "generation-pipeline-lattice"
    provider_scope: str = "provider-neutral"
    backend: str = "local-pixel"
    metadata_used_for_verdict: bool = False
    pixels_preserved: bool = True
    # Consumers cannot be expected to read a caveat in prose, so the two measured
    # failure modes travel with every verdict.
    tile_aligned_crop_required: bool = True
    identifies_watermark: bool = False

    @property
    def detected(self) -> bool:
        """Whether the supported carrier crossed its frozen threshold."""
        return self.status == "detected"

    def to_dict(self) -> dict[str, str | int | float | bool | None]:
        """Return a JSON-safe result without a local file path."""
        return {
            "status": self.status,
            "width": self.width,
            "height": self.height,
            "score": self.score,
            "threshold": self.threshold,
            "detector": self.detector,
            "reason": self.reason,
            "signal_family": self.signal_family,
            "provider_scope": self.provider_scope,
            "backend": self.backend,
            "metadata_used_for_verdict": self.metadata_used_for_verdict,
            "pixels_preserved": self.pixels_preserved,
            "tile_aligned_crop_required": self.tile_aligned_crop_required,
            "identifies_watermark": self.identifies_watermark,
        }


@dataclass(frozen=True)
class LargeImageComponents:
    """Auditable margins for the calibrated large-image carrier branch."""

    width: int
    height: int
    minimum_fixed_score: float
    minimum_red_green_spatial: float
    minimum_blue_yellow_spatial: float
    minimum_blue_yellow_mid_band: float
    maximum_green_mid_band: float

    @property
    def decision_score(self) -> float:
        """Return the minimum normalized gate margin; one is the boundary."""
        margins = [
            self.minimum_fixed_score / LARGE_FIXED_SCORE_MIN,
            self.minimum_red_green_spatial / LARGE_RED_GREEN_SPATIAL_MIN,
            self.minimum_blue_yellow_spatial / LARGE_BLUE_YELLOW_SPATIAL_MIN,
            self.minimum_blue_yellow_mid_band / LARGE_BLUE_YELLOW_MID_BAND_MAX,
        ]
        if (self.width, self.height) == LARGE_PORTRAIT_GEOMETRY:
            margins.append(1.0 + LARGE_PORTRAIT_GREEN_MID_BAND_MAX - self.maximum_green_mid_band)
        return min(margins)


def is_available() -> bool:
    """True when numpy and OpenCV import."""
    import importlib.util

    return importlib.util.find_spec("cv2") is not None and importlib.util.find_spec("numpy") is not None


@lru_cache(maxsize=1)
def _load_template() -> tuple[NDArray[Any], float, int, int, int, int]:
    """Load and validate the bundled pickle-free detector model."""
    import numpy as np

    model_path = Path(__file__).resolve().parent / MODEL_FILENAME
    with np.load(model_path, allow_pickle=False) as artifact:
        if int(artifact["format_version"]) != 1:
            raise RuntimeError("unsupported SynthID detector model format")
        height = int(artifact["height"])
        width = int(artifact["width"])
        tile_height = int(artifact["tile_height"])
        tile_width = int(artifact["tile_width"])
        denoise_sigma = float(artifact["denoise_sigma"])
        template = np.asarray(artifact["template"], dtype=np.float64)
    if not _geometry_supported(width, height):
        raise RuntimeError("bundled SynthID detector has unexpected geometry")
    if template.shape != (tile_height, tile_width, 3):
        raise RuntimeError("bundled SynthID detector has an invalid template shape")
    if not np.all(np.isfinite(template)) or not np.isclose(np.linalg.norm(template), 1.0):
        raise RuntimeError("bundled SynthID detector has an invalid template")
    if not np.isfinite(denoise_sigma) or denoise_sigma <= 0.0:
        raise RuntimeError("bundled SynthID detector has an invalid denoise sigma")
    return template, denoise_sigma, height, width, tile_height, tile_width


def fold_residual_template(
    pixels: NDArray[Any],
    *,
    tile_height: int,
    tile_width: int,
    denoise_sigma: float,
) -> NDArray[Any]:
    """Estimate a zero-mean periodic residual template by modulo folding."""
    import cv2
    import numpy as np

    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("pixels must have shape (height, width, 3)")
    if tile_height < 1 or tile_width < 1 or denoise_sigma <= 0.0:
        raise ValueError("tile dimensions and denoise sigma must be positive")
    height, width = pixels.shape[:2]
    if height < tile_height or width < tile_width:
        raise ValueError("image geometry must be at least as large as the tile geometry")
    divisible = height % tile_height == 0 and width % tile_width == 0
    full_height = height - height % tile_height
    full_width = width - width % tile_width
    repeats_y = full_height // tile_height
    repeats_x = full_width // tile_width
    remaining_height = height - full_height
    remaining_width = width - full_width
    counts = np.full((tile_height, tile_width), repeats_y * repeats_x, dtype=np.int64)
    counts[:remaining_height] += repeats_x
    counts[:, :remaining_width] += repeats_y
    counts[:remaining_height, :remaining_width] += 1

    # OpenCV filters channels independently. Processing one channel at a time
    # keeps the 18 MP upper bound from requiring two full three-channel float32
    # buffers in addition to the decoded image.
    folded = np.empty((tile_height, tile_width, 3), dtype=np.float64)
    for channel in range(3):
        residual = pixels[:, :, channel].astype(np.float32)
        residual -= cv2.GaussianBlur(
            residual,
            (0, 0),
            sigmaX=denoise_sigma,
            sigmaY=denoise_sigma,
            borderType=cv2.BORDER_REFLECT_101,
        )
        if divisible:
            folded[:, :, channel] = residual.reshape(
                repeats_y,
                tile_height,
                repeats_x,
                tile_width,
            ).mean(axis=(0, 2), dtype=np.float64)
            continue
        folded_sum = (
            residual[:full_height, :full_width]
            .reshape(
                repeats_y,
                tile_height,
                repeats_x,
                tile_width,
            )
            .sum(axis=(0, 2), dtype=np.float64)
        )
        if remaining_height:
            bottom = residual[full_height:, :full_width].reshape(
                remaining_height,
                repeats_x,
                tile_width,
            )
            folded_sum[:remaining_height] += bottom.sum(axis=1, dtype=np.float64)
        if remaining_width:
            right = residual[:full_height, full_width:].reshape(
                repeats_y,
                tile_height,
                remaining_width,
            )
            folded_sum[:, :remaining_width] += right.sum(axis=0, dtype=np.float64)
        if remaining_height and remaining_width:
            folded_sum[:remaining_height, :remaining_width] += residual[
                full_height:,
                full_width:,
            ]
        folded[:, :, channel] = folded_sum / counts
    return folded - np.mean(folded, axis=(0, 1), keepdims=True)


def unit_tile(tile: NDArray[Any]) -> tuple[NDArray[Any], float]:
    """Return TILE normalized by its L2 norm and the original norm."""
    import numpy as np

    norm = float(np.linalg.norm(tile))
    if norm == 0.0:
        return np.zeros_like(tile, dtype=np.float64), 0.0
    return np.asarray(tile, dtype=np.float64) / norm, norm


def _image_size(image_path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(image_path) as image:
        return image.size


def _geometry_supported(width: int, height: int) -> bool:
    """Whether the image has a calibrated number of periodic-tile samples."""
    pixels = width * height
    return MIN_SUPPORTED_PIXELS <= pixels <= MAX_SUPPORTED_PIXELS


def _registered_geometry_supported(width: int, height: int) -> bool:
    """Whether scale registration was challenged at this decoded size."""
    pixels = width * height
    return (
        min(width, height) >= REGISTERED_MIN_SIDE
        and REGISTERED_MIN_SUPPORTED_PIXELS <= pixels <= REGISTERED_MAX_SUPPORTED_PIXELS
    )


def _large_geometry_supported(width: int, height: int) -> bool:
    """Whether fixed phase-aligned windows cover the calibrated large range."""
    pixels = width * height
    return min(width, height) >= LARGE_WINDOW and LARGE_MIN_PIXELS < pixels <= LARGE_MAX_PIXELS


def _opponent_registered_geometry_supported(width: int, height: int) -> bool:
    """Whether the opponent-color fallback passed its frozen geometry challenge."""
    pixels = width * height
    return (
        min(width, height) >= OPPONENT_REGISTERED_MIN_SIDE
        and OPPONENT_REGISTERED_MIN_PIXELS <= pixels <= REGISTERED_MAX_SUPPORTED_PIXELS
    )


def _fine_opponent_registered_geometry_supported(width: int, height: int) -> bool:
    """Whether the fine-period selector passed its frozen geometry challenge."""
    pixels = width * height
    return (
        min(width, height) >= FINE_OPPONENT_REGISTERED_MIN_SIDE
        and FINE_OPPONENT_REGISTERED_MIN_PIXELS <= pixels <= FINE_OPPONENT_REGISTERED_MAX_PIXELS
    )


def folded_template_score(
    pixels: NDArray[Any],
    template: NDArray[Any],
    denoise_sigma: float,
) -> tuple[float, NDArray[Any]]:
    """Fold PIXELS at the model geometry and score the normalized tile."""
    tile_height, tile_width = template.shape[:2]
    folded = fold_residual_template(
        pixels,
        tile_height=tile_height,
        tile_width=tile_width,
        denoise_sigma=denoise_sigma,
    )
    normalized, _norm = unit_tile(folded)
    return float((template * normalized).sum()), folded


def _large_window_starts(length: int) -> tuple[int, ...]:
    """Return phase-aligned starts that cover both edges without resampling."""
    if length < LARGE_WINDOW:
        raise ValueError("large-image sides must be at least 2,048 pixels")
    last = ((length - LARGE_WINDOW) // LARGE_PHASE) * LARGE_PHASE
    starts = list(range(0, last + 1, LARGE_WINDOW))
    if starts[-1] != last:
        starts.append(last)
    return tuple(starts)


def _correlation(left: NDArray[Any], right: NDArray[Any]) -> float:
    import numpy as np

    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.real(np.vdot(right, left)) / denominator) if denominator > 0.0 else 0.0


def _large_window_components(
    folded: NDArray[Any],
    template: NDArray[Any],
) -> tuple[float, float, float, float]:
    """Measure the four color-phase features used by the large branch."""
    import numpy as np

    folded_red_green = folded[:, :, 0] - folded[:, :, 1]
    template_red_green = template[:, :, 0] - template[:, :, 1]
    folded_blue_yellow = folded[:, :, 2] - 0.5 * (folded[:, :, 0] + folded[:, :, 1])
    template_blue_yellow = template[:, :, 2] - 0.5 * (template[:, :, 0] + template[:, :, 1])

    height, width = folded.shape[:2]
    y_coordinates = np.minimum(np.arange(height), height - np.arange(height))
    x_coordinates = np.minimum(np.arange(width), width - np.arange(width))
    radius = np.sqrt(y_coordinates[:, None] ** 2 + x_coordinates[None, :] ** 2)
    mid_band = (radius >= 4.5) & (radius < 6.5)
    blue_yellow_mid = _correlation(
        np.fft.fft2(folded_blue_yellow)[mid_band],
        np.fft.fft2(template_blue_yellow)[mid_band],
    )
    green_mid = _correlation(
        np.fft.fft2(folded[:, :, 1])[mid_band],
        np.fft.fft2(template[:, :, 1])[mid_band],
    )
    return (
        _correlation(folded_red_green, template_red_green),
        _correlation(folded_blue_yellow, template_blue_yellow),
        blue_yellow_mid,
        green_mid,
    )


def large_image_components(
    pixels: NDArray[Any],
    template: NDArray[Any],
    denoise_sigma: float,
) -> LargeImageComponents:
    """Score all phase-aligned 2,048-pixel windows of one large RGB image."""
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("pixels must have shape (height, width, 3)")
    height, width = pixels.shape[:2]
    if not _large_geometry_supported(width, height):
        raise ValueError("image geometry is outside the calibrated large-image range")

    minimum_fixed = float("inf")
    minimum_red_green = float("inf")
    minimum_blue_yellow = float("inf")
    minimum_blue_yellow_mid = float("inf")
    maximum_green_mid = -float("inf")
    for y in _large_window_starts(height):
        for x in _large_window_starts(width):
            window = pixels[y : y + LARGE_WINDOW, x : x + LARGE_WINDOW]
            fixed_score, folded = folded_template_score(window, template, denoise_sigma)
            red_green, blue_yellow, blue_yellow_mid, green_mid = _large_window_components(
                folded,
                template,
            )
            minimum_fixed = min(minimum_fixed, fixed_score)
            minimum_red_green = min(minimum_red_green, red_green)
            minimum_blue_yellow = min(minimum_blue_yellow, blue_yellow)
            minimum_blue_yellow_mid = min(minimum_blue_yellow_mid, blue_yellow_mid)
            maximum_green_mid = max(maximum_green_mid, green_mid)
    return LargeImageComponents(
        width=width,
        height=height,
        minimum_fixed_score=minimum_fixed,
        minimum_red_green_spatial=minimum_red_green,
        minimum_blue_yellow_spatial=minimum_blue_yellow,
        minimum_blue_yellow_mid_band=minimum_blue_yellow_mid,
        maximum_green_mid_band=maximum_green_mid,
    )


def detect_synthid(
    image_path: str | Path,
    *,
    image: NDArray[Any] | None = None,
    register_scale: bool | None = None,
) -> SynthIDDetection:
    """Detect the supported periodic carrier in IMAGE_PATH.

    ``indeterminate`` means that the frozen periodic carrier did not cross its
    calibrated threshold; it is not a clean-image guarantee. The default
    production router uses scale registration through 10 megapixels and the
    native large-image expert above that boundary. Set ``register_scale`` to
    ``True`` to force registration or ``False`` to run the legacy fixed-period
    diagnostic below the large-image boundary.
    """
    path = Path(image_path)
    if image is None:
        width, height = _image_size(path)
    else:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be a three-channel BGR array")
        height, width = image.shape[:2]
    large_mode = register_scale is not True and width * height > LARGE_MIN_PIXELS
    registered_mode = register_scale is True or (register_scale is None and not large_mode)
    if registered_mode:
        geometry_supported = _registered_geometry_supported(width, height)
        threshold = REGISTERED_THRESHOLD
        detector_id = REGISTERED_DETECTOR_ID
        unsupported_reason = (
            "registered-v3 requires 250,000-10,000,000 decoded pixels and both dimensions to be at least 256 pixels"
        )
    elif large_mode:
        geometry_supported = _large_geometry_supported(width, height)
        threshold = LARGE_THRESHOLD
        detector_id = LARGE_DETECTOR_ID
        unsupported_reason = (
            "large-v1 requires more than 10,000,000 through 18,000,000 decoded pixels "
            "and at least two phase-aligned 2048-pixel windows"
        )
    else:
        geometry_supported = _geometry_supported(width, height)
        threshold = TILE_THRESHOLD
        detector_id = DETECTOR_ID
        unsupported_reason = "fixed-v2 requires 1,000,000-18,000,000 decoded pixels"
    if not geometry_supported:
        return SynthIDDetection(
            status="unsupported",
            width=width,
            height=height,
            score=None,
            threshold=threshold,
            detector=detector_id,
            reason=unsupported_reason,
        )
    if not is_available():
        raise RuntimeError(f"SynthID pixel detection needs numpy and OpenCV; {INSTALL_HINT}")

    import numpy as np
    from PIL import Image

    template, sigma, *_model = _load_template()
    if image is None:
        with Image.open(path) as source:
            pixels = np.asarray(source.convert("RGB"), dtype=np.uint8)
    else:
        pixels = np.asarray(image[:, :, ::-1], dtype=np.uint8)
    if pixels.shape != (height, width, 3):
        raise RuntimeError("decoded image geometry does not match its header")
    if registered_mode:
        from synthid_runtime._synthid_registered import (
            fine_opponent_registered_score,
            opponent_registered_score,
            registered_score,
        )

        score = registered_score(pixels, template, sigma)
        if score < REGISTERED_THRESHOLD and _opponent_registered_geometry_supported(width, height):
            opponent_score = opponent_registered_score(pixels, template, sigma)
            if opponent_score >= OPPONENT_REGISTERED_THRESHOLD:
                score = opponent_score
                threshold = OPPONENT_REGISTERED_THRESHOLD
                detector_id = OPPONENT_REGISTERED_DETECTOR_ID
        if score < threshold and _fine_opponent_registered_geometry_supported(width, height):
            fine_score = fine_opponent_registered_score(pixels, template, sigma)
            if fine_score >= FINE_OPPONENT_REGISTERED_THRESHOLD:
                score = fine_score
                threshold = FINE_OPPONENT_REGISTERED_THRESHOLD
                detector_id = FINE_OPPONENT_REGISTERED_DETECTOR_ID
    elif large_mode:
        score = large_image_components(pixels, template, sigma).decision_score
    else:
        score, _folded = folded_template_score(pixels, template, sigma)
    detected = score >= threshold
    return SynthIDDetection(
        status="detected" if detected else "indeterminate",
        width=width,
        height=height,
        score=score,
        threshold=threshold,
        detector=detector_id,
        reason=None if detected else "the selected carrier expert did not cross every calibrated gate",
    )
