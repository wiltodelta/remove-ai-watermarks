"""Gemini visible watermark removal engine.

Port of the GeminiWatermarkTool reverse-alpha-blending algorithm from C++ to Python.
Original author: Allen Kuo (allenk) — https://github.com/allenk/GeminiWatermarkTool

The Gemini AI watermark is applied using alpha blending:
    watermarked = a * logo + (1 - a) * original

We reverse this to recover the original:
    original = (watermarked - a * logo) / (1 - a)

The alpha maps are derived from background captures of the Gemini watermark
on pure-black backgrounds (48x48 for small images, 96x96 for large images).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class WatermarkSize(Enum):
    """Watermark size mode based on image dimensions."""

    SMALL = "small"  # 48x48, for images <= 1024x1024
    LARGE = "large"  # 96x96, for images > 1024x1024


@dataclass
class DetectionResult:
    """Result of watermark detection."""

    detected: bool = False
    confidence: float = 0.0
    region: tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h
    size: WatermarkSize = WatermarkSize.SMALL

    # stage scores
    spatial_score: float = 0.0
    gradient_score: float = 0.0
    variance_score: float = 0.0


@dataclass
class WatermarkPosition:
    """Watermark position configuration."""

    margin_right: int
    margin_bottom: int
    logo_size: int

    def get_position(self, image_width: int, image_height: int) -> tuple[int, int]:
        """Get top-left position for a given image size."""
        x = image_width - self.margin_right - self.logo_size
        y = image_height - self.margin_bottom - self.logo_size
        return (x, y)


def get_watermark_config(width: int, height: int) -> WatermarkPosition:
    """Get the appropriate watermark configuration based on image size.

    Rules discovered from Gemini:
      - W > 1024 AND H > 1024: 96x96 logo at (W-64-96, H-64-96)
      - Otherwise:              48x48 logo at (W-32-48, H-32-48)
    """
    if width > 1024 and height > 1024:
        return WatermarkPosition(margin_right=64, margin_bottom=64, logo_size=96)
    return WatermarkPosition(margin_right=32, margin_bottom=32, logo_size=48)


def get_watermark_size(width: int, height: int) -> WatermarkSize:
    """Determine watermark size mode from image dimensions."""
    if width > 1024 and height > 1024:
        return WatermarkSize.LARGE
    return WatermarkSize.SMALL


def _calculate_alpha_map(bg_capture: NDArray) -> NDArray:
    """Calculate alpha map from a background capture.

    The alpha map represents how much the watermark affects each pixel.
    alpha = max(R, G, B) / 255.0
    """
    if len(bg_capture.shape) == 2:
        gray = bg_capture.astype(np.float32)
    elif bg_capture.shape[2] >= 3:
        # Use max of channels for brightness
        gray = np.max(bg_capture[:, :, :3], axis=2).astype(np.float32)
    else:
        gray = bg_capture[:, :, 0].astype(np.float32)

    return gray / 255.0


def _load_embedded_asset(name: str) -> NDArray:
    """Load an embedded PNG asset and decode it with OpenCV."""
    asset_path = Path(__file__).parent / "assets" / name
    if not asset_path.exists():
        raise FileNotFoundError(f"Embedded asset not found: {asset_path}")

    data = asset_path.read_bytes()
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to decode embedded asset: {name}")
    return img


class GeminiEngine:
    """Engine for removing visible Gemini watermarks via reverse alpha blending.

    This is a Python port of the GeminiWatermarkTool C++ engine.
    """

    def __init__(self, logo_value: float = 255.0) -> None:
        """Initialize the engine with embedded alpha maps.

        Args:
            logo_value: The logo brightness value (default 255.0 = white).
        """
        self.logo_value = logo_value

        # Load embedded background captures
        bg_small = _load_embedded_asset("gemini_bg_48.png")
        bg_large = _load_embedded_asset("gemini_bg_96.png")

        # Ensure correct sizes
        if bg_small.shape[:2] != (48, 48):
            bg_small = cv2.resize(bg_small, (48, 48), interpolation=cv2.INTER_AREA)
        if bg_large.shape[:2] != (96, 96):
            bg_large = cv2.resize(bg_large, (96, 96), interpolation=cv2.INTER_AREA)

        # Calculate alpha maps
        self._alpha_small = _calculate_alpha_map(bg_small)
        self._alpha_large = _calculate_alpha_map(bg_large)

        logger.debug(
            "Alpha maps loaded: small=%s, large=%s",
            self._alpha_small.shape,
            self._alpha_large.shape,
        )

    def get_alpha_map(self, size: WatermarkSize) -> NDArray:
        """Get the base alpha map for a specific standard size."""
        if size == WatermarkSize.SMALL:
            return self._alpha_small
        return self._alpha_large

    def get_interpolated_alpha(self, size_px: int) -> NDArray:
        """Create an interpolated alpha map dynamically scaled from the high-res 96x96 base."""
        source = self._alpha_large
        if size_px == source.shape[1]:
            return source.copy()

        interp = cv2.INTER_LINEAR if size_px > source.shape[1] else cv2.INTER_AREA
        return cv2.resize(source, (size_px, size_px), interpolation=interp)

    # ── Detection ────────────────────────────────────────────────────

    def detect_watermark(
        self,
        image: NDArray,
        force_size: WatermarkSize | None = None,
    ) -> DetectionResult:
        """Detect Gemini watermark using multi-scale Snap Engine logic (ported from C++ vendor algorithm)."""
        result = DetectionResult()

        if image is None or image.size == 0:
            return result

        h, w = image.shape[:2]
        base_size = force_size or get_watermark_size(w, h)
        result.size = base_size

        # Use large alpha template (96x96) as the high-quality source for downscaling
        source_alpha = self._alpha_large

        # Dynamically search bottom-right corner (search up to 256x256 region)
        search_size = int(min(min(w, h), 256))
        sx1 = max(0, w - search_size)
        sy1 = max(0, h - search_size)

        search_region = image[sy1:h, sx1:w]
        if len(search_region.shape) == 3 and search_region.shape[2] >= 3:
            gray_sr = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_sr = search_region.copy()

        gray_sr_f = gray_sr.astype(np.float32) / 255.0

        # Phase 1 & 2: Multi-scale spatial NCC search
        best_scale = 0
        best_score = -1.0
        best_raw_ncc = -1.0
        best_loc = (0, 0)

        # Search scales from 16 to 120 (covering aggressively downscaled or slightly upscaled logos)
        for scale in range(16, 120, 2):
            if scale > search_region.shape[0] or scale > search_region.shape[1]:
                continue

            tmpl = cv2.resize(source_alpha, (scale, scale), interpolation=cv2.INTER_AREA)
            match_res = cv2.matchTemplate(gray_sr_f, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(match_res)

            # Size-adjusted score to overcome NCC bias toward tiny patches (mimics C++ weight)
            weight = min(1.0, (scale / 96.0) ** 0.5)
            adj_val = max_val * weight

            if adj_val > best_score:
                best_score = adj_val
                best_scale = scale
                best_loc = max_loc
                best_raw_ncc = max_val

        # Exact dynamic location & size
        pos_x = sx1 + best_loc[0]
        pos_y = sy1 + best_loc[1]
        result.region = (pos_x, pos_y, best_scale, best_scale)
        result.spatial_score = float(best_raw_ncc)

        # Generate exact alpha map for matched size
        alpha_region = self.get_interpolated_alpha(best_scale)

        # Extract exactly the matched region for Gradient & Variance analysis
        x1 = pos_x
        y1 = pos_y
        x2 = min(w, x1 + best_scale)
        y2 = min(h, y1 + best_scale)

        region = image[y1:y2, x1:x2]
        if len(region.shape) == 3 and region.shape[2] >= 3:
            gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray_region = region.copy()

        gray_f = gray_region.astype(np.float32) / 255.0

        # Adjust alpha_region if clipped by image boundary (rare, but possible)
        ay1, ax1 = 0, 0
        alpha_region = alpha_region[ay1 : ay1 + (y2 - y1), ax1 : ax1 + (x2 - x1)]

        if result.spatial_score < 0.25:
            result.confidence = float(max(0.0, result.spatial_score * 0.5))
            return result

        # ── Stage 2: Gradient NCC ────────────────────────────────────
        img_gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
        img_gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
        img_gmag = cv2.magnitude(img_gx, img_gy)

        alpha_gx = cv2.Sobel(alpha_region, cv2.CV_32F, 1, 0, ksize=3)
        alpha_gy = cv2.Sobel(alpha_region, cv2.CV_32F, 0, 1, ksize=3)
        alpha_gmag = cv2.magnitude(alpha_gx, alpha_gy)

        grad_match = cv2.matchTemplate(img_gmag, alpha_gmag, cv2.TM_CCOEFF_NORMED)
        _, grad_score, _, _ = cv2.minMaxLoc(grad_match)
        result.gradient_score = float(grad_score)

        # ── Stage 3: Variance Analysis ───────────────────────────────
        var_score = 0.0
        ref_h = min(y1, best_scale)

        if ref_h > 8:
            ref_region = image[y1 - ref_h : y1, x1:x2]
            gray_ref = cv2.cvtColor(ref_region, cv2.COLOR_BGR2GRAY) if len(ref_region.shape) == 3 else ref_region

            _, s_wm = cv2.meanStdDev(gray_region)
            _, s_ref = cv2.meanStdDev(gray_ref)

            if s_ref[0][0] > 5.0:
                var_score = max(0.0, min(1.0, 1.0 - (s_wm[0][0] / s_ref[0][0])))

        result.variance_score = float(var_score)

        # ── Fusion ───────────────────────────────────────────────────
        confidence = result.spatial_score * 0.50 + result.gradient_score * 0.30 + var_score * 0.20
        result.confidence = float(max(0.0, min(1.0, confidence)))
        result.detected = result.confidence >= 0.35

        logger.debug(
            "Detection: spatial=%.3f, grad=%.3f, var=%.3f → conf=%.3f (%s)",
            result.spatial_score,
            result.gradient_score,
            var_score,
            result.confidence,
            "DETECTED" if result.detected else "not detected",
        )

        return result

    # ── Removal ──────────────────────────────────────────────────────

    def remove_watermark(
        self,
        image: NDArray,
        force_size: WatermarkSize | None = None,
    ) -> NDArray:
        """Remove Gemini visible watermark from an image using reverse alpha blending.

        No-op when the detector does not find a watermark: returns an unmodified
        copy. Reverse alpha blending applied where no sparkle exists creates a
        visible inverse artifact, so we refuse to touch pixels without a positive
        detection. To bypass detection (e.g. you know the exact region), use
        ``remove_watermark_custom``.

        Args:
            image: BGR image as numpy array (will NOT be modified in-place).
            force_size: Force a specific watermark size (auto-detect if None).

        Returns:
            Cleaned BGR image as numpy array, or an unmodified copy when no
            watermark is detected.
        """
        result = image.copy()

        # Handle alpha channel
        if result.shape[2] == 4:
            result = cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)
        elif result.shape[2] == 1:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        size = force_size or get_watermark_size(result.shape[1], result.shape[0])

        # Detect dynamic position & size
        detection = self.detect_watermark(image, force_size=size)

        if not detection.detected:
            logger.debug(
                "No watermark detected (conf=%.3f); returning image unchanged.",
                detection.confidence,
            )
            return result

        pos = (detection.region[0], detection.region[1])
        alpha_map = self.get_interpolated_alpha(detection.region[2])
        logger.debug(
            "Removing watermark at (%d, %d) size %dx%d [conf=%.3f]",
            pos[0],
            pos[1],
            detection.region[2],
            detection.region[3],
            detection.confidence,
        )

        self._reverse_alpha_blend(result, alpha_map, pos)
        return result

    def remove_watermark_custom(
        self,
        image: NDArray,
        region: tuple[int, int, int, int],
    ) -> NDArray:
        """Remove watermark from a custom region with interpolated alpha map.

        Args:
            image: BGR image (will NOT be modified in-place).
            region: (x, y, width, height) of the watermark region.

        Returns:
            Cleaned BGR image.
        """
        result = image.copy()
        x, y, rw, rh = region

        # Check standard sizes
        if rw == 48 and rh == 48:
            self._reverse_alpha_blend(result, self._alpha_small, (x, y))
            return result
        if rw == 96 and rh == 96:
            self._reverse_alpha_blend(result, self._alpha_large, (x, y))
            return result

        # Interpolate alpha map for custom size
        interp = cv2.INTER_LINEAR if rw > 96 else cv2.INTER_AREA
        alpha = cv2.resize(self._alpha_large, (rw, rh), interpolation=interp)
        self._reverse_alpha_blend(result, alpha, (x, y))
        return result

    def _reverse_alpha_blend(
        self,
        image: NDArray,
        alpha_map: NDArray,
        position: tuple[int, int],
    ) -> None:
        """Apply reverse alpha blending in-place.

        Formula: original = (watermarked - a * logo) / (1 - a)
        """
        x, y = position
        ah, aw = alpha_map.shape[:2]
        ih, iw = image.shape[:2]

        # Clip to bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(iw, x + aw)
        y2 = min(ih, y + ah)

        if x1 >= x2 or y1 >= y2:
            return

        # Get ROIs
        ax1, ay1 = x1 - x, y1 - y
        alpha_roi = alpha_map[ay1 : ay1 + (y2 - y1), ax1 : ax1 + (x2 - x1)]
        image_roi = image[y1:y2, x1:x2].astype(np.float32)

        alpha_threshold = 0.002
        max_alpha = 0.99

        # Vectorized reverse alpha blending
        alpha = alpha_roi.copy()
        mask = alpha >= alpha_threshold
        alpha = np.clip(alpha, 0.0, max_alpha)
        one_minus_alpha = 1.0 - alpha

        # Expand alpha for 3-channel broadcast
        alpha_3d = alpha[:, :, np.newaxis]
        one_minus_3d = one_minus_alpha[:, :, np.newaxis]
        mask_3d = mask[:, :, np.newaxis]

        # original = (watermarked - alpha * logo) / (1 - alpha)
        restored = (image_roi - alpha_3d * self.logo_value) / one_minus_3d
        restored = np.clip(restored, 0.0, 255.0)

        # Apply only where alpha is significant
        image_roi = np.where(mask_3d, restored, image_roi)
        image[y1:y2, x1:x2] = image_roi.astype(np.uint8)

    # ── Inpainting cleanup ───────────────────────────────────────────

    def inpaint_residual(
        self,
        image: NDArray,
        region: tuple[int, int, int, int],
        strength: float = 0.85,
        method: Literal["gaussian", "telea", "ns"] = "ns",
        inpaint_radius: int = 10,
        padding: int = 32,
    ) -> NDArray:
        """Apply inpaint cleanup on residual artifacts after reverse alpha blend.

        Uses a sparse mask derived from alpha map gradient to repair only
        the sparkle-edge pixels where interpolation broke the math.

        Args:
            image: BGR image (will NOT be modified in-place).
            region: (x, y, w, h) of the watermark region.
            strength: Blend strength (0.0 = keep original, 1.0 = fully inpainted).
            method: Inpaint method ("gaussian", "telea", or "ns").
            inpaint_radius: Radius for cv2.inpaint.
            padding: Context padding around region in pixels.

        Returns:
            Cleaned BGR image.
        """
        result = image.copy()
        x, y, rw, rh = region

        if rw < 4 or rh < 4:
            return result

        strength = max(0.0, min(1.0, strength))
        if strength < 0.001:
            return result

        # Padded region
        px1 = max(0, x - padding)
        py1 = max(0, y - padding)
        px2 = min(image.shape[1], x + rw + padding)
        py2 = min(image.shape[0], y + rh + padding)

        if (px2 - px1) < 8 or (py2 - py1) < 8:
            return result

        # Inner rect relative to padded
        ix1 = x - px1
        iy1 = y - py1

        # Get alpha map (interpolated if needed)
        source_alpha = self._alpha_large
        interp = cv2.INTER_LINEAR if rw > source_alpha.shape[1] else cv2.INTER_AREA
        alpha_resized = cv2.resize(source_alpha, (rw, rh), interpolation=interp)

        # Compute gradient mask from alpha
        grad_x = cv2.Sobel(alpha_resized, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(alpha_resized, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)

        grad_min, grad_max = grad_mag.min(), grad_mag.max()
        if grad_max <= grad_min:
            return result

        # Normalize and apply gamma correction
        grad_norm = (grad_mag - grad_min) / (grad_max - grad_min)
        grad_weight = np.sqrt(grad_norm)

        # Dilate the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        grad_weight = cv2.dilate(grad_weight, kernel)

        if method == "gaussian":
            # Soft blend with Gaussian blur
            padded_roi = result[py1:py2, px1:px2].copy()
            blurred = cv2.GaussianBlur(padded_roi, (0, 0), sigmaX=2.0)

            # Create weight mask on padded area (only inner region has weights)
            weight_full = np.zeros((py2 - py1, px2 - px1), dtype=np.float32)
            weight_full[iy1 : iy1 + rh, ix1 : ix1 + rw] = grad_weight * strength

            weight_3d = weight_full[:, :, np.newaxis]
            blended = padded_roi.astype(np.float32) * (1 - weight_3d) + blurred.astype(np.float32) * weight_3d
            result[py1:py2, px1:px2] = blended.astype(np.uint8)
        else:
            # OpenCV inpainting (TELEA or NS)
            inpaint_flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS

            # Create binary mask from gradient weight
            binary_mask = (grad_weight * 255).astype(np.uint8)
            _, binary_mask = cv2.threshold(binary_mask, 30, 255, cv2.THRESH_BINARY)

            # Expand mask to padded region
            mask_full = np.zeros((py2 - py1, px2 - px1), dtype=np.uint8)
            mask_full[iy1 : iy1 + rh, ix1 : ix1 + rw] = binary_mask

            padded_roi = result[py1:py2, px1:px2].copy()
            inpainted = cv2.inpaint(padded_roi, mask_full, inpaint_radius, inpaint_flag)

            # Blend with strength
            weight_full = np.zeros((py2 - py1, px2 - px1), dtype=np.float32)
            weight_full[iy1 : iy1 + rh, ix1 : ix1 + rw] = grad_weight * strength
            weight_3d = weight_full[:, :, np.newaxis]

            blended = padded_roi.astype(np.float32) * (1 - weight_3d) + inpainted.astype(np.float32) * weight_3d
            result[py1:py2, px1:px2] = blended.astype(np.uint8)

        return result
