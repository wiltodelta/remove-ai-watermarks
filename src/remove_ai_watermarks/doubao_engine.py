"""Doubao visible watermark removal engine.

Doubao (ByteDance) stamps every generated image with a visible "豆包AI生成"
(Doubao AI generated) text strip in the bottom-right corner. This is the
explicit AIGC label mandated by China's TC260 standard, rendered as a
near-white / light-gray, low-saturation text overlay.

Unlike the Gemini sparkle (a fixed square logo removed by reverse alpha
blending against a captured alpha map), the Doubao mark is a text strip whose
exact alpha map we do not yet have. This engine therefore removes it by:

    locate -> mask -> inpaint

1. Locate: the mark scales with image WIDTH and sits in the bottom-right at a
   fixed margin, so we anchor a generous box there (geometry only -- no bundled
   template). Constants below are derived from measured Doubao output.
2. Mask: within the box, extract the light, low-saturation glyph pixels with a
   polarity-aware rule (the mark is brighter than dark backgrounds and a
   distinct off-white gray against light backgrounds).
3. Inpaint: cv2 inpainting (TELEA / NS) reconstructs the covered pixels.

This is fast, offline, deterministic, and needs no GPU. A future upgrade path
is per-pixel reverse alpha blending once a Doubao alpha map is captured on a
controlled black background (see data/doubao_capture/), which would recover the
true pixels instead of hallucinating them -- the same approach as the Gemini
engine.
"""

# cv2/numpy boundary: third-party libs ship no usable element types; relax the
# unknown-type rules for this file only.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportAttributeAccessIssue=false, reportPrivateImportUsage=false, reportPrivateUsage=false, reportInvalidTypeForm=false, reportConstantRedefinition=false, reportUnnecessaryComparison=false
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# Geometry as a fraction of image WIDTH. The Doubao mark scales with width and
# is anchored bottom-right. The box is intentionally generous (the glyph mask
# tightens it); values cover measured outputs across resolutions and aspect
# ratios (square 2048, portrait, ultra-wide). Margins are width-relative too.
WM_WIDTH_FRAC = 0.185
WM_HEIGHT_FRAC = 0.065
MARGIN_RIGHT_FRAC = 0.012
MARGIN_BOTTOM_FRAC = 0.014

# Glyph appearance: the label is a low-saturation light gray, rendered brighter
# than the surrounding content (the common case: a generated photo/illustration).
# We detect it as a local bright feature (white top-hat: brighter than a blurred
# local background) intersected with the grayish + minimum-brightness tests.
# This is polarity-correct for bright-on-darker backgrounds and, crucially,
# leaves white-paper documents untouched (there the mark is not brighter than
# its surroundings, so nothing is masked rather than damaging the document text).
MAX_SATURATION = 55  # max channel spread to count a pixel as "grayish"
LOGO_MIN_LUMA = 150  # glyphs are at least this bright in absolute terms
TOPHAT_DELTA = 12  # glyph must exceed the local background by this many levels

# Detection: a genuine label fills a meaningful fraction of the box. Measured
# coverage is >=0.20 on real Doubao outputs; random/textured corners stay <=0.06
# on large images but can spike to ~0.15 on tiny ones (small box -> high variance),
# so the threshold sits above that spike and below the real-mark floor.
DETECT_MIN_COVERAGE = 0.16

# Safety: a text strip fills a modest slice of the (generous) box. When the box
# is over a dense-text / document background the mask explodes and cv2 inpainting
# would smear the real content. Above this coverage we refuse to inpaint and
# leave the image untouched -- that hard case needs the neural path, not a guess.
MAX_INPAINT_COVERAGE = 0.50


@dataclass(frozen=True)
class DoubaoLocation:
    """Located watermark box (bottom-right), in absolute pixel coordinates."""

    x: int
    y: int
    w: int
    h: int
    is_fallback: bool = True  # geometry anchor (no template match) -> always True for now

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


@dataclass
class DoubaoDetection:
    """Result of visible Doubao watermark detection."""

    detected: bool = False
    confidence: float = 0.0
    region: tuple[int, int, int, int] = (0, 0, 0, 0)
    coverage: float = 0.0  # fraction of the box occupied by glyph pixels


class DoubaoEngine:
    """Remove the visible Doubao "豆包AI生成" watermark (locate -> mask -> inpaint)."""

    def __init__(
        self,
        *,
        width_frac: float = WM_WIDTH_FRAC,
        height_frac: float = WM_HEIGHT_FRAC,
        margin_right_frac: float = MARGIN_RIGHT_FRAC,
        margin_bottom_frac: float = MARGIN_BOTTOM_FRAC,
    ) -> None:
        self.width_frac = width_frac
        self.height_frac = height_frac
        self.margin_right_frac = margin_right_frac
        self.margin_bottom_frac = margin_bottom_frac

    # ── Locate ────────────────────────────────────────────────────────

    def locate(self, image: NDArray[Any]) -> DoubaoLocation:
        """Anchor the watermark box in the bottom-right corner by geometry."""
        h, w = image.shape[:2]
        wm_w = max(40, int(w * self.width_frac))
        wm_h = max(16, int(w * self.height_frac))
        margin_r = max(4, int(w * self.margin_right_frac))
        margin_b = max(4, int(w * self.margin_bottom_frac))
        x = max(0, w - margin_r - wm_w)
        y = max(0, h - margin_b - wm_h)
        wm_w = min(wm_w, w - x)
        wm_h = min(wm_h, h - y)
        return DoubaoLocation(x=x, y=y, w=wm_w, h=wm_h, is_fallback=True)

    # ── Mask ──────────────────────────────────────────────────────────

    def extract_mask(self, image: NDArray[Any], loc: DoubaoLocation) -> NDArray[Any]:
        """Build a full-image uint8 mask (255 = watermark glyph) for the box.

        Polarity-aware: the mark is a light, low-saturation gray. On a dark
        background it is the bright region; on a light background it is the
        off-white gray below paper-white. Both cases are captured by the logo
        luminance band intersected with the grayish constraint, plus a
        brighter-than-local-background test on dark backgrounds.
        """
        h, w = image.shape[:2]
        x, y, bw, bh = loc.bbox
        roi = image[y : y + bh, x : x + bw].astype(np.float32)

        luma = roi.mean(axis=2)
        sat = roi.max(axis=2) - roi.min(axis=2)
        grayish = sat < MAX_SATURATION

        # Local background model: a strong Gaussian blur (sigma ~ box height)
        # approximates the content under the glyphs. The white top-hat
        # (luma - local_bg) lights up bright thin strokes regardless of the
        # absolute background level.
        sigma = max(4.0, bh * 0.4)
        local_bg = cv2.GaussianBlur(luma, (0, 0), sigmaX=sigma, sigmaY=sigma)
        tophat = luma - local_bg

        cand = grayish & (tophat > TOPHAT_DELTA) & (luma > LOGO_MIN_LUMA)
        glyph = cand.astype(np.uint8) * 255
        # Connect glyph parts, then drop isolated specks (5x5 open clears the
        # scattered grayish pixels that random/textured corners produce).
        glyph = cv2.morphologyEx(glyph, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        glyph = cv2.morphologyEx(glyph, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        mask = np.zeros((h, w), np.uint8)
        mask[y : y + bh, x : x + bw] = glyph
        return mask

    # ── Detect ────────────────────────────────────────────────────────

    def detect(self, image: NDArray[Any]) -> DoubaoDetection:
        """Detect the visible Doubao mark by glyph coverage in the corner box.

        Heuristic: a genuine label fills a meaningful fraction of the box with
        text-like glyph pixels. Coverage maps to a confidence score.
        """
        det = DoubaoDetection()
        if image is None or image.size == 0:
            return det
        loc = self.locate(image)
        mask = self.extract_mask(image, loc)
        x, y, bw, bh = loc.bbox
        box = mask[y : y + bh, x : x + bw]
        coverage = float((box > 0).sum()) / float(max(1, bw * bh))
        det.region = loc.bbox
        det.coverage = coverage
        # Map coverage to a 0-1 confidence: ~0.06 (noise floor) -> 0, ~0.26 -> 1.
        det.confidence = float(max(0.0, min(1.0, (coverage - 0.06) / 0.20)))
        det.detected = coverage >= DETECT_MIN_COVERAGE
        logger.debug("Doubao detect: coverage=%.3f conf=%.3f", coverage, det.confidence)
        return det

    # ── Remove ────────────────────────────────────────────────────────

    def remove_watermark(
        self,
        image: NDArray[Any],
        *,
        inpaint_method: Literal["telea", "ns"] = "telea",
        inpaint_radius: int = 6,
        dilate: int = 3,
    ) -> NDArray[Any]:
        """Remove the visible Doubao watermark by inpainting the glyph mask.

        Returns an unmodified copy when no glyph pixels are found (so we never
        smear a clean corner). ``dilate`` grows the mask to cover anti-aliased
        glyph edges before inpainting.
        """
        if image is None or image.size == 0:
            return image
        loc = self.locate(image)
        mask = self.extract_mask(image, loc)
        if not mask.any():
            logger.debug("Doubao remove: no glyph pixels found; returning copy")
            return image.copy()

        x, y, bw, bh = loc.bbox
        coverage = float((mask[y : y + bh, x : x + bw] > 0).sum()) / float(max(1, bw * bh))
        if coverage > MAX_INPAINT_COVERAGE:
            logger.warning(
                "Doubao remove: box coverage %.2f exceeds %.2f (dense-text/document "
                "background); leaving image untouched to avoid smearing content",
                coverage,
                MAX_INPAINT_COVERAGE,
            )
            return image.copy()

        if dilate > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
            mask = cv2.dilate(mask, k)

        flag = cv2.INPAINT_TELEA if inpaint_method == "telea" else cv2.INPAINT_NS
        return cv2.inpaint(image, mask, inpaint_radius, flag)


def load_image_bgr(path: str | Path) -> NDArray[Any]:
    """Read an image as BGR ndarray (helper for scripts/tests)."""
    from remove_ai_watermarks import image_io

    img = image_io.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img
