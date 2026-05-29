"""Samsung Galaxy AI visible-badge removal engine.

Samsung's Galaxy AI editing tools (Generative Edit, Sketch to Image, Portrait
Studio, Drawing Assist) stamp a visible badge in the BOTTOM-LEFT corner: a small
sparkle icon followed by a localized "AI-generated content" string (e.g. English
"AI-generated content", Turkish "AI tarafindan olusturulan icerik"). It is a
low-saturation gray/white text strip whose polarity flips with the background --
light on a dark image, a darker gray on a light one.

Like the Doubao mark (and unlike the fixed-template Gemini sparkle) the exact
alpha map is unknown, so removal is:

    locate -> mask -> inpaint

1. Locate: anchor a generous bottom-left box. The badge text width varies a lot
   with locale, so the box is wide; the glyph mask tightens it.
2. Mask: dual-polarity -- a low-saturation pixel that is either brighter OR
   darker than its local background by a margin is a glyph candidate (covers
   light-on-dark and dark-on-light). The structural gate in ``detect`` rejects
   textured corners that merely look glyph-like.
3. Inpaint: cv2 inpainting (TELEA / NS) reconstructs the covered pixels.

Fast, offline, deterministic, no GPU. The visible badge is only present on some
Galaxy AI outputs (full generations carry it; some edits do not), so detection
is gated -- a badge-free image is returned untouched.
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
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Geometry as a fraction of image WIDTH (the badge scales with the image and is
# anchored bottom-left). The box is intentionally wide -- the localized text
# length varies a lot (English "AI-generated content" ~0.28w, Turkish ~0.40w) --
# because the glyph mask tightens the actual extent. Margins are width-relative.
WM_WIDTH_FRAC = 0.46
WM_HEIGHT_FRAC = 0.055
MARGIN_LEFT_FRAC = 0.010
MARGIN_BOTTOM_FRAC = 0.010

# Glyph appearance: a low-saturation gray/white badge whose polarity flips with
# the background, so we accept a pixel that is either brighter OR darker than a
# blurred local background by TOPHAT_DELTA, intersected with the grayish test.
MAX_SATURATION = 60  # max channel spread to count a pixel as "grayish"
TOPHAT_DELTA = 10  # glyph must differ from the local background by this many levels

# Detection. The badge is a thin, left-anchored, low-saturation text strip. The
# Doubao CJK gate (many separated glyph components) does NOT transfer: the badge
# is short Latin text whose anti-aliased letters connect into a few blobs, so we
# gate on its actual signature instead -- glyph mass concentrated in a thin
# horizontal band (`DETECT_MIN_BAND_FRAC`), beginning near the left edge
# (`DETECT_MAX_LEFT_START`), filling a moderate slice of the wide box
# (`DETECT_MIN/MAX_COVERAGE`). Corpus-tuned on the captured badges (band
# 0.80-0.99, left-start <=0.06) vs textured-corner false positives. NOTE: only a
# handful of real badges were available, so the thresholds are conservative and
# this detector drives the EXPLICIT `--mark samsung` path, not `--mark auto`.
DETECT_MIN_COVERAGE = 0.05
DETECT_MAX_COVERAGE = 0.5
DETECT_MIN_BAND_FRAC = 0.78
DETECT_MAX_LEFT_START = 0.12

# Safety: above this box coverage the mask is exploding over real content, so we
# refuse to inpaint and leave the image untouched rather than smear it.
MAX_INPAINT_COVERAGE = 0.45


@dataclass(frozen=True)
class SamsungLocation:
    """Located badge box (bottom-left), in absolute pixel coordinates."""

    x: int
    y: int
    w: int
    h: int

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


@dataclass
class SamsungDetection:
    """Result of visible Samsung Galaxy AI badge detection."""

    detected: bool = False
    confidence: float = 0.0
    region: tuple[int, int, int, int] = (0, 0, 0, 0)
    coverage: float = 0.0


def _badge_structure(box: NDArray[Any]) -> tuple[float, float]:
    """Return ``(band_fraction, left_start_fraction)`` of a binary glyph mask.

    ``band_fraction``: glyph mass within the densest half-box-height horizontal
    window (a one-line badge -> high; spread texture -> low). ``left_start_fraction``:
    column of the leftmost glyph pixel / box width (the left-anchored badge -> ~0).
    """
    total = int((box > 0).sum())
    if total == 0:
        return 0.0, 1.0
    bh, bw = box.shape[:2]
    row_glyphs = (box > 0).sum(axis=1).astype(np.float64)
    win = max(1, int(bh * 0.5))
    best_band = max((row_glyphs[i : i + win].sum() for i in range(max(1, bh - win + 1))), default=0.0)
    band_frac = float(best_band) / float(total)
    cols = np.where(box.any(axis=0))[0]
    left_start = float(cols.min()) / float(max(1, bw))
    return band_frac, left_start


class SamsungEngine:
    """Remove the visible Samsung Galaxy AI badge (locate -> mask -> inpaint)."""

    def __init__(
        self,
        *,
        width_frac: float = WM_WIDTH_FRAC,
        height_frac: float = WM_HEIGHT_FRAC,
        margin_left_frac: float = MARGIN_LEFT_FRAC,
        margin_bottom_frac: float = MARGIN_BOTTOM_FRAC,
    ) -> None:
        self.width_frac = width_frac
        self.height_frac = height_frac
        self.margin_left_frac = margin_left_frac
        self.margin_bottom_frac = margin_bottom_frac

    # ── Locate ────────────────────────────────────────────────────────

    def locate(self, image: NDArray[Any]) -> SamsungLocation:
        """Anchor the badge box in the bottom-left corner by geometry."""
        h, w = image.shape[:2]
        wm_w = max(60, int(w * self.width_frac))
        wm_h = max(16, int(w * self.height_frac))
        margin_l = max(2, int(w * self.margin_left_frac))
        margin_b = max(2, int(w * self.margin_bottom_frac))
        x = margin_l
        y = max(0, h - margin_b - wm_h)
        wm_w = min(wm_w, w - x)
        wm_h = min(wm_h, h - y)
        return SamsungLocation(x=x, y=y, w=wm_w, h=wm_h)

    # ── Mask ──────────────────────────────────────────────────────────

    def extract_mask(self, image: NDArray[Any], loc: SamsungLocation) -> NDArray[Any]:
        """Build a full-image uint8 mask (255 = badge glyph) for the box.

        Dual-polarity: a low-saturation pixel that is either brighter or darker
        than its blurred local background by TOPHAT_DELTA is a glyph candidate.
        """
        h, w = image.shape[:2]
        x, y, bw, bh = loc.bbox
        roi = image[y : y + bh, x : x + bw].astype(np.float32)

        luma = roi.mean(axis=2)
        sat = roi.max(axis=2) - roi.min(axis=2)
        grayish = sat < MAX_SATURATION

        sigma = max(4.0, bh * 0.4)
        local_bg = cv2.GaussianBlur(luma, (0, 0), sigmaX=sigma, sigmaY=sigma)
        diff = luma - local_bg
        cand = grayish & (np.abs(diff) > TOPHAT_DELTA)
        glyph = cand.astype(np.uint8) * 255
        glyph = cv2.morphologyEx(glyph, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        glyph = cv2.morphologyEx(glyph, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        mask = np.zeros((h, w), np.uint8)
        mask[y : y + bh, x : x + bw] = glyph
        return mask

    # ── Detect ────────────────────────────────────────────────────────

    def detect(self, image: NDArray[Any]) -> SamsungDetection:
        """Detect the visible badge by glyph coverage + text structure."""
        det = SamsungDetection()
        if image is None or image.size == 0:
            return det
        loc = self.locate(image)
        mask = self.extract_mask(image, loc)
        x, y, bw, bh = loc.bbox
        box = mask[y : y + bh, x : x + bw]
        coverage = float((box > 0).sum()) / float(max(1, bw * bh))
        det.region = loc.bbox
        det.coverage = coverage
        det.confidence = float(max(0.0, min(1.0, (coverage - 0.02) / 0.08)))
        if DETECT_MIN_COVERAGE <= coverage <= DETECT_MAX_COVERAGE:
            band_frac, left_start = _badge_structure(box)
            det.detected = band_frac >= DETECT_MIN_BAND_FRAC and left_start <= DETECT_MAX_LEFT_START
            logger.debug(
                "Samsung detect: coverage=%.3f band=%.2f left_start=%.2f detected=%s",
                coverage,
                band_frac,
                left_start,
                det.detected,
            )
        return det

    # ── Remove ────────────────────────────────────────────────────────

    def remove_watermark(
        self,
        image: NDArray[Any],
        *,
        inpaint_method: Literal["telea", "ns"] = "telea",
        inpaint_radius: int = 6,
        dilate: int = 3,
        force: bool = False,
    ) -> NDArray[Any]:
        """Remove the visible badge by inpainting the glyph mask.

        Returns an unmodified copy when the badge is not detected (so a badge-free
        image -- or a textured corner -- is never smeared), unless ``force`` is set
        (the ``--no-detect`` path: inpaint the bottom-left box regardless).
        """
        if image is None or image.size == 0:
            return image
        if not force and not self.detect(image).detected:
            logger.debug("Samsung remove: badge not detected; returning copy")
            return image.copy()
        loc = self.locate(image)
        mask = self.extract_mask(image, loc)
        x, y, bw, bh = loc.bbox
        coverage = float((mask[y : y + bh, x : x + bw] > 0).sum()) / float(max(1, bw * bh))
        if coverage > MAX_INPAINT_COVERAGE:
            logger.warning(
                "Samsung remove: box coverage %.2f exceeds %.2f; leaving image untouched",
                coverage,
                MAX_INPAINT_COVERAGE,
            )
            return image.copy()
        if dilate > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
            mask = cv2.dilate(mask, k)
        flag = cv2.INPAINT_TELEA if inpaint_method == "telea" else cv2.INPAINT_NS
        return cv2.inpaint(image, mask, inpaint_radius, flag)
