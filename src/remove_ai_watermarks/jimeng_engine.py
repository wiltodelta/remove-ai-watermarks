"""Jimeng (即梦AI) visible watermark removal engine.

Jimeng / Dreamina (ByteDance's image generator, distinct from Doubao) stamps a
visible "★ 即梦AI" wordmark -- a four-point sparkle icon followed by the 即梦AI
characters -- in the bottom-right corner: a near-white semi-transparent overlay,
the explicit AIGC label under China's TC260 standard.

Like the Gemini sparkle and the Doubao strip, it is a fixed overlay, so removal
starts from **reverse-alpha blending** against a captured alpha map
(``remove_watermark_reverse_alpha``): ``original = (wm - a*logo)/(1-a)``. The logo
is pure white (255,255,255); the alpha map was solved from the GRAY Jimeng capture
(see data/jimeng_capture/), bundled as ``assets/jimeng_alpha.png`` -- a careful
build (cubic-background fit, mean over channels, full halo extent, unblurred) that
drops the self-residual to ~1.3. Gray is the chosen background because the mark
sits on bright photo content in real use, not on black.

Unlike the Doubao mark, Jimeng re-rasterizes its mark per generation AND jitters
its position a few px (the alpha maps solved from independent captures correlate
0.998 but not 1.0), so a single 2048 alpha map does not pixel-cancel the mark on
every image/resolution the way Doubao's deterministic overlay does. Removal
therefore NCC-aligns the alpha to the actual mark (always, not only off-native),
reverse-alphas, then clears the residual with a deliberately THIN inpaint over the
glyph footprint. The reverse-alpha pre-step recovers the true background (including
edges) under the semi-transparent mark, so the thin inpaint only finishes the
residual edges rather than smearing the whole footprint -- a wide full-footprint
pass blurred the texture/edges under the mark. Verified clean on the solid captures
(native 2048) and on a real 1440-wide Jimeng download (off-native, table edge kept).

Detection (``detect``) matches the bundled "即梦AI" glyph silhouette against the
corner candidate via normalized correlation, so it keys on the actual mark shape
(real marks score >=0.81, the Doubao strip 0.21, other AI output 0.0) rather than
coverage heuristics, and does not hijack ``--mark auto`` on a Doubao image.

``locate`` (geometry box, scales with image WIDTH) and ``extract_mask`` (the
candidate glyph mask the detector correlates) mirror the Doubao engine. Fast,
offline, no GPU. Arbitrary-region inpainting still lives in ``region_eraser`` /
the ``erase`` command.
"""

# cv2/numpy boundary: third-party libs ship no usable element types; relax the
# unknown-type rules for this file only.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportAttributeAccessIssue=false, reportPrivateImportUsage=false, reportPrivateUsage=false, reportInvalidTypeForm=false, reportConstantRedefinition=false, reportUnnecessaryComparison=false
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# Geometry as a fraction of image WIDTH. The Jimeng mark scales with width and is
# anchored bottom-right. The box is intentionally generous (the glyph mask
# tightens it); values cover the measured 2048 captures plus a real 1440 download.
WM_WIDTH_FRAC = 0.27
WM_HEIGHT_FRAC = 0.092
MARGIN_RIGHT_FRAC = 0.008
MARGIN_BOTTOM_FRAC = 0.010

# Glyph appearance: a low-saturation light gray rendered brighter than the
# surrounding content (white top-hat: brighter than a blurred local background)
# intersected with the grayish + minimum-brightness tests. Same polarity logic as
# the Doubao engine: leaves white-paper documents untouched (the mark is not
# brighter than its surroundings there, so nothing is masked).
MAX_SATURATION = 55  # max channel spread to count a pixel as "grayish"
LOGO_MIN_LUMA = 150  # glyphs are at least this bright in absolute terms
TOPHAT_DELTA = 12  # glyph must exceed the local background by this many levels

# Detection matches the bundled alpha-template glyph silhouette
# (assets/jimeng_alpha.png) against the candidate via zero-mean normalized
# correlation (cv2 TM_CCOEFF_NORMED). Real Jimeng marks score >=0.83, the Doubao
# strip 0.22, other AI output 0.0 -> threshold 0.45 separates cleanly while
# keeping `--mark auto` from confusing Jimeng with Doubao. A small coverage floor
# skips the template match on a near-empty candidate box.
DETECT_MIN_COVERAGE = 0.02
DETECT_NCC_THRESHOLD = 0.45

# ── Reverse-alpha (recovery, Gemini/Doubao-style) ────────────────────
# The Jimeng mark is a fixed semi-transparent white overlay; given its alpha map
# the original pixels are recovered by inverting the blend. The logo is pure white
# (the white capture confirms L=255 and a pair-solve of L lands at ~254.6). The
# alpha map was solved from the GRAY capture: a = (I - B)/(255 - B) with B a
# per-capture CUBIC background fit over the non-glyph pixels, averaged over the
# three channels, kept at full halo extent (down to a~0.02) and UNBLURRED. Gray
# (background ~132, mark contrast ~120) is chosen over black because it is the
# best proxy for real content, where the mark sits on bright photo areas, not on
# black; the careful build drops the gray self-residual to ~1.3 (the earlier
# max-channel / quadratic-bg / blurred / halo-truncated build was visibly worse --
# the mask, not the method, was the limit). The bundled asset
# (assets/jimeng_alpha.png) is the alpha template (a*255) at the captured width.
# The mark scales with image WIDTH; a pure width-scale is only sub-pixel-accurate
# at the captured width, so removal also registers the template to the actual mark
# via a TM_CCOEFF_NORMED scale+position search (`_aligned_alpha_map`) off it.
_ALPHA_NATIVE_WIDTH = 2048
_ALPHA_LOGO_BGR: tuple[float, float, float] = (255.0, 255.0, 255.0)
# Geometry below is emitted by scripts/visible_alpha_solve.py for the bundled
# asset -- keep them in sync when the asset is rebuilt.
_ALPHA_WIDTH_FRAC = 0.2021  # asset width / image width -- the alignment scale seed
_ALPHA_HEIGHT_FRAC = 0.0576
# Margins (of image WIDTH) of the captured mark -- the geometry record / where to
# seed; alignment refines the actual position, so these are not load-bearing.
_ALPHA_MARGIN_RIGHT_FRAC = 0.0288
_ALPHA_MARGIN_BOTTOM_FRAC = 0.0288
# Alignment scale search (np.linspace args) around the width-scaled glyph size --
# fine enough that a per-image scale/position jitter does not leave a thick
# edge-misalignment outline (a coarse step left ~4px slop at the mark ends).
_ALPHA_ALIGN_SEARCH = (0.90, 1.12, 23)
# Residual inpaint footprint: unlike Doubao, Jimeng's per-image render variation
# leaves a faint outline even at native, so the glyph footprint (alpha above this)
# is always inpainted after reverse-alpha (dilated by this kernel, INPAINT_NS).
# Kept deliberately THIN -- the careful alpha map (cubic-background, mean-channel,
# full-halo solve) knocks the mark down far enough that a tight footprint clears
# it, so the inpaint does not smear the texture/edges under the mark the way a
# wide full-footprint pass did.
_RESIDUAL_ALPHA_FLOOR = 0.05
_RESIDUAL_DILATE = 5
_RESIDUAL_INPAINT_RADIUS = 2
_alpha_template_cache: NDArray[Any] | None = None


def _alpha_template() -> NDArray[Any] | None:
    """Lazily load the bundled Jimeng alpha template (float [0,1]), or None."""
    global _alpha_template_cache
    if _alpha_template_cache is None:
        from pathlib import Path

        from remove_ai_watermarks import image_io

        path = Path(__file__).parent / "assets" / "jimeng_alpha.png"
        img = image_io.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        _alpha_template_cache = img.astype(np.float32) / 255.0
    return _alpha_template_cache


@dataclass(frozen=True)
class JimengLocation:
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
class JimengDetection:
    """Result of visible Jimeng watermark detection."""

    detected: bool = False
    confidence: float = 0.0
    region: tuple[int, int, int, int] = (0, 0, 0, 0)
    coverage: float = 0.0  # fraction of the box occupied by glyph pixels


_silhouette_cache: NDArray[Any] | None = None


def _glyph_silhouette() -> NDArray[Any] | None:
    """Binary "即梦AI" silhouette (255 = glyph) from the bundled alpha map, used
    as the detection template. None if the alpha asset is missing."""
    global _silhouette_cache
    if _silhouette_cache is None:
        at = _alpha_template()
        if at is None:
            return None
        _silhouette_cache = (at > 0.15).astype(np.uint8) * 255
    return _silhouette_cache


def _template_match_score(box_mask: NDArray[Any], image_width: int) -> float:
    """Zero-mean normalized correlation of the alpha-template glyph silhouette
    (scaled to the mark's expected size) against the candidate ``box_mask``."""
    sil = _glyph_silhouette()
    if sil is None or box_mask.size == 0:
        return 0.0
    gw = min(box_mask.shape[1] - 1, max(8, int(_ALPHA_WIDTH_FRAC * image_width)))
    gh = min(box_mask.shape[0] - 1, max(4, int(_ALPHA_HEIGHT_FRAC * image_width)))
    if gw < 8 or gh < 4:
        return 0.0
    template = cv2.resize(sil, (gw, gh), interpolation=cv2.INTER_NEAREST)
    return float(cv2.matchTemplate(box_mask, template, cv2.TM_CCOEFF_NORMED).max())


class JimengEngine:
    """Remove the visible Jimeng "即梦AI" watermark (locate -> mask -> reverse-alpha)."""

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

    def locate(self, image: NDArray[Any]) -> JimengLocation:
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
        return JimengLocation(x=x, y=y, w=wm_w, h=wm_h, is_fallback=True)

    # ── Mask ──────────────────────────────────────────────────────────

    def extract_mask(self, image: NDArray[Any], loc: JimengLocation) -> NDArray[Any]:
        """Build a full-image uint8 mask (255 = watermark glyph) for the box.

        Polarity-aware: the mark is a light, low-saturation gray rendered brighter
        than the local background (white top-hat), so a white-paper document is
        left untouched (nothing brighter than its surroundings is masked there).
        """
        h, w = image.shape[:2]
        x, y, bw, bh = loc.bbox
        # Normalize the ROI to 3-channel BGR: a 2D grayscale or 4-channel BGRA
        # input would otherwise break the axis=2 channel reductions below.
        roi = image[y : y + bh, x : x + bw]
        if roi.ndim == 2:
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        elif roi.shape[2] == 4:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2BGR)
        roi = roi.astype(np.float32)

        luma = roi.mean(axis=2)
        sat = roi.max(axis=2) - roi.min(axis=2)
        grayish = sat < MAX_SATURATION

        sigma = max(4.0, bh * 0.4)
        local_bg = cv2.GaussianBlur(luma, (0, 0), sigmaX=sigma, sigmaY=sigma)
        tophat = luma - local_bg

        cand = grayish & (tophat > TOPHAT_DELTA) & (luma > LOGO_MIN_LUMA)
        glyph = cand.astype(np.uint8) * 255
        glyph = cv2.morphologyEx(glyph, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        glyph = cv2.morphologyEx(glyph, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        mask = np.zeros((h, w), np.uint8)
        mask[y : y + bh, x : x + bw] = glyph
        return mask

    # ── Detect ────────────────────────────────────────────────────────

    def detect(self, image: NDArray[Any]) -> JimengDetection:
        """Detect the visible Jimeng mark by matching the alpha-template glyph
        silhouette against the corner candidate (TM_CCOEFF_NORMED)."""
        det = JimengDetection()
        if image is None or image.size == 0:
            return det
        loc = self.locate(image)
        mask = self.extract_mask(image, loc)
        x, y, bw, bh = loc.bbox
        box = mask[y : y + bh, x : x + bw]
        coverage = float((box > 0).sum()) / float(max(1, bw * bh))
        det.region = loc.bbox
        det.coverage = coverage
        if coverage >= DETECT_MIN_COVERAGE:
            score = _template_match_score(box, image.shape[1])
            det.confidence = score
            det.detected = score >= DETECT_NCC_THRESHOLD
            logger.debug("Jimeng detect: coverage=%.3f ncc=%.2f detected=%s", coverage, score, det.detected)
        return det

    # ── Reverse-alpha (recovery + residual inpaint) ───────────────────

    def reverse_alpha_available(self, image: NDArray[Any]) -> bool:
        """True if the bundled alpha map is loadable (NCC alignment places it at
        any resolution; the caller still gates on ``detect``)."""
        return image is not None and image.size > 0 and _alpha_template() is not None

    def _fixed_alpha_map(self, image: NDArray[Any]) -> tuple[NDArray[Any], tuple[int, int, int, int]] | None:
        """Place the template by fixed width-relative geometry."""
        at = _alpha_template()
        if at is None:
            return None
        h, w = image.shape[:2]
        gw = min(w, max(1, int(_ALPHA_WIDTH_FRAC * w)))
        gh = min(h, max(1, int(_ALPHA_HEIGHT_FRAC * w)))
        ax = max(0, w - int(_ALPHA_MARGIN_RIGHT_FRAC * w) - gw)
        ay = max(0, h - int(_ALPHA_MARGIN_BOTTOM_FRAC * w) - gh)
        amap = np.zeros((h, w), np.float32)
        amap[ay : ay + gh, ax : ax + gw] = cv2.resize(at, (gw, gh), interpolation=cv2.INTER_LINEAR)
        return amap, (ax, ay, gw, gh)

    def _aligned_alpha_map(self, image: NDArray[Any]) -> tuple[NDArray[Any], tuple[int, int, int, int]] | None:
        """Register the captured template to the actual mark via a
        TM_CCOEFF_NORMED scale + position search -- so the single capture works
        off the captured width. Returns ``(alpha_map, glyph_bbox)`` or None."""
        at = _alpha_template()
        sil = _glyph_silhouette()
        if at is None or sil is None:
            return None
        h, w = image.shape[:2]
        loc = self.locate(image)
        bx, by, bw, bh = loc.bbox
        box_mask = self.extract_mask(image, loc)[by : by + bh, bx : bx + bw]
        expected = _ALPHA_WIDTH_FRAC * w
        best: tuple[float, int, int, int, int] | None = None
        for scale in np.linspace(*_ALPHA_ALIGN_SEARCH):
            gw, gh = int(expected * scale), int(_ALPHA_HEIGHT_FRAC * w * scale)
            if gw < 8 or gh < 4 or gw >= bw or gh >= bh:
                continue
            t = cv2.resize(sil, (gw, gh), interpolation=cv2.INTER_NEAREST)
            _, score, _, top_left = cv2.minMaxLoc(cv2.matchTemplate(box_mask, t, cv2.TM_CCOEFF_NORMED))
            if best is None or score > best[0]:
                best = (score, gw, gh, top_left[0], top_left[1])
        if best is None:
            return None
        _, gw, gh, ox, oy = best
        ax, ay = bx + ox, by + oy
        amap = np.zeros((h, w), np.float32)
        amap[ay : ay + gh, ax : ax + gw] = cv2.resize(at, (gw, gh), interpolation=cv2.INTER_LINEAR)
        return amap, (ax, ay, gw, gh)

    def _apply_reverse_alpha(self, image: NDArray[Any], amap: NDArray[Any]) -> NDArray[Any]:
        """Invert the alpha blend with ``amap``: ``original = (wm - a*logo)/(1-a)``."""
        a3 = np.clip(amap, 0.0, 1.0)[:, :, None]
        logo = np.array(_ALPHA_LOGO_BGR, np.float32)
        return np.clip((image.astype(np.float32) - a3 * logo) / np.clip(1.0 - a3, 0.25, 1.0), 0, 255).astype(np.uint8)

    def remove_watermark_reverse_alpha(self, image: NDArray[Any], *, residual_inpaint: bool = True) -> NDArray[Any]:
        """Recover the original pixels by inverting the alpha blend, then clear
        the residual outline with a thin inpaint over the glyph footprint.

        Placement: fixed geometry AND the NCC-aligned placement are always tried
        and the one leaving the least residual mark (lowest re-``detect``
        confidence) is kept -- Jimeng jitters the mark a few px per image even at
        the captured width, so fixed geometry alone is not reliable. A single 2048
        alpha cannot pixel-cancel the mark re-rasterized at another resolution, so a
        deliberately THIN residual inpaint (``_RESIDUAL_*``) follows: reverse-alpha
        has already recovered the true background (edges included) under the mark,
        so the inpaint only finishes the residual edges instead of smearing the
        whole footprint. Call only when :meth:`reverse_alpha_available` and the mark
        is detected.
        """
        # Normalize to 3-channel BGR so a 2D grayscale or 4-channel BGRA input
        # does not break the reverse-alpha math (which assumes a 3-channel logo).
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        # Always try fixed geometry AND the NCC-aligned placement and keep
        # whichever leaves the least residual mark (re-detect confidence on the
        # bare reverse-alpha). Unlike Doubao's deterministic overlay, Jimeng jitters
        # the mark's position a few px PER IMAGE even at the captured width, so
        # fixed geometry alone misses there too -- the NCC search registers the
        # template to the actual mark; fixed stays as a fallback if the search has
        # no saliency to lock onto (a flat/contrastless mark).
        maps = [c for c in (self._fixed_alpha_map(image), self._aligned_alpha_map(image)) if c is not None]
        if not maps:
            return image.copy()
        best_out: NDArray[Any] | None = None
        best_amap: NDArray[Any] | None = None
        best_residual = float("inf")
        for amap, _region in maps:
            out = self._apply_reverse_alpha(image, amap)
            residual = self.detect(out).confidence
            if residual < best_residual:
                best_residual, best_out, best_amap = residual, out, amap
        if best_out is None or best_amap is None:  # pragma: no cover - maps is non-empty
            return image.copy()
        if residual_inpaint:
            kernel = np.ones((_RESIDUAL_DILATE, _RESIDUAL_DILATE), np.uint8)
            rm = cv2.dilate((best_amap > _RESIDUAL_ALPHA_FLOOR).astype(np.uint8) * 255, kernel)
            best_out = cv2.inpaint(best_out, rm, _RESIDUAL_INPAINT_RADIUS, cv2.INPAINT_NS)
        return best_out


def load_image_bgr(path: str | Path) -> NDArray[Any]:
    """Read an image as BGR ndarray (helper for scripts/tests)."""
    from remove_ai_watermarks import image_io

    img = image_io.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img
