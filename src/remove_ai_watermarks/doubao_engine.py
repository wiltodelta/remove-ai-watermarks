"""Doubao visible watermark removal engine.

Doubao (ByteDance) stamps every generated image with a visible "豆包AI生成"
(Doubao AI generated) text strip in the bottom-right corner -- the explicit AIGC
label mandated by China's TC260 standard, a near-white semi-transparent overlay.

Like the Gemini sparkle, it is a fixed overlay, so it is removed by **exact
reverse-alpha blending** against a captured alpha map (``remove_watermark_reverse_alpha``):
``original = (wm - a*logo)/(1-a)`` -- recovering the true pixels, not an inpaint
guess. The alpha map + logo colour were solved from black+gray Doubao captures
(see data/doubao_capture/ and the reverse-alpha section below) and bundled as
``assets/doubao_alpha.png``.

Detection (``detect``) is reverse-alpha-consistent: it matches that same alpha
glyph silhouette against the corner via normalized correlation, so it keys on
the actual "豆包AI生成" shape rather than coverage/structure heuristics.

``locate`` (geometry box, scales with image WIDTH) and ``extract_mask`` (the
candidate glyph mask the detector correlates) remain; there is no inpaint-based
removal here -- arbitrary-region inpainting lives in ``region_eraser`` / the
``erase`` command. Fast, offline, no GPU.
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

# Detection is reverse-alpha-consistent: the mark is recognized by matching the
# bundled alpha-template glyph silhouette (assets/doubao_alpha.png -- the exact
# shape we invert) against the extracted candidate mask via zero-mean normalized
# correlation (cv2 TM_CCOEFF_NORMED). It keys on the actual "豆包AI生成" glyph
# SHAPE, not on coverage/structure heuristics, so a merely-textured corner does
# not fire (the old coverage detector false-positived on ~28% of images; #23).
# Corpus-tuned: real marks score median ~0.61, arbitrary corners <=0.17 (p99);
# threshold 0.4 -> false positives 7/1243 (0.6%). A small coverage floor skips
# the template match on a near-empty candidate box.
DETECT_MIN_COVERAGE = 0.04
DETECT_NCC_THRESHOLD = 0.4

# ── Reverse-alpha (exact recovery, Gemini-style) ─────────────────────
# The Doubao mark is a fixed semi-transparent white overlay, so given its alpha
# map the original pixels are recovered exactly: original = (wm - a*logo)/(1-a).
# The alpha map + logo colour were solved from black+gray Doubao captures on a
# controlled background (data/doubao_capture/): on black, captured = a*logo, and
# the black/gray pair solves a per-pixel WITHOUT assuming the logo colour. The
# bundled asset (assets/doubao_alpha.png) is the alpha template (a*255) at the
# captured width. The mark scales with image WIDTH, but a pure width-scale is
# only sub-pixel-accurate at the captured width and ghosts elsewhere, so removal
# does NOT trust fixed geometry: `_aligned_alpha_map` registers the template to
# the actual mark by a TM_CCOEFF_NORMED scale+position search, which makes the
# single capture work at any resolution (verified clean on 1773x2364). Verified
# 2026-05-29: white-capture cross-check -> mark vanishes to a flat fill; clean on
# doubao-1.png (2048) and the 3:4 portrait corpus size.
_ALPHA_NATIVE_WIDTH = 2048
_ALPHA_LOGO_BGR: tuple[float, float, float] = (252.0, 255.0, 255.0)
_ALPHA_WIDTH_FRAC = 0.1572  # glyph width / image width -- the alignment scale seed
_ALPHA_HEIGHT_FRAC = 0.0347
# Margins (of image WIDTH) of the captured mark -- the geometry record / where to
# seed; alignment refines the actual position, so these are not load-bearing.
_ALPHA_MARGIN_RIGHT_FRAC = 0.0166
_ALPHA_MARGIN_BOTTOM_FRAC = 0.0195
# Alignment scale search (np.linspace args) around the width-scaled glyph size.
_ALPHA_ALIGN_SEARCH = (0.88, 1.12, 13)
# At (near) the captured width the fixed geometry is pixel-exact, so we use it
# directly there -- NCC alignment is integer-pixel and would land ~1px off,
# degrading the otherwise-exact native recovery. Off this band, alignment wins.
_ALPHA_NATIVE_BAND = 0.03
_alpha_template_cache: NDArray[Any] | None = None


def _alpha_template() -> NDArray[Any] | None:
    """Lazily load the bundled Doubao alpha template (float [0,1]), or None."""
    global _alpha_template_cache
    if _alpha_template_cache is None:
        from pathlib import Path

        from remove_ai_watermarks import image_io

        path = Path(__file__).parent / "assets" / "doubao_alpha.png"
        img = image_io.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        _alpha_template_cache = img.astype(np.float32) / 255.0
    return _alpha_template_cache


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


_silhouette_cache: NDArray[Any] | None = None


def _glyph_silhouette() -> NDArray[Any] | None:
    """Binary "豆包AI生成" silhouette (255 = glyph) from the bundled alpha map,
    used as the detection template. None if the alpha asset is missing."""
    global _silhouette_cache
    if _silhouette_cache is None:
        at = _alpha_template()
        if at is None:
            return None
        _silhouette_cache = (at > 0.15).astype(np.uint8) * 255
    return _silhouette_cache


def _template_match_score(box_mask: NDArray[Any], image_width: int) -> float:
    """Zero-mean normalized correlation of the alpha-template glyph silhouette
    (scaled to the mark's expected size) against the candidate ``box_mask``.

    TM_CCOEFF_NORMED keys on glyph SHAPE, not coverage, so a dense textured
    corner does not score highly -- only the actual "豆包AI生成" shape does.
    """
    sil = _glyph_silhouette()
    if sil is None or box_mask.size == 0:
        return 0.0
    gw = min(box_mask.shape[1] - 1, max(8, int(_ALPHA_WIDTH_FRAC * image_width)))
    gh = min(box_mask.shape[0] - 1, max(4, int(_ALPHA_HEIGHT_FRAC * image_width)))
    if gw < 8 or gh < 4:
        return 0.0
    template = cv2.resize(sil, (gw, gh), interpolation=cv2.INTER_NEAREST)
    return float(cv2.matchTemplate(box_mask, template, cv2.TM_CCOEFF_NORMED).max())


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
        """Detect the visible Doubao mark by matching the alpha-template glyph
        silhouette against the corner candidate (TM_CCOEFF_NORMED).

        Keys on the "豆包AI生成" SHAPE, not coverage, so a textured corner does
        not fire. ``confidence`` is the correlation score; ``detected`` is it
        clearing ``DETECT_NCC_THRESHOLD``.
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
        if coverage >= DETECT_MIN_COVERAGE:
            score = _template_match_score(box, image.shape[1])
            det.confidence = score
            det.detected = score >= DETECT_NCC_THRESHOLD
            logger.debug("Doubao detect: coverage=%.3f ncc=%.2f detected=%s", coverage, score, det.detected)
        return det

    # ── Reverse-alpha (exact recovery) ────────────────────────────────

    def reverse_alpha_available(self, image: NDArray[Any]) -> bool:
        """True if the bundled alpha map is loadable. Sub-pixel NCC alignment
        (see ``_aligned_alpha_map``) places it on the actual mark at ANY
        resolution, so there is no width gate -- the caller still gates on
        ``detect`` so a clean corner is never touched."""
        return image is not None and image.size > 0 and _alpha_template() is not None

    def _fixed_alpha_map(self, image: NDArray[Any]) -> tuple[NDArray[Any], tuple[int, int, int, int]] | None:
        """Place the template by fixed width-relative geometry -- pixel-exact at
        the captured width (used there instead of integer-pixel NCC alignment)."""
        at = _alpha_template()
        if at is None:
            return None
        h, w = image.shape[:2]
        gw, gh = max(1, int(_ALPHA_WIDTH_FRAC * w)), max(1, int(_ALPHA_HEIGHT_FRAC * w))
        ax = max(0, w - int(_ALPHA_MARGIN_RIGHT_FRAC * w) - gw)
        ay = max(0, h - int(_ALPHA_MARGIN_BOTTOM_FRAC * w) - gh)
        amap = np.zeros((h, w), np.float32)
        amap[ay : ay + gh, ax : ax + gw] = cv2.resize(at, (gw, gh), interpolation=cv2.INTER_LINEAR)
        return amap, (ax, ay, gw, gh)

    def _aligned_alpha_map(self, image: NDArray[Any]) -> tuple[NDArray[Any], tuple[int, int, int, int]] | None:
        """Build a full-image alpha map with the captured template registered to
        the actual mark via a TM_CCOEFF_NORMED scale + position search -- so the
        single capture works off the captured width (a pure width-scale ghosts).
        Returns ``(alpha_map, glyph_bbox)`` or None."""
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
        """Recover the original pixels by inverting the alpha blend
        ``original = (wm - a*logo)/(1-a)``.

        Placement: at (near) the captured width the fixed geometry is pixel-exact,
        so the recovery is returned UNTOUCHED -- inpainting over exactly-recovered
        interior pixels only swaps them for a cv2 hallucination (measured worse on
        textured backgrounds: native error vs true bg 1.6 reverse-alpha-only vs
        2.6 with full-footprint inpaint). Off-native, NCC alignment registers the
        template to the real mark; the alignment is only sub-pixel-approximate, so
        the interior recovery is no longer exact and the seam can re-trip the
        detector. There we try BOTH placements and keep whichever leaves the least
        residual mark (on a faint/busy-background mark the NCC peak can wander a
        few px, where geometry wins; on a clear mark alignment wins) -- no magic
        threshold, it just picks the better removal -- then a residual inpaint over
        the glyph footprint cleans the seam (the interior is approximate anyway, so
        inpaint there costs nothing and reliably clears the mark).
        Call only when :meth:`reverse_alpha_available` and the mark is detected.
        """
        at_native = abs(image.shape[1] / _ALPHA_NATIVE_WIDTH - 1.0) <= _ALPHA_NATIVE_BAND
        if at_native:
            amap = self._fixed_alpha_map(image)
            return self._apply_reverse_alpha(image, amap[0]) if amap is not None else image.copy()
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
            rm = cv2.dilate((best_amap > 0.10).astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
            best_out = cv2.inpaint(best_out, rm, 3, cv2.INPAINT_TELEA)
        return best_out


def load_image_bgr(path: str | Path) -> NDArray[Any]:
    """Read an image as BGR ndarray (helper for scripts/tests)."""
    from remove_ai_watermarks import image_io

    img = image_io.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img
