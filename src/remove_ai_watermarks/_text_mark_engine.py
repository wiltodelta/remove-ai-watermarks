"""Shared base for the reverse-alpha visible text-mark engines.

The Doubao "豆包AI生成", Jimeng "★ 即梦AI", and Samsung "✦ Contenuti generati
dall'AI" marks are the SAME algorithm: anchor a bottom-corner box by width-relative
geometry, extract the light low-saturation glyph candidate, detect by matching the
bundled alpha-glyph silhouette via ``TM_CCOEFF_NORMED``, and remove by inverting the
alpha blend ``original = (wm - a*logo)/(1-a)`` (always trying fixed AND NCC-aligned
placement, keeping the lower-residual one) plus a thin footprint inpaint.

They differ ONLY in a bounded set of tuned values captured by :class:`TextMarkConfig`:
the constants, the bundled asset, the corner (Doubao/Jimeng bottom-right, Samsung
bottom-left), and a few structural knobs (the morphology-open kernel size and the
minimum glyph width used by the alignment / template-match). Each engine module is a
thin :class:`TextMarkEngine` subclass plus the test-facing module constants/helpers.

Gemini stays a SEPARATE engine (``gemini_engine``): its multi-size fixed-slot sparkle
model is genuinely different, not a tuned variant of this one.
"""

# cv2/numpy boundary: third-party libs ship no usable element types; relax the
# unknown-type rules for this file only.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportAttributeAccessIssue=false, reportPrivateImportUsage=false, reportPrivateUsage=false, reportInvalidTypeForm=false, reportConstantRedefinition=false, reportUnnecessaryComparison=false
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np

from remove_ai_watermarks import image_io

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Reverse-alpha over-subtraction guard (ported from gemini_engine, 2026-06-20).
# The reverse-alpha blend ``(wm - a*logo)/(1-a)`` over-subtracts when the captured
# alpha over-estimates THIS image's mark opacity: on a dark or mid-tone background
# it drives the glyph footprint into a visibly DARKER-than-background ghost (a
# "dark pit") instead of recovering the true pixels. The retained-corpus mining
# (2026-06-20) showed the sparkle-only fix (commit 41f6797) left this unhandled
# for the Doubao/Jimeng text marks. Mirror the sparkle gate: when the recovered
# glyph body lands more than this many gray levels below the local background
# ring, abandon the reverse-alpha output for the footprint and inpaint it from
# the surroundings instead. Calibrated to the same 25-level margin the sparkle
# gate uses -- clean text-mark removals recover within ~10 of the ring, the dark
# pit lands tens of levels below.
_OVERSUB_DARK_MARGIN = 25.0
# Glyph-body / background-ring sampling for the guard. The ring is a pad around
# the glyph box (excluding the box); the body is the bright-core glyph pixels.
_OVERSUB_RING_PAD_FRAC = 0.6  # ring pad as a fraction of the glyph-box height
_OVERSUB_BODY_ALPHA_FLOOR = 0.15  # alpha above which a block pixel counts as glyph body
# Footprint inpaint when the guard trips: dilate the glyph mask wider than the
# thin residual pass so the whole darkened ghost is reconstructed, not just its edge.
_OVERSUB_INPAINT_DILATE = 9
_OVERSUB_INPAINT_RADIUS = 4

# Minimum image short side (px) for text-mark DETECTION. Below this the glyph
# template degrades to the ``min_gw`` floor (~8 px) and TM_CCOEFF_NORMED on a few
# pixels is noise, so an unrelated small geometric shape can spuriously correlate
# with the CJK silhouette (2026-06-26 FP: a 48x48 app icon -- a blue chevron --
# scored Doubao 0.41 / Jimeng 0.47, both above their thresholds). The FP is purely
# a small-size artifact: the same icon upscaled collapses to ~0.06-0.10 NCC at 256
# px and above. A real AI-generation text label is stamped on a full-resolution
# render (the captured samples are 1086-2048 px wide), so 200 px sits far below any
# genuine mark while killing the icon/thumbnail noise band (<=96 px). Detection is
# skipped (verdict stays "unknown", the safe default) rather than risk a false
# positive; removal is gated on detection, so it is suppressed too.
_MIN_DETECT_SHORT_SIDE = 200


@dataclass(frozen=True)
class TextMarkConfig:
    """All per-mark tuning for a reverse-alpha text-mark engine."""

    name: str  # short label for log lines (e.g. "Doubao")
    asset_name: str  # bundled alpha PNG under assets/ (e.g. "doubao_alpha.png")
    corner: Literal["br", "bl"]  # bottom-right (Doubao/Jimeng) or bottom-left (Samsung)
    margin_floor: int  # min margin in px for locate (4 for br marks, 2 for Samsung)
    # locate geometry (fraction of image WIDTH)
    width_frac: float
    height_frac: float
    margin_x_frac: float  # right margin (br) or left margin (bl)
    margin_bottom_frac: float
    # glyph appearance
    max_saturation: float
    logo_min_luma: float
    tophat_delta: float
    morph_open_size: int  # MORPH_OPEN kernel side (5 for br marks, 3 for Samsung)
    # detection
    detect_min_coverage: float
    detect_ncc_threshold: float
    # alpha-map geometry (fraction of WIDTH) emitted by scripts/visible_alpha_solve.py
    alpha_width_frac: float
    alpha_height_frac: float
    alpha_margin_x_frac: float
    alpha_margin_bottom_frac: float
    alpha_align_search: tuple[float, float, int]  # np.linspace(start, stop, num) scale search
    min_gw: int  # minimum glyph width for the template match / align search (8 br, 16 Samsung)
    alpha_logo_bgr: tuple[float, float, float] = (255.0, 255.0, 255.0)
    # residual inpaint over the glyph footprint (thin)
    residual_alpha_floor: float = 0.05
    residual_dilate: int = 5
    residual_inpaint_radius: int = 2


@dataclass
class TextMarkLocation:
    """Located watermark box, in absolute pixel coordinates."""

    x: int
    y: int
    w: int
    h: int
    is_fallback: bool = True  # geometry anchor (no template match) -> always True for now

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


@dataclass
class TextMarkDetection:
    """Result of visible text-mark detection."""

    detected: bool = False
    confidence: float = 0.0
    region: tuple[int, int, int, int] = (0, 0, 0, 0)
    coverage: float = 0.0  # fraction of the box occupied by glyph pixels


# Alpha / silhouette templates, cached per asset name (the originals cached per
# module global; this keys by asset so the three engines share the loader without
# re-reading). Only SUCCESSFUL loads are cached, so a missing asset is retried.
_alpha_cache: dict[str, NDArray[Any]] = {}
_silhouette_cache: dict[str, NDArray[Any]] = {}


def load_alpha_template(asset_name: str) -> NDArray[Any] | None:
    """Lazily load the bundled alpha template (float [0,1]) for ``asset_name``, or None."""
    cached = _alpha_cache.get(asset_name)
    if cached is not None:
        return cached
    path = Path(__file__).parent / "assets" / asset_name
    img = image_io.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    _alpha_cache[asset_name] = img.astype(np.float32) / 255.0
    return _alpha_cache[asset_name]


def glyph_silhouette(asset_name: str) -> NDArray[Any] | None:
    """Binary glyph silhouette (255 = glyph) from the bundled alpha map, or None."""
    cached = _silhouette_cache.get(asset_name)
    if cached is not None:
        return cached
    at = load_alpha_template(asset_name)
    if at is None:
        return None
    _silhouette_cache[asset_name] = (at > 0.15).astype(np.uint8) * 255
    return _silhouette_cache[asset_name]


def template_match_score(box_mask: NDArray[Any], image_width: int, config: TextMarkConfig) -> float:
    """Zero-mean normalized correlation of the alpha-template glyph silhouette
    (scaled to the mark's expected size) against the candidate ``box_mask``.

    ``TM_CCOEFF_NORMED`` keys on glyph SHAPE, not coverage, so a dense textured
    corner does not score highly -- only the actual glyph shape does.
    """
    sil = glyph_silhouette(config.asset_name)
    if sil is None or box_mask.size == 0:
        return 0.0
    gw = min(box_mask.shape[1] - 1, max(config.min_gw, int(config.alpha_width_frac * image_width)))
    gh = min(box_mask.shape[0] - 1, max(4, int(config.alpha_height_frac * image_width)))
    if gw < config.min_gw or gh < 4:
        return 0.0
    template = cv2.resize(sil, (gw, gh), interpolation=cv2.INTER_NEAREST)
    return float(cv2.matchTemplate(box_mask, template, cv2.TM_CCOEFF_NORMED).max())


class TextMarkEngine:
    """Reverse-alpha visible text-mark remover (locate -> mask -> detect -> reverse-alpha)."""

    def __init__(self, config: TextMarkConfig) -> None:
        self.config = config

    # ── Templates (delegate to the asset-keyed module cache) ────────────

    def _alpha_template(self) -> NDArray[Any] | None:
        return load_alpha_template(self.config.asset_name)

    def _glyph_silhouette(self) -> NDArray[Any] | None:
        return glyph_silhouette(self.config.asset_name)

    def _template_match_score(self, box_mask: NDArray[Any], image_width: int) -> float:
        return template_match_score(box_mask, image_width, self.config)

    # ── Locate ──────────────────────────────────────────────────────────

    def locate(self, image: NDArray[Any]) -> TextMarkLocation:
        """Anchor the watermark box in the configured bottom corner by geometry."""
        c = self.config
        h, w = image.shape[:2]
        wm_w = max(40, int(w * c.width_frac))
        wm_h = max(16, int(w * c.height_frac))
        margin_x = max(c.margin_floor, int(w * c.margin_x_frac))
        margin_b = max(c.margin_floor, int(w * c.margin_bottom_frac))
        x = max(0, w - margin_x - wm_w) if c.corner == "br" else min(margin_x, max(0, w - wm_w))
        y = max(0, h - margin_b - wm_h)
        wm_w = min(wm_w, w - x)
        wm_h = min(wm_h, h - y)
        return TextMarkLocation(x=x, y=y, w=wm_w, h=wm_h, is_fallback=True)

    # ── Mask ────────────────────────────────────────────────────────────

    def extract_mask(self, image: NDArray[Any], loc: TextMarkLocation) -> NDArray[Any]:
        """Build a box-sized uint8 mask (255 = watermark glyph) for ``loc``.

        Returns just the glyph mask of the located box (shape ``(loc.h, loc.w)``),
        not a full-frame array: every caller immediately crops to ``loc.bbox``, so
        allocating a full ``(h, w)`` mask and embedding the box was O(image) work
        and memory for an O(box) result -- a wasted full-frame uint8 allocation on
        each detect (~12 MB on a 12 MP frame, recomputed per text-mark detector on
        the memory-tight identify path). The box mask is byte-identical to the old
        full-frame mask cropped to ``loc.bbox``.

        Polarity-aware: the mark is a light, low-saturation gray rendered brighter
        than the local background (white top-hat), so a white-paper document is left
        untouched (nothing brighter than its surroundings is masked there).
        """
        c = self.config
        x, y, bw, bh = loc.bbox
        # A degenerate ROI (a sliver from an extremely wide/short image) cannot hold
        # the mark and would feed cv2's GaussianBlur/morphology a ~1-px-tall array,
        # which can fault native code on some platforms. Skip the cv2 pipeline.
        if bh < 16 or bw < 16:
            return np.zeros((bh, bw), np.uint8)
        # Normalize the ROI to 3-channel BGR (grayscale / BGRA would break axis=2).
        roi = image_io.to_bgr(image[y : y + bh, x : x + bw]).astype(np.float32)

        luma = roi.mean(axis=2)
        sat = roi.max(axis=2) - roi.min(axis=2)
        grayish = sat < c.max_saturation

        # Local background model: a strong Gaussian blur (sigma ~ box height); the
        # white top-hat (luma - local_bg) lights up bright thin strokes regardless
        # of the absolute background level.
        sigma = max(4.0, bh * 0.4)
        local_bg = cv2.GaussianBlur(luma, (0, 0), sigmaX=sigma, sigmaY=sigma)
        tophat = luma - local_bg

        cand = grayish & (tophat > c.tophat_delta) & (luma > c.logo_min_luma)
        glyph = cand.astype(np.uint8) * 255
        glyph = cv2.morphologyEx(glyph, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        k = c.morph_open_size
        return cv2.morphologyEx(glyph, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))

    # ── Detect ──────────────────────────────────────────────────────────

    def detect(self, image: NDArray[Any]) -> TextMarkDetection:
        """Detect the mark by matching the alpha-template glyph silhouette against
        the corner candidate (``TM_CCOEFF_NORMED``); keys on glyph SHAPE, not coverage."""
        c = self.config
        det = TextMarkDetection()
        if image is None or image.size == 0:
            return det
        # Guard against the small-image NCC-noise false positive (see
        # _MIN_DETECT_SHORT_SIDE): an icon/thumbnail is too small to carry a real
        # text label, and the degraded few-pixel template spuriously correlates.
        if min(image.shape[:2]) < _MIN_DETECT_SHORT_SIDE:
            logger.debug(
                "%s detect: image short side %d < %d; too small to carry the mark, skipping.",
                c.name,
                min(image.shape[:2]),
                _MIN_DETECT_SHORT_SIDE,
            )
            return det
        loc = self.locate(image)
        box = self.extract_mask(image, loc)  # box-sized mask (== old full-frame cropped to bbox)
        _x, _y, bw, bh = loc.bbox
        coverage = float((box > 0).sum()) / float(max(1, bw * bh))
        det.region = loc.bbox
        det.coverage = coverage
        if coverage >= c.detect_min_coverage:
            score = self._template_match_score(box, image.shape[1])
            det.confidence = score
            det.detected = score >= c.detect_ncc_threshold
            logger.debug("%s detect: coverage=%.3f ncc=%.2f detected=%s", c.name, coverage, score, det.detected)
        return det

    # ── Reverse-alpha (recovery + thin residual inpaint) ────────────────

    def reverse_alpha_available(self, image: NDArray[Any]) -> bool:
        """True if the bundled alpha map is loadable (NCC alignment places it at any
        resolution; the caller still gates on ``detect`` so a clean corner is untouched)."""
        return image is not None and image.size > 0 and self._alpha_template() is not None

    def _fixed_alpha_map(self, image: NDArray[Any]) -> tuple[NDArray[Any], tuple[int, int, int, int]] | None:
        """Place the template by fixed width-relative geometry (pixel-exact at the
        captured width).

        Returns the glyph-sized alpha BLOCK (shape ``(gh, gw)``) plus its placement
        ``(ax, ay, gw, gh)``, not a full-frame ``(h, w)`` map. The map is non-zero
        only inside the glyph box and every consumer reads exactly that box, so a
        full-frame float32 map was O(image*4 bytes) of mostly zeros -- ~48 MB on a
        12 MP frame, and two were held at once (fixed + aligned). The block is
        byte-identical to the old full-frame map's ``[ay:ay+gh, ax:ax+gw]`` slice.
        """
        c = self.config
        at = self._alpha_template()
        if at is None:
            return None
        h, w = image.shape[:2]
        # Clamp both dims so a wide/short image cannot overflow the slice assignment.
        gw = min(w, max(1, int(c.alpha_width_frac * w)))
        gh = min(h, max(1, int(c.alpha_height_frac * w)))
        if c.corner == "br":
            ax = max(0, w - int(c.alpha_margin_x_frac * w) - gw)
        else:  # bottom-left
            ax = min(max(0, int(c.alpha_margin_x_frac * w)), max(0, w - gw))
        ay = max(0, h - int(c.alpha_margin_bottom_frac * w) - gh)
        block = cv2.resize(at, (gw, gh), interpolation=cv2.INTER_LINEAR)
        return block, (ax, ay, gw, gh)

    def _aligned_alpha_map(self, image: NDArray[Any]) -> tuple[NDArray[Any], tuple[int, int, int, int]] | None:
        """Register the captured template to the actual mark via a TM_CCOEFF_NORMED
        scale + position search. Returns the glyph-sized alpha BLOCK and its
        placement ``(ax, ay, gw, gh)`` (see :meth:`_fixed_alpha_map` for why the
        block, not a full-frame map), or None."""
        c = self.config
        at = self._alpha_template()
        sil = self._glyph_silhouette()
        if at is None or sil is None:
            return None
        w = image.shape[1]
        loc = self.locate(image)
        bx, by, bw, bh = loc.bbox
        box_mask = self.extract_mask(image, loc)  # box-sized (== old full-frame cropped to bbox)
        expected = c.alpha_width_frac * w
        best: tuple[float, int, int, int, int] | None = None
        for scale in np.linspace(*c.alpha_align_search):
            gw, gh = int(expected * scale), int(c.alpha_height_frac * w * scale)
            if gw < c.min_gw or gh < 4 or gw >= bw or gh >= bh:
                continue
            t = cv2.resize(sil, (gw, gh), interpolation=cv2.INTER_NEAREST)
            _, score, _, top_left = cv2.minMaxLoc(cv2.matchTemplate(box_mask, t, cv2.TM_CCOEFF_NORMED))
            if best is None or score > best[0]:
                best = (score, gw, gh, top_left[0], top_left[1])
        if best is None:
            return None
        _, gw, gh, ox, oy = best
        ax, ay = bx + ox, by + oy
        block = cv2.resize(at, (gw, gh), interpolation=cv2.INTER_LINEAR)
        return block, (ax, ay, gw, gh)

    def _apply_reverse_alpha(
        self, image: NDArray[Any], amap: NDArray[Any], region: tuple[int, int, int, int]
    ) -> NDArray[Any]:
        """Invert the alpha blend with ``amap``: ``original = (wm - a*logo)/(1-a)``.

        ``amap`` is the glyph-sized alpha BLOCK for ``region`` (x, y, w, h); outside
        it the blend is a no-op (``(wm - 0)/(1 - 0) == wm``). Compute the math on the
        glyph crop only and copy the rest through unchanged -- byte-identical to a
        full-frame pass (a uint8 round-trip through float32 is exact), but O(glyph)
        instead of O(image): a full-frame pass costs ~275 ms on a 12 MP frame for a
        glyph that is <0.1% of it, and it runs once per candidate placement.
        """
        out = image.copy()
        x1, y1, gw, gh = region
        x2, y2 = x1 + gw, y1 + gh
        if y1 >= y2 or x1 >= x2:
            return out
        a3 = np.clip(amap, 0.0, 1.0)[:, :, None]
        logo = np.array(self.config.alpha_logo_bgr, np.float32)
        roi = out[y1:y2, x1:x2].astype(np.float32)
        out[y1:y2, x1:x2] = np.clip((roi - a3 * logo) / np.clip(1.0 - a3, 0.25, 1.0), 0, 255).astype(np.uint8)
        return out

    def _reverse_alpha_oversubtracts(
        self, image: NDArray[Any], amap: NDArray[Any], region: tuple[int, int, int, int]
    ) -> bool:
        """True when reverse-alpha would darken the glyph footprint into a dark pit.

        Ported from ``gemini_engine._reverse_alpha_oversubtracts`` (2026-06-20):
        PREDICT the reverse-alpha output at the bright glyph core directly from the
        INPUT and the captured alpha, ``(core_obs - a*logo)/(1-a)``, and trip when it
        lands more than ``_OVERSUB_DARK_MARGIN`` gray levels below the local
        background ring. Predicting from the input (not the produced output) keeps the
        gate independent of which placement the reverse-alpha picked, so a clean
        full-strength mark (whose strokes predict back to the background) never trips,
        while a mark fainter than the capture (over-subtracted into a ghost) does.
        """
        ax, ay, gw, gh = region
        ih, iw = image.shape[:2]
        if gw < 4 or gh < 4:
            return False
        if float(amap.max()) < 0.2:  # too faint a capture to over-subtract meaningfully
            return False
        body_box = amap >= _OVERSUB_BODY_ALPHA_FLOOR  # glyph strokes
        if not bool(body_box.any()):
            return False
        pad = max(4, int(gh * _OVERSUB_RING_PAD_FRAC))
        ry1, ry2 = max(0, ay - pad), min(ih, ay + gh + pad)
        rx1, rx2 = max(0, ax - pad), min(iw, ax + gw + pad)
        ring = image[ry1:ry2, rx1:rx2].astype(np.float32).mean(axis=2)
        fy1, fy2, fx1, fx2 = ay - ry1, ay - ry1 + gh, ax - rx1, ax - rx1 + gw
        ring_mask = np.ones(ring.shape, dtype=bool)
        ring_mask[fy1:fy2, fx1:fx2] = False
        if int(ring_mask.sum()) < 10:
            return False
        # Predict the reverse-alpha output PER PIXEL over the glyph body -- exactly
        # the (obs - a*logo)/(1-a) math the remover applies -- so a cleanly captured
        # mark predicts back to the true background everywhere (no trip), while a mark
        # fainter than the capture predicts a body far below the local ring. The
        # per-pixel alpha (not a single peak value) keeps the prediction faithful
        # across the glyph's anti-aliased alpha gradient.
        obs = ring[fy1:fy2, fx1:fx2]
        a = np.clip(amap, 0.0, 0.99)
        logo = float(np.mean(self.config.alpha_logo_bgr))
        predicted = (obs - a * logo) / (1.0 - a)
        predicted_core = float(np.median(predicted[body_box]))
        bg = float(np.median(ring[ring_mask]))
        oversub = predicted_core < bg - _OVERSUB_DARK_MARGIN
        if oversub:
            logger.debug(
                "%s reverse-alpha over-subtracts: predicted core=%.1f bg=%.1f (margin %.0f) -> footprint inpaint",
                self.config.name,
                predicted_core,
                bg,
                _OVERSUB_DARK_MARGIN,
            )
        return oversub

    def _inpaint_footprint(
        self, image: NDArray[Any], amap: NDArray[Any], region: tuple[int, int, int, int]
    ) -> NDArray[Any]:
        """Reconstruct the glyph footprint from its surroundings (used when
        reverse-alpha would over-subtract into a dark pit). Inpaints the ORIGINAL
        image over a dilated glyph mask, so the result never contains the darkened
        reverse-alpha pixels."""
        ax, ay, gw, gh = region
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[ay : ay + gh, ax : ax + gw] = (amap > self.config.residual_alpha_floor).astype(np.uint8) * 255
        mask = cv2.dilate(mask, np.ones((_OVERSUB_INPAINT_DILATE, _OVERSUB_INPAINT_DILATE), np.uint8))
        return cv2.inpaint(image, mask, _OVERSUB_INPAINT_RADIUS, cv2.INPAINT_NS)

    def remove_watermark_reverse_alpha(self, image: NDArray[Any], *, residual_inpaint: bool = True) -> NDArray[Any]:
        """Recover the original pixels by inverting the alpha blend, then clear the
        residual outline with a thin inpaint over the glyph footprint.

        Placement: fixed geometry AND the NCC-aligned placement are always tried and
        the one leaving the least residual mark (lowest re-``detect`` confidence) is
        kept -- the mark re-rasterizes a few px per image, so fixed geometry alone is
        not reliable. A single capture cannot pixel-cancel the mark on every image, so
        a deliberately THIN residual inpaint (``residual_*``) follows: reverse-alpha
        has already recovered the true background under the mark, so the inpaint only
        finishes the residual edges instead of smearing the whole footprint. Call only
        when :meth:`reverse_alpha_available` and the mark is detected.
        """
        c = self.config
        # Normalize to 3-channel BGR (the reverse-alpha math assumes a 3-channel logo).
        image = image_io.to_bgr(image)
        # An image too small to hold the mark would make the geometry boxes degenerate
        # and feed cv2.resize a ~1-px-tall target; skip cv2 entirely.
        h, w = image.shape[:2]
        if h < 32 or w < 64:
            return image.copy()
        maps = [m for m in (self._fixed_alpha_map(image), self._aligned_alpha_map(image)) if m is not None]
        if not maps:
            return image.copy()
        best_out: NDArray[Any] | None = None
        best_amap: NDArray[Any] | None = None
        best_region: tuple[int, int, int, int] | None = None
        best_residual = float("inf")
        for amap, region in maps:
            out = self._apply_reverse_alpha(image, amap, region)
            residual = self.detect(out).confidence
            if residual < best_residual:
                best_residual, best_out, best_amap, best_region = residual, out, amap, region
        if best_out is None or best_amap is None or best_region is None:  # pragma: no cover - maps is non-empty
            return image.copy()
        # Over-subtraction guard: on a dark/mid-tone background the captured alpha can
        # over-estimate the mark's opacity and reverse-alpha leaves a darker-than-
        # background ghost. When the recovered glyph body sits far below the local
        # ring, reconstruct the footprint from its surroundings instead of shipping the
        # dark pit (the thin residual inpaint cannot fix a footprint-wide darkening).
        if self._reverse_alpha_oversubtracts(image, best_amap, best_region):
            return self._inpaint_footprint(image, best_amap, best_region)
        if residual_inpaint:
            # Embed the glyph-sized alpha block into a full-frame uint8 mask only for
            # the inpaint (cv2.inpaint needs a mask matching best_out). One uint8
            # full-frame array, built once, vs the old two full-frame float32 maps;
            # byte-identical to thresholding the old full-frame float32 map (zero
            # outside the block, so the dilate/inpaint see the same mask).
            ax, ay, gw, gh = best_region
            rm = np.zeros(best_out.shape[:2], np.uint8)
            rm[ay : ay + gh, ax : ax + gw] = (best_amap > c.residual_alpha_floor).astype(np.uint8) * 255
            kernel = np.ones((c.residual_dilate, c.residual_dilate), np.uint8)
            rm = cv2.dilate(rm, kernel)
            best_out = cv2.inpaint(best_out, rm, c.residual_inpaint_radius, cv2.INPAINT_NS)
        return best_out

    # ── Inpaint footprint (for the inpaint-fallback removal path) ────────

    def footprint_mask(
        self, image: NDArray[Any], *, force: bool = False, dilate: int | None = None
    ) -> NDArray[Any] | None:
        """Full-frame uint8 mask (255 = mark) of the mark footprint, for the
        inpaint-fallback removal path (LaMa / cv2), or None if no placement fits.

        ``force`` is accepted for a uniform engine signature (the caller passes it to
        every engine) but ignored here -- the text-mark footprint is always the
        geometry-placed captured silhouette, present with or without a detection.

        Uses the NCC-ALIGNED captured silhouette, NOT the per-image
        :meth:`extract_mask` signature: the signature under-segments the glyphs, so
        inpainting it leaves a residual ghost (corpus-validated 2026-07 -- Doubao
        left a "三包" remnant). The mask is dilated to absorb alpha-alignment slop
        (a scale/position mismatch at low detect confidence otherwise leaves a thin
        residual ring); ``dilate`` defaults to a mark-relative margin.

        The caller gates on detection -- this returns the geometric footprint
        regardless, so a clean corner would be masked too.
        """
        image = image_io.to_bgr(image)
        h, w = image.shape[:2]
        if h < 32 or w < 64:
            return None
        placed = self._aligned_alpha_map(image) or self._fixed_alpha_map(image)
        if placed is None:
            return None
        block, (ax, ay, gw, gh) = placed
        sil = (block > self.config.residual_alpha_floor).astype(np.uint8) * 255
        if int((sil > 0).sum()) == 0:
            return None
        mask = np.zeros((h, w), np.uint8)
        ch, cw = min(gh, h - ay), min(gw, w - ax)
        mask[ay : ay + ch, ax : ax + cw] = sil[:ch, :cw]
        d = dilate if dilate is not None else max(9, int(0.05 * gw))
        if d > 0:
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1)))
        return mask
