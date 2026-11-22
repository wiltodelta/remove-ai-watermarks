"""Doubao visible watermark detector/localizer.

Doubao (ByteDance) stamps every generated image with a visible "豆包AI生成"
(Doubao AI generated) text strip in the bottom-right corner -- the explicit AIGC
label mandated by China's TC260 standard, a near-white semi-transparent overlay.

Detection matches the bundled glyph silhouette against the corner candidate; removal
is the shared **localize -> fill**. Doubao reuses that aligned silhouette as a sparse
glyph footprint instead of bounding the thresholded corner response: bright scene
texture behind this bottom-edge mark can otherwise expand a solid mask to the frame
edges, where classical inpainting has no outside context. This module shares
:class:`remove_ai_watermarks._text_mark_engine.TextMarkEngine` and
supplies only Doubao's tuned :class:`TextMarkConfig` (bottom-right corner,
``assets/doubao_alpha.png`` -- the detection silhouette, rebuilt by
``scripts/visible_alpha_solve.py``). Arbitrary-region inpainting still lives in
``region_eraser`` / the ``erase`` command.
"""
# The module-level _alpha_template / _glyph_silhouette / _template_match_score below
# are thin test-facing shims (imported by tests/), so pyright's src-only pass sees them
# as unused; the use is cross-module.
# pyright: reportUnusedFunction=false

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from remove_ai_watermarks import _text_mark_engine, image_io
from remove_ai_watermarks._text_mark_engine import TextMarkConfig, TextMarkDetection, TextMarkEngine, TextMarkScan
from remove_ai_watermarks._text_mark_variants import BottomRightBoxTextMarkEngine, TextMarkVariantsMixin

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Locate geometry as a fraction of image WIDTH (the mark scales with width, anchored
# bottom-right). The box is GENEROUSLY wider than the mark and reaches close to the
# corner so a per-image re-rasterization shift stays inside the NCC alignment search.
WM_WIDTH_FRAC = 0.22
WM_HEIGHT_FRAC = 0.075
MARGIN_RIGHT_FRAC = 0.004
MARGIN_BOTTOM_FRAC = 0.004

# Glyph appearance: a light glyph rendered brighter than the local background
# (white top-hat). Preserve the original low-saturation front end, which can
# score better on pale backgrounds than a wider color filter.
MAX_SATURATION = 55  # max channel spread to count a pixel as "grayish"
_TINTED_MAX_SATURATION = 100
_TINTED_CORNER_FRACTION = 0.10
LOGO_MIN_LUMA = 150  # glyphs are at least this bright in absolute terms
TOPHAT_DELTA = 12  # glyph must exceed the local background by this many levels

# Shape-consistent detection: match the bundled alpha glyph silhouette against the
# corner candidate via TM_CCOEFF_NORMED (keys on glyph SHAPE, not coverage; #23).
DETECT_MIN_COVERAGE = 0.04
# NOTE: this gate is FRONT-END SPECIFIC. The continuous top-hat front-end scores higher
# overall than the binary one (mean 0.809 vs 0.723 on the same 90 positives), so the
# binary-era 0.40 left the provenance-relaxed gate (x0.7) far too low and admitted false
# fires. Calibrated on the 240-image unbiased recall sample, full auto path:
#
#   gate   relaxed   recall   precision   true   false
#   0.40     0.280      96%         91%     86       8
#   0.45     0.315      94%         93%     85       6
#   0.50     0.350      92%         99%     83       1   <- chosen
#   0.60     0.420      87%         99%     78       1
#
# 0.50 beats the binary front-end on recall (92% vs 89%) at identical precision (99%),
# which is the only reason the front-end switch is worth it. Do not port this number to
# a binary-front-end mark; re-calibrate per front-end.
DETECT_NCC_THRESHOLD = 0.50

# Detection-silhouette geometry, emitted by scripts/visible_alpha_solve.py at the
# captured width. Sizes the glyph silhouette for the TM_CCOEFF_NORMED detection match;
# Doubao also reuses the aligned alpha as its sparse removal footprint.
_ALPHA_NATIVE_WIDTH = 2048
_ALPHA_WIDTH_FRAC = 0.1636  # asset width / image width -- sizes the detection silhouette
_ALPHA_HEIGHT_FRAC = 0.0405

# The captured alpha is also the removal footprint after the detector aligns it.
# Keep faint anti-aliased edges, then grow by one pixel so cv2 does not leave a halo.
# A solid enclosing rectangle is unsafe here: the mark sits close to two frame edges,
# and a textured branch in the canonical sample made the thresholded response reach
# both edges before OpenCV inpainted the whole block into triangular wedges.
_FOOTPRINT_ALPHA_FLOOR = 0.05
_FOOTPRINT_DILATE = 1

_CONFIG = TextMarkConfig(
    name="Doubao",
    asset_name="doubao_alpha.png",
    corner="br",
    margin_floor=4,
    width_frac=WM_WIDTH_FRAC,
    height_frac=WM_HEIGHT_FRAC,
    margin_x_frac=MARGIN_RIGHT_FRAC,
    margin_bottom_frac=MARGIN_BOTTOM_FRAC,
    max_saturation=MAX_SATURATION,
    logo_min_luma=LOGO_MIN_LUMA,
    tophat_delta=TOPHAT_DELTA,
    morph_open_size=5,
    detect_min_coverage=DETECT_MIN_COVERAGE,
    detect_ncc_threshold=DETECT_NCC_THRESHOLD,
    detect_frontend="tophat",
    scale_basis="short",  # measured: recovers 56% of landscape misses (see scale_base)
    # No rival margin: measured 2026-07-18, the symmetric gate cost Doubao 7 genuine
    # detections to prevent 5 false ones (1.4:1 against). Doubao's absolute detector
    # is already 86% precise, so it has nothing to buy; Jimeng's is 38% and gains 25pp
    # for free. The confusion is asymmetric, so the remedy is too.
    alpha_width_frac=_ALPHA_WIDTH_FRAC,
    alpha_height_frac=_ALPHA_HEIGHT_FRAC,
    min_gw=8,
)

# A translucent white mark over yellow/orange imagery can retain enough of the
# background color to miss the original 55-level filter. Search with a wider
# filter only when the corner is substantially colorful. Keep already accepted
# detections and otherwise retain the stronger of the two scans.
_TINTED_ENGINE = TextMarkEngine(dataclasses.replace(_CONFIG, max_saturation=_TINTED_MAX_SATURATION))


class _OutlinedDoubaoEngine(BottomRightBoxTextMarkEngine):
    # The outlined layout's provenance-relaxed NCC floor overlaps ordinary corner
    # texture. Require bright neutral strokes and edge detail in the winning box
    # to distinguish the label from unmarked corners. Metadata confirms
    # Doubao as the producer, not that the export retained a visible label.
    _MIN_BRIGHT_NEUTRAL_FRACTION = 0.29
    _MIN_MEAN_LAPLACIAN = 10.0
    _BOX_PAD_LEFT_HEIGHT = 1.2
    _BOX_PAD_RIGHT_HEIGHT = 0.7

    @classmethod
    def _outline_pixels_ok(cls, patch: NDArray[Any]) -> bool:
        if patch.size == 0:
            return False
        patch = image_io.to_bgr(patch)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        saturation = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 1]
        bright_neutral = float(np.mean((gray > 160) & (saturation < 60)))
        edge_detail = float(np.mean(np.abs(cv2.Laplacian(gray, cv2.CV_32F))))
        return bright_neutral >= cls._MIN_BRIGHT_NEUTRAL_FRACTION and edge_detail >= cls._MIN_MEAN_LAPLACIAN

    def _scan(self, image: NDArray[Any] | None) -> TextMarkScan:
        scan = super()._scan(image)
        if image is None or scan.loc is None or scan.match_box is None or scan.score is None:
            return scan
        x0, y0, x1, y1 = scan.match_box
        patch = image[
            scan.loc.y + y0 : scan.loc.y + y1 + 1,
            scan.loc.x + x0 : scan.loc.x + x1 + 1,
        ]
        if not self._outline_pixels_ok(patch):
            return dataclasses.replace(scan, score=0.0)
        return scan


def _alpha_template() -> NDArray[Any] | None:
    """The bundled Doubao alpha template (float [0,1]), or None."""
    return _text_mark_engine.load_alpha_template(_CONFIG.asset_name)


def _glyph_silhouette() -> NDArray[Any] | None:
    """Binary "豆包AI生成" silhouette (255 = glyph) from the alpha map, or None."""
    return _text_mark_engine.glyph_silhouette(_CONFIG.asset_name)


def _template_match_score(box_mask: NDArray[Any], scale_base: int) -> float:
    """TM_CCOEFF_NORMED of the Doubao glyph silhouette against ``box_mask``."""
    return _text_mark_engine.template_match_score(box_mask, scale_base, _CONFIG)


class DoubaoEngine(TextMarkVariantsMixin):
    """Detect/localize the visible Doubao "豆包AI生成" watermark (locate -> mask; mask feeds the fill)."""

    def __init__(self) -> None:
        super().__init__(_CONFIG)
        self._variants = (
            _OutlinedDoubaoEngine(
                dataclasses.replace(
                    _CONFIG,
                    asset_name="doubao_outline_alpha.png",
                    width_frac=0.28,
                    height_frac=0.13,
                    scale_basis="width",
                    alpha_width_frac=0.14,
                    alpha_height_frac=0.14 * 0.75 * 77 / 346,
                    max_saturation=255,
                    detect_frontend="contrast",
                    ladder=(0.95, 1.0, 1.05),
                    detect_ncc_threshold=0.50,
                    provenance_ncc_factor=0.25,
                )
            ),
        )

    def _scan(self, image: NDArray[Any] | None) -> TextMarkScan:
        base = super()._scan(image)
        if base.loc is None or image is None or (base.score or 0) >= DETECT_NCC_THRESHOLD:
            return base

        x, y, w, h = base.loc.bbox
        corner = image_io.to_bgr(image[y : y + h, x : x + w])
        channel_spread = corner.max(axis=2) - corner.min(axis=2)
        colored = np.count_nonzero(channel_spread >= _TINTED_MAX_SATURATION)
        if colored < _TINTED_CORNER_FRACTION * channel_spread.size:
            return base

        tinted = _TINTED_ENGINE._scan(image)
        return tinted if (tinted.score or 0) > (base.score or 0) else base

    def footprint_mask(
        self,
        image: NDArray[Any] | None,
        *,
        force: bool = False,
        dilate: int | None = None,
        detection: TextMarkDetection | None = None,
    ) -> NDArray[Any] | None:
        """Return the detector-aligned Doubao glyph footprint.

        The continuous top-hat response is suitable for locating the wordmark but is
        not a safe removal mask: unrelated bright texture can join the response and
        enlarge its bounding rectangle. The detector already carries the winning
        template box, so resize the captured alpha to that exact box and mask only its
        glyphs. Explicit ``force`` has no trustworthy alignment and retains the shared
        geometry-box behavior.
        """
        if force:
            return super().footprint_mask(image, force=True, dilate=dilate, detection=detection)
        if image is None or image.size == 0:
            return None

        image = image_io.to_bgr(image)
        det = detection if detection is not None else self.detect(image)
        if not det.detected or det.match_box is None:
            return super().footprint_mask(image, force=False, dilate=dilate, detection=det)
        engine = self.variant_engine(det)
        if engine is not self:
            return engine.footprint_mask(image, force=False, dilate=dilate, detection=det)
        alpha = _text_mark_engine.load_alpha_template(engine.config.asset_name)
        if alpha is None:
            return super().footprint_mask(image, force=False, dilate=dilate, detection=det)

        radius = _FOOTPRINT_DILATE if dilate is None else max(0, dilate)
        base_mask = engine._aligned_alpha_mask(
            image,
            det,
            alpha,
            alpha_floor=_FOOTPRINT_ALPHA_FLOOR,
            dilate=radius,
        )
        if dilate is not None or base_mask is None:
            return base_mask

        # Some single-line Doubao exports render a dark border stroke around the
        # standard ``豆包AI生成`` glyphs over light backgrounds. A 1-px dilation masks
        # only the white interior and leaves the dark stroke ring intact. Check a
        # bounded corner slice around the aligned match box: when the immediate 4-px
        # stroke ring dips substantially darker than the surrounding outer ring,
        # grow the aligned alpha mask across the dark stroke while leaving
        # un-outlined white marks at their 1-px footprint.
        loc = engine.locate(image)
        mx0, my0, mx1, my1 = det.match_box
        h, w = image.shape[:2]
        pad = 16
        ry0 = max(0, loc.y + my0 - pad)
        ry1 = min(h, loc.y + my1 + 1 + pad)
        rx0 = max(0, loc.x + mx0 - pad)
        rx1 = min(w, loc.x + mx1 + 1 + pad)
        m_roi = base_mask[ry0:ry1, rx0:rx1]
        if not m_roi.any():
            return base_mask
        d_stroke = cv2.dilate(m_roi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        d_outer = cv2.dilate(m_roi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
        ring_stroke = (d_stroke > 0) & (m_roi == 0)
        ring_outer = (d_outer > 0) & (d_stroke == 0)
        if np.count_nonzero(ring_stroke) >= 32 and np.count_nonzero(ring_outer) >= 32:
            gray = cv2.cvtColor(image_io.to_bgr(image[ry0:ry1, rx0:rx1]), cv2.COLOR_BGR2GRAY).astype(np.float32)
            q10_out, q25_out = np.percentile(gray[ring_outer], (10, 25))
            q10_str, q25_str = np.percentile(gray[ring_stroke], (10, 25))
            if float(q10_out - q10_str) >= 14.0 and float(q25_out - q25_str) >= 5.0:
                mh = my1 - my0 + 1
                stroke_radius = max(_FOOTPRINT_DILATE + 1, round(0.075 * mh))
                return engine._aligned_alpha_mask(
                    image,
                    det,
                    alpha,
                    alpha_floor=_FOOTPRINT_ALPHA_FLOOR,
                    dilate=stroke_radius,
                )
        return base_mask
