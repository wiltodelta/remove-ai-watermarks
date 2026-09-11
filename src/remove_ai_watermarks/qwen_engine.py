"""Qwen (Alibaba) visible watermark detector/localizer.

Qwen stamps its generations with either a visible "千问AI生成" text strip or a
small three-lobe symbol in the bottom-right corner. The text is the explicit AIGC
label mandated by China's GB 45438-2025
(the same 6-glyph house style as Doubao's "豆包AI生成": a 2-glyph vendor prefix
plus the shared `AI生成` tail), preceded by the vendor's tri-lobe logo. Qwen
Create's global consumer surface uses the symbol alone; it has its own square
template and search geometry.

Detection matches the bundled glyph silhouette against the corner; removal is the
shared **localize -> fill** (the glyph-bbox :meth:`footprint_mask` feeds
``region_eraser``), NOT reverse-alpha. This module supplies only Qwen's tuned
:class:`TextMarkConfig` (``assets/qwen_alpha.png`` -- a font-rendered synthetic
silhouette from ``scripts/render_vendor_silhouettes.py``, never cut from an
upload). It also feeds ``identify`` as the medium-confidence ``visible_qwen``
signal via the registry.

EVERY tuned number below was measured on the vendor cohort (117 TC260 carriers
whose producer USCC 91440101MA9Y9T4H7A names the entity, 2026-07-21; harness
``scripts/vendor_mark_calibrate.py``), NOT inherited from Doubao:

  * The mark sits in TWO size modes (frac of the short side ~0.124 and ~0.203,
    ratio 1.64 -- wider than the shared 3-rung ladder's 1.5625 span), so a single
    fraction on the shared ladder covers ~75% of marks and the rest land in the
    comb's collapse zone. Qwen therefore carries its OWN 2-rung ladder
    (``TextMarkConfig.ladder``), one rung centered on each mode; the shared
    default is untouched for every other mark.
  * The mark also sits FARTHER off the corner than Doubao's box assumes (right
    margin ~0.025 vs 0.004 of the short side), so Doubao's locate box clipped the
    first glyph and collapsed an exact-size template to 0.26; the box fractions
    below are fitted from the measured absolute mark rects.
  * ``alpha_height_frac`` comes from the aspect fit at the winning width (p50
    aspect 0.26), not from the silhouette's own aspect (0.2219) and not from
    Doubao's ratio.
  * STRICT ONLY (``provenance_ncc_factor`` 1.0): the score band just below the
    gate is dominated by non-Qwen banners on same-cohort frames (a 夸克
    anti-forgery strip at 0.274, a 造点 mark at 0.253), so a provenance-relaxed
    arm would be mostly false fills. No provenance relaxation exists for this
    mark.
  * No rival margin: at the shipped gate the template fires on 0 of 400
    Doubao-marked frames, 0 of 298 Jimeng-marked frames and 0 of 286 hand-labeled
    clean frames (the shared tail correlates at ~0.22, far below the gate), while
    a 0.10 rival margin would have suppressed ~10% of genuine Qwen detections.
"""
# The module-level _alpha_template / _glyph_silhouette / _template_match_score below
# are thin test-facing shims (imported by tests/), so pyright's src-only pass sees them
# as unused; the use is cross-module.
# pyright: reportUnusedFunction=false

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from remove_ai_watermarks import _text_mark_engine, image_io
from remove_ai_watermarks._text_mark_engine import TextMarkConfig, TextMarkDetection, TextMarkEngine

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Locate geometry as a fraction of the image SHORT side (measured basis -- see
# scale_base). The box is fitted to the measured mark rects: the mark's right
# margin is ~0.025 of the short side (not Doubao's 0.004), so the box anchor is
# wider off the corner; width/height cover the big size mode plus NCC slack.
WM_WIDTH_FRAC = 0.231
WM_HEIGHT_FRAC = 0.074
MARGIN_RIGHT_FRAC = 0.0203
MARGIN_BOTTOM_FRAC = 0.0218

# Glyph appearance: a light, low-saturation gray rendered brighter than the local
# background (white top-hat), same overlay class as Doubao -- inherited, and
# harmless because the tophat front-end turns these gates into weights.
MAX_SATURATION = 55
LOGO_MIN_LUMA = 150
TOPHAT_DELTA = 12

DETECT_MIN_COVERAGE = 0.04  # unused by the tophat front-end (kept for config parity)
# Calibrated 2026-07-21 on the vendor cohort vs 286 hand-labeled clean frames
# (cohort-contamination-guarded): clean p99 0.301 / max 0.316, and every cohort
# frame scoring >= 0.45 carries a visible 千问AI生成 mark (86% of the eyeballed
# visible marks fire, the misses being white-on-near-white contrast losses).
# 0.45 was picked over 0.32 (identical clean fire) for margin against unseen
# clean content at zero measured recall cost.
DETECT_NCC_THRESHOLD = 0.45

# Detection-silhouette geometry (fraction of the short side), fitted on the
# cohort: the mark's width modes and its aspect (0.26) at the winning width.
_ALPHA_WIDTH_FRAC = 0.160
_ALPHA_HEIGHT_FRAC = 0.0416

# The two measured size modes as scale rungs: 0.124 and 0.203 of the short side,
# expressed against the 0.160 nominal. Measured, not rounded: off-mode rungs drop
# NCC from ~0.73 to ~0.37 on real marks (the comb), and a 4-rung variant scored
# strictly worse (the extra rungs cover nothing and the big mode lands 4.6% off
# its nearest rung).
_LADDER = (0.78, 1.27)

# Qwen Create symbol-only mark, measured on the cleared 2048-square provider
# fixture. The mark body is ~2.7% of the short side, inset ~3% from both edges.
# The strict 0.65 gate leaves 0.13 NCC margin over the tracked-control maximum
# (0.516 across 132 decoded images other than the positive on 2026-09-10).
_SYMBOL_ASSET = "qwen_symbol_alpha.png"
_SYMBOL_SEARCH_FRAC = 0.11
_SYMBOL_MARGIN_FRAC = 0.012
_SYMBOL_SIZE_FRAC = 0.03125
_SYMBOL_LADDER = (0.75, 0.875, 1.0, 1.125, 1.25)
_SYMBOL_NCC_THRESHOLD = 0.65
_SYMBOL_MIN_SHORT_SIDE = 256
_SYMBOL_MASK_ALPHA = 4 / 255
_SYMBOL_MASK_DILATE_FRAC = 0.06

_CONFIG = TextMarkConfig(
    name="Qwen",
    asset_name="qwen_alpha.png",
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
    scale_basis="short",  # measured: frac_short CV 0.189 vs width 0.273
    ladder=_LADDER,
    alpha_width_frac=_ALPHA_WIDTH_FRAC,
    alpha_height_frac=_ALPHA_HEIGHT_FRAC,
    min_gw=8,
    # STRICT ONLY: the sub-gate band is dominated by non-Qwen banners, so
    # provenance relaxation is disabled outright (factor 1.0 = never relaxed).
    provenance_ncc_factor=1.0,
)


def _alpha_template() -> NDArray[Any] | None:
    """The bundled Qwen alpha template (float [0,1]), or None."""
    return _text_mark_engine.load_alpha_template(_CONFIG.asset_name)


def _glyph_silhouette() -> NDArray[Any] | None:
    """Binary "千问AI生成" silhouette (255 = glyph) from the alpha map, or None."""
    return _text_mark_engine.glyph_silhouette(_CONFIG.asset_name)


class QwenSymbolDetection(TextMarkDetection):
    """Explicit marker for the symbol variant's distinct footprint policy."""


class QwenEngine(TextMarkEngine):
    """Detect and localize Qwen text and symbol visible marks."""

    def __init__(self) -> None:
        super().__init__(_CONFIG)
        template = _text_mark_engine.load_alpha_template(_SYMBOL_ASSET)
        if template is None:
            raise RuntimeError(f"failed to decode embedded asset: {_SYMBOL_ASSET}")
        self._symbol_template = template
        self._symbol_template_scales: dict[int, NDArray[Any]] = {}

    def _scaled_symbol_template(self, side: int) -> NDArray[Any]:
        """Return a cached square symbol template for a measured ladder size."""
        template = self._symbol_template_scales.get(side)
        if template is None:
            if len(self._symbol_template_scales) >= len(_SYMBOL_LADDER) * 4:
                self._symbol_template_scales.clear()
            template = cv2.resize(self._symbol_template, (side, side), interpolation=cv2.INTER_AREA)
            self._symbol_template_scales[side] = template
        return template

    def _symbol_scan(self, image: NDArray[Any] | None) -> QwenSymbolDetection:
        """Locate the strongest symbol-shaped candidate in the lower-right search box."""
        det = QwenSymbolDetection()
        if image is None or image.size == 0:
            return det
        height, width = image.shape[:2]
        base = min(height, width)
        if base < _SYMBOL_MIN_SHORT_SIDE:
            return det
        search_side = max(80, int(base * _SYMBOL_SEARCH_FRAC))
        margin = int(base * _SYMBOL_MARGIN_FRAC)
        origin_x = max(0, width - margin - search_side)
        origin_y = max(0, height - margin - search_side)
        roi = image_io.to_bgr(image[origin_y : origin_y + search_side, origin_x : origin_x + search_side])
        luma: NDArray[Any] = roi.mean(axis=2, dtype=np.float32)
        background: NDArray[Any] = cv2.GaussianBlur(luma, (0, 0), sigmaX=max(4.0, search_side * 0.20))
        response: NDArray[Any] = np.clip(luma - background, 0.0, None)

        best_score = 0.0
        best_region = (0, 0, 0, 0)
        for rung in _SYMBOL_LADDER:
            side = max(12, int(base * _SYMBOL_SIZE_FRAC * rung))
            if side >= search_side:
                continue
            template = self._scaled_symbol_template(side)
            scores = cv2.matchTemplate(response, template, cv2.TM_CCOEFF_NORMED)
            _minimum, score, _min_location, location = cv2.minMaxLoc(scores)
            if score > best_score:
                best_score = float(score)
                best_region = (origin_x + location[0], origin_y + location[1], side, side)
        det.confidence = best_score
        det.detected = best_score >= _SYMBOL_NCC_THRESHOLD
        det.region = best_region
        return det

    def detect(self, image: NDArray[Any], *, provenance: bool = False) -> TextMarkDetection:
        """Return the stronger text or symbol variant."""
        text = super().detect(image, provenance=provenance)
        symbol = self._symbol_scan(image)
        symbol.provenance = provenance
        return _text_mark_engine.best_detection(text, symbol)

    def detect_both(self, image: NDArray[Any] | None) -> tuple[TextMarkDetection, TextMarkDetection]:
        """Return both trust verdicts from one text scan and one symbol scan."""
        strict_text, relaxed_text = super().detect_both(image)
        strict_symbol = self._symbol_scan(image)
        relaxed_symbol = replace(strict_symbol, provenance=True)
        return _text_mark_engine.best_detection(strict_text, strict_symbol), _text_mark_engine.best_detection(
            relaxed_text, relaxed_symbol
        )

    @staticmethod
    def _is_symbol_detection(detection: TextMarkDetection | None) -> bool:
        return isinstance(detection, QwenSymbolDetection) and detection.detected

    def footprint_mask(
        self,
        image: NDArray[Any] | None,
        *,
        force: bool = False,
        dilate: int | None = None,
        detection: TextMarkDetection | None = None,
    ) -> NDArray[Any] | None:
        """Mask the detected symbol box or delegate to the text-mark footprint."""
        if image is None or image.size == 0:
            return None
        resolved = detection if detection is not None else self.detect(image)
        if self._is_symbol_detection(resolved):
            x, y, width, height = resolved.region
            alpha = cv2.resize(self._symbol_template, (width, height), interpolation=cv2.INTER_LINEAR)
            radius = dilate if dilate is not None else max(2, int(max(width, height) * _SYMBOL_MASK_DILATE_FRAC))
            local = np.pad((alpha > _SYMBOL_MASK_ALPHA).astype(np.uint8) * 255, radius)
            if radius > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
                local = cv2.dilate(local, kernel)
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            target_x, target_y = x - radius, y - radius
            x1, y1 = max(0, target_x), max(0, target_y)
            x2, y2 = min(image.shape[1], target_x + local.shape[1]), min(image.shape[0], target_y + local.shape[0])
            local_x, local_y = x1 - target_x, y1 - target_y
            mask[y1:y2, x1:x2] = local[local_y : local_y + y2 - y1, local_x : local_x + x2 - x1]
            return mask
        return super().footprint_mask(image, force=force, dilate=dilate, detection=resolved)
