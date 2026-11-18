"""Tencent Yuanbao visible watermark detector and localizer.

Yuanbao stamps a compact italic two-line mark, ``元宝`` over ``AI生成``, in the
bottom-right corner. The same silhouette is rendered light on dark scenes and
dark on pale scenes, so a one-polarity white top-hat cannot detect it reliably.
This engine uses the shared text-mark pipeline with the ``contrast`` front-end:
normalized absolute local-luma residual followed by silhouette NCC.

The bundled silhouette is synthetic and font-rendered by
``scripts/render_vendor_silhouettes.py``. Removal follows the shared
localize-then-fill path and uses the detector's own match box.

Calibration (2026-07-25) used the metadata-harvested Tencent cohort after byte
deduplication and visual adjudication. The standard two-line variant was detected
on 26 of 28 unique marked carriers (92.9%, lowest positive 0.3975), with 0 fires
on 286 hand-labeled clean frames. Raising the strict gate inside the empty
[0.38, 0.3975) band to 0.395 and requiring both stacked lines plus 2D glyph
detail in ``_post_gate`` eliminates bottom-right shadow, railing, and caption
false positives (0.403-0.439) while keeping ``provenance_ncc_factor=1.0`` and
retaining every positive carrier. The separate photographer-overlay variant is
not covered by this silhouette.
"""

# The module-level helpers are imported by tests.
# pyright: reportUnusedFunction=false

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import cv2

from remove_ai_watermarks import _text_mark_engine
from remove_ai_watermarks._text_mark_engine import (
    BottomRightAnchoredEngine,
    TextMarkConfig,
    TextMarkDetection,
    TextMarkScan,
    glyph_detail_ncc,
)
from remove_ai_watermarks._text_mark_variants import BottomRightBoxTextMarkEngine, TextMarkVariantsMixin

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

WM_WIDTH_FRAC = 0.20
WM_HEIGHT_FRAC = 0.15
MARGIN_RIGHT_FRAC = 0.002
MARGIN_BOTTOM_FRAC = 0.002

MAX_SATURATION = 55
LOGO_MIN_LUMA = 150
TOPHAT_DELTA = 12

DETECT_MIN_COVERAGE = 0.04
DETECT_NCC_THRESHOLD = 0.395

_ALPHA_WIDTH_FRAC = 0.08
_ALPHA_HEIGHT_FRAC = 0.0446
_LADDER = (0.95, 1.0, 1.05)

_CONFIG = TextMarkConfig(
    name="Tencent Yuanbao",
    asset_name="yuanbao_alpha.png",
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
    detect_frontend="contrast",
    scale_basis="short",
    ladder=_LADDER,
    alpha_width_frac=_ALPHA_WIDTH_FRAC,
    alpha_height_frac=_ALPHA_HEIGHT_FRAC,
    min_gw=32,
    provenance_ncc_factor=1.0,
)


def _alpha_template() -> NDArray[Any] | None:
    """The bundled Yuanbao alpha template (float [0,1]), or None."""
    return _text_mark_engine.load_alpha_template(_CONFIG.asset_name)


class YuanbaoEngine(TextMarkVariantsMixin, BottomRightAnchoredEngine):
    """Detect and localize the bottom-right Yuanbao mark."""

    _ANCHOR_MAX_RIGHT = 0.04
    _ANCHOR_MAX_BOTTOM = 0.04
    # Both stacked rows ("元宝" over "AI生成") and row/column mean-subtracted 2D glyph
    # detail must correlate with the silhouette. Measured across 16 real/fixture
    # positives (min line NCC 0.384, min detail NCC 0.326) vs WildAIGC-Bench and
    # corpus clean corner false positives (line min 0.221-0.344, detail 0.214-0.289).
    _MIN_LINE_NCC = 0.36
    _MIN_DETAIL_NCC = 0.31

    def __init__(self) -> None:
        super().__init__(_CONFIG)
        self._variants = (
            BottomRightBoxTextMarkEngine(
                replace(
                    _CONFIG,
                    asset_name="yuanbao_compact_alpha.png",
                    width_frac=0.28,
                    height_frac=0.13,
                    scale_basis="width",
                    alpha_width_frac=0.09,
                    alpha_height_frac=0.09 * 238 / 387 * 0.75,
                    ladder=(0.95, 1.0, 1.05),
                    detect_ncc_threshold=0.50,
                    provenance_ncc_factor=0.62,
                    max_saturation=255,
                )
            ),
            BottomRightBoxTextMarkEngine(
                replace(
                    _CONFIG,
                    asset_name="yuanbao_compact_gray_alpha.png",
                    width_frac=0.28,
                    height_frac=0.13,
                    scale_basis="width",
                    alpha_width_frac=0.06,
                    alpha_height_frac=0.06 * 238 / 387 * 0.90,
                    ladder=(0.95, 1.0, 1.05),
                    detect_frontend="gray",
                    detect_ncc_threshold=0.50,
                    provenance_ncc_factor=0.50,
                    max_saturation=255,
                )
            ),
        )

    def _two_line_glyph_ok(self, patch: NDArray[Any] | None) -> tuple[bool, float, float]:
        sil = self._glyph_silhouette()
        if patch is None or sil is None or patch.shape[0] < 4 or patch.shape[1] < 4:
            return False, 0.0, 0.0
        gh, gw = int(patch.shape[0]), int(patch.shape[1])
        tmpl = cv2.resize(sil, (gw, gh), interpolation=cv2.INTER_AREA)
        mid = gh // 2
        top_ncc = float(cv2.matchTemplate(patch[:mid, :], tmpl[:mid, :], cv2.TM_CCOEFF_NORMED)[0, 0])
        bot_ncc = float(cv2.matchTemplate(patch[mid:, :], tmpl[mid:, :], cv2.TM_CCOEFF_NORMED)[0, 0])
        line_min = min(top_ncc, bot_ncc)
        detail = glyph_detail_ncc(patch, tmpl)
        return line_min >= self._MIN_LINE_NCC and detail >= self._MIN_DETAIL_NCC, line_min, detail

    def _post_gate(self, det: TextMarkDetection, scan: TextMarkScan) -> TextMarkDetection:
        det = super()._post_gate(det, scan)
        if not det.detected:
            return det
        ok, line_min, detail = self._two_line_glyph_ok(scan.match_patch)
        if not ok:
            logger.debug(
                "%s detect: score %.3f failed two-line/detail gate (line_min=%.3f detail=%.3f); demoting.",
                self.config.name,
                det.confidence,
                line_min,
                detail,
            )
            det.detected = False
        return det
