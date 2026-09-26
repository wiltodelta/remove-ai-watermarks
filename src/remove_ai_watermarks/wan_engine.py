"""Alibaba Wan (通义万相) visible watermark detector/localizer.

Wan image exports carry a white, semi-transparent logo plus "Wan" wordmark in the
bottom-right corner, alongside the TC260 label in XMP whose ContentProducer names
Tongyi Yunqi (Hangzhou) Information Technology (USCC 91330106MA2CFLDG4R, owned by
Alibaba Cloud and Tongyi Lab). The service API's ``noWaterMark: true`` flag did not remove it on
the measured export, so the flag is not evidence of a clean file.

Detection matches the bundled synthetic silhouette (``assets/wan_alpha.png``, drawn
by ``scripts/render_vendor_silhouettes.py::draw_wan``, no export pixels) against the
corner; removal is the shared localize -> fill.

**Calibration is PROVISIONAL.** One real positive exists (a 2048 px square Wan 2.7
Pro export, 2026-09-23): the mark measures 0.131 x 0.050 of the short side with
~0.005 margins and scores 0.61-0.67 at 512-2048 px. NCC alone does not separate
it well: over 3092 local images (every tracked fixture plus 3000 sampled from the
local corpus, 2026-09-24) the maximum was 0.519. The mark hugs the corner, and
requiring the match to end within 0.015 of the short side from the right and
bottom edges leaves no negative at 0.45 or above. Both gates apply. Re-measure on
a real Wan cohort before any precision claim.
"""

from __future__ import annotations

import logging

from remove_ai_watermarks._text_mark_engine import (
    TextMarkConfig,
    TextMarkDetection,
    TextMarkEngine,
    TextMarkScan,
)

logger = logging.getLogger(__name__)

# Locate box as a fraction of the SHORT side: the mark plus slack for the ladder.
WM_WIDTH_FRAC = 0.17
WM_HEIGHT_FRAC = 0.075
MARGIN_RIGHT_FRAC = 0.002
MARGIN_BOTTOM_FRAC = 0.002

MAX_SATURATION = 55
LOGO_MIN_LUMA = 150
TOPHAT_DELTA = 12

DETECT_MIN_COVERAGE = 0.04  # unused by the tophat front-end (kept for config parity)
DETECT_NCC_THRESHOLD = 0.55  # provisional; see the module docstring

_ALPHA_WIDTH_FRAC = 268 / 2048
_ALPHA_HEIGHT_FRAC = 103 / 2048

_CONFIG = TextMarkConfig(
    name="Wan",
    asset_name="wan_alpha.png",
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
    scale_basis="short",
    alpha_width_frac=_ALPHA_WIDTH_FRAC,
    alpha_height_frac=_ALPHA_HEIGHT_FRAC,
    min_gw=8,
    # STRICT ONLY: one positive cannot calibrate a relaxed provenance band.
    provenance_ncc_factor=1.0,
)


class WanEngine(TextMarkEngine):
    """Detect/localize the bottom-right Wan logo plus wordmark (locate -> mask -> fill)."""

    # Match must end this close to the right and bottom edges (fraction of the short
    # side). Measured positive: 0.005-0.008 at 512-2048 px.
    _ANCHOR_MAX_RIGHT = 0.015
    _ANCHOR_MAX_BOTTOM = 0.015

    def __init__(self) -> None:
        super().__init__(_CONFIG)

    def _post_gate(self, det: TextMarkDetection, scan: TextMarkScan) -> TextMarkDetection:
        """Demote a match that does not hug the bottom-right corner.

        A shared post-gate rather than a ``detect`` override, so the single-pass
        perception path (``detect_both``) cannot skip it.
        """
        if not det.detected or scan.loc is None:
            return det
        box = det.match_box
        if box is None:
            det.detected = False
            return det
        h, w = scan.frame
        base = min(h, w)
        right = (w - (scan.loc.x + box[2] + 1)) / base
        bottom = (h - (scan.loc.y + box[3] + 1)) / base
        if not (0 <= right <= self._ANCHOR_MAX_RIGHT and 0 <= bottom <= self._ANCHOR_MAX_BOTTOM):
            logger.debug(
                "Wan detect: score %.3f but match off-anchor (right=%.3f bottom=%.3f); demoting.",
                det.confidence,
                right,
                bottom,
            )
            det.detected = False
        return det
