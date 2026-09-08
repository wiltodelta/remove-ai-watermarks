"""LiblibAI visible watermark detector/localizer.

LiblibAI (哩布哩布AI, USCC 91110105MACJ6K1C8A) stamps its generations with a
white triangle logo + "LiblibAI" Latin wordmark at **bottom-center** (not a
corner -- the locate box is horizontally centered). Current outputs can instead
carry a compact top-left ``AI生成`` pill. The wordmark detector matches the
bundled font-rendered "LiblibAI" silhouette (the triangle logo is NOT rendered
-- logos vary, the wordmark discriminates); removal is the shared **localize ->
fill** (the glyph blob covers logo + wordmark, both bright).

This module supplies LiblibAI's tuned :class:`TextMarkConfig`
(``assets/liblib_alpha.png`` from ``scripts/render_vendor_silhouettes.py``,
never cut from an upload) and the corroboration-only compact-pill adapter.

The detector uses an Arial-class synthetic silhouette, width-based geometry, a
strict confidence gate, and a minimum image size. The footprint includes both
the logo and wordmark.
"""
# The module-level _alpha_template / _glyph_silhouette / _template_match_score below
# are thin test-facing shims (imported by tests/), so pyright's src-only pass sees them
# as unused; the use is cross-module.
# pyright: reportUnusedFunction=false

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from remove_ai_watermarks import _text_mark_engine
from remove_ai_watermarks._text_mark_engine import (
    TextMarkConfig,
    TextMarkDetection,
    TextMarkEngine,
    TextMarkLocation,
    TextMarkScan,
)
from remove_ai_watermarks.pill_engine import PillDetection, PillEngine

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Locate geometry as a fraction of the image WIDTH (measured basis). The box is
# horizontally centered (corner="bc") and covers the logo + wordmark with NCC
# slack around the measured 0.10 width.
WM_WIDTH_FRAC = 0.20
WM_HEIGHT_FRAC = 0.09
MARGIN_BOTTOM_FRAC = 0.02

# Glyph appearance: white wordmark on a usually-darker background (white
# top-hat), same overlay class as Doubao -- inherited, harmless because the
# tophat front-end turns these gates into weights.
MAX_SATURATION = 55
LOGO_MIN_LUMA = 150
TOPHAT_DELTA = 12

DETECT_MIN_COVERAGE = 0.04  # unused by the tophat front-end (kept for config parity)
# Calibrated against vendor and clean compatibility examples. The Arial-class
# silhouette separates the wordmark from generic Latin UI text.
DETECT_NCC_THRESHOLD = 0.42

# Detection-silhouette geometry (fraction of the frame width): the wordmark,
# measured 0.10 wide with aspect 0.26.
_ALPHA_WIDTH_FRAC = 0.10
_ALPHA_HEIGHT_FRAC = 0.026

# Tight ladder: the NCC comb is sharp in size (see runninghub_engine).
_LADDER = (0.9, 1.0, 1.1)

# LiblibAI's compact top-left ``AI生成`` pill is a separate current-output
# variant. Its shape scores just below the standalone Jimeng pill gate because
# the Liblib render is shorter. LiblibAI metadata or the reliable wordmark is the
# product anchor here, so this lower component gate never acts on its own.
_TOP_LEFT_PILL_MIN_CONFIDENCE = 0.20
_TOP_LEFT_PILL_X_FRAC = 0.015
_TOP_LEFT_PILL_Y_FRAC = 0.015
_TOP_LEFT_PILL_WIDTH_FRAC = 0.17
_TOP_LEFT_PILL_HEIGHT_FRAC = 0.08

_CONFIG = TextMarkConfig(
    name="LiblibAI",
    asset_name="liblib_alpha.png",
    corner="bc",
    margin_floor=4,
    width_frac=WM_WIDTH_FRAC,
    height_frac=WM_HEIGHT_FRAC,
    margin_x_frac=0.0,  # unused for corner="bc" (horizontally centered)
    margin_bottom_frac=MARGIN_BOTTOM_FRAC,
    max_saturation=MAX_SATURATION,
    logo_min_luma=LOGO_MIN_LUMA,
    tophat_delta=TOPHAT_DELTA,
    morph_open_size=5,
    detect_min_coverage=DETECT_MIN_COVERAGE,
    detect_ncc_threshold=DETECT_NCC_THRESHOLD,
    detect_frontend="tophat",
    scale_basis="width",
    ladder=_LADDER,
    alpha_width_frac=_ALPHA_WIDTH_FRAC,
    alpha_height_frac=_ALPHA_HEIGHT_FRAC,
    min_gw=8,
    # STRICT ONLY: small cohort, the relaxed band is unmeasured.
    provenance_ncc_factor=1.0,
)


def _alpha_template() -> NDArray[Any] | None:
    """The bundled LiblibAI alpha template (float [0,1]), or None."""
    return _text_mark_engine.load_alpha_template(_CONFIG.asset_name)


class LibLibEngine(TextMarkEngine):
    """Detect/localize the visible LiblibAI wordmark (bottom-center; localize -> fill)."""

    # Per-mark size floor prevents small generic icons from matching the wordmark.
    _MIN_SHORT_SIDE = 480

    def __init__(self) -> None:
        super().__init__(_CONFIG)

    def _scan(self, image: NDArray[Any] | None) -> TextMarkScan:
        """Skip the scan entirely below the size floor.

        Gating the SCAN rather than overriding ``detect`` is what keeps the floor on the
        single-pass perception path too, and it means a small image costs nothing.
        """
        if image is None or not image.size or min(image.shape[:2]) < self._MIN_SHORT_SIDE:
            return TextMarkScan(None, None, 0)
        return super()._scan(image)

    def _footprint_rect(
        self,
        image: NDArray[Any],
        loc: TextMarkLocation,
        *,
        force: bool,
        detection: TextMarkDetection | None,
    ) -> tuple[int, int, int, int] | None:
        """Bound the fill by the detector's match box, never by the binary glyph blob.

        The base class's blob-bbox footprint is wrong in both directions here: the
        blob bleeds UP into bright background structure (on the 768x1024 cohort
        frame it reached y 931 and the fill ate the shirt's own print) and it does
        not own the triangle logo anyway.
        """
        return self._match_box_rect(image, loc, force=force, detection=detection)

    def _extend_match_box(
        self, box: tuple[int, int, int, int], loc: TextMarkLocation, frame: tuple[int, int]
    ) -> tuple[int, int, int, int]:
        """Extend the match box LEFT to take in the triangle logo.

        The match box bounds the wordmark exactly (that is what the NCC localized);
        the logo sits its own height to the LEFT of the text (measured on the cohort
        zoom: logo ~1.0x the glyph height, gap ~0.3x), so the footprint is the match
        box extended left by ~1.3 heights.
        """
        gx0, gy0, gx1, gy1 = box
        bx, by, _bw, _bh = loc.bbox
        h, w = frame
        gh = gy1 - gy0 + 1
        pad = max(3, int(0.25 * gh))
        return (
            max(0, bx + gx0 - int(1.3 * gh)),  # the triangle logo, left of the text
            max(0, by + gy0 - pad),
            min(w, bx + gx1 + 1 + pad),
            min(h, by + gy1 + 1 + pad),
        )


class LibLibPillEngine:
    """Detect the compact top-left pill only when LiblibAI is corroborated."""

    def __init__(self) -> None:
        self._pill = PillEngine()

    def _raw(self, image: NDArray[Any]) -> PillDetection:
        return self._pill.detect(image)

    def _relaxed_detected(self, image: NDArray[Any], confidence: float) -> bool:
        return confidence >= _TOP_LEFT_PILL_MIN_CONFIDENCE and self._pill.footprint_is_flat(image)

    def detect(self, image: NDArray[Any], *, provenance: bool = False) -> PillDetection:
        """Return a verdict; the pill is actionable only with LiblibAI evidence."""
        raw = self._raw(image)
        detected = provenance and self._relaxed_detected(image, raw.confidence)
        return PillDetection(detected, raw.confidence, raw.region)

    def detect_both(self, image: NDArray[Any]) -> tuple[PillDetection, PillDetection]:
        """Return strict and LiblibAI-corroborated verdicts from one raw match."""
        raw = self._raw(image)
        strict = PillDetection(False, raw.confidence, raw.region)
        relaxed = PillDetection(self._relaxed_detected(image, raw.confidence), raw.confidence, raw.region)
        return strict, relaxed

    def footprint_mask(
        self,
        image: NDArray[Any],
        *,
        force: bool = False,
        detection: PillDetection | None = None,
    ) -> NDArray[Any] | None:
        """Mask the compact pill's fixed top-left footprint."""
        if not image.size:
            return None
        if detection is None and not force:
            raw = self._raw(image)
            if not raw.detected:
                return None

        frame_h, frame_w = int(image.shape[0]), int(image.shape[1])
        x = int(_TOP_LEFT_PILL_X_FRAC * frame_w)
        y = int(_TOP_LEFT_PILL_Y_FRAC * frame_h)
        width = int(_TOP_LEFT_PILL_WIDTH_FRAC * frame_w)
        height = int(_TOP_LEFT_PILL_HEIGHT_FRAC * frame_w)
        if width <= 0 or height <= 0:
            return None

        from remove_ai_watermarks import region_eraser

        return region_eraser.boxes_to_mask(
            (frame_h, frame_w),
            [(x, y, width, height)],
            dilate=0,
        )
