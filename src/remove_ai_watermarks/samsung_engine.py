"""Samsung Galaxy AI visible watermark detector/localizer.

Samsung's on-device Generative AI photo edits burn a visible "✦ Contenuti generati
dall'AI" wordmark into the bottom-LEFT corner (the Italian locale variant calibrated
here; the string is locale-specific -- DETECTION only matches this locale's silhouette,
so other locales are not yet detected, though the fill mask itself is locale-agnostic).
It is a faint, near-white semi-transparent overlay, the same overlay class as the
Doubao/Jimeng marks but bottom-left.

Detection matches the bundled glyph silhouette against the corner; removal is the
shared **localize -> fill** (a detector-aligned alpha :meth:`footprint_mask` feeds
``region_eraser``), NOT reverse-alpha. This module shares
:class:`remove_ai_watermarks._text_mark_engine.TextMarkEngine` and
supplies only Samsung's tuned :class:`TextMarkConfig` (bottom-LEFT corner, a lower glyph
luma since the mark is faint, ``assets/samsung_alpha.png`` -- the detection silhouette,
solved from the flat captures by ``scripts/visible_alpha_solve.py``). Samsung Galaxy AI
edits are also caught by C2PA + the ``genAIType`` marker, so this is the visible-mark
*removal* path; it also feeds ``identify`` as the medium-confidence ``visible_samsung``
signal via the registry.
"""
# The module-level _alpha_template / _glyph_silhouette / _template_match_score below
# are thin test-facing shims (imported by tests/), so pyright's src-only pass sees them
# as unused; the use is cross-module.
# pyright: reportUnusedFunction=false

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from remove_ai_watermarks import _text_mark_engine, image_io
from remove_ai_watermarks._text_mark_engine import TextMarkConfig, TextMarkDetection
from remove_ai_watermarks._text_mark_variants import GlyphTextMarkEngine, TextMarkVariantsMixin

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Locate geometry as a fraction of image WIDTH (mark scales with width, bottom-LEFT).
WM_WIDTH_FRAC = 0.40
WM_HEIGHT_FRAC = 0.060
MARGIN_LEFT_FRAC = 0.004
MARGIN_BOTTOM_FRAC = 0.002

# Glyph appearance: a light, low-saturation gray. LOGO_MIN_LUMA is lower than Jimeng's
# because the mark is faint (peak alpha ~0.38), so on a mid/dark background its glyph
# luma is lower; a white-paper document is still left untouched.
MAX_SATURATION = 55
LOGO_MIN_LUMA = 110
TOPHAT_DELTA = 8

# Shape-consistent detection. Retained real positives score 0.69-0.90 (0.82-0.90 on
# the three primary calibration captures) and clean non-caption controls stay <=0.39,
# while archival bottom-left postcard captions and negative borders in WildAIGC-Bench
# reach 0.403-0.434 (and synthetic caption decoys reach 0.435). Raising the strict gate
# to 0.50 clears captions with >=0.065 headroom while retaining 305/421 nominal
# constructed marks; ``PROVENANCE_NCC_FACTOR = 0.40 / DETECT_NCC_THRESHOLD`` keeps the
# ``samsung_genai`` metadata-corroborated gate at 0.40 (above clean texture, avoiding
# the 0.28 clean-fire band). The exact one-rung ladder matches both measured widths and
# avoids handing clean corners unneeded scale trials.
DETECT_MIN_COVERAGE = 0.01
DETECT_NCC_THRESHOLD = 0.50
PROVENANCE_NCC_FACTOR = 0.40 / DETECT_NCC_THRESHOLD

# Detection-silhouette geometry, solved by scripts/visible_alpha_solve.py from the flat
# gray capture (native width 1086). Real photos are ~2958 wide, so the captured glyph is
# upscaled; width-scale + NCC-align sizes the silhouette for the detection match (removal
# aligns this alpha to the detector's match box; see footprint_mask).
_ALPHA_NATIVE_WIDTH = 1086
_ALPHA_WIDTH_FRAC = 0.3195  # asset width / image width -- sizes the detection silhouette
_ALPHA_HEIGHT_FRAC = 0.0378

# The solved alpha carries a wide low-opacity halo. A 0.05 floor plus one-pixel
# dilation covered the visible strokes on both retained provider captures while
# reducing the fill from ~19.7k pixels to ~4.9k at 1086 px width. It also covers
# >=94% of every changed pixel on both deterministic gallery backgrounds.
_FOOTPRINT_ALPHA_FLOOR = 0.05
_FOOTPRINT_DILATE = 1

_CONFIG = TextMarkConfig(
    name="Samsung Galaxy AI",
    asset_name="samsung_alpha.png",
    corner="bl",
    margin_floor=2,
    width_frac=WM_WIDTH_FRAC,
    height_frac=WM_HEIGHT_FRAC,
    margin_x_frac=MARGIN_LEFT_FRAC,
    margin_bottom_frac=MARGIN_BOTTOM_FRAC,
    max_saturation=MAX_SATURATION,
    logo_min_luma=LOGO_MIN_LUMA,
    tophat_delta=TOPHAT_DELTA,
    morph_open_size=3,
    detect_min_coverage=DETECT_MIN_COVERAGE,
    detect_ncc_threshold=DETECT_NCC_THRESHOLD,
    detect_frontend="tophat",
    ladder=(1.0,),
    provenance_ncc_factor=PROVENANCE_NCC_FACTOR,
    alpha_width_frac=_ALPHA_WIDTH_FRAC,
    alpha_height_frac=_ALPHA_HEIGHT_FRAC,
    min_gw=16,
)


def _alpha_template() -> NDArray[Any] | None:
    """The bundled Samsung alpha template (float [0,1]), or None."""
    return _text_mark_engine.load_alpha_template(_CONFIG.asset_name)


def _glyph_silhouette() -> NDArray[Any] | None:
    """Binary "Contenuti generati dall'AI" silhouette (255 = glyph), or None."""
    return _text_mark_engine.glyph_silhouette(_CONFIG.asset_name)


def _template_match_score(box_mask: NDArray[Any], scale_base: int) -> float:
    """TM_CCOEFF_NORMED of the Samsung glyph silhouette against ``box_mask``."""
    return _text_mark_engine.template_match_score(box_mask, scale_base, _CONFIG)


class SamsungEngine(TextMarkVariantsMixin):
    """Detect/localize the visible Samsung Galaxy AI text mark (locate -> mask; mask feeds the fill)."""

    def __init__(self) -> None:
        super().__init__(_CONFIG)
        self._variants = (
            GlyphTextMarkEngine(
                replace(
                    _CONFIG,
                    asset_name="samsung_ko_alpha.png",
                    width_frac=0.28,
                    height_frac=0.065,
                    alpha_width_frac=0.20,
                    alpha_height_frac=0.20 * 49 / 344,
                    ladder=(0.95, 1.0, 1.05),
                    max_saturation=255,
                    detect_frontend="contrast",
                    detect_ncc_threshold=0.46,
                    provenance_ncc_factor=1.0,
                )
            ),
        )

    def footprint_mask(
        self,
        image: NDArray[Any] | None,
        *,
        force: bool = False,
        dilate: int | None = None,
        detection: TextMarkDetection | None = None,
    ) -> NDArray[Any] | None:
        """Return the detector-aligned Samsung glyph footprint.

        Samsung's peak opacity is only about 0.38, so filling the enclosing wordmark
        rectangle destroys substantially more real content than the overlay damaged.
        The continuous detector already found the captured silhouette at one exact box;
        align the solved alpha to that box and mask its strokes only. Explicit
        ``force`` has no trustworthy alignment and retains the shared geometry box.
        """
        if force or image is None or image.size == 0:
            return super().footprint_mask(image, force=force, dilate=dilate, detection=detection)

        image = image_io.to_bgr(image)
        det = detection if detection is not None else self.detect(image)
        engine = self.variant_engine(det)
        radius = _FOOTPRINT_DILATE if dilate is None else max(0, dilate)
        if engine is not self and det.detected and det.match_box is not None:
            from remove_ai_watermarks import region_eraser

            x0, y0, x1, y1 = det.match_box
            loc = engine.locate(image)
            glyph_height = y1 - y0 + 1
            return region_eraser.boxes_to_mask(
                image.shape[:2],
                [
                    (
                        max(0, loc.x + x0 - round(1.3 * glyph_height)),
                        loc.y + y0,
                        x1 - x0 + 1 + round(1.3 * glyph_height),
                        glyph_height,
                    )
                ],
                dilate=radius,
            )
        alpha = _text_mark_engine.load_alpha_template(engine.config.asset_name)
        if alpha is not None:
            aligned = engine._aligned_alpha_mask(
                image,
                det,
                alpha,
                alpha_floor=_FOOTPRINT_ALPHA_FLOOR,
                dilate=radius,
            )
            if aligned is not None:
                return aligned
        return super().footprint_mask(image, force=False, dilate=dilate, detection=det)
