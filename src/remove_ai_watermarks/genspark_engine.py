"""Genspark visible watermark detector/localizer.

Genspark stamps free-tier AI image exports with a centered semi-transparent white
composite mark: a rounded-square badge carrying transparent four-point sparkle
cutouts above a horizontal rounded bar, followed by the geometric sans-serif
"Genspark" wordmark across the middle of the frame (`corner == "cc"`). Observed
exports ship without C2PA, EXIF generator tags, IPTC AI disclosures, or China
TC260 metadata, making the visible watermark the sole local provenance signal.

Geometry and silhouette proportions were measured on a confirmed 960x1280
portrait carrier (badge 106x106 px, 36 px gap, "Genspark" 324 px wide with a
55 px cap height; 466x106 px overall, centered at 49.8% width and 50.0% height)
and rendered synthetically via ``scripts/render_vendor_silhouettes.py``
(``assets/genspark_alpha.png``, containing zero source-image pixels). The real
carrier scores 0.836 NCC on the continuous white top-hat front-end while 98
non-Genspark evaluation and gallery images peak at 0.550; ``DETECT_NCC_THRESHOLD``
is set conservatively at 0.65 because a center-frame locate box sits over varied
subject content where false fills are destructive.
"""

# pyright: reportUnusedFunction=false

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2

from remove_ai_watermarks import _text_mark_engine
from remove_ai_watermarks._text_mark_engine import TextMarkConfig, TextMarkEngine, TextMarkScan

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Cap scan resolution at the carrier's native 960 px short side so high-megapixel
# photos do not pay multi-second GaussianBlur passes on a 0.72 x 0.24 center box.
_MAX_SCAN_SHORT_SIDE = 960

WM_WIDTH_FRAC = 0.72
WM_HEIGHT_FRAC = 0.24
MARGIN_X_FRAC = 0.0
MARGIN_BOTTOM_FRAC = 0.0

MAX_SATURATION = 55
LOGO_MIN_LUMA = 150
TOPHAT_DELTA = 12

DETECT_MIN_COVERAGE = 0.02
DETECT_NCC_THRESHOLD = 0.65
PROVENANCE_NCC_FACTOR = 1.0

_ALPHA_WIDTH_FRAC = 466 / 960
_ALPHA_HEIGHT_FRAC = (82 / 360) * _ALPHA_WIDTH_FRAC
_ALPHA_NATIVE_SHORT_SIDE = round(360 / _ALPHA_WIDTH_FRAC)

_CONFIG = TextMarkConfig(
    name="Genspark",
    asset_name="genspark_alpha.png",
    corner="cc",
    margin_floor=4,
    width_frac=WM_WIDTH_FRAC,
    height_frac=WM_HEIGHT_FRAC,
    margin_x_frac=MARGIN_X_FRAC,
    margin_bottom_frac=MARGIN_BOTTOM_FRAC,
    max_saturation=MAX_SATURATION,
    logo_min_luma=LOGO_MIN_LUMA,
    tophat_delta=TOPHAT_DELTA,
    morph_open_size=5,
    detect_min_coverage=DETECT_MIN_COVERAGE,
    detect_ncc_threshold=DETECT_NCC_THRESHOLD,
    provenance_ncc_factor=PROVENANCE_NCC_FACTOR,
    detect_frontend="tophat",
    scale_basis="short",
    alpha_width_frac=_ALPHA_WIDTH_FRAC,
    alpha_height_frac=_ALPHA_HEIGHT_FRAC,
    min_gw=16,
)


def _alpha_template() -> NDArray[Any] | None:
    """The bundled synthetic Genspark alpha template (float [0,1]), or None."""
    return _text_mark_engine.load_alpha_template(_CONFIG.asset_name)


def _glyph_silhouette() -> NDArray[Any] | None:
    """Binary Genspark silhouette (255 = glyph) from the bundled alpha map, or None."""
    return _text_mark_engine.glyph_silhouette(_CONFIG.asset_name)


def _template_match_score(box_mask: NDArray[Any], scale_base: int) -> float:
    """TM_CCOEFF_NORMED of the Genspark silhouette against ``box_mask``."""
    return _text_mark_engine.template_match_score(box_mask, scale_base, _CONFIG)


class GensparkEngine(TextMarkEngine):
    """Detect/localize the center-frame Genspark watermark (locate -> mask -> fill)."""

    def __init__(self) -> None:
        super().__init__(_CONFIG)

    def _scan(self, image: NDArray[Any] | None) -> TextMarkScan:
        if image is None or image.size == 0:
            return super()._scan(image)
        h, w = image.shape[:2]
        short = min(h, w)
        if short <= _MAX_SCAN_SHORT_SIDE:
            return super()._scan(image)
        scale = _MAX_SCAN_SHORT_SIDE / float(short)
        sw, sh = max(200, round(w * scale)), max(200, round(h * scale))
        small = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
        sub = super()._scan(small)
        if sub.loc is None or sub.box is None:
            return sub
        loc = self.locate(image)
        inv_x = loc.w / float(max(1, sub.loc.w))
        inv_y = loc.h / float(max(1, sub.loc.h))
        full_box = cv2.resize(sub.box, (loc.w, loc.h), interpolation=cv2.INTER_NEAREST)
        full_match: tuple[int, int, int, int] | None = None
        if sub.match_box is not None:
            mx0, my0, mx1, my1 = sub.match_box
            full_match = (
                int(mx0 * inv_x),
                int(my0 * inv_y),
                min(loc.w - 1, round((mx1 + 1) * inv_x) - 1),
                min(loc.h - 1, round((my1 + 1) * inv_y) - 1),
            )
        return TextMarkScan(
            loc,
            full_box,
            self.scale_base(image),
            frame=(h, w),
            coverage=sub.coverage,
            score=sub.score,
            match_box=full_match,
            match_patch=sub.match_patch,
        )
