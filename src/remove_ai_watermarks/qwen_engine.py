"""Qwen (Tongyi Qianwen, Alibaba) visible watermark detector/localizer.

Qwen stamps its generations with a visible "千问AI生成" text strip in the
bottom-right corner -- the explicit AIGC label mandated by China's GB 45438-2025
(the same 6-glyph house style as Doubao's "豆包AI生成": a 2-glyph vendor prefix
plus the shared `AI生成` tail), preceded by the vendor's tri-lobe logo (not part
of the detection silhouette -- logos vary between releases, the CJK run is what
discriminates).

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
    (``TextMarkConfig.ladder``), one rung centred on each mode; the shared
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
    Doubao-marked frames, 0 of 298 Jimeng-marked frames and 0 of 286 hand-labelled
    clean frames (the shared tail correlates at ~0.22, far below the gate), while
    a 0.10 rival margin would have suppressed ~10% of genuine Qwen detections.
"""
# The module-level _alpha_template / _glyph_silhouette / _template_match_score below
# are thin test-facing shims (imported by tests/), so pyright's src-only pass sees them
# as unused; the use is cross-module.
# pyright: reportUnusedFunction=false

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from remove_ai_watermarks import _text_mark_engine
from remove_ai_watermarks._text_mark_engine import TextMarkConfig, TextMarkDetection, TextMarkEngine

if TYPE_CHECKING:
    from pathlib import Path

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
# Calibrated 2026-07-21 on the vendor cohort vs 286 hand-labelled clean frames
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

QwenDetection = TextMarkDetection


def _alpha_template() -> NDArray[Any] | None:
    """The bundled Qwen alpha template (float [0,1]), or None."""
    return _text_mark_engine.load_alpha_template(_CONFIG.asset_name)


def _glyph_silhouette() -> NDArray[Any] | None:
    """Binary "千问AI生成" silhouette (255 = glyph) from the alpha map, or None."""
    return _text_mark_engine.glyph_silhouette(_CONFIG.asset_name)


def _template_match_score(box_mask: NDArray[Any], scale_base: int) -> float:
    """TM_CCOEFF_NORMED of the Qwen glyph silhouette against ``box_mask``."""
    return _text_mark_engine.template_match_score(box_mask, scale_base, _CONFIG)


class QwenEngine(TextMarkEngine):
    """Detect/localize the visible Qwen "千问AI生成" watermark (locate -> mask; mask feeds the fill)."""

    def __init__(self) -> None:
        super().__init__(_CONFIG)


def load_image_bgr(path: str | Path) -> NDArray[Any]:
    """Read an image as BGR ndarray (helper for scripts/tests)."""
    from remove_ai_watermarks import image_io

    img = image_io.imread(path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img
