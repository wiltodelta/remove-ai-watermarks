"""Kling AI (可灵AI, Kuaishou) visible watermark detector/localizer.

Kling AI stamps its generations with a thin, light-gray text strip in the
bottom-right corner, preceded by the vendor's spiral logo (not part of the
detection silhouette -- logos vary between releases, the text run is what
discriminates). Separate silhouettes cover the older "可灵AI 3.0" release and
the current IMAGE 3.0 "KlingAI 3.0" release. An "Omni" suffix and a version-less
"可灵AI" remain outside the calibrated pair.

Detection matches the bundled glyph silhouette against the corner. Removal unions
the pixel-derived footprint with the known silhouettes aligned to the winning match box,
then sends the hybrid mask through the shared **localize -> fill** path in
``region_eraser``; it is not reverse-alpha. This module supplies two tuned
:class:`TextMarkConfig` instances (``assets/kling_alpha.png`` and
``assets/kling_latin_alpha.png`` -- font-rendered synthetic silhouettes from
``scripts/render_vendor_silhouettes.py``, never cut from an upload). The detector
also feeds ``identify`` as the medium-confidence ``visible_kling`` signal via the
registry.

EVERY tuned number below was measured on the vendor cohort (30 TC260 carriers whose
producer USCC 91110108335469089C names the entity, 2026-07-21; harness
``scripts/vendor_mark_calibrate.py``), NOT inherited from Doubao:

  * The mark scales with the SHORT side at ~0.12 of it (mark_w/short measured
    0.118-0.122 across portrait AND landscape carriers -- unimodal, so the shipped
    3-rung ladder covers it) and sits ~0.03 off the right/bottom edges; the locate
    box fractions below are fitted from the measured absolute mark rects.
  * ``alpha_height_frac`` comes from the silhouette aspect (0.239) at the fitted
    width, matching the aspect the fit converged on (0.25).
  * The CJK gate is 0.35, one step above the clean arm's max: on the cohort-vs-clean run
    (cohort-contamination-guarded, 286 hand-labeled clean frames) the clean arm
    scored p99 0.304 / max 0.320, and every cohort frame >= 0.35 carries a visible
    可灵AI 3.0 mark (9 of ~19 eyeballed visible marks fire = ~47% recall of visible
    marks; the misses are the faint "Omni"-suffix release and the version-less
    "可灵AI", which score 0.17-0.25 and cannot be reached without engulfing the
    clean arm). The separately measured Latin IMAGE 3.0 variant is documented
    beside ``_LATIN_CONFIG`` below.
  * STRICT ONLY (``provenance_ncc_factor`` 1.0): the sub-gate band holds real Kling
    variants AND the clean arm's top (clean p90 0.220 vs variant marks at 0.17-0.25
    -- they overlap), so a provenance-relaxed arm cannot separate them. No
    provenance relaxation exists for this mark.
  * No rival margin: at the shipped gate the template fires on 1 of 400
    Doubao-marked frames (0.2%, a 豆包 frame sitting INSIDE the Kling cohort, still
    below the gate), 0 of 298 Jimeng-marked frames and 0 of 286 hand-labeled clean
    frames, and a 0.10 rival margin costs zero genuine Kling detections -- so it is
    simply unnecessary (same conclusion shape as Qwen).
"""
# The module-level _alpha_template / _glyph_silhouette / _template_match_score below
# are thin test-facing shims (imported by tests/), so pyright's src-only pass sees them
# as unused; the use is cross-module.
# pyright: reportUnusedFunction=false

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import cv2

from remove_ai_watermarks import _text_mark_engine
from remove_ai_watermarks._text_mark_engine import TextMarkConfig, TextMarkDetection, TextMarkEngine

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Locate geometry as a fraction of the image SHORT side (measured basis -- see
# scale_base). The box is fitted to the measured mark rects: the mark's right
# margin is ~0.034 of the short side and its bottom margin ~0.027; width/height
# cover the mark plus NCC slack.
WM_WIDTH_FRAC = 0.19
WM_HEIGHT_FRAC = 0.05
MARGIN_RIGHT_FRAC = 0.03
MARGIN_BOTTOM_FRAC = 0.023

# Glyph appearance: a light, low-saturation gray rendered brighter than the local
# background (white top-hat), same overlay class as Doubao -- inherited, and
# harmless because the tophat front-end turns these gates into weights.
MAX_SATURATION = 55
LOGO_MIN_LUMA = 150
TOPHAT_DELTA = 12

DETECT_MIN_COVERAGE = 0.04  # unused by the tophat front-end (kept for config parity)
# Calibrated 2026-07-21 on the vendor cohort vs 286 hand-labeled clean frames
# (cohort-contamination-guarded): clean p99 0.304 / max 0.320, and every cohort
# frame scoring >= 0.35 carries a visible 可灵AI 3.0 mark. 0.35 was picked over
# 0.33 (also zero clean fires) for margin against unseen clean content at a cost
# of zero measured cohort detections.
DETECT_NCC_THRESHOLD = 0.35

# Detection-silhouette geometry (fraction of the short side), fitted on the
# cohort: the mark's width (0.12, unimodal) and the silhouette aspect (0.239).
_ALPHA_WIDTH_FRAC = 0.12
_ALPHA_HEIGHT_FRAC = 0.0287

# A faint Kling overlay can yield a partial thresholded blob even though the
# continuous detector aligned the complete text core. Keep that pixel-derived mask
# for release-specific strokes, then add the aligned synthetic core with a one-pixel
# halo so every glyph that justified the detection is removed on the first fill.
_FOOTPRINT_ALPHA_FLOOR = 0.05
_FOOTPRINT_DILATE = 1

_CONFIG = TextMarkConfig(
    name="Kling AI",
    asset_name="kling_alpha.png",
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
    scale_basis="short",  # measured: mark_w/short 0.118-0.122 across orientations
    alpha_width_frac=_ALPHA_WIDTH_FRAC,
    alpha_height_frac=_ALPHA_HEIGHT_FRAC,
    min_gw=8,
    # STRICT ONLY: the sub-gate band (real Kling variants at 0.17-0.25) overlaps
    # the clean arm's top (p90 0.220), so provenance relaxation is disabled
    # outright (factor 1.0 = never relaxed).
    provenance_ncc_factor=1.0,
)

# Direct Kling IMAGE 3.0 export, 2026-09-04: the current Latin run measures
# 0.095 x 0.028 of the short side inside the same bottom-right locate box. The
# narrow ladder covers small rasterization shifts without admitting any of the
# 94 available neighboring real/synthetic image controls (max 0.349, versus
# 0.429 on the provider original). A deliberately adversarial solid corner blob
# reaches 0.379, so the 0.40 gate stays above both measured negative arms. Keep
# this a separate silhouette and gate:
# replacing the CJK template would discard the older measured release.
_LATIN_CONFIG = replace(
    _CONFIG,
    asset_name="kling_latin_alpha.png",
    alpha_width_frac=0.095,
    alpha_height_frac=0.028,
    ladder=(0.9, 1.0, 1.1),
    detect_ncc_threshold=0.40,
)


def _alpha_template() -> NDArray[Any] | None:
    """The bundled Kling AI alpha template (float [0,1]), or None."""
    return _text_mark_engine.load_alpha_template(_CONFIG.asset_name)


def _glyph_silhouette() -> NDArray[Any] | None:
    """Binary "可灵AI 3.0" silhouette (255 = glyph) from the alpha map, or None."""
    return _text_mark_engine.glyph_silhouette(_CONFIG.asset_name)


class KlingEngine(TextMarkEngine):
    """Detect/localize Kling AI's CJK and Latin 3.0 marks (locate -> mask -> fill)."""

    def __init__(self) -> None:
        super().__init__(_CONFIG)
        self._latin = TextMarkEngine(_LATIN_CONFIG)

    def detect(self, image: NDArray[Any], *, provenance: bool = False) -> TextMarkDetection:
        """Return the strongest CJK or Latin Kling wordmark verdict."""
        return _text_mark_engine.best_detection(
            super().detect(image, provenance=provenance),
            self._latin.detect(image, provenance=provenance),
        )

    def detect_both(self, image: NDArray[Any] | None) -> tuple[TextMarkDetection, TextMarkDetection]:
        """Scan each immutable variant once and combine strict/relaxed verdicts."""
        cjk_strict, cjk_relaxed = super().detect_both(image)
        latin_strict, latin_relaxed = self._latin.detect_both(image)
        return _text_mark_engine.best_detection(cjk_strict, latin_strict), _text_mark_engine.best_detection(
            cjk_relaxed, latin_relaxed
        )

    def footprint_mask(
        self,
        image: NDArray[Any] | None,
        *,
        force: bool = False,
        dilate: int | None = None,
        detection: TextMarkDetection | None = None,
    ) -> NDArray[Any] | None:
        """Add detector-aligned Kling cores to the pixel-derived footprint.

        The legacy mask remains in the union because it can cover variant-specific
        strokes outside the synthetic core. The aligned alpha closes holes caused by
        a faint top-hat blob without replacing that evidence or triggering another
        detector pass. Explicit ``force`` has no trustworthy alignment and retains the
        shared geometry-box behavior.
        """
        if force or image is None or image.size == 0:
            return super().footprint_mask(image, force=force, dilate=dilate, detection=detection)

        det = detection if detection is not None else self.detect(image)
        legacy = super().footprint_mask(image, force=False, dilate=dilate, detection=det)
        radius = _FOOTPRINT_DILATE if dilate is None else max(0, dilate)
        result = legacy
        for config in (_CONFIG, _LATIN_CONFIG):
            alpha = _text_mark_engine.load_alpha_template(config.asset_name)
            if alpha is None:
                continue
            core = self._aligned_alpha_mask(
                image,
                det,
                alpha,
                alpha_floor=_FOOTPRINT_ALPHA_FLOOR,
                dilate=radius,
            )
            if core is not None:
                result = core if result is None else cv2.bitwise_or(result, core)
        return result
