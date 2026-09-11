"""Shared base for visible text-mark detectors and localizers.

Each mark supplies a :class:`TextMarkConfig` with its silhouette, expected area,
scale ladder, detector frontend, and calibrated gates. The shared engine locates the
candidate, scores the silhouette with normalized correlation, and builds a removal
footprint for the common fill backend. Individual engines may override detection or
footprint behavior when their measured variant requires it.

The removal path never performs reverse-alpha pixel recovery. Gemini and the Jimeng
pill remain separate engines because their geometry and gating differ from text marks.
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

# Provenance-confirmed NCC relaxation. When external metadata already confirms the
# vendor (so the mark is present with high prior), a faint or slightly re-rendered
# glyph that scores just below the standard NCC gate is still trusted. The relaxed
# gate is ``detect_ncc_threshold * provenance_ncc_factor``; the coverage gate still
# applies on top.
#
# This used to be ONE shared 0.7 for every text mark. Measured 2026-07-18 on the
# `auto` path (the default -- no flag, driven by TC260 metadata), it turned out to
# mean two completely different things per mark. Blind hand-label of the ADDITIONS
# (accepted with provenance, rejected without) over a labeled TC260 evaluation set,
# two-sided control (labeler sensitivity 100%/96%, specificity 100%/100%):
#
#   mark     band            precision   95% CI     n
#   doubao   whole arm             76%    61-87%    42
#            [0.280,0.340)         58%    36-77%    19
#            [0.340,0.400)         91%    73-98%    23
#   jimeng   whole arm             17%    10-27%    82
#            [0.315,0.383)         12%     6-22%    68
#            [0.383,0.450)         43%    21-67%    14
#
# Doubao stays at 0.70: both its bands return more true marks than false fills, so
# tightening would cost 11 genuine recoveries to prevent 8 false ones.
#
# Jimeng moves to 0.85. Its relaxed detector does not key on the "★ 即梦AI" wordmark
# any more -- it keys on "some text in the bottom-right corner": of 68 false
# additions, 33 were DOUBAO marks and 17 were other vendors' AI labels (千问, 百度,
# 星绘, 抖音). 45 of those 68 fill a corner nothing else would touch (the other 23
# are harmless -- doubao fires strictly there and fills the same box anyway). At
# 0.85 the [0.315,0.383) band is dropped: 8 genuine recoveries lost, 60 false fills
# prevented (7.5:1). A false fill is the worse error -- it destroys pixels AND makes
# the caller report a removal that did not happen, while a miss leaves the image
# untouched.
#
# NOTE: 0.85 is a patch on a detector problem, not a fix. Jimeng's silhouette is not
# discriminative against Doubao's (same corner, same script, both ByteDance), and no
# threshold repairs that -- it needs a better detection silhouette.
_DEFAULT_PROVENANCE_NCC_FACTOR = 0.7


@dataclass(frozen=True)
class TextMarkConfig:
    """All per-mark tuning for a text-mark detector/localizer."""

    name: str  # short label for log lines (e.g. "Doubao")
    asset_name: str  # bundled alpha PNG under assets/ (e.g. "doubao_alpha.png")
    corner: Literal[
        "br", "bl", "tl", "tr", "bc"
    ]  # br (Doubao/Jimeng), bl (Samsung), tl (RunningHub), tr (Microsoft), bc (LiblibAI)
    margin_floor: int  # min margin in px for locate (4 for br marks, 2 for Samsung)
    # locate geometry (fraction of scale_base -- see scale_base())
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
    # alpha-map glyph geometry (fraction of scale_base) emitted by
    # scripts/visible_alpha_solve.py, sizing the detection silhouette for
    # template_match_score
    alpha_width_frac: float
    alpha_height_frac: float
    min_gw: int  # minimum glyph width for the template match (8 br, 16 Samsung)
    # Asset names of RIVAL marks that occupy the same corner and can therefore be
    # scored against the same glyph blob. Detection becomes COMPETITIVE: this mark's
    # template must beat every rival's by `rival_margin`. See _rival_margin_ok.
    # Detection front-end. "binary" thresholds the top-hat into a glyph blob and
    # correlates a binary silhouette against it; "tophat" correlates the CONTINUOUS
    # top-hat response against a soft template and never binarizes. See
    # TextMarkEngine.tophat_response for the measurement that motivated the split.
    # "gray" correlates the silhouette against the raw GRAYSCALE of the locate box:
    # for a faint mid-gray mark (RunningHub) the top-hat's background-subtraction and
    # max-normalization suppress the response to clean-arm levels (positives 0.16-0.23
    # vs clean p99 0.31), while raw gray NCC separates (positives 0.38-0.54 vs clean
    # p99 0.264 / max 0.304, measured 2026-07-22). Contrast-DEPENDENT, unlike tophat.
    # "contrast" correlates against the ABSOLUTE local-luma residual. It is for a mark
    # whose renderer switches between light-on-dark and dark-on-light while preserving
    # one silhouette (Tencent Yuanbao); a one-polarity white top-hat misses the latter.
    detect_frontend: Literal["binary", "tophat", "gray", "contrast"] = "binary"
    # Gaussian sigma applied to the template in the "tophat" front-end (0 = none).
    template_blur: float = 0.0
    # Which image dimension the mark's size and margins scale with. VENDOR-SPECIFIC,
    # measured, not assumed -- see TextMarkEngine.scale_base. "short" = min(h, w), "width" = w.
    scale_basis: Literal["short", "width", "long"] = "width"
    # Scale rungs ``_ladder_best`` sweeps (the detection comb). PER-MARK: a vendor
    # whose stamp sizes do not land on the shared 3-rung comb carries its own ladder
    # (measured for 千问, whose marks sit in two size modes ~1.6x apart -- one fraction
    # on 3 rungs covers only ~75% of them). Densifying the SHARED ladder for everyone
    # was measured and rejected (false fire 2.52% -> 3.05%; see docs/verification-plan.md
    # B2), so the default stays the shipped 3 rungs and a deviation must be calibrated
    # per mark on real positives, never ported.
    ladder: tuple[float, ...] = (0.8, 1.0, 1.25)
    rivals: tuple[str, ...] = ()
    rival_margin: float = 0.10
    # Multiplier applied to detect_ncc_threshold when provenance confirms the vendor.
    # Per-mark, NOT shared: see _DEFAULT_PROVENANCE_NCC_FACTOR for the measured
    # precision that forced the split. Last field so it can carry a default.
    provenance_ncc_factor: float = _DEFAULT_PROVENANCE_NCC_FACTOR


@dataclass
class TextMarkLocation:
    """Located watermark box, in absolute pixel coordinates."""

    x: int
    y: int
    w: int
    h: int

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
    # ROI-local (x0, y0, x1, y1) of the template's best match, in the LOCATED BOX's
    # coordinates. Continuous front ends get it from the ladder sweep; ``binary`` gets
    # it from its single nominal-size match. None whenever no template matched.
    # ``footprint_mask`` bounds the fill with it, so carrying it here is what stops the
    # mask path re-running a search the detector already ran.
    match_box: tuple[int, int, int, int] | None = None
    # The trust level this detection was taken at, mirroring detect()'s ``provenance``.
    # ``footprint_mask`` reuses a threaded detection only when it matches the STRICT
    # level its own re-detect would have used -- see TextMarkEngine._strict_detection.
    provenance: bool = False


def best_detection(*detections: TextMarkDetection) -> TextMarkDetection:
    """Prefer an accepted variant, then retain the strongest rejected score."""
    return max(detections, key=lambda detection: (detection.detected, detection.confidence))


@dataclass(frozen=True)
class TextMarkScan:
    """The trust-level-BLIND half of text-mark detection, reusable across both levels.

    ``loc is None`` means detection stopped before any scan (empty or too-small image).
    ``score is None`` means the binary front-end fell below its coverage gate, which is
    a verdict of "not detected, confidence 0.0" without consulting the rival margin.
    """

    loc: TextMarkLocation | None
    box: NDArray[Any] | None  # box-sized binary glyph mask
    base: int  # scale_base(image)
    frame: tuple[int, int] = (0, 0)  # (h, w) of the scanned image
    coverage: float = 0.0
    score: float | None = None
    match_box: tuple[int, int, int, int] | None = None


# Alpha / silhouette templates, cached per asset name. This shared cache lets every
# text-mark engine reuse the loader without re-reading an asset. Only SUCCESSFUL loads
# are cached, so a missing asset is retried.
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


_RIVAL_MODULES = {
    "doubao_alpha.png": "remove_ai_watermarks.doubao_engine",
    "jimeng_alpha.png": "remove_ai_watermarks.jimeng_engine",
    "qwen_alpha.png": "remove_ai_watermarks.qwen_engine",
    "samsung_alpha.png": "remove_ai_watermarks.samsung_engine",
}


def _rival_config(asset_name: str) -> TextMarkConfig | None:
    """The rival mark's own config, for scoring its template on a shared blob.

    Looked up LAZILY by asset name: a rival's template geometry
    (``alpha_*_frac`` / ``min_gw``) is its own, and scoring it with this mark's
    geometry would compare a correctly-sized template against a mis-sized one and
    hand the margin a free win. Lazy because the engine modules import this one.
    """
    mod_path = _RIVAL_MODULES.get(asset_name)
    if mod_path is None:
        return None
    from importlib import import_module

    try:
        return import_module(mod_path)._CONFIG
    except Exception:  # a missing/renamed engine must not break detection
        logger.debug("rival config %s unavailable; skipping its margin check.", asset_name)
        return None


def _template_match_best(
    box_mask: NDArray[Any], scale_base: int, config: TextMarkConfig
) -> tuple[float, tuple[int, int, int, int] | None]:
    """Best nominal-size template score and its inclusive ROI-local box.

    ``TM_CCOEFF_NORMED`` keys on glyph SHAPE, not coverage, so a dense textured
    corner does not score highly -- only the actual glyph shape does.

    ``scale_base`` is the mark's own scaling dimension (:meth:`TextMarkEngine.scale_base`),
    not always the width: sizing the template on the wrong basis stretches it by the
    aspect ratio on landscape inputs and the correlation collapses.
    """
    sil = glyph_silhouette(config.asset_name)
    if sil is None or box_mask.size == 0:
        return (0.0, None)
    gw = min(box_mask.shape[1] - 1, max(config.min_gw, int(config.alpha_width_frac * scale_base)))
    gh = min(box_mask.shape[0] - 1, max(4, int(config.alpha_height_frac * scale_base)))
    if gw < config.min_gw or gh < 4:
        return (0.0, None)
    template = cv2.resize(sil, (gw, gh), interpolation=cv2.INTER_NEAREST)
    result = cv2.matchTemplate(box_mask, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, top_left = cv2.minMaxLoc(result)
    x, y = int(top_left[0]), int(top_left[1])
    return (float(score), (x, y, x + gw - 1, y + gh - 1))


def template_match_score(box_mask: NDArray[Any], scale_base: int, config: TextMarkConfig) -> float:
    """Zero-mean normalized correlation of the expected glyph against ``box_mask``."""
    return _template_match_best(box_mask, scale_base, config)[0]


class TextMarkEngine:
    """Visible text-mark detector/localizer (locate -> mask -> detect; mask feeds the fill)."""

    def __init__(self, config: TextMarkConfig) -> None:
        self.config = config

    # ── Templates (delegate to the asset-keyed module cache) ────────────

    def _alpha_template(self) -> NDArray[Any] | None:
        return load_alpha_template(self.config.asset_name)

    def _glyph_silhouette(self) -> NDArray[Any] | None:
        return glyph_silhouette(self.config.asset_name)

    def _template_match_score(self, box_mask: NDArray[Any], scale_base: int) -> float:
        return template_match_score(box_mask, scale_base, self.config)

    def _rival_margin_ok(self, score: float, box_mask: NDArray[Any], scale_base: int) -> bool:
        """Whether this mark's template beats every same-corner RIVAL's on the SAME blob.

        Detection was purely ABSOLUTE -- each engine scored its own template and
        compared against its own threshold, so nothing ever asked the discriminative
        question "does this blob look more like the neighbour's mark than like mine?".
        Two marks sharing a corner and a script (Doubao "豆包AI生成" and Jimeng
        "★ 即梦AI", both bottom-right, both near-white CJK) survive binarization into
        very similar blobs, so an absolute gate cannot separate them -- and under the
        provenance relaxation it stopped trying because many Jimeng false additions
        were actually Doubao marks.

        Measured separability on hand-labeled examples, scoring BOTH templates
        against the same glyph blob:

            feature                       separability   (0.5 = useless, 1.0 = perfect)
            absolute ncc_jimeng                   0.96
            ncc_jimeng MINUS ncc_doubao           0.99

        A 0.10 margin separated Jimeng wordmarks from the rival and clean examples
        without reducing recall, unlike raising the absolute threshold.

        Marks with no same-corner rival declare `rivals=()` and are unaffected.
        """
        c = self.config
        if not c.rivals:
            return True
        for rival_asset in c.rivals:
            rival = _rival_config(rival_asset)
            if rival is None:
                continue
            if score - template_match_score(box_mask, scale_base, rival) < c.rival_margin:
                logger.debug("%s detect: loses the %s rival margin; rejecting.", c.name, rival_asset)
                return False
        return True

    # ── Locate ──────────────────────────────────────────────────────────

    def _roi_fields(
        self, image: NDArray[Any], loc: TextMarkLocation
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        """``(luma, saturation, local_background)`` for the located box, all float32.

        The ROI is normalized to 3-channel BGR first (grayscale / BGRA would break
        ``axis=2``).

        Local background model: a strong Gaussian blur (sigma ~ box height); the white
        top-hat (``luma - local_bg``) lights up bright thin strokes regardless of the
        absolute background level.

        The callers each keep their own ``bh < 16 or bw < 16`` guard: they return three
        different sentinels for a degenerate ROI, so the check cannot move in here.
        """
        x, y, bw, bh = loc.bbox
        roi = image_io.to_bgr(image[y : y + bh, x : x + bw]).astype(np.float32)
        luma = roi.mean(axis=2)
        sat = roi.max(axis=2) - roi.min(axis=2)
        sigma = max(4.0, bh * 0.4)  # 0.4 factor and 4.0 floor are calibrated; do not retune
        return luma, sat, cv2.GaussianBlur(luma, (0, 0), sigmaX=sigma, sigmaY=sigma)

    def tophat_response(self, image: NDArray[Any], loc: TextMarkLocation) -> NDArray[Any] | None:
        """The CONTINUOUS white top-hat in the located box -- the glyph signal, unbinarized.

        :meth:`extract_mask` thresholds this same response into a 0/255 glyph blob. That
        is fine for a mark stamped bold and opaque, and destructive for a faint one: a
        thin translucent overlay shatters into specks under the threshold, and no
        template can match a blob that is not there.

        Calibration on hand-verified examples, scoring each mark with its own template:

            front-end   doubao   clean neg   AUC doubao/neg
            binary       0.723       ~0.12             --
            tophat       0.781        0.122          1.00

        The gates that were hard cuts in the binary path (saturation, absolute luma)
        become WEIGHTS here, so a faint stroke contributes in proportion to its strength
        instead of being dropped at a threshold. The response is max-normalized, which
        makes the score contrast-invariant -- the point of the exercise.

        Kept per-mark (``detect_frontend``) rather than switched globally, because a
        front-end change must be measured per mark before it ships.
        """
        return self._residual_response(image, loc, absolute=False)

    def _residual_response(self, image: NDArray[Any], loc: TextMarkLocation, *, absolute: bool) -> NDArray[Any] | None:
        """Max-normalized uint8 local-luma residual in the located box, saturation-weighted.

        ``absolute=False`` keeps only the POSITIVE side -- the white top-hat, for a mark
        always rendered brighter than its background. ``absolute=True`` takes the
        magnitude, for a renderer that switches between light-on-dark and dark-on-light
        while preserving one silhouette; a one-polarity top-hat misses the latter.
        """
        c = self.config
        _x, _y, bw, bh = loc.bbox
        if bh < 16 or bw < 16:
            return None
        luma, sat, local_bg = self._roi_fields(image, loc)
        residual = luma - local_bg
        resp = (np.abs(residual) if absolute else np.clip(residual, 0, None)) * (sat < c.max_saturation)
        peak = float(resp.max())
        if peak <= 1e-6:
            return None
        return (resp / peak * 255).astype(np.uint8)

    def _detect_response(self, image: NDArray[Any], loc: TextMarkLocation) -> NDArray[Any] | None:
        """The uint8 image the ladder sweep correlates the silhouette against, chosen by
        ``TextMarkConfig.detect_frontend``. ``binary`` runs no sweep and never reaches here."""
        frontend = self.config.detect_frontend
        if frontend == "gray":
            x, y, bw, bh = loc.bbox
            if bh < 16 or bw < 16:
                return None
            return cv2.cvtColor(image_io.to_bgr(image[y : y + bh, x : x + bw]), cv2.COLOR_BGR2GRAY)
        if frontend == "contrast":
            return self._residual_response(image, loc, absolute=True)
        if frontend == "tophat":
            return self._residual_response(image, loc, absolute=False)
        raise ValueError(f"{frontend!r} has no ladder response (binary runs no sweep)")

    def _ladder_best(
        self, image: NDArray[Any], loc: TextMarkLocation
    ) -> tuple[float, tuple[int, int, int, int] | None]:
        """Best TM_CCOEFF_NORMED of the mark's silhouette against its front-end response,
        and the ROI-local box (x0, y0, x1, y1) where that best match sits.

        Sweeps the mark's scale ladder: the nominal glyph size is derived from the mark's
        geometry, but a vendor re-rasterization shifts it by a few percent and the
        continuous response is sharp enough that an exact-size template would miss. The
        ladder is per-mark (``TextMarkConfig.ladder``), defaulting to the shipped 3 rungs.

        Detection and the removal mask BOTH read this one method -- the score gates
        detection, the box bounds the fill. Sharing it is deliberate: the standing rule is
        that detection and the mask use the same front-end, and the way that rule was last
        broken was a drift between two separate implementations. One method makes the drift
        impossible instead of merely discouraged -- which is why the three continuous
        front-ends (tophat / contrast / gray) sweep here rather than in a copy each.

        Tie-breaking is load-bearing: the ``0.0`` seed plus the STRICT ``>`` means the
        EARLIEST ladder rung wins a tie, and a sweep whose maximum is not above 0.0
        returns no box at all.
        """
        c = self.config
        resp = self._detect_response(image, loc)
        sil = self._glyph_silhouette()
        if resp is None or sil is None:
            return (0.0, None)
        base = self.scale_base(image)
        best_score = 0.0
        best_box: tuple[int, int, int, int] | None = None
        for scale in c.ladder:
            gw = max(c.min_gw, int(c.alpha_width_frac * base * scale))
            gh = max(4, int(c.alpha_height_frac * base * scale))
            if gw >= resp.shape[1] or gh >= resp.shape[0]:
                continue
            tmpl = cv2.resize(sil, (gw, gh), interpolation=cv2.INTER_AREA)
            if c.detect_frontend == "tophat" and c.template_blur > 0:
                tmpl = cv2.GaussianBlur(
                    tmpl.astype(np.float32), (0, 0), sigmaX=c.template_blur, sigmaY=c.template_blur
                ).astype(np.uint8)
            result = cv2.matchTemplate(resp, tmpl, cv2.TM_CCOEFF_NORMED)
            _, score, _, top_left = cv2.minMaxLoc(result)
            if score > best_score:
                tx, ty = int(top_left[0]), int(top_left[1])
                best_score, best_box = float(score), (tx, ty, tx + gw - 1, ty + gh - 1)
        return (best_score, best_box)

    def scale_base(self, image: NDArray[Any]) -> int:
        """The image dimension this mark's geometry scales with.

        Per-mark, and MEASURED -- a single shared basis is wrong. The tuned fractions
        were all calibrated on PORTRAIT captures, where width and short side coincide,
        so the basis was never exercised until landscape inputs were measured.

        Detector calibration showed that a width-scaled box is inflated by the aspect
        ratio on a wide image and can miss the glyph entirely. A short-side basis
        recovered the affected Doubao landscape cases.

        The same switch broke Jimeng landscape positives because its wordmark tracks
        the WIDTH. Both marks are ByteDance and share a
        corner, and they still scale differently -- so this is a per-mark measurement,
        not a house rule to generalize. Samsung keeps ``width`` because it has not been
        calibrated for a different basis, and an unmeasured change is not an improvement.

        China's GB 45438-2025 clause 5.2(e) mandates glyph height >= 5% of "the
        shortest side" for CN marks, which is why a short-side basis is the natural
        prior -- but Jimeng's measured behavior overrides the prior, and measurement
        wins over the standard's wording.

        "long" (max of the two sides) is the Microsoft badge's measured basis: the
        pill tracks the RENDER dimension, so on a 1024x1536 portrait it scales with
        the 1536 (a width basis undersized the template by the aspect ratio and the
        portrait carriers fell to 0.15-0.32 NCC; measured 2026-08-27).
        """
        if self.config.scale_basis == "short":
            return min(image.shape[:2])
        if self.config.scale_basis == "long":
            return max(image.shape[:2])
        return image.shape[1]

    def locate(self, image: NDArray[Any]) -> TextMarkLocation:
        """Anchor the watermark box in the configured corner, scaled by ``scale_basis``.

        Every fraction is taken against ``scale_base(image)`` -- see
        :data:`TextMarkConfig.scale_basis`, which is per-mark because the vendors
        genuinely differ.
        """
        c = self.config
        h, w = image.shape[:2]
        base = self.scale_base(image)
        wm_w = max(40, int(base * c.width_frac))
        wm_h = max(16, int(base * c.height_frac))
        margin_x = max(c.margin_floor, int(base * c.margin_x_frac))
        margin_b = max(c.margin_floor, int(base * c.margin_bottom_frac))
        if c.corner == "br" or c.corner == "tr":
            x = max(0, w - margin_x - wm_w)
        elif c.corner == "bc":  # bottom-center: horizontally centered, margin_x unused
            x = max(0, (w - wm_w) // 2)
        else:
            x = min(margin_x, max(0, w - wm_w))
        # "tl"/"tr" anchor at the top instead: margin_bottom_frac is then the TOP margin.
        y = min(margin_b, max(0, h - wm_h)) if c.corner in ("tl", "tr") else max(0, h - margin_b - wm_h)
        wm_w = min(wm_w, w - x)
        wm_h = min(wm_h, h - y)
        return TextMarkLocation(x=x, y=y, w=wm_w, h=wm_h)

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

    def detect(self, image: NDArray[Any], *, provenance: bool = False) -> TextMarkDetection:
        """Detect the mark by matching the alpha-template glyph silhouette against
        the corner candidate (``TM_CCOEFF_NORMED``); keys on glyph SHAPE, not coverage.

        ``provenance`` signals that external metadata already confirms this vendor
        (China-AIGC / byteimg for Doubao/Jimeng, ``samsung_genai`` for Samsung); the
        NCC gate exists to keep a corner texture on an UNRELATED image from matching
        the glyph silhouette, so when provenance confirms the vendor it is relaxed by
        the mark's own ``provenance_ncc_factor`` to recover a faint or slightly
        re-rendered mark (per-mark, not shared -- see _DEFAULT_PROVENANCE_NCC_FACTOR).
        """
        scan = self._scan(image)
        return self._verdict(scan, provenance=provenance)

    def detect_both(self, image: NDArray[Any] | None) -> tuple[TextMarkDetection, TextMarkDetection]:
        """``(strict, relaxed)`` from ONE scan of the image.

        ``provenance`` scales the acceptance THRESHOLD and nothing else -- the locate
        box, the glyph mask, the coverage and the front-end ladder score are computed
        identically at either trust level. Two ``detect`` calls therefore ran the same
        expensive sweep twice to reach two verdicts, which is what the arbiter's
        perception pass did for every mark on every image.

        Returns two DISTINCT objects: subclasses demote a verdict by mutating it.
        """
        scan = self._scan(image)
        return self._verdict(scan, provenance=False), self._verdict(scan, provenance=True)

    def _scan(self, image: NDArray[Any] | None) -> TextMarkScan:
        """Everything in detection that does not depend on the trust level.

        Per-CALL only, never memoized on ``self``: ``remove_auto_marks`` re-invokes each
        engine on a progressively cleaned frame inside one process, so a cached scan
        would answer for the wrong pixels.
        """
        c = self.config
        if image is None or image.size == 0:
            return TextMarkScan(None, None, 0)
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
            return TextMarkScan(None, None, 0)
        loc = self.locate(image)
        box = self.extract_mask(image, loc)  # box-sized mask (== old full-frame cropped to bbox)
        _x, _y, bw, bh = loc.bbox
        coverage = float((box > 0).sum()) / float(max(1, bw * bh))
        base = self.scale_base(image)
        match_box: tuple[int, int, int, int] | None = None
        if c.detect_frontend == "binary":
            # The coverage gate is a blob-AREA heuristic, so it applies only to the
            # front-end that binarizes; the continuous ones never build a blob. Below
            # the gate the detection stays at confidence 0.0 and the rival margin is
            # never consulted, so the score stays None here.
            if coverage >= c.detect_min_coverage:
                score, match_box = _template_match_best(box, base, c)
            else:
                score = None
        else:
            score, match_box = self._ladder_best(image, loc)
        return TextMarkScan(loc, box, base, frame=image.shape[:2], coverage=coverage, score=score, match_box=match_box)

    def _verdict(self, scan: TextMarkScan, *, provenance: bool) -> TextMarkDetection:
        """Apply the trust-level-dependent tail to a scan, as a fresh result object."""
        c = self.config
        det = TextMarkDetection(provenance=provenance)
        if scan.loc is None or scan.box is None:
            return det
        det.region = scan.loc.bbox
        det.coverage = scan.coverage
        det.match_box = scan.match_box
        if scan.score is None:  # binary front-end below the coverage gate
            return det
        threshold = c.detect_ncc_threshold * (c.provenance_ncc_factor if provenance else 1.0)
        det.confidence = scan.score
        # Short-circuit is load-bearing: _rival_margin_ok scores every rival template
        # and logs its own rejection line, so it must stay unevaluated below threshold.
        det.detected = scan.score >= threshold and self._rival_margin_ok(scan.score, scan.box, scan.base)
        logger.debug(
            "%s detect (%s): coverage=%.3f ncc=%.2f thr=%.2f detected=%s",
            c.name,
            c.detect_frontend,
            scan.coverage,
            scan.score,
            threshold,
            det.detected,
        )
        return self._post_gate(det, scan)

    def _post_gate(self, det: TextMarkDetection, scan: TextMarkScan) -> TextMarkDetection:
        """Per-mark demotion applied after the shared threshold, for both trust levels.

        OVERRIDABLE. It lives here rather than in a ``detect`` override so a mark's gate
        cannot be silently skipped by the single-pass ``detect_both`` path -- which is
        exactly what happened while the anchor demotions were ``detect`` overrides.
        """
        return det

    # ── Inpaint footprint (for the inpaint-fallback removal path) ────────

    # Minimum glyph pixels for a template-free footprint. Below this the corner has
    # no real wordmark (a few top-hat specks), so without ``force`` there is nothing
    # to mask. A real strip covers hundreds of pixels.
    _MIN_GLYPH_PIXELS = 20

    def _strict_detection(self, image: NDArray[Any], detection: TextMarkDetection | None) -> TextMarkDetection:
        """The STRICT detection the footprint is bounded by.

        A threaded detection is reused only when it was taken at the same strict level
        this method would have used itself. A provenance-RELAXED detection is NOT
        reused: a strict re-detect can demote a mark the relaxed gate accepted, which
        for a continuous front-end means no mask at all. That is a MEASURED difference,
        not a refactor, so the strict semantics stay.

        Reuse is safe against ``remove_auto_marks`` chaining marks on a progressively
        cleaned frame: the registry re-detects on that same cleaned array before
        threading (``KnownMark.localize``), so a threaded detection is never stale.
        """
        if detection is not None and not detection.provenance:
            return detection
        return self.detect(image)  # polymorphic: a subclass gate must still apply

    def _geometry_rect(self, loc: TextMarkLocation, frame: tuple[int, int]) -> tuple[int, int, int, int]:
        """The whole locate box, clamped to the frame -- the ``force`` footprint."""
        bx, by, bw, bh = loc.bbox
        h, w = frame
        return (bx, by, min(w, bx + bw), min(h, by + bh))

    def _extend_match_box(
        self, box: tuple[int, int, int, int], loc: TextMarkLocation, frame: tuple[int, int]
    ) -> tuple[int, int, int, int]:
        """Grow an ROI-local box into the absolute fill rectangle by a symmetric pad.

        OVERRIDABLE, and the override contract is specifically the DETECTOR'S MATCH BOX:
        a mark whose removable footprint reaches beyond what the NCC localizes -- Baidu's
        flat white tag right of the text run, LiblibAI's triangle logo left of the
        wordmark -- supplies its own extension here and inherits the rest of the
        footprint path. The blob-bbox branch never routes through an override.
        """
        gx0, gy0, gx1, gy1 = box
        bx, by, _bw, bh = loc.bbox
        h, w = frame
        pad = max(4, int(0.10 * bh))
        return (
            max(0, bx + gx0 - pad),
            max(0, by + gy0 - pad),
            min(w, bx + gx1 + 1 + pad),
            min(h, by + gy1 + 1 + pad),
        )

    def _footprint_rect(
        self,
        image: NDArray[Any],
        loc: TextMarkLocation,
        *,
        force: bool,
        detection: TextMarkDetection | None,
    ) -> tuple[int, int, int, int] | None:
        """Default footprint policy: the binary glyph blob's bbox, else the detector's
        own match box for the front-ends that under-segment, else the geometry box
        under ``force``.

        The gray front-end exists for marks the top-hat under-segments, so the binary
        blob is NOT authoritative there: trusting it first bounded the fill by a PARTIAL
        blob (the faint head glyphs dropped out) and left the leftmost "Runni" of
        "RunningHub AI生成" unremoved (2026-07-22).

        A dark-on-light Yuanbao mark has no WHITE top-hat blob at all, so the contrast
        front-end is bounded by the polarity-independent detector's match box too.

        A tophat mark found only by the CONTINUOUS front-end has no binary glyph blob to
        bound, so the mask came back empty and removal was a silent no-op while
        ``identify`` still reported the mark. Use the DETECTOR'S OWN best-match box: the
        correlation already located the mark at a position and scale, and thresholding
        the response was a strictly worse proxy for that. An earlier fix thresholded the
        max-normalized uint8 response at 0.5 -- which selects every non-zero pixel, not
        "half the peak" as its comment claimed -- and filled ~120% of the corner box on
        textured frames (measured: whole corner vs 58.7% for the match box, both
        detector-clean). Gated on an actual detection: on a clean corner the box would
        be spurious.
        """
        ys, xs = np.where(self.extract_mask(image, loc) > 0)
        blob = (
            (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if xs.size >= self._MIN_GLYPH_PIXELS else None
        )
        frontend = self.config.detect_frontend
        if frontend in ("gray", "contrast"):
            det = self._strict_detection(image, detection)
            box = det.match_box if det.detected else blob
        elif blob is not None:
            box = blob
        elif frontend == "tophat":
            det = self._strict_detection(image, detection)
            box = det.match_box if det.detected else None
        else:
            box = None
        if box is not None:
            return self._extend_match_box(box, loc, image.shape[:2])
        return self._geometry_rect(loc, image.shape[:2]) if force else None

    def _match_box_rect(
        self,
        image: NDArray[Any],
        loc: TextMarkLocation,
        *,
        force: bool,
        detection: TextMarkDetection | None,
    ) -> tuple[int, int, int, int] | None:
        """Footprint policy for a mark whose fill must be bounded by the DETECTOR's match
        box and never by the binary glyph blob.

        Baidu's white tag has a flat interior a top-hat cannot answer, and LiblibAI's
        blob bleeds up into background structure; in both cases the blob bbox is
        measurably wrong and the NCC match box is right. ``force`` takes priority here,
        unlike the default policy: a ``--no-detect`` caller named the mark, so the whole
        geometry box is the honest footprint.
        """
        if force:
            return self._geometry_rect(loc, image.shape[:2])
        det = self._strict_detection(image, detection)
        if not det.detected or det.match_box is None:
            return None
        return self._extend_match_box(det.match_box, loc, image.shape[:2])

    def _aligned_alpha_mask(
        self,
        image: NDArray[Any],
        detection: TextMarkDetection,
        alpha: NDArray[Any],
        *,
        alpha_floor: float,
        dilate: int,
    ) -> NDArray[Any] | None:
        """Place an alpha footprint on the detector's winning template box.

        Continuous front ends locate faint glyphs that the thresholded pixel response
        can only partly recover. The winning match already fixes the template's position
        and scale, so engines with a trustworthy alpha asset can reuse that alignment
        for their removal mask instead of running another search.
        """
        if not detection.detected or detection.match_box is None:
            return None
        source = image_io.to_bgr(image)
        loc = self.locate(source)
        x0, y0, x1, y1 = detection.match_box  # inclusive ROI-local coordinates
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(loc.w - 1, x1), min(loc.h - 1, y1)
        if x1 < x0 or y1 < y0:
            return None

        glyph_w, glyph_h = x1 - x0 + 1, y1 - y0 + 1
        aligned = cv2.resize(alpha, (glyph_w, glyph_h), interpolation=cv2.INTER_LINEAR)
        local_mask = np.zeros((loc.h, loc.w), np.uint8)
        local_mask[y0 : y0 + glyph_h, x0 : x0 + glyph_w] = (aligned > alpha_floor).astype(np.uint8) * 255
        if dilate > 0:
            kernel = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
            local_mask = cv2.dilate(local_mask, kernel)

        mask = np.zeros(source.shape[:2], np.uint8)
        mask[loc.y : loc.y + loc.h, loc.x : loc.x + loc.w] = local_mask
        return mask

    def footprint_mask(
        self,
        image: NDArray[Any] | None,
        *,
        force: bool = False,
        dilate: int | None = None,
        detection: TextMarkDetection | None = None,
    ) -> NDArray[Any] | None:
        """Full-frame uint8 mask (255 = mark) of the mark footprint, for the shared
        fill removal path (cv2 / MI-GAN / LaMa), or None if no glyph is found.

        Template-FREE: localize the glyph blob with the top-hat :meth:`extract_mask`,
        take its bounding box in the corner, and fill that box solid (plus a small
        margin + dilation). Filling the enclosing rectangle -- not the sparse glyph
        strokes -- is what makes it robust: the top-hat under-segments individual
        strokes (which used to leave a "三包"-style residual ghost when the strokes
        themselves were the mask), but the inpaint reconstructs the whole wordmark
        rectangle from its surroundings, so a stroke missed by the top-hat is still
        covered. This drops the fixed alpha-template dependency, so a re-rendered or
        differently-rendered mark can still be masked by engines that keep this default
        rectangular policy. Engines with measured alpha footprints override this method.

        With ``force`` and no glyph found, falls back to the whole geometry box (the
        ``--no-detect`` path). The caller gates on detection.

        ``detection`` is the caller's already-computed detection, threaded in so the
        footprint does not re-run a sweep the detector already ran. See
        :meth:`_strict_detection` for when it is reused.
        """
        if image is None or image.size == 0:
            return None  # guard before to_bgr (cvtColor raises on an empty Mat); mirror detect()
        image = image_io.to_bgr(image)
        h, w = image.shape[:2]
        if h < 32 or w < 64:
            return None
        loc = self.locate(image)
        rect = self._footprint_rect(image, loc, force=force, detection=detection)
        if rect is None:
            return None
        rx1, ry1, rx2, ry2 = rect
        if rx1 >= rx2 or ry1 >= ry2:
            return None
        # Rectangular footprint + dilation is exactly region_eraser.boxes_to_mask (the
        # same primitive the shared fill uses); reuse it instead of re-inlining the
        # zeros/fill/MORPH_ELLIPSE-dilate here.
        from remove_ai_watermarks import region_eraser

        d = dilate if dilate is not None else max(3, int(0.02 * loc.w))
        return region_eraser.boxes_to_mask((h, w), [(rx1, ry1, rx2 - rx1, ry2 - ry1)], dilate=d)
