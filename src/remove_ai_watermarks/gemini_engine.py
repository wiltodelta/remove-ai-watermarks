"""Gemini visible-sparkle detector and localizer (cv2/numpy, no GPU).

Locates the Google Gemini / Nano Banana sparkle so the shared fill (region_eraser)
can inpaint it. Detection is a multi-scale NCC search against the captured sparkle
alpha template (ported from GeminiWatermarkTool's Snap Engine; original author
Allen Kuo (allenk), https://github.com/allenk/GeminiWatermarkTool), scored by a
spatial + gradient + variance fusion with a false-positive gate. ``footprint_mask``
returns the sparkle footprint (captured alpha thresholded low to include the halo,
then dilated) as a full-frame mask for the fill.

The captured alpha maps are background captures of the sparkle on pure-black
backgrounds (48x48 for small images, 96x96 for large). NB: they are used here only
to DETECT and to shape the removal mask -- the old reverse-alpha pixel recovery
(``original = (watermarked - a*logo)/(1-a)``) is gone; removal is localize -> fill.
"""

# cv2/numpy boundary: cv2 and numpy ship no usable type info for the array ops
# below, so strict pyright cannot know their element types. Relax the unknown-type
# rules for this file only; the public signatures are still annotated with NDArray[Any].
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportAttributeAccessIssue=false, reportPrivateImportUsage=false, reportPrivateUsage=false, reportInvalidTypeForm=false, reportConstantRedefinition=false, reportUnnecessaryComparison=false
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from remove_ai_watermarks import image_io

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class WatermarkSize(Enum):
    """Watermark size mode based on image dimensions."""

    SMALL = "small"  # 48x48, for images <= 1024x1024
    LARGE = "large"  # 96x96, for images > 1024x1024


@dataclass
class DetectionResult:
    """Result of watermark detection."""

    detected: bool = False
    confidence: float = 0.0
    region: tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h
    size: WatermarkSize = WatermarkSize.SMALL

    # stage scores
    spatial_score: float = 0.0
    gradient_score: float = 0.0
    variance_score: float = 0.0


@dataclass
class WatermarkPosition:
    """Watermark position configuration."""

    margin_right: int
    margin_bottom: int
    logo_size: int

    def get_position(self, image_width: int, image_height: int) -> tuple[int, int]:
        """Get top-left position for a given image size."""
        x = image_width - self.margin_right - self.logo_size
        y = image_height - self.margin_bottom - self.logo_size
        return (x, y)


def get_watermark_config(width: int, height: int) -> WatermarkPosition:
    """Get the appropriate watermark configuration based on image size.

    Rules discovered from Gemini:
      - W > 1024 AND H > 1024: 96x96 logo at (W-64-96, H-64-96)
      - Otherwise:              48x48 logo at (W-32-48, H-32-48)
    """
    if width > 1024 and height > 1024:
        return WatermarkPosition(margin_right=64, margin_bottom=64, logo_size=96)
    return WatermarkPosition(margin_right=32, margin_bottom=32, logo_size=48)


def get_watermark_size(width: int, height: int) -> WatermarkSize:
    """Determine watermark size mode from image dimensions."""
    if width > 1024 and height > 1024:
        return WatermarkSize.LARGE
    return WatermarkSize.SMALL


def _calculate_alpha_map(bg_capture: NDArray[Any]) -> NDArray[Any]:
    """Calculate alpha map from a background capture.

    The alpha map represents how much the watermark affects each pixel.
    alpha = max(R, G, B) / 255.0
    """
    if len(bg_capture.shape) == 2:
        gray = bg_capture.astype(np.float32)
    elif bg_capture.shape[2] >= 3:
        # Use max of channels for brightness
        gray = np.max(bg_capture[:, :, :3], axis=2).astype(np.float32)
    else:
        gray = bg_capture[:, :, 0].astype(np.float32)

    return gray / 255.0


def _load_embedded_asset(name: str) -> NDArray[Any]:
    """Load an embedded PNG asset and decode it with OpenCV."""
    asset_path = Path(__file__).parent / "assets" / name
    if not asset_path.exists():
        raise FileNotFoundError(f"Embedded asset not found: {asset_path}")

    data = asset_path.read_bytes()
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to decode embedded asset: {name}")
    return img


class GeminiEngine:
    """Detects and localizes the visible Gemini sparkle for the shared fill removal.

    The multi-scale NCC detection is a Python port of the GeminiWatermarkTool C++
    Snap Engine; ``footprint_mask`` turns a detection into a removal mask.
    """

    # Body pixels at >= this fraction of the peak captured alpha define the sparkle
    # "core", sampled by the detection FP-gate's core-vs-ring brightness margin
    # (:meth:`_core_and_bg`).
    _CORE_ALPHA_FRAC = 0.8

    # Sparkle false-positive gate. A real Gemini sparkle is a bright WHITE overlay,
    # so its core sits above the local background; a shape-only NCC match on ornate
    # or flat content (text, banners, hatching) can score >0.5 without that lift.
    # Demote a detection that is BOTH low-confidence AND low core-ring brightness
    # margin -- the joint signature of a content false positive (verified on the
    # spaces corpus: of 16 demoted, 13 carried no AI metadata and the 3 AI-meta ones
    # were visually FPs / a near-invisible white-on-white sparkle whose AI verdict is
    # held by metadata anyway). Real sparkles escape via EITHER high confidence
    # (white-bg sparkles score >=0.79 despite a low margin) OR high margin (dark/mid
    # backgrounds, incl. the #36 faint-corner case, lift well clear), so both must
    # fail to demote.
    _SPARKLE_FP_CONF = 0.65
    _SPARKLE_FP_MARGIN = 5.0
    # Bright-background content false positives (2026-06-26 landing-page FPs: a snow+sky
    # photo and a white-background product render both scored ~0.51). The margin gate
    # above cannot catch them -- a bright background gives the "core" a HIGH core-ring
    # margin (it is genuinely brighter than its surroundings), so the brightness check
    # reads it as a real overlay. The discriminating signature is the GRADIENT NCC: a
    # real white sparkle is a crisp star silhouette (grad ~0.97-1.0 on the synthetic
    # composites, ~0.96 on the real #36 corner sparkle), while a smooth luminance blob
    # that shape-NCC-matches the rough outline has low gradient fidelity (the two FPs
    # measured 0.105 and 0.463). So ALSO demote a low-confidence match whose gradient
    # NCC is below this floor, regardless of margin -- 0.55 sits well above the worst FP
    # (0.463) and far below every real sparkle (>=0.8). This only ADDS demotions on
    # bright backgrounds (a real bright-bg sparkle keeps grad ~0.97), so it cannot
    # regress a dark/mid sparkle (already kept by margin) or a white-bg one (kept by
    # confidence >= 0.65, above the gate).
    _SPARKLE_FP_GRAD = 0.55

    # White-core rescue for the gate above. A real but FAINT sparkle -- a soft white
    # star on a bright/textured background -- has a high core-ring margin but low
    # gradient fidelity, the SAME signature the grad gate uses to demote the smooth
    # colored-corner FP, so faint real sparkles get demoted with it. The separator the
    # grad gate discards is the CORE COLOR: a real Gemini sparkle core is near-WHITE
    # (low saturation), while a clean bright corner that shape-matches (sky, sun, a warm
    # light) is COLORED. So do NOT demote a low-grad match that already clears the trust
    # confidence (_SPARKLE_KEEP_CONF -- the registry's 0.5 sparkle gate plus a small
    # margin so the ~0.51 bright-background FPs the grad gate was added for stay demoted)
    # AND has a bright (margin) near-neutral core (_core_saturation <= _SPARKLE_WHITE_SAT).
    # Corpus-measured on metadata-stripped faint sparkles: recovers ~14/20 the low-grad
    # demotion would drop, at ~0.8% clean false-fire vs the ~0.55% baseline.
    _SPARKLE_KEEP_CONF = 0.52
    _SPARKLE_WHITE_SAT = 0.20

    # Corner promotion (issue #36): the size weight that suppresses tiny-patch
    # false positives also buries a small, near-perfect sparkle when a larger,
    # mediocre match sits elsewhere (e.g. a bright collar in a portrait). A small
    # faint sparkle on a busy background therefore loses the global argmax and the
    # image reads as clean -- the regression osachub reported when the search
    # window widened 256px -> 512px (v0.7.2's tighter window still found it).
    # Remedy: if the bottom-right corner holds a very-high-fidelity raw-NCC match,
    # trust it regardless of size, without reverting the wider window (which is
    # needed for variant margins). The threshold sits midway between the worst
    # real-photo corner match (~0.78 across native + downscaled real photos) and a
    # genuine faint sparkle (~0.93), so it adds true detections without adding
    # false ones; it only ever overrides a lower-fidelity global pick, so it cannot
    # weaken an existing detection.
    _CORNER_PROMOTE_NCC = 0.85
    # Bottom-right corner side for the promotion search, as a fraction of the
    # image's short side, clamped to an absolute pixel band. Relative so the corner
    # stays a true corner at every scale: a fixed 256 px is a genuine corner on a
    # large image but covers ~70% of a small portrait, where a busy real photo can
    # then raw-match the star template at ~0.81 (only 0.04 below the promote gate).
    # Scaling the side down on small images drops that worst case to ~0.69, while
    # the upper clamp stops it ballooning on huge images (more corner area = more
    # random texture to false-match -- a real photo reached ~0.83 at 512 px). The
    # Gemini sparkle sits ~60-160 px from the corner (fixed margins, not
    # proportional), and the [96, 384] band covers that at every measured size.
    _CORNER_PROMOTE_FRAC = 0.20
    _CORNER_PROMOTE_MIN = 96
    _CORNER_PROMOTE_MAX = 384

    # Number of top size-weighted spatial candidates scored by full fusion before one
    # is selected. The single size-weighted argmax can bury a genuine mid-size sparkle
    # under a LARGER, lower-fidelity shape match (the 256->512 search-widening
    # regression: a real corner sparkle at raw ~0.77 lost to a decoy at raw ~0.63).
    # Scoring the top-K by gradient-bearing fusion rescues it. Top-K (NOT the raw-NCC
    # argmax) keeps the tiny-patch suppression intact: a coincidental 16 px match never
    # ranks in the size-weighted top-K, so widening selection cannot add a false
    # positive on non-Gemini content (verified on the doubao/jimeng visible corpora).
    _SELECT_TOPK = 3

    def __init__(self, logo_value: float = 255.0) -> None:
        """Initialize the engine with embedded alpha maps.

        Args:
            logo_value: The logo brightness value (default 255.0 = white).
        """
        self.logo_value = logo_value

        # Load embedded background captures
        bg_small = _load_embedded_asset("gemini_bg_48.png")
        bg_large = _load_embedded_asset("gemini_bg_96.png")

        # Ensure correct sizes
        if bg_small.shape[:2] != (48, 48):
            bg_small = cv2.resize(bg_small, (48, 48), interpolation=cv2.INTER_AREA)
        if bg_large.shape[:2] != (96, 96):
            bg_large = cv2.resize(bg_large, (96, 96), interpolation=cv2.INTER_AREA)

        # Calculate alpha maps
        self._alpha_small = _calculate_alpha_map(bg_small)
        self._alpha_large = _calculate_alpha_map(bg_large)

        logger.debug(
            "Alpha maps loaded: small=%s, large=%s",
            self._alpha_small.shape,
            self._alpha_large.shape,
        )

    def get_alpha_map(self, size: WatermarkSize) -> NDArray[Any]:
        """Get the base alpha map for a specific standard size."""
        if size == WatermarkSize.SMALL:
            return self._alpha_small
        return self._alpha_large

    def get_interpolated_alpha(self, size_px: int) -> NDArray[Any]:
        """Create an interpolated alpha map dynamically scaled from the high-res 96x96 base."""
        source = self._alpha_large
        if size_px == source.shape[1]:
            return source.copy()

        interp = cv2.INTER_LINEAR if size_px > source.shape[1] else cv2.INTER_AREA
        return cv2.resize(source, (size_px, size_px), interpolation=interp)

    # ── Detection ────────────────────────────────────────────────────

    def _scan_scales(self, gray: NDArray[Any]) -> Iterator[tuple[int, float, tuple[int, int]]]:
        """Yield ``(scale, max_ncc, max_loc)`` for the alpha template matched at each scale.

        Shared multi-scale ``TM_CCOEFF_NORMED`` primitive over a normalized [0, 1]
        grayscale region, used by both the size-weighted global search in
        ``detect_watermark`` and the raw-NCC corner pass in ``_corner_promote`` --
        each applies its own scoring/argmax to the yielded values. The 96x96
        ``_alpha_large`` is the high-quality source downscaled per scale; the range
        covers aggressively downscaled to slightly upscaled logos.
        """
        for scale in range(16, 120, 2):
            if scale > gray.shape[0] or scale > gray.shape[1]:
                continue
            tmpl = cv2.resize(self._alpha_large, (scale, scale), interpolation=cv2.INTER_AREA)
            match_res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(match_res)
            yield scale, float(max_val), max_loc

    def detect_watermark(
        self,
        image: NDArray[Any],
        force_size: WatermarkSize | None = None,
        *,
        trust_provenance: bool = False,
    ) -> DetectionResult:
        """Detect Gemini watermark using multi-scale Snap Engine logic (ported from C++ vendor algorithm).

        ``trust_provenance`` signals that external metadata already proves this is a
        Google generation (C2PA issuer "Google"/"Gemini"). The false-positive gate
        exists only to reject content that shape-matches the sparkle on NON-Google
        images (Doubao text, ornate corners); when provenance confirms Google, that
        gate would demote a genuine sparkle the vendor moved/re-rendered (bigger,
        lighter, shifted), so it is skipped. The caller (registry) still applies the
        relaxed provenance trust gate to the returned confidence."""
        result = DetectionResult()

        if image is None or image.size == 0:
            return result

        # Normalize to 3-channel BGR: the multi-scale search tolerates grayscale, but
        # the FP-gate / alpha-gain helpers (_core_and_bg) reduce over axis=2 and would
        # crash on a 2D/BGRA input reaching this public entry point (e.g. via the
        # registry detect adapter or the library API).
        image = image_io.to_bgr(image)
        h, w = image.shape[:2]
        base_size = force_size or get_watermark_size(w, h)
        result.size = base_size

        # Dynamically search bottom-right corner. 512 covers up to 512px from the
        # corner -- enough for known Gemini margin variations (standard: 64+96=160px;
        # observed variants up to ~300px). 256 was too tight and caused misses.
        search_size = int(min(min(w, h), 512))
        sx1 = max(0, w - search_size)
        sy1 = max(0, h - search_size)

        search_region = image[sy1:h, sx1:w]
        if len(search_region.shape) == 3 and search_region.shape[2] >= 3:
            gray_sr = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_sr = search_region.copy()

        gray_sr_f = gray_sr.astype(np.float32) / 255.0

        # Phase 1 & 2: multi-scale spatial NCC search. The size weight (mimicking the
        # C++ vendor weight) overcomes the NCC bias toward tiny patches, but its single
        # argmax can bury a genuine mid-size sparkle under a LARGER, lower-fidelity
        # shape match (the 256->512 search-widening regression). So score the top-K
        # size-weighted candidates by the FULL fusion and keep the highest -- the
        # gradient term separates a true white sparkle from a shape-only decoy. See
        # _SELECT_TOPK for why top-K (not the raw-NCC argmax) preserves tiny-patch
        # suppression and so cannot add a false positive on non-Gemini content.
        scored: list[tuple[float, int, int, int, float]] = []  # (adj, scale, raw, x, y)
        for scale, max_val, max_loc in self._scan_scales(gray_sr_f):
            adj_val = max_val * min(1.0, (scale / 96.0) ** 0.5)
            scored.append((adj_val, scale, max_val, sx1 + max_loc[0], sy1 + max_loc[1]))
        scored.sort(reverse=True)

        # Top-K candidates at distinct locations (NMS: drop a lower-ranked match that
        # overlaps an already-kept one -- the same sparkle matches at adjacent scales).
        candidates: list[tuple[int, int, int, float]] = []
        for _adj, scale, raw, x, y in scored:
            if any(
                abs(x - px) < 0.5 * max(scale, ps) and abs(y - py) < 0.5 * max(scale, ps)
                for ps, px, py, _ in candidates
            ):
                continue
            candidates.append((scale, x, y, raw))
            if len(candidates) >= self._SELECT_TOPK:
                break

        # Corner promotion: a near-perfect small bottom-right sparkle the size weight
        # buries even below the top-K (see _CORNER_PROMOTE_NCC) -- add it as a candidate.
        promoted = self._corner_promote(image, candidates[0][3] if candidates else -1.0)
        if promoted is not None:
            candidates.append(promoted)

        # Select the candidate with the highest full-fusion confidence (pre-FP-gate).
        best_scale, pos_x, pos_y, best_raw_ncc = candidates[0]
        grad_score, var_score, best_fused = 0.0, 0.0, -1.0
        for c_scale, c_x, c_y, c_raw in candidates:
            if c_raw < 0.25:
                c_grad, c_var, c_fused = 0.0, 0.0, max(0.0, c_raw * 0.5)
            else:
                c_grad, c_var = self._grad_var_scores(image, c_scale, c_x, c_y)
                c_fused = c_raw * 0.50 + c_grad * 0.30 + c_var * 0.20
            if c_fused > best_fused:
                best_fused = c_fused
                best_scale, pos_x, pos_y = c_scale, c_x, c_y
                best_raw_ncc, grad_score, var_score = c_raw, c_grad, c_var

        result.region = (pos_x, pos_y, best_scale, best_scale)
        result.spatial_score = float(best_raw_ncc)
        result.gradient_score = float(grad_score)
        result.variance_score = float(var_score)

        if result.spatial_score < 0.25:
            result.confidence = float(max(0.0, result.spatial_score * 0.5))
            return result

        # ── Fusion ───────────────────────────────────────────────────
        # best_fused is the selected candidate's spatial*0.5 + grad*0.3 + var*0.2.
        confidence = best_fused

        # False-positive gate: a low-confidence match that shows NEITHER real-sparkle
        # signature is a content false positive, not a white sparkle overlay. A real
        # sparkle proves itself by a bright core (high core-ring margin, on dark/mid
        # backgrounds) OR a crisp star silhouette (high gradient NCC, on any background
        # incl. bright). Demote when both are weak -- this catches the dark/mid no-core
        # FP (low margin) AND the bright-background smooth-blob FP (high margin but low
        # gradient), which the margin check alone misses. See _SPARKLE_FP_GRAD.
        if confidence < self._SPARKLE_FP_CONF and not trust_provenance:
            alpha = self.get_interpolated_alpha(best_scale)
            pos = (pos_x, pos_y)
            margin = self._core_ring_margin(image, alpha, pos)
            low_margin = margin is not None and margin < self._SPARKLE_FP_MARGIN
            low_grad = grad_score < self._SPARKLE_FP_GRAD
            if low_margin or low_grad:
                # White-core rescue: a real faint sparkle clears the trust confidence,
                # has a bright core (not low_margin), and a near-WHITE core -- unlike the
                # colored-corner FP the low-grad demotion targets. See _SPARKLE_WHITE_SAT.
                core_sat = self._core_saturation(image, alpha, pos)
                white_core = not low_margin and core_sat is not None and core_sat <= self._SPARKLE_WHITE_SAT
                if not (confidence >= self._SPARKLE_KEEP_CONF and white_core):
                    logger.debug(
                        "Sparkle FP gate: conf=%.3f, margin=%s, grad=%.3f, core_sat=%s; demoting.",
                        confidence,
                        f"{margin:.1f}" if margin is not None else "n/a",
                        grad_score,
                        f"{core_sat:.2f}" if core_sat is not None else "n/a",
                    )
                    confidence = min(confidence, 0.30)

        result.confidence = float(max(0.0, min(1.0, confidence)))
        result.detected = result.confidence >= 0.35

        logger.debug(
            "Detection: spatial=%.3f, grad=%.3f, var=%.3f → conf=%.3f (%s)",
            result.spatial_score,
            result.gradient_score,
            var_score,
            result.confidence,
            "DETECTED" if result.detected else "not detected",
        )

        return result

    def _grad_var_scores(
        self,
        image: NDArray[Any],
        scale: int,
        pos_x: int,
        pos_y: int,
    ) -> tuple[float, float]:
        """Return ``(gradient_score, variance_score)`` for a candidate sparkle.

        Factored out of ``detect_watermark`` so each top-K candidate can be scored by
        the full fusion before one is selected. The gradient NCC correlates
        Sobel-magnitude maps (shape fidelity, contrast-robust); the variance score
        rewards a flat overlay region against the row band above it.
        """
        h, w = image.shape[:2]
        x1, y1 = pos_x, pos_y
        x2, y2 = min(w, x1 + scale), min(h, y1 + scale)
        region = image[y1:y2, x1:x2]
        gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 and region.shape[2] >= 3 else region
        gray_f = gray_region.astype(np.float32) / 255.0
        alpha_region = self.get_interpolated_alpha(scale)[: y2 - y1, : x2 - x1]

        # ── Gradient NCC ──
        img_gmag = cv2.magnitude(
            cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
        )
        alpha_gmag = cv2.magnitude(
            cv2.Sobel(alpha_region, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(alpha_region, cv2.CV_32F, 0, 1, ksize=3)
        )
        _, grad_score, _, _ = cv2.minMaxLoc(cv2.matchTemplate(img_gmag, alpha_gmag, cv2.TM_CCOEFF_NORMED))

        # ── Variance ──
        var_score = 0.0
        ref_h = min(y1, scale)
        if ref_h > 8:
            ref_region = image[y1 - ref_h : y1, x1:x2]
            gray_ref = cv2.cvtColor(ref_region, cv2.COLOR_BGR2GRAY) if ref_region.ndim == 3 else ref_region
            _, s_wm = cv2.meanStdDev(gray_region)
            _, s_ref = cv2.meanStdDev(gray_ref)
            if s_ref[0][0] > 5.0:
                var_score = max(0.0, min(1.0, 1.0 - (s_wm[0][0] / s_ref[0][0])))
        return float(grad_score), float(var_score)

    def _corner_promote(
        self,
        image: NDArray[Any],
        current_raw_ncc: float,
    ) -> tuple[int, int, int, float] | None:
        """Search the bottom-right corner for a very-high-fidelity sparkle match.

        Returns ``(scale, x, y, raw_ncc)`` when the corner holds a match with raw
        NCC >= ``_CORNER_PROMOTE_NCC`` that beats the global pick's ``current_raw_ncc``,
        else None. Used to rescue a small sparkle that the size weight buried under
        a larger, lower-fidelity match elsewhere. See ``_CORNER_PROMOTE_NCC`` and
        ``_CORNER_PROMOTE_FRAC`` for the corner sizing.
        """
        h, w = image.shape[:2]
        side = max(
            self._CORNER_PROMOTE_MIN, min(self._CORNER_PROMOTE_MAX, round(min(w, h) * self._CORNER_PROMOTE_FRAC))
        )
        cs = int(min(min(w, h), side))
        cx1, cy1 = max(0, w - cs), max(0, h - cs)
        corner = image[cy1:h, cx1:w]
        gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY) if corner.ndim == 3 and corner.shape[2] >= 3 else corner
        gray = gray.astype(np.float32) / 255.0

        best_raw = -1.0
        best_scale = 0
        best_loc = (0, 0)
        for scale, max_val, max_loc in self._scan_scales(gray):
            if max_val > best_raw:
                best_raw = max_val
                best_scale = scale
                best_loc = max_loc

        if best_raw >= self._CORNER_PROMOTE_NCC and best_raw > current_raw_ncc:
            return best_scale, cx1 + best_loc[0], cy1 + best_loc[1], float(best_raw)
        return None

    # ── Removal ──────────────────────────────────────────────────────

    # Footprint mask for the localize -> fill removal path. The mask must cover the
    # WHOLE sparkle including its faint semi-transparent halo, not just the bright
    # core, or the fill leaves a visible ring. Threshold the captured alpha low
    # (>_MASK_ALPHA catches the halo the core-only 0.10 misses) then dilate by a
    # sparkle-relative margin so alignment slop and the outermost halo are absorbed.
    _MASK_ALPHA = 0.04
    _MASK_DILATE_FRAC = 0.18  # dilation radius as a fraction of the sparkle scale

    def footprint_mask(
        self,
        image: NDArray[Any],
        *,
        force: bool = False,
        dilate: int | None = None,
        region: tuple[int, int, int, int] | None = None,
    ) -> NDArray[Any] | None:
        """Full-frame uint8 mask (255 = sparkle) of the sparkle footprint, for the
        shared fill removal path (cv2 / MI-GAN / LaMa), or None.

        The footprint is the interpolated captured alpha at the detected scale,
        thresholded LOW so the faint halo is included, then dilated by a
        sparkle-relative margin. When ``force`` and nothing is detected, falls back to
        the default sparkle slot for the image size (the ``--no-detect`` path).

        ``region`` is the already-resolved ``(x, y, scale)`` from the caller's detection
        (the registry passes the decision's provenance-aware region). When given, the
        mask is built from it directly WITHOUT a second internal detect -- otherwise a
        provenance/assume-relaxed sparkle would be re-demoted by the strict re-detect and
        yield no mask (reported-removed-but-unchanged). Absent ``region``, direct callers
        keep the detect-then-force behavior.
        """
        image = image_io.to_bgr(image)
        h, w = image.shape[:2]
        if region is not None:
            x, y, scale = region[0], region[1], region[2]
        else:
            det = self.detect_watermark(image)
            if det.detected:
                x, y, scale = det.region[0], det.region[1], det.region[2]
            elif force:
                cfg = get_watermark_config(w, h)
                x, y = cfg.get_position(w, h)
                scale = cfg.logo_size
            else:
                return None
        alpha = self.get_interpolated_alpha(scale)
        fp = self._footprint_indices(alpha, (x, y), image.shape)
        if fp is None:
            return None
        aroi, (y1, y2, x1, x2) = fp
        sil = (aroi > self._MASK_ALPHA).astype(np.uint8) * 255
        if int((sil > 0).sum()) == 0:
            return None
        mask = np.zeros((h, w), np.uint8)
        mask[y1:y2, x1:x2] = sil
        d = dilate if dilate is not None else max(13, int(scale * self._MASK_DILATE_FRAC))
        if d > 0:
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1)))
        return mask

    def _footprint_indices(
        self,
        alpha_map: NDArray[Any],
        position: tuple[int, int],
        image_shape: tuple[int, ...],
    ) -> tuple[NDArray[Any], tuple[int, int, int, int]] | None:
        """Return (alpha_roi, (y1, y2, x1, x2)) for the placed footprint, or None.

        Shared by the over-subtraction test and the inpaint mask so both operate on
        exactly the same clipped, in-bounds region.
        """
        x, y = position
        ah, aw = alpha_map.shape[:2]
        ih, iw = image_shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(iw, x + aw), min(ih, y + ah)
        if x1 >= x2 or y1 >= y2:
            return None
        ax1, ay1 = x1 - x, y1 - y
        alpha_roi = alpha_map[ay1 : ay1 + (y2 - y1), ax1 : ax1 + (x2 - x1)]
        return alpha_roi, (y1, y2, x1, x2)

    def _core_and_bg(
        self,
        image: NDArray[Any],
        alpha_map: NDArray[Any],
        position: tuple[int, int],
    ) -> tuple[float, float, float] | None:
        """Return ``(core_obs, bg, a_cap)`` for the placed sparkle, or None.

        ``core_obs`` is the bright-core brightness (75th pct over the high-alpha
        core), ``bg`` the local background ring median, ``a_cap`` the captured peak
        alpha. Shared by the alpha-gain estimate and the false-positive margin gate.
        None when the footprint or the background ring cannot be sampled.
        """
        placed = self._footprint_indices(alpha_map, position, image.shape)
        if placed is None:
            return None
        alpha_roi, (y1, y2, x1, x2) = placed
        a_cap = float(alpha_roi.max())
        if a_cap < 0.2:
            return None
        core = alpha_roi >= a_cap * self._CORE_ALPHA_FRAC
        if not bool(core.any()):
            return None
        # Convert only the footprint+ring crop to gray, not the whole image: every
        # sample below lives inside the ring box, so a full-image mean is wasted work
        # that scales with resolution (~70 ms on a 12 MP image, recomputed for both
        # the alpha-gain estimate and the over-subtraction gate). The crop is sized by
        # the footprint, so this is O(footprint^2) regardless of image size.
        ih, iw = image.shape[:2]
        pad = int((x2 - x1) * 0.7)
        ry1, ry2 = max(0, y1 - pad), min(ih, y2 + pad)
        rx1, rx2 = max(0, x1 - pad), min(iw, x2 + pad)
        ring = image[ry1:ry2, rx1:rx2].astype(np.float32).mean(axis=2)
        # Footprint box expressed in ring-crop coordinates.
        fy1, fy2, fx1, fx2 = y1 - ry1, y2 - ry1, x1 - rx1, x2 - rx1
        core_obs = float(np.percentile(ring[fy1:fy2, fx1:fx2][core], 75))
        ring_mask = np.ones(ring.shape, dtype=bool)
        ring_mask[fy1:fy2, fx1:fx2] = False
        if int(ring_mask.sum()) < 10:
            return None
        return core_obs, float(np.median(ring[ring_mask])), a_cap

    def _core_ring_margin(
        self,
        image: NDArray[Any],
        alpha_map: NDArray[Any],
        position: tuple[int, int],
    ) -> float | None:
        """Bright-core brightness minus the local background ring (gray levels).

        A real white sparkle overlay lifts its core above the surroundings; a
        shape-only NCC false positive on ornate/flat content does not. None when the
        background ring cannot be sampled.
        """
        cb = self._core_and_bg(image, alpha_map, position)
        return None if cb is None else cb[0] - cb[1]

    def _core_saturation(
        self,
        image: NDArray[Any],
        alpha_map: NDArray[Any],
        position: tuple[int, int],
    ) -> float | None:
        """Median color saturation of the sparkle core (0 = white/neutral, higher =
        colored). A real Gemini sparkle is a white star, so its core is near-neutral;
        a clean bright corner that shape-matches (sky, sun, a warm light) is colored,
        so a high core saturation flags the false positive the brightness/gradient
        gates miss. Samples the same high-alpha core pixels as :meth:`_core_and_bg`.
        None when the footprint cannot be placed or the core is empty.
        """
        placed = self._footprint_indices(alpha_map, position, image.shape)
        if placed is None:
            return None
        alpha_roi, (y1, y2, x1, x2) = placed
        a_cap = float(alpha_roi.max())
        if a_cap < 0.2:
            return None
        core = alpha_roi >= a_cap * self._CORE_ALPHA_FRAC
        box = image[y1:y2, x1:x2]
        if box.shape[:2] != core.shape or not bool(core.any()):
            return None
        px = box[core].astype(np.float32)  # (N, 3) BGR core pixels
        hi = px.max(axis=1)
        lo = px.min(axis=1)
        return float(np.median((hi - lo) / (hi + 1.0)))


def detect_sparkle_confidence(image_path: Path, *, image: NDArray[Any] | None = None) -> float | None:
    """Visible-sparkle detection confidence for a file, for provenance use.

    Loads the image with cv2 and runs :meth:`GeminiEngine.detect_watermark`.
    Returns the NCC confidence in [0, 1], or None if the image cannot be read
    (cv2 returns None for unsupported containers such as HEIC). Kept here so the
    cv2 dependency stays in this module; callers apply their own threshold.

    ``image`` lets a caller that has already decoded the file (e.g. ``identify``
    running several visible-mark detectors) pass the BGR array to avoid a second
    full decode; when None the file is read from ``image_path``.
    """
    from remove_ai_watermarks import image_io

    img = image if image is not None else image_io.imread(image_path)
    if img is None:
        return None
    return float(GeminiEngine().detect_watermark(img).confidence)
