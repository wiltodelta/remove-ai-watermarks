"""Gemini visible watermark removal engine.

Port of the GeminiWatermarkTool reverse-alpha-blending algorithm from C++ to Python.
Original author: Allen Kuo (allenk) — https://github.com/allenk/GeminiWatermarkTool

The Gemini AI watermark is applied using alpha blending:
    watermarked = a * logo + (1 - a) * original

We reverse this to recover the original:
    original = (watermarked - a * logo) / (1 - a)

The alpha maps are derived from background captures of the Gemini watermark
on pure-black backgrounds (48x48 for small images, 96x96 for large images).
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
from typing import TYPE_CHECKING, Any, Literal

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
    """Engine for removing visible Gemini watermarks via reverse alpha blending.

    This is a Python port of the GeminiWatermarkTool C++ engine.
    """

    # Footprint pixels with alpha at/above this are the sparkle body; below it the
    # mark barely affects the pixel, so those are excluded from both the
    # over-subtraction test and the inpaint mask.
    _FOOTPRINT_ALPHA = 0.1
    # If more than this fraction of footprint pixels over-subtract (numerator < 0),
    # the fixed alpha does not match this image's sparkle and reverse-alpha would
    # punch a dark pit -- inpaint instead. demo_banana measures 0.0 (reverse-alpha
    # kept), the issue #30 dark-grass image measures ~0.61 (inpaint), so the 0.05
    # gate separates them with a wide margin.
    _OVERSUB_FOOTPRINT_FRAC = 0.05

    # Mid-tone over-subtraction (2026-06-18 prod "the color just changed, not removed"
    # report). The numerator fraction above only trips when reverse-alpha drives a
    # footprint pixel fully NEGATIVE -- the dark-background black-pit case. On a MID-TONE
    # background a sparkle fainter than the captured alpha is over-subtracted into a
    # visibly DARKER-than-background diamond while no pixel ever crosses zero, so the
    # numerator gate misses it and ships the dark mark. Predict the reverse-alpha output
    # at the bright core, (core - a*logo)/(1-a); when it lands more than this many gray
    # levels BELOW the local background ring, reverse-alpha would leave a dark diamond --
    # inpaint instead. Calibrated wide: clean removals predict within ~12 of background
    # (demo_banana ~-1, a bright-bg sparkle ~-12), the prod regression predicts ~-40 and
    # the issue #30 dark case ~-82, so 25 separates keep-vs-inpaint with margin.
    _OVERSUB_DARK_MARGIN = 25.0

    # Per-image alpha gain (under-subtraction fix). The captured alpha peaks ~0.51
    # (a ~51%-opaque sparkle). Some real Gemini sparkles are rendered MORE opaque,
    # so the fixed alpha under-subtracts and reverse-alpha leaves a bright residual
    # the detector still fires on (~11% of marks on the spaces corpus). Estimate
    # this image's effective sparkle opacity from the bright core vs the local
    # background and scale the alpha to match, capped so alpha stays < 0.99. The
    # gain is clamped to >= 1.0 so it only ever STRENGTHENS removal: ~1.0 when the
    # sparkle matches the capture (working cases unchanged), >1 when more opaque.
    # On the spaces corpus the gain cleanly separates -- under-removed marks ~1.47,
    # cleanly-removed ~1.00. 1.94 is the cap that reaches alpha 0.99 from 0.51.
    _ALPHA_GAIN_MAX = 1.94
    _ALPHA_GAIN_CORE_FRAC = 0.8  # body pixels at >= this * peak alpha define the core
    # Deadband: apply the gain only above this, so a sparkle that already matches the
    # capture (estimated gain ~1.0-1.04 from background noise) stays byte-identical to
    # the pre-fix output. Under-removed marks estimate >= 1.26, well clear of the band.
    _ALPHA_GAIN_DEADBAND = 1.05

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

    # Self-verify fallback. The gain estimate corrects most under-subtractions, but
    # on the spaces corpus a tail of strong sparkles still survived reverse-alpha
    # (a few px of position jitter or a gain estimate the [1.0, 1.94] clamp could
    # not fully reach). After the reverse blend, re-detect; if a sparkle this strong
    # remains, inpaint the footprint and keep that ONLY when it lowers the re-detect
    # confidence. Purely additive: the common clean removal re-detects below this and
    # is returned untouched. Threshold matches the registry's real fail line (0.5),
    # so it triggers exactly on the cases that would otherwise read as not-removed
    # (rescued 4 of 15 corpus fails, 0 regressions). An offset+scale alignment search
    # was prototyped on the remaining 11 but REJECTED: it only lowered the shape-NCC by
    # moving the reverse-alpha to a different placement that left the sparkle as bright
    # or brighter (NCC-gaming, not removal), so a brightness sanity check rejected every
    # one. The footprint inpaint physically reconstructs the slot from its surroundings,
    # so its rescues are genuine; the survivors are near-white ill-conditioning or
    # detector false positives that no reverse-alpha placement fixes.
    _VERIFY_FALLBACK_CONF = 0.5

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
    ) -> DetectionResult:
        """Detect Gemini watermark using multi-scale Snap Engine logic (ported from C++ vendor algorithm)."""
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
        if confidence < self._SPARKLE_FP_CONF:
            margin = self._core_ring_margin(image, self.get_interpolated_alpha(best_scale), (pos_x, pos_y))
            low_margin = margin is not None and margin < self._SPARKLE_FP_MARGIN
            low_grad = grad_score < self._SPARKLE_FP_GRAD
            if low_margin or low_grad:
                logger.debug(
                    "Sparkle FP gate: conf=%.3f, core-ring margin=%s, grad=%.3f < %.2f; demoting.",
                    confidence,
                    f"{margin:.1f}" if margin is not None else "n/a",
                    grad_score,
                    self._SPARKLE_FP_GRAD,
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

    def remove_watermark(
        self,
        image: NDArray[Any],
        force_size: WatermarkSize | None = None,
    ) -> NDArray[Any]:
        """Remove Gemini visible watermark from an image using reverse alpha blending.

        No-op when the detector does not find a watermark: returns an unmodified
        copy. Reverse alpha blending applied where no sparkle exists creates a
        visible inverse artifact, so we refuse to touch pixels without a positive
        detection. To bypass detection (e.g. you know the exact region), use
        ``remove_watermark_custom``.

        Args:
            image: BGR image as numpy array (will NOT be modified in-place).
            force_size: Force a specific watermark size (auto-detect if None).

        Returns:
            Cleaned BGR image as numpy array, or an unmodified copy when no
            watermark is detected.
        """
        # Normalize to 3-channel BGR up front: 2D grayscale (no channel axis) and
        # 4-channel BGRA both reach this public entry point and would otherwise
        # crash on the channel-count checks / downstream 3-channel math.
        result = image_io.to_bgr(image.copy())

        size = force_size or get_watermark_size(result.shape[1], result.shape[0])

        # Detect dynamic position & size (on the normalized 3-channel image so a
        # grayscale/BGRA input does not crash the detector).
        detection = self.detect_watermark(result, force_size=size)

        if not detection.detected:
            logger.debug(
                "No watermark detected (conf=%.3f); returning image unchanged.",
                detection.confidence,
            )
            return result

        pos = (detection.region[0], detection.region[1])
        alpha_map = self.get_interpolated_alpha(detection.region[2])
        # Match the captured alpha to this image's sparkle opacity (under-subtraction
        # fix): a more-opaque-than-captured sparkle would otherwise leave a bright
        # residual. gain == 1.0 leaves the working cases byte-identical.
        gain = self._estimate_alpha_gain(result, alpha_map, pos)
        if gain > self._ALPHA_GAIN_DEADBAND:
            alpha_map = np.clip(alpha_map * gain, 0.0, 0.99)
        logger.debug(
            "Removing watermark at (%d, %d) size %dx%d [conf=%.3f]",
            pos[0],
            pos[1],
            detection.region[2],
            detection.region[3],
            detection.confidence,
        )

        # The captured alpha map (max ~0.51 = a ~50%-opaque white sparkle) is exact
        # only when the real mark's effective opacity matches it. On a dark/textured
        # background the sparkle's effective alpha is lower than the capture, so the
        # fixed-alpha reverse blend OVER-subtracts and drives the footprint to black --
        # the "white sparkle turns into a black pit" bug (issue #30). The signature is
        # a large fraction of footprint pixels whose numerator (watermarked - a*logo)
        # goes negative, which is physically impossible under a brightening overlay.
        # In that case inpaint the small footprint from the surrounding pixels instead;
        # on a bright background no pixel over-subtracts, so reverse-alpha is used and
        # the result is byte-identical to before (verified on demo_banana: 0% vs 61%).
        if self._reverse_alpha_oversubtracts(result, alpha_map, pos):
            logger.debug("Reverse-alpha over-subtracts on this background; inpainting sparkle footprint.")
            self._inpaint_footprint(result, alpha_map, pos)
        else:
            self._reverse_alpha_blend(result, alpha_map, pos)
        return self._verify_and_repair(result, alpha_map, pos, size)

    def footprint_mask(self, image: NDArray[Any], *, force: bool = False, dilate: int = 13) -> NDArray[Any] | None:
        """Full-frame uint8 mask (255 = sparkle) of the sparkle footprint, for the
        inpaint-fallback removal path (LaMa / cv2), or None.

        The footprint is the interpolated captured alpha at the detected scale --
        the same region reverse-alpha operates on. When ``force`` and nothing is
        detected, falls back to the default sparkle slot for the image size (the
        ``--no-detect`` path). The caller gates on the trust-confidence detection.
        """
        image = image_io.to_bgr(image)
        h, w = image.shape[:2]
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
        sil = (aroi > 0.10).astype(np.uint8) * 255
        if int((sil > 0).sum()) == 0:
            return None
        mask = np.zeros((h, w), np.uint8)
        mask[y1:y2, x1:x2] = sil
        if dilate > 0:
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1)))
        return mask

    def remove_watermark_custom(
        self,
        image: NDArray[Any],
        region: tuple[int, int, int, int],
    ) -> NDArray[Any]:
        """Remove watermark from a custom region with interpolated alpha map.

        Args:
            image: BGR image (will NOT be modified in-place).
            region: (x, y, width, height) of the watermark region.

        Returns:
            Cleaned BGR image.
        """
        # Same channel normalization as remove_watermark: the reverse-alpha blend
        # assumes 3-channel BGR (a grayscale/BGRA input would mis-broadcast).
        result = image_io.to_bgr(image.copy())
        x, y, rw, rh = region

        # Check standard sizes
        if rw == 48 and rh == 48:
            self._reverse_alpha_blend(result, self._alpha_small, (x, y))
            return result
        if rw == 96 and rh == 96:
            self._reverse_alpha_blend(result, self._alpha_large, (x, y))
            return result

        # Interpolate alpha map for custom size
        interp = cv2.INTER_LINEAR if rw > 96 else cv2.INTER_AREA
        alpha = cv2.resize(self._alpha_large, (rw, rh), interpolation=interp)
        self._reverse_alpha_blend(result, alpha, (x, y))
        return result

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
        core = alpha_roi >= a_cap * self._ALPHA_GAIN_CORE_FRAC
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

    def _estimate_alpha_gain(
        self,
        image: NDArray[Any],
        alpha_map: NDArray[Any],
        position: tuple[int, int],
    ) -> float:
        """Scale factor matching the captured alpha to this image's sparkle opacity.

        The captured alpha (peak ~0.51) under-represents sparkles rendered more
        opaque; reverse-alpha then leaves a bright residual. Estimate the effective
        opacity at the sparkle core (observed brightness vs the local background
        ring) and return ``a_eff / a_capture``, clamped to ``[1.0, _ALPHA_GAIN_MAX]``
        so it only ever STRENGTHENS removal (1.0 = no change on a matching sparkle).
        Returns 1.0 when the background cannot be estimated reliably.
        """
        cb = self._core_and_bg(image, alpha_map, position)
        if cb is None:
            return 1.0
        core_obs, bg, a_cap = cb
        if 255.0 - bg < 5.0:
            return 1.0
        a_eff = float(np.clip((core_obs - bg) / (255.0 - bg), 0.0, 0.99))
        return float(np.clip(a_eff / a_cap, 1.0, self._ALPHA_GAIN_MAX))

    def _reverse_alpha_oversubtracts(
        self,
        image: NDArray[Any],
        alpha_map: NDArray[Any],
        position: tuple[int, int],
    ) -> bool:
        """True when reverse-alpha would drive the footprint dark.

        Two signatures of the captured alpha over-estimating this image's sparkle
        opacity, either of which means reverse-alpha would leave a dark mark:

        1. Dark-background black pit (issue #30): the numerator
           ``watermarked - alpha*logo`` over the sparkle body. A brightening overlay
           can never make it negative, so a large negative fraction means the fixed
           alpha over-subtracts past black.
        2. Mid-tone dark diamond (see ``_OVERSUB_DARK_MARGIN``): on a mid-tone
           background the over-subtraction darkens the core well below the background
           without any pixel crossing zero, so case 1 misses it. Predict the
           reverse-alpha core output and trip when it lands far below the local ring.
        """
        placed = self._footprint_indices(alpha_map, position, image.shape)
        if placed is None:
            return False
        alpha_roi, (y1, y2, x1, x2) = placed
        body = alpha_roi >= self._FOOTPRINT_ALPHA
        if not bool(body.any()):
            return False
        roi = image[y1:y2, x1:x2].astype(np.float32)
        numerator = roi.mean(axis=2) - np.clip(alpha_roi, 0.0, 0.99) * self.logo_value
        frac = float((numerator[body] < 0).sum()) / float(body.sum())
        if frac > self._OVERSUB_FOOTPRINT_FRAC:
            return True

        # Mid-tone darkening: predict the reverse-alpha output at the bright core and
        # compare to the local background ring (reuses the FP-gate / alpha-gain machinery).
        cb = self._core_and_bg(image, alpha_map, position)
        if cb is None:
            return False
        core_obs, bg, a_cap = cb
        a = min(a_cap, 0.99)
        predicted_core = (core_obs - a * self.logo_value) / (1.0 - a)
        return predicted_core < bg - self._OVERSUB_DARK_MARGIN

    def _inpaint_footprint(
        self,
        image: NDArray[Any],
        alpha_map: NDArray[Any],
        position: tuple[int, int],
    ) -> None:
        """Inpaint the sparkle body from surrounding pixels, in-place.

        Fallback for backgrounds where reverse-alpha over-subtracts: a small mask of
        the footprint (alpha >= threshold, dilated) is reconstructed by cv2 NS inpaint
        from the continuous surroundings, so the sparkle is replaced by plausible
        background instead of a black pit.
        """
        placed = self._footprint_indices(alpha_map, position, image.shape)
        if placed is None:
            return
        alpha_roi, (y1, y2, x1, x2) = placed
        # Inpaint only a padded crop around the footprint, not the whole image: the
        # mask is zero outside a ~96x96 corner, so inpainting the full (multi-MP)
        # image would be ~hundreds of times more work for an identical result. The
        # padding gives cv2 enough surrounding context to reconstruct the sparkle.
        ih, iw = image.shape[:2]
        pad = 24
        cy1, cy2 = max(0, y1 - pad), min(ih, y2 + pad)
        cx1, cx2 = max(0, x1 - pad), min(iw, x2 + pad)
        crop = image[cy1:cy2, cx1:cx2]
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        mask[y1 - cy1 : y2 - cy1, x1 - cx1 : x2 - cx1] = (alpha_roi >= self._FOOTPRINT_ALPHA).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=2)
        image[cy1:cy2, cx1:cx2] = cv2.inpaint(crop, mask, 6, cv2.INPAINT_NS)

    def _verify_and_repair(
        self,
        result: NDArray[Any],
        alpha_map: NDArray[Any],
        position: tuple[int, int],
        size: WatermarkSize,
    ) -> NDArray[Any]:
        """Inpaint-repair a sparkle that survived reverse-alpha, keeping the better.

        Re-detect on the reverse-alpha output; if a sparkle this strong remains (an
        alpha mismatch the gain estimate could not fully correct), inpaint the
        footprint and return that ONLY when it lowers the re-detect confidence. The
        footprint inpaint reconstructs from the (darker) surroundings, so it physically
        removes the bright sparkle rather than gaming the shape-NCC. Returns ``result``
        unchanged when the removal is already clean (the common case) or when the
        inpaint does not improve it, so it can never regress.
        """
        residual = self.detect_watermark(result, force_size=size).confidence
        if residual < self._VERIFY_FALLBACK_CONF:
            return result
        candidate = result.copy()
        self._inpaint_footprint(candidate, alpha_map, position)
        if self.detect_watermark(candidate, force_size=size).confidence < residual:
            logger.debug("Sparkle survived reverse-alpha (conf=%.3f); footprint inpaint improved it.", residual)
            return candidate
        return result

    def _reverse_alpha_blend(
        self,
        image: NDArray[Any],
        alpha_map: NDArray[Any],
        position: tuple[int, int],
    ) -> None:
        """Apply reverse alpha blending in-place.

        Formula: original = (watermarked - a * logo) / (1 - a)
        """
        placed = self._footprint_indices(alpha_map, position, image.shape)
        if placed is None:
            return
        alpha_roi, (y1, y2, x1, x2) = placed
        image_roi = image[y1:y2, x1:x2].astype(np.float32)

        alpha_threshold = 0.002
        max_alpha = 0.99

        # Vectorized reverse alpha blending
        alpha = alpha_roi.copy()
        mask = alpha >= alpha_threshold
        alpha = np.clip(alpha, 0.0, max_alpha)
        one_minus_alpha = 1.0 - alpha

        # Expand alpha for 3-channel broadcast
        alpha_3d = alpha[:, :, np.newaxis]
        one_minus_3d = one_minus_alpha[:, :, np.newaxis]
        mask_3d = mask[:, :, np.newaxis]

        # original = (watermarked - alpha * logo) / (1 - alpha)
        restored = (image_roi - alpha_3d * self.logo_value) / one_minus_3d
        restored = np.clip(restored, 0.0, 255.0)

        # Apply only where alpha is significant
        image_roi = np.where(mask_3d, restored, image_roi)
        image[y1:y2, x1:x2] = image_roi.astype(np.uint8)

    # ── Inpainting cleanup ───────────────────────────────────────────

    def inpaint_residual(
        self,
        image: NDArray[Any],
        region: tuple[int, int, int, int],
        strength: float = 0.85,
        method: Literal["gaussian", "telea", "ns"] = "ns",
        inpaint_radius: int = 10,
        padding: int = 32,
    ) -> NDArray[Any]:
        """Apply inpaint cleanup on residual artifacts after reverse alpha blend.

        Uses a sparse mask derived from alpha map gradient to repair only
        the sparkle-edge pixels where interpolation broke the math.

        Args:
            image: BGR image (will NOT be modified in-place).
            region: (x, y, w, h) of the watermark region.
            strength: Blend strength (0.0 = keep original, 1.0 = fully inpainted).
            method: Inpaint method ("gaussian", "telea", or "ns").
            inpaint_radius: Radius for cv2.inpaint.
            padding: Context padding around region in pixels.

        Returns:
            Cleaned BGR image.
        """
        result = image.copy()
        x, y, rw, rh = region

        if rw < 4 or rh < 4:
            return result

        strength = max(0.0, min(1.0, strength))
        if strength < 0.001:
            return result

        # Padded region
        px1 = max(0, x - padding)
        py1 = max(0, y - padding)
        px2 = min(image.shape[1], x + rw + padding)
        py2 = min(image.shape[0], y + rh + padding)

        if (px2 - px1) < 8 or (py2 - py1) < 8:
            return result

        # Inner rect relative to padded
        ix1 = x - px1
        iy1 = y - py1

        # Get alpha map (interpolated if needed)
        source_alpha = self._alpha_large
        interp = cv2.INTER_LINEAR if rw > source_alpha.shape[1] else cv2.INTER_AREA
        alpha_resized = cv2.resize(source_alpha, (rw, rh), interpolation=interp)

        # Compute gradient mask from alpha
        grad_x = cv2.Sobel(alpha_resized, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(alpha_resized, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)

        grad_min, grad_max = grad_mag.min(), grad_mag.max()
        if grad_max <= grad_min:
            return result

        # Normalize and apply gamma correction
        grad_norm = (grad_mag - grad_min) / (grad_max - grad_min)
        grad_weight = np.sqrt(grad_norm)

        # Dilate the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        grad_weight = cv2.dilate(grad_weight, kernel)

        if method == "gaussian":
            # Soft blend with Gaussian blur
            padded_roi = result[py1:py2, px1:px2].copy()
            blurred = cv2.GaussianBlur(padded_roi, (0, 0), sigmaX=2.0)

            # Create weight mask on padded area (only inner region has weights)
            weight_full = np.zeros((py2 - py1, px2 - px1), dtype=np.float32)
            weight_full[iy1 : iy1 + rh, ix1 : ix1 + rw] = grad_weight * strength

            weight_3d = weight_full[:, :, np.newaxis]
            blended = padded_roi.astype(np.float32) * (1 - weight_3d) + blurred.astype(np.float32) * weight_3d
            result[py1:py2, px1:px2] = blended.astype(np.uint8)
        else:
            # OpenCV inpainting (TELEA or NS)
            inpaint_flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS

            # Create binary mask from gradient weight
            binary_mask = (grad_weight * 255).astype(np.uint8)
            _, binary_mask = cv2.threshold(binary_mask, 30, 255, cv2.THRESH_BINARY)

            # Expand mask to padded region
            mask_full = np.zeros((py2 - py1, px2 - px1), dtype=np.uint8)
            mask_full[iy1 : iy1 + rh, ix1 : ix1 + rw] = binary_mask

            padded_roi = result[py1:py2, px1:px2].copy()
            inpainted = cv2.inpaint(padded_roi, mask_full, inpaint_radius, inpaint_flag)

            # Blend with strength
            weight_full = np.zeros((py2 - py1, px2 - px1), dtype=np.float32)
            weight_full[iy1 : iy1 + rh, ix1 : ix1 + rw] = grad_weight * strength
            weight_3d = weight_full[:, :, np.newaxis]

            blended = padded_roi.astype(np.float32) * (1 - weight_3d) + inpainted.astype(np.float32) * weight_3d
            result[py1:py2, px1:px2] = blended.astype(np.uint8)

        return result


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
