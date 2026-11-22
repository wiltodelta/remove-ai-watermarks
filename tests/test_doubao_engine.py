"""Tests for the Doubao visible-watermark engine (localize -> fill)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from remove_ai_watermarks import watermark_registry as registry
from remove_ai_watermarks._text_mark_engine import TextMarkEngine
from remove_ai_watermarks.doubao_engine import (
    _ALPHA_HEIGHT_FRAC,
    _ALPHA_NATIVE_WIDTH,
    _ALPHA_WIDTH_FRAC,
    _CONFIG,
    DETECT_NCC_THRESHOLD,
    DoubaoEngine,
    _alpha_template,
    _glyph_silhouette,
    _template_match_score,
)
from remove_ai_watermarks.image_io import load_image_bgr

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "provenance" / "doubao-1.png"


def _compose(w: int, h: int, bg: float | tuple[int, int, int] = 100.0):
    """Composite the real alpha (scaled to width ``w``) onto a flat bg.
    Returns ``(watermarked_uint8, mark_bool_mask)``."""
    img = np.empty((h, w, 3), np.float32)
    img[:] = bg
    at = _alpha_template()
    gw, gh = int(_ALPHA_WIDTH_FRAC * w), int(_ALPHA_HEIGHT_FRAC * w)
    margin = int(0.015 * w)
    ax = w - margin - gw
    ay = h - margin - gh
    amap = np.zeros((h, w), np.float32)
    amap[ay : ay + gh, ax : ax + gw] = cv2.resize(at, (gw, gh))
    a3 = amap[:, :, None]
    wm = (a3 * 255.0 + (1 - a3) * img).clip(0, 255).astype(np.uint8)
    return wm, amap > 0.2


class TestLocate:
    def test_box_anchored_bottom_right(self):
        eng = DoubaoEngine()
        img = np.zeros((2048, 2048, 3), np.uint8)
        loc = eng.locate(img)
        assert 2048 - (loc.x + loc.w) < int(2048 * 0.03)
        assert 2048 - (loc.y + loc.h) < int(2048 * 0.03)

    def test_box_scales_with_width(self):
        eng = DoubaoEngine()
        small = eng.locate(np.zeros((1024, 1024, 3), np.uint8))
        large = eng.locate(np.zeros((2048, 2048, 3), np.uint8))
        assert large.w == pytest.approx(small.w * 2, rel=0.1)


# ── Detection: alpha-template NCC ───────────────────────────────────


class TestDetect:
    def test_tinted_glyph_on_saturated_background(self):
        """A translucent white glyph stays recognizable after a yellow background tints it."""
        wm, _mark = _compose(_ALPHA_NATIVE_WIDTH, _ALPHA_NATIVE_WIDTH, bg=(20, 190, 240))
        eng = DoubaoEngine()
        clean = np.empty_like(wm)
        clean[:] = (20, 190, 240)
        assert not eng.detect(clean).detected
        assert eng.detect(wm).detected
        out, region = registry.get_mark("doubao").remove(wm, backend="cv2")
        assert region is not None
        assert not eng.detect(out).detected
        assert np.array_equal(wm[:1024, :1024], out[:1024, :1024])

    def test_clean_gradient_not_detected(self):
        eng = DoubaoEngine()
        ramp = np.tile(np.linspace(0, 255, 1024, dtype=np.uint8), (1024, 1))
        img = cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR)
        assert not eng.detect(img).detected

    def test_solid_blob_corner_not_detected(self):
        """A bright blob is not the glyph shape -> low correlation, not detected."""
        eng = DoubaoEngine()
        img = np.zeros((1024, 1024, 3), np.uint8)
        x, y, bw, bh = eng.locate(img).bbox
        img[y + bh // 4 : y + bh * 3 // 4, x : x + bw // 2] = 200
        assert not eng.detect(img).detected

    def test_silhouette_loads(self):
        sil = _glyph_silhouette()
        assert sil is not None
        assert set(np.unique(sil)).issubset({0, 255})

    def test_match_score_shape_sensitive(self):
        """The glyph silhouette correlates with itself, not with a filled block."""
        sil = _glyph_silhouette()
        h, w = sil.shape
        # box that contains the silhouette -> high score
        box = np.zeros((h + 8, int(w / _ALPHA_WIDTH_FRAC * 0.2) + w), np.uint8)
        box[4 : 4 + h, 4 : 4 + w] = sil
        assert _template_match_score(box, _ALPHA_NATIVE_WIDTH) >= DETECT_NCC_THRESHOLD
        # a uniformly filled box has no glyph structure -> low score
        solid = np.full_like(box, 255)
        assert _template_match_score(solid, _ALPHA_NATIVE_WIDTH) < DETECT_NCC_THRESHOLD

    def test_small_image_guarded_from_false_positive(self):
        """Below the minimum short side a tiny geometric shape spuriously NCC-matches
        the CJK silhouette (2026-06-26 FP: a 48x48 app-icon chevron scored 0.41). The
        size guard suppresses detection there. Bracket it: a real mark is detected at
        native size, but the same content downscaled below the guard is not."""
        wm, _mark = _compose(_ALPHA_NATIVE_WIDTH, _ALPHA_NATIVE_WIDTH)
        eng = DoubaoEngine()
        assert eng.detect(wm).detected  # native: real mark detected
        assert not eng.detect(cv2.resize(wm, (150, 150))).detected  # below guard: suppressed


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample image not present")
class TestRealSample:
    def test_detects_watermark(self):
        image = load_image_bgr(SAMPLE)
        det = DoubaoEngine().detect(image)
        assert det.detected
        assert det.confidence >= DETECT_NCC_THRESHOLD
        assert det.confidence == TextMarkEngine(_CONFIG).detect(image).confidence

    def test_fill_lowers_confidence(self):
        img = load_image_bgr(SAMPLE)
        before = DoubaoEngine().detect(img)
        assert before.detected
        out, region = registry.get_mark("doubao").remove(img, backend="cv2")
        assert region is not None
        after = DoubaoEngine().detect(out)
        assert not after.detected
        assert after.confidence < before.confidence

    def test_far_region_untouched(self):
        img = load_image_bgr(SAMPLE)
        out, _ = registry.get_mark("doubao").remove(img, backend="cv2")
        h, w = img.shape[:2]
        assert np.array_equal(img[: h // 2, : w // 2], out[: h // 2, : w // 2])

    def test_footprint_stays_on_glyphs_and_off_frame_edges(self):
        """Do not turn textured content behind the mark into a solid edge mask.

        The sample's branch contributes bright top-hat pixels below the wordmark.
        Bounding that response and dilating it used to clamp a solid 404x134 mask
        to both frame edges, so OpenCV propagated the remaining context into large
        triangular artifacts. The detector already aligned the captured glyph; the
        removal footprint must follow that glyph instead of the background response.
        """
        img = load_image_bgr(SAMPLE)
        mask = DoubaoEngine().footprint_mask(img)
        assert mask is not None
        assert not mask[-1].any()
        assert not mask[:, -1].any()
        ys, xs = np.where(mask > 0)
        bounding_area = (int(xs.max()) - int(xs.min()) + 1) * (int(ys.max()) - int(ys.min()) + 1)
        assert np.count_nonzero(mask) < 0.65 * bounding_area


# ── Alpha asset + footprint mask + localize -> fill removal ─────────


class TestAlphaAsset:
    def test_alpha_asset_loads(self):
        at = _alpha_template()
        assert at is not None
        assert at.dtype.kind == "f"
        assert float(at.min()) >= 0.0
        assert float(at.max()) <= 1.0


class TestFootprintMaskAndRemoval:
    def test_footprint_mask_in_bottom_right(self):
        """A composed mark yields a footprint mask localized to the bottom-right."""
        wm, _mark = _compose(_ALPHA_NATIVE_WIDTH, _ALPHA_NATIVE_WIDTH)
        mask = DoubaoEngine().footprint_mask(wm)
        assert mask is not None
        assert mask.shape == wm.shape[:2]
        ys, xs = np.where(mask > 0)
        assert ys.mean() > wm.shape[0] / 2
        assert xs.mean() > wm.shape[1] / 2

    def test_removes_synthetic_mark(self):
        """localize -> cv2 fill clears a mark composed from the real alpha
        (re-detect no longer fires)."""
        wm, _mark = _compose(_ALPHA_NATIVE_WIDTH, _ALPHA_NATIVE_WIDTH)
        assert DoubaoEngine().detect(wm).detected
        out, region = registry.get_mark("doubao").remove(wm, backend="cv2")
        assert region is not None
        assert not DoubaoEngine().detect(out).detected

    @pytest.mark.parametrize(
        ("w", "h"),
        [
            (_ALPHA_NATIVE_WIDTH, _ALPHA_NATIVE_WIDTH),  # captured width
            (1773, 2364),  # 3:4 portrait -> aligned footprint generalizes
        ],
    )
    def test_fill_removes_and_leaves_far_region(self, w, h):
        """The fill lowers re-detect confidence and leaves the far corner exact."""
        wm, mark = _compose(w, h)
        assert float(np.abs(wm.astype(np.float32)[mark] - 100.0).mean()) > 15  # mark visible
        before = DoubaoEngine().detect(wm)
        out, _ = registry.get_mark("doubao").remove(wm, backend="cv2")
        assert DoubaoEngine().detect(out).confidence < before.confidence
        assert np.array_equal(out[: h // 2, : w // 2], wm[: h // 2, : w // 2])

    def test_footprint_expands_over_dark_outline_stroke_without_growing_plain_glyphs(self):
        """A dark border stroke around single-line Doubao glyphs is masked and filled,
        while un-outlined white glyphs retain the compact 1-px dilation."""
        w = h = _ALPHA_NATIVE_WIDTH
        bg = np.full((h, w, 3), 214.0, np.float32)
        at = _alpha_template()
        assert at is not None
        gw, gh = int(_ALPHA_WIDTH_FRAC * w), int(_ALPHA_HEIGHT_FRAC * w)
        margin = int(0.015 * w)
        ax, ay = w - margin - gw, h - margin - gh
        resized = cv2.resize(at, (gw, gh))
        core = (resized > 0.15).astype(np.uint8)
        dilated = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        stroke_ring = (dilated > 0) & (core == 0)

        roi = bg[ay : ay + gh, ax : ax + gw].copy()
        roi[stroke_ring] = 168.0
        a3 = resized[:, :, None]
        roi = a3 * 242.0 + (1.0 - a3) * roi
        outlined = bg.copy()
        outlined[ay : ay + gh, ax : ax + gw] = roi
        outlined_u8 = outlined.clip(0, 255).astype(np.uint8)

        plain_u8, _ = _compose(w, h, bg=214.0)
        eng = DoubaoEngine()
        plain_mask = eng.footprint_mask(plain_u8)
        outlined_mask = eng.footprint_mask(outlined_u8)
        assert plain_mask is not None
        assert outlined_mask is not None
        assert np.count_nonzero(outlined_mask) > int(np.count_nonzero(plain_mask) * 1.35)
        stroke_full = np.zeros((h, w), dtype=bool)
        stroke_full[ay : ay + gh, ax : ax + gw] = stroke_ring
        assert float(np.mean(outlined_mask[stroke_full] > 0)) >= 0.85

        cleaned, region = registry.get_mark("doubao").remove(outlined_u8, backend="cv2")
        assert region is not None
        assert not eng.detect(cleaned).detected
        # Dark stroke residual must be lifted back near the 214 background level.
        assert float(np.median(cleaned[stroke_full])) >= 205.0


class TestDegenerateAndChannelInputs:
    """footprint_mask must not crash on degenerate sizes or non-3-channel inputs."""

    @pytest.mark.parametrize(("w", "h"), [(2048, 1), (1, 2048), (2048, 8)])
    def test_wide_short_does_not_raise(self, w, h):
        """A wide/short image at native width makes the width-derived glyph box
        taller than the image; masking must not ValueError."""
        eng = DoubaoEngine()
        img = np.zeros((h, w, 3), np.uint8)
        mask = eng.footprint_mask(img, force=True)
        assert mask is None or mask.shape == (h, w)

    def test_grayscale_2d_does_not_raise(self):
        eng = DoubaoEngine()
        gray = np.zeros((2048, 2048), np.uint8)
        mask = eng.footprint_mask(gray, force=True)
        assert mask is None or mask.shape == (2048, 2048)

    def test_bgra_4channel_does_not_raise(self):
        eng = DoubaoEngine()
        bgra = np.zeros((2048, 2048, 4), np.uint8)
        mask = eng.footprint_mask(bgra, force=True)
        assert mask is None or mask.shape == (2048, 2048)

    def test_template_match_score_guards_return_zero(self):
        # Guards return 0.0 (never a false positive) for a mask that cannot hold a
        # glyph: empty, narrower than min_gw, or shorter than the 4-px floor.
        assert _template_match_score(np.zeros((0, 5), np.uint8), 1000) == 0.0
        assert _template_match_score(np.zeros((10, 3), np.uint8), 1000) == 0.0  # width-1 < min_gw
        assert _template_match_score(np.zeros((3, 200), np.uint8), 1000) == 0.0  # height-1 < 4

    @pytest.mark.parametrize("shape", [(20, 20, 3), (10, 400, 3), (400, 10, 3), (1, 1, 3), (2000, 2000, 3)])
    def test_locate_box_stays_in_bounds(self, shape):
        """locate() must clamp its geometry box inside the image for ANY size/aspect --
        wide-short, tall-narrow, 1x1, huge -- for both bottom corners (br + bl)."""
        from remove_ai_watermarks._text_mark_engine import TextMarkEngine
        from remove_ai_watermarks.doubao_engine import _CONFIG as BR_CONFIG
        from remove_ai_watermarks.samsung_engine import _CONFIG as BL_CONFIG

        h, w = shape[:2]
        img = np.zeros(shape, np.uint8)
        for cfg in (BR_CONFIG, BL_CONFIG):
            loc = TextMarkEngine(cfg).locate(img)
            assert loc.x >= 0
            assert loc.y >= 0
            assert loc.x + loc.w <= w
            assert loc.y + loc.h <= h
            assert loc.w > 0
            assert loc.h > 0
