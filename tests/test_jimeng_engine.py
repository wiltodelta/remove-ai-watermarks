"""Tests for the Jimeng (即梦AI) visible-watermark engine.

No real Jimeng sample is committed (the captures are gitignored, repo is public),
so detection/removal is exercised against a watermark synthesized from the bundled
alpha asset itself -- self-consistent and download-free.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from remove_ai_watermarks.jimeng_engine import (
    _ALPHA_HEIGHT_FRAC,
    _ALPHA_LOGO_BGR,
    _ALPHA_MARGIN_BOTTOM_FRAC,
    _ALPHA_MARGIN_RIGHT_FRAC,
    _ALPHA_NATIVE_WIDTH,
    _ALPHA_WIDTH_FRAC,
    DETECT_NCC_THRESHOLD,
    JimengEngine,
    _alpha_template,
    _glyph_silhouette,
    _template_match_score,
)


def _compose(w: int, h: int, bg: float = 100.0):
    """Composite the real alpha (scaled to width ``w``) onto a flat bg by the
    engine's fixed geometry. Returns ``(watermarked_uint8, mark_bool_mask)``."""
    img = np.full((h, w, 3), bg, np.float32)
    at = _alpha_template()
    gw, gh = int(_ALPHA_WIDTH_FRAC * w), int(_ALPHA_HEIGHT_FRAC * w)
    ax = w - int(_ALPHA_MARGIN_RIGHT_FRAC * w) - gw
    ay = h - int(_ALPHA_MARGIN_BOTTOM_FRAC * w) - gh
    amap = np.zeros((h, w), np.float32)
    amap[ay : ay + gh, ax : ax + gw] = cv2.resize(at, (gw, gh))
    a3 = amap[:, :, None]
    wm = (a3 * np.array(_ALPHA_LOGO_BGR, np.float32) + (1 - a3) * img).clip(0, 255).astype(np.uint8)
    return wm, amap > 0.2


class TestLocate:
    def test_box_anchored_bottom_right(self):
        eng = JimengEngine()
        img = np.zeros((2048, 2048, 3), np.uint8)
        loc = eng.locate(img)
        assert 2048 - (loc.x + loc.w) < int(2048 * 0.03)
        assert 2048 - (loc.y + loc.h) < int(2048 * 0.03)

    def test_box_scales_with_width(self):
        eng = JimengEngine()
        small = eng.locate(np.zeros((1024, 1024, 3), np.uint8))
        large = eng.locate(np.zeros((2048, 2048, 3), np.uint8))
        assert large.w == pytest.approx(small.w * 2, rel=0.1)


class TestDetect:
    def test_clean_gradient_not_detected(self):
        eng = JimengEngine()
        ramp = np.tile(np.linspace(0, 255, 1024, dtype=np.uint8), (1024, 1))
        img = cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR)
        assert not eng.detect(img).detected

    def test_solid_blob_corner_not_detected(self):
        """A bright blob is not the glyph shape -> low correlation, not detected."""
        eng = JimengEngine()
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
        box = np.zeros((h + 8, int(w / _ALPHA_WIDTH_FRAC * 0.2) + w), np.uint8)
        box[4 : 4 + h, 4 : 4 + w] = sil
        assert _template_match_score(box, _ALPHA_NATIVE_WIDTH) >= DETECT_NCC_THRESHOLD
        solid = np.full_like(box, 255)
        assert _template_match_score(solid, _ALPHA_NATIVE_WIDTH) < DETECT_NCC_THRESHOLD

    def test_synthetic_mark_detected(self):
        """A watermark composed from the real alpha is detected at its threshold."""
        eng = JimengEngine()
        wm, _mark = _compose(_ALPHA_NATIVE_WIDTH, _ALPHA_NATIVE_WIDTH)
        det = eng.detect(wm)
        assert det.detected
        assert det.confidence >= DETECT_NCC_THRESHOLD


class TestReverseAlpha:
    def test_alpha_asset_loads(self):
        at = _alpha_template()
        assert at is not None
        assert at.dtype.kind == "f"
        assert float(at.min()) >= 0.0
        assert float(at.max()) <= 1.0

    def test_logo_is_white(self):
        assert _ALPHA_LOGO_BGR == (255.0, 255.0, 255.0)

    def test_available_whenever_asset_present(self):
        eng = JimengEngine()
        assert eng.reverse_alpha_available(np.zeros((1024, 1024, 3), np.uint8))
        assert eng.reverse_alpha_available(np.zeros((1440, 2560, 3), np.uint8))
        assert not eng.reverse_alpha_available(np.zeros((0, 0, 3), np.uint8))

    def test_removes_synthetic_mark(self):
        """Reverse-alpha + residual inpaint clears the composed mark (re-detect
        no longer fires)."""
        eng = JimengEngine()
        wm, _mark = _compose(_ALPHA_NATIVE_WIDTH, _ALPHA_NATIVE_WIDTH)
        assert eng.detect(wm).detected
        out = eng.remove_watermark_reverse_alpha(wm)
        assert not eng.detect(out).detected

    @pytest.mark.parametrize(
        ("w", "h", "max_err"),
        [
            (_ALPHA_NATIVE_WIDTH, _ALPHA_NATIVE_WIDTH, 4.0),  # captured width
            (1440, 2560, 8.0),  # off-native -> NCC alignment generalizes the capture
        ],
    )
    def test_recovers_flat_background(self, w, h, max_err):
        eng = JimengEngine()
        wm, mark = _compose(w, h)
        assert float(np.abs(wm.astype(np.float32)[mark] - 100.0).mean()) > 15  # mark visible
        out = eng.remove_watermark_reverse_alpha(wm).astype(np.float32)
        assert float(np.abs(out[mark] - 100.0).mean()) < max_err

    def test_far_region_untouched(self):
        """The residual inpaint only touches the bottom-right footprint; the
        opposite corner stays pixel-identical."""
        eng = JimengEngine()
        wm, _mark = _compose(_ALPHA_NATIVE_WIDTH, _ALPHA_NATIVE_WIDTH)
        out = eng.remove_watermark_reverse_alpha(wm)
        h, w = wm.shape[:2]
        assert np.array_equal(wm[: h // 2, : w // 2], out[: h // 2, : w // 2])

    def test_recovers_shifted_mark_on_texture(self):
        """A real mark is re-rasterized a few px off its fixed slot, so removal
        must NCC-align to it (a too-tight locate box would let a corner-ward shift
        escape the search and leave a readable outline). Composes the real alpha
        SHIFTED on a known texture and asserts the texture is recovered."""
        eng = JimengEngine()
        w = h = _ALPHA_NATIVE_WIDTH
        at = _alpha_template()
        gw, gh = int(_ALPHA_WIDTH_FRAC * w), int(_ALPHA_HEIGHT_FRAC * w)
        ax = w - int(_ALPHA_MARGIN_RIGHT_FRAC * w) - gw + 12  # shift toward the corner
        ay = h - int(_ALPHA_MARGIN_BOTTOM_FRAC * w) - gh + 8
        amap = np.zeros((h, w), np.float32)
        amap[ay : ay + gh, ax : ax + gw] = cv2.resize(at, (gw, gh))
        a3 = amap[:, :, None]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        base = 120 + 40 * np.sin(xx / 90.0) + 30 * np.cos(yy / 70.0)
        bg = np.clip(np.stack([base, base * 0.95, base * 1.05], axis=-1), 0, 255)
        wm = (a3 * np.array(_ALPHA_LOGO_BGR, np.float32) + (1 - a3) * bg).clip(0, 255).astype(np.uint8)
        mark = amap > 0.15
        assert float(np.abs(wm.astype(np.float32)[mark] - bg[mark]).mean()) > 30  # mark clearly visible
        out = eng.remove_watermark_reverse_alpha(wm).astype(np.float32)
        assert float(np.abs(out[mark] - bg[mark]).mean()) < 8.0  # texture recovered, no outline


class TestDegenerateAndChannelInputs:
    """Removal must not crash on degenerate sizes or non-3-channel inputs."""

    @pytest.mark.parametrize(("w", "h"), [(2048, 1), (1, 2048), (2048, 8)])
    def test_wide_short_does_not_raise(self, w, h):
        eng = JimengEngine()
        img = np.zeros((h, w, 3), np.uint8)
        out = eng.remove_watermark_reverse_alpha(img)
        assert out.shape == img.shape

    def test_grayscale_2d_does_not_raise(self):
        eng = JimengEngine()
        gray = np.zeros((2048, 2048), np.uint8)
        out = eng.remove_watermark_reverse_alpha(gray)
        assert out.shape == (2048, 2048, 3)

    def test_bgra_4channel_does_not_raise(self):
        eng = JimengEngine()
        bgra = np.zeros((2048, 2048, 4), np.uint8)
        out = eng.remove_watermark_reverse_alpha(bgra)
        assert out.shape == (2048, 2048, 3)
