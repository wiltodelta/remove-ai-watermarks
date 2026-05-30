"""Tests for the Doubao visible-watermark engine (reverse-alpha only)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from remove_ai_watermarks.doubao_engine import (
    _ALPHA_HEIGHT_FRAC,
    _ALPHA_LOGO_BGR,
    _ALPHA_MARGIN_BOTTOM_FRAC,
    _ALPHA_MARGIN_RIGHT_FRAC,
    _ALPHA_NATIVE_WIDTH,
    _ALPHA_WIDTH_FRAC,
    DETECT_NCC_THRESHOLD,
    DoubaoEngine,
    _alpha_template,
    _glyph_silhouette,
    _template_match_score,
    load_image_bgr,
)

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "doubao-1.png"


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


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample image not present")
class TestRealSample:
    def test_detects_watermark(self):
        det = DoubaoEngine().detect(load_image_bgr(SAMPLE))
        assert det.detected
        assert det.confidence >= DETECT_NCC_THRESHOLD

    def test_reverse_alpha_removes_mark(self):
        eng = DoubaoEngine()
        img = load_image_bgr(SAMPLE)
        assert eng.reverse_alpha_available(img)  # sample is at the captured width
        out = eng.remove_watermark_reverse_alpha(img)
        assert not eng.detect(out).detected  # mark gone after recovery

    def test_far_region_untouched(self):
        eng = DoubaoEngine()
        img = load_image_bgr(SAMPLE)
        out = eng.remove_watermark_reverse_alpha(img)
        h, w = img.shape[:2]
        assert np.array_equal(img[: h // 2, : w // 2], out[: h // 2, : w // 2])


# ── Reverse-alpha (exact recovery) ──────────────────────────────────


class TestReverseAlpha:
    def test_alpha_asset_loads(self):
        at = _alpha_template()
        assert at is not None
        assert at.dtype.kind == "f"
        assert float(at.min()) >= 0.0
        assert float(at.max()) <= 1.0

    def test_availability_gated_by_width(self):
        eng = DoubaoEngine()
        native = np.zeros((_ALPHA_NATIVE_WIDTH, _ALPHA_NATIVE_WIDTH, 3), np.uint8)
        far = np.zeros((1024, 1024, 3), np.uint8)  # ratio 0.5 -> out of band
        assert eng.reverse_alpha_available(native)
        assert not eng.reverse_alpha_available(far)

    def test_recovers_flat_background(self):
        """Compose the real alpha onto a flat background, then recover it."""
        eng = DoubaoEngine()
        w = _ALPHA_NATIVE_WIDTH
        bg = 100.0
        img = np.full((w, w, 3), bg, np.float32)
        at = _alpha_template()
        gw, gh = int(_ALPHA_WIDTH_FRAC * w), int(_ALPHA_HEIGHT_FRAC * w)
        ax = w - int(_ALPHA_MARGIN_RIGHT_FRAC * w) - gw
        ay = w - int(_ALPHA_MARGIN_BOTTOM_FRAC * w) - gh
        amap = np.zeros((w, w), np.float32)
        amap[ay : ay + gh, ax : ax + gw] = cv2.resize(at, (gw, gh))
        a3 = amap[:, :, None]
        logo = np.array(_ALPHA_LOGO_BGR, np.float32)
        wm = (a3 * logo + (1 - a3) * img).clip(0, 255).astype(np.uint8)
        mark = amap > 0.2
        assert float(np.abs(wm.astype(np.float32)[mark] - bg).mean()) > 15
        out = eng.remove_watermark_reverse_alpha(wm).astype(np.float32)
        assert float(np.abs(out[mark] - bg).mean()) < 6
