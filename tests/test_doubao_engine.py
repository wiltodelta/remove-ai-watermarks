"""Tests for the Doubao visible-watermark engine."""

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
    DoubaoEngine,
    _alpha_template,
    _glyph_structure,
    load_image_bgr,
)

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "doubao-1.png"


# ── Locate ──────────────────────────────────────────────────────────


class TestLocate:
    def test_box_anchored_bottom_right(self):
        eng = DoubaoEngine()
        img = np.zeros((2048, 2048, 3), np.uint8)
        loc = eng.locate(img)
        # right and bottom edges sit close to the image corner (within margins)
        assert 2048 - (loc.x + loc.w) < int(2048 * 0.03)
        assert 2048 - (loc.y + loc.h) < int(2048 * 0.03)
        assert loc.is_fallback  # geometry anchor, no bundled template yet

    def test_box_scales_with_width(self):
        eng = DoubaoEngine()
        small = eng.locate(np.zeros((1024, 1024, 3), np.uint8))
        large = eng.locate(np.zeros((2048, 2048, 3), np.uint8))
        # width-relative geometry: 2x wider image -> ~2x wider box
        assert large.w == pytest.approx(small.w * 2, rel=0.1)


# ── Detect + remove on the real sample ──────────────────────────────


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample image not present")
class TestRealSample:
    def test_detects_watermark(self):
        eng = DoubaoEngine()
        det = eng.detect(load_image_bgr(SAMPLE))
        assert det.detected
        assert det.confidence > 0.0
        assert det.coverage > 0.04

    def test_remove_reduces_glyph_coverage(self):
        eng = DoubaoEngine()
        img = load_image_bgr(SAMPLE)
        before = eng.detect(img).coverage
        out = eng.remove_watermark(img)
        after = eng.detect(out).coverage
        # the inpaint should clear most glyph pixels from the corner box
        assert after < before * 0.5

    def test_pixels_outside_box_untouched(self):
        eng = DoubaoEngine()
        img = load_image_bgr(SAMPLE)
        out = eng.remove_watermark(img)
        # top-left quadrant is far from the bottom-right mark: must be identical
        h, w = img.shape[:2]
        assert np.array_equal(img[: h // 2, : w // 2], out[: h // 2, : w // 2])


# ── Negative + safety guard ─────────────────────────────────────────


class TestNegativeAndGuard:
    def test_clean_image_not_detected(self):
        eng = DoubaoEngine()
        # smooth gradient, no watermark
        ramp = np.tile(np.linspace(0, 255, 1024, dtype=np.uint8), (1024, 1))
        img = cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR)
        det = eng.detect(img)
        assert not det.detected

    def test_clean_image_returned_unchanged(self):
        eng = DoubaoEngine()
        ramp = np.tile(np.linspace(0, 255, 1024, dtype=np.uint8), (1024, 1))
        img = cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR)
        out = eng.remove_watermark(img)
        assert np.array_equal(img, out)

    def _bright_corner(self, eng: DoubaoEngine, draw) -> np.ndarray:
        """Dark 1024² image with the bottom-right box painted by ``draw(box_view)``."""
        img = np.zeros((1024, 1024, 3), np.uint8)
        x, y, bw, bh = eng.locate(img).bbox
        draw(img[y : y + bh, x : x + bw])
        return img

    def test_solid_blob_corner_not_detected(self):
        """A single bright blob clears coverage but lacks text structure (one
        dominant component) -- the issue #23 false positive, now rejected."""
        eng = DoubaoEngine()

        def fill(box):
            box[box.shape[0] // 4 : box.shape[0] * 3 // 4, : box.shape[1] // 2] = 200

        assert not eng.detect(self._bright_corner(eng, fill)).detected

    def test_text_like_glyph_row_detected(self):
        """Several small bright bars in one horizontal row (glyph-like) pass the
        structural gate (many components, no dominant blob, banded)."""
        eng = DoubaoEngine()

        def glyphs(box):
            bh, bw = box.shape[:2]
            gh = int(bh * 0.55)
            gw = max(3, int(bw * 0.07))
            y0 = (bh - gh) // 2
            for i in range(6):
                cx = int(bw * (0.06 + i * 0.15))
                box[y0 : y0 + gh, cx : cx + gw] = 200

        assert eng.detect(self._bright_corner(eng, glyphs)).detected

    def test_glyph_structure_descriptors(self):
        # One filled blob -> 1 component, dominant, fully banded.
        blob = np.zeros((40, 200), np.uint8)
        blob[10:30, 20:120] = 255
        ncomp, top1, _ = _glyph_structure(blob)
        assert ncomp == 1
        assert top1 > 0.95
        # Several separated bars -> many components, none dominant.
        bars = np.zeros((40, 200), np.uint8)
        for i in range(6):
            bars[15:25, 10 + i * 30 : 16 + i * 30] = 255
        ncomp2, top1_2, _ = _glyph_structure(bars)
        assert ncomp2 >= 4
        assert top1_2 < 0.5
        assert _glyph_structure(np.zeros((40, 200), np.uint8)) == (0, 1.0, 0.0)

    def test_document_background_guard(self):
        """A dense high-frequency corner (document-like) trips the coverage
        guard, so the image is left untouched rather than smeared."""
        eng = DoubaoEngine()
        rng = np.random.default_rng(0)
        img = np.full((1024, 1024, 3), 255, np.uint8)
        # fill the bottom-right box area with random grayish text-like noise
        loc = eng.locate(img)
        x, y, bw, bh = loc.bbox
        noise = rng.integers(150, 246, size=(bh, bw), dtype=np.uint8)
        img[y : y + bh, x : x + bw] = noise[:, :, None]
        out = eng.remove_watermark(img)
        assert np.array_equal(img, out)


class TestReverseAlpha:
    """Exact reverse-alpha recovery from the bundled Doubao alpha map."""

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
        # the mark must be visible before removal, gone after
        mark = amap > 0.2
        assert float(np.abs(wm.astype(np.float32)[mark] - bg).mean()) > 15
        out = eng.remove_watermark_reverse_alpha(wm).astype(np.float32)
        assert float(np.abs(out[mark] - bg).mean()) < 6  # recovered close to flat bg
