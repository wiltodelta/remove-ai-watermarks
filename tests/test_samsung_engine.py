"""Tests for the Samsung Galaxy AI visible-badge engine.

Synthetic fixtures only -- real Galaxy AI badge images are user content and not
shipped (public repo), mirroring the Grok/Doubao byte-blob discipline. The
synthetic badge reproduces the badge's signature (a thin, left-anchored,
low-saturation horizontal strip) without any real image.
"""

from __future__ import annotations

import numpy as np

from remove_ai_watermarks.samsung_engine import SamsungEngine, _badge_structure


def _draw_badge_strip(eng: SamsungEngine, h: int = 1024, w: int = 1024) -> np.ndarray:
    """Dark image with a text-like row of thin gray strokes in the bottom-left
    box -- mimics the badge's glyph signature (the top-hat mask keys on thin
    strokes, not a solid bar)."""
    img = np.full((h, w, 3), 120, np.uint8)  # mid-gray background (a flat photo area)
    x, y, bw, bh = eng.locate(img).bbox
    cy = y + bh // 2
    sh = max(8, bh // 3)  # one text line, ~third of the box height
    x0 = x + int(bw * 0.04)  # starts near the left edge
    for i in range(30):  # ~one row of letter strokes, brighter than the background
        sx = x0 + i * 14
        img[cy - sh // 2 : cy + sh // 2, sx : sx + 4] = 210
    return img


class TestLocate:
    def test_box_anchored_bottom_left(self):
        eng = SamsungEngine()
        loc = eng.locate(np.zeros((1024, 1024, 3), np.uint8))
        assert loc.x < int(1024 * 0.03)  # hugs the left edge
        assert 1024 - (loc.y + loc.h) < int(1024 * 0.03)  # hugs the bottom

    def test_box_scales_with_width(self):
        eng = SamsungEngine()
        small = eng.locate(np.zeros((512, 512, 3), np.uint8))
        large = eng.locate(np.zeros((1024, 1024, 3), np.uint8))
        assert large.w == small.w * 2 or abs(large.w - small.w * 2) <= 2


class TestDetect:
    def test_badge_strip_detected(self):
        eng = SamsungEngine()
        assert eng.detect(_draw_badge_strip(eng)).detected

    def test_clean_image_not_detected(self):
        eng = SamsungEngine()
        assert not eng.detect(np.zeros((1024, 1024, 3), np.uint8)).detected

    def test_centered_strip_not_detected(self):
        """A strip not anchored to the left edge fails the left-start gate."""
        eng = SamsungEngine()
        img = np.full((1024, 1024, 3), 120, np.uint8)
        x, y, bw, bh = eng.locate(img).bbox
        cy = y + bh // 2
        for i in range(15):  # strokes in the right half of the box, not left-anchored
            sx = x + int(bw * 0.5) + i * 14
            img[cy - bh // 6 : cy + bh // 6, sx : sx + 4] = 210
        assert not eng.detect(img).detected

    def test_full_box_fill_not_detected(self):
        """A solid low-saturation fill (no thin band) is rejected by coverage."""
        eng = SamsungEngine()
        img = np.zeros((1024, 1024, 3), np.uint8)
        x, y, bw, bh = eng.locate(img).bbox
        img[y : y + bh, x : x + bw] = 130
        assert not eng.detect(img).detected


class TestRemove:
    def test_removal_reduces_badge_coverage(self):
        eng = SamsungEngine()
        img = _draw_badge_strip(eng)
        before = eng.detect(img).coverage
        after = eng.detect(eng.remove_watermark(img)).coverage
        assert after < before * 0.5

    def test_clean_image_returned_unchanged(self):
        eng = SamsungEngine()
        img = np.zeros((1024, 1024, 3), np.uint8)
        assert np.array_equal(img, eng.remove_watermark(img))

    def test_far_region_untouched(self):
        eng = SamsungEngine()
        img = _draw_badge_strip(eng)
        out = eng.remove_watermark(img)
        h, w = img.shape[:2]
        # top-right quadrant is far from the bottom-left badge
        assert np.array_equal(img[: h // 2, w // 2 :], out[: h // 2, w // 2 :])


class TestBadgeStructure:
    def test_thin_left_strip(self):
        box = np.zeros((56, 470), np.uint8)
        box[24:32, 10:300] = 255  # thin row starting at the left
        band, left = _badge_structure(box)
        assert band > 0.9
        assert left < 0.1

    def test_spread_fill_low_band(self):
        box = np.zeros((56, 470), np.uint8)
        box[:, :200] = 255  # fills full height -> not a thin band
        band, _ = _badge_structure(box)
        assert band < 0.7

    def test_empty(self):
        assert _badge_structure(np.zeros((56, 470), np.uint8)) == (0.0, 1.0)
