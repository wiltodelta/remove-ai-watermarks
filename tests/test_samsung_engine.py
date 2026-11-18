"""Tests for the Samsung Galaxy AI visible-watermark engine.

Detection and removal use both controlled flat provider captures and marks synthesized
from the bundled alpha asset. The mark is anchored bottom-LEFT (unlike the bottom-right
Doubao/Jimeng marks).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from remove_ai_watermarks import watermark_registry as registry
from remove_ai_watermarks.region_eraser import erase_cv2
from remove_ai_watermarks.samsung_engine import (
    _ALPHA_HEIGHT_FRAC,
    _ALPHA_NATIVE_WIDTH,
    _ALPHA_WIDTH_FRAC,
    DETECT_NCC_THRESHOLD,
    SamsungEngine,
    _alpha_template,
    _glyph_silhouette,
    _template_match_score,
)


def _compose(w: int, h: int, bg: float | np.ndarray = 100.0):
    """Composite the real alpha (scaled to width ``w``) onto ``bg`` by the
    engine's fixed bottom-left geometry. Returns ``(watermarked_uint8, mark_bool_mask)``."""
    img = np.broadcast_to(bg, (h, w, 3)).astype(np.float32)
    at = _alpha_template()
    assert at is not None
    gw, gh = int(_ALPHA_WIDTH_FRAC * w), int(_ALPHA_HEIGHT_FRAC * w)
    margin = int(0.015 * w)
    ax = margin
    ay = h - margin - gh
    amap = np.zeros((h, w), np.float32)
    amap[ay : ay + gh, ax : ax + gw] = cv2.resize(at, (gw, gh))
    a3 = amap[:, :, None]
    wm = (a3 * 255.0 + (1 - a3) * img).clip(0, 255).astype(np.uint8)
    return wm, amap > 0.15


class TestLocate:
    def test_box_anchored_bottom_left(self):
        eng = SamsungEngine()
        img = np.zeros((1448, 1086, 3), np.uint8)
        loc = eng.locate(img)
        assert loc.x < int(1086 * 0.03)  # hugs the left edge
        assert 1448 - (loc.y + loc.h) < int(1086 * 0.03)  # hugs the bottom

    def test_box_scales_with_width(self):
        eng = SamsungEngine()
        small = eng.locate(np.zeros((1024, 1024, 3), np.uint8))
        large = eng.locate(np.zeros((2048, 2048, 3), np.uint8))
        assert large.w == pytest.approx(small.w * 2, rel=0.1)


class TestDetect:
    def test_clean_gradient_not_detected(self):
        eng = SamsungEngine()
        ramp = np.tile(np.linspace(0, 255, 1086, dtype=np.uint8), (1086, 1))
        img = cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR)
        assert not eng.detect(img).detected

    def test_solid_blob_corner_not_detected(self):
        """A bright blob is not the glyph shape -> low correlation, not detected."""
        eng = SamsungEngine()
        img = np.zeros((1086, 1086, 3), np.uint8)
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

    def test_small_image_guarded_from_false_positive(self):
        """Below the minimum short side a tiny geometric shape spuriously NCC-matches
        the glyph silhouette (the 2026-06-26 small-icon FP class). The size guard
        suppresses detection there. Bracket it: a real mark is detected at native
        size, but the same content downscaled below the guard is not."""
        wm, _mark = _compose(_ALPHA_NATIVE_WIDTH, int(_ALPHA_NATIVE_WIDTH * 1.33))
        eng = SamsungEngine()
        assert eng.detect(wm).detected  # native: real mark detected
        assert not eng.detect(cv2.resize(wm, (150, 150))).detected  # below guard: suppressed

    def test_synthetic_mark_detected(self):
        """A watermark composed from the real alpha is detected at its threshold."""
        eng = SamsungEngine()
        wm, _mark = _compose(_ALPHA_NATIVE_WIDTH, int(_ALPHA_NATIVE_WIDTH * 1.33))
        det = eng.detect(wm)
        assert det.detected
        assert det.confidence >= DETECT_NCC_THRESHOLD

    def test_archival_postcard_caption_decoy_not_detected(self):
        """A bottom-left archival postcard caption strip (0.40-0.44 NCC) stays rejected.

        Constructed deterministically from integer bitmap blocks (no cv2.putText, whose
        Hershey glyph rasterization differs between OpenCV 4 and 5).
        """
        h, w = 583, 780
        yy, xx = np.ogrid[:h, :w]
        bg = (110 + 25 * np.sin(xx / 47.0) + 20 * np.cos(yy / 39.0)).clip(20, 235).astype(np.uint8)
        card = np.repeat(bg[:, :, None], 3, axis=2)
        bar_top, bar_bot, left_b = h - 64, h - 14, 11
        card[bar_top:, :] = 12
        card[bar_top:bar_bot, left_b : w - left_b] = 244
        pattern = "101 010 100 111 101 111 010 101 111  011 111 110 010 111 011  101 010  111 111 111"
        strip = np.concatenate([np.full((14, 2), 25 if c == "1" else 244, np.uint8) for c in pattern], axis=1)
        cy, cx = bar_top + 9, left_b + 54
        card[cy : cy + strip.shape[0], cx : cx + strip.shape[1]] = strip[:, :, None]

        mark = registry.get_mark("samsung")
        loc = mark.localize(card)
        assert 0.40 <= loc.confidence < DETECT_NCC_THRESHOLD
        assert not loc.detected
        assert loc.mask is None
        assert registry.decide(registry._build_candidates(card), registry.Context()) == []

    def test_provenance_keeps_samsung_genai_gate_at_040_without_relaxing_korean_variant(self):
        """samsung_genai metadata relaxes the Italian wordmark back to 0.40 and aligns its
        sparse mask, while the Korean variant stays strict-only at 0.46."""
        eng = SamsungEngine()
        h, w = 1448, 1086
        yy, xx = np.ogrid[:h, :w]
        tex = 125.0 + 68.0 * np.sin(xx / 4.5) * np.cos(yy / 4.5)
        wm, _ = _compose(w, h, bg=tex.clip(0, 255).astype(np.float32)[:, :, None])

        strict, relaxed = eng.detect_both(wm)
        assert 0.40 <= strict.confidence < DETECT_NCC_THRESHOLD
        assert not strict.detected
        assert relaxed.detected

        uncorroborated = registry.remove_auto_marks_detailed(wm, provenance=frozenset(), backend="cv2")
        assert uncorroborated.marks == ()
        corroborated = registry.remove_auto_marks_detailed(wm, provenance=frozenset({"samsung"}), backend="cv2")
        assert [m.key for m in corroborated.marks] == ["samsung"]
        assert corroborated.marks[0].status == "cleaned"


class TestAlphaAssetAndRemoval:
    def test_alpha_asset_loads(self):
        at = _alpha_template()
        assert at is not None
        assert at.dtype.kind == "f"
        assert float(at.min()) >= 0.0
        assert float(at.max()) <= 1.0

    def test_footprint_mask_in_bottom_left(self):
        wm, _mark = _compose(_ALPHA_NATIVE_WIDTH, int(_ALPHA_NATIVE_WIDTH * 1.33))
        mask = SamsungEngine().footprint_mask(wm)
        assert mask is not None
        assert mask.shape == wm.shape[:2]
        ys, xs = np.where(mask > 0)
        assert ys.mean() > wm.shape[0] / 2  # bottom
        assert xs.mean() < wm.shape[1] / 2  # left

    def test_removes_synthetic_mark(self):
        """localize -> cv2 fill clears the composed mark (re-detect no longer fires)."""
        wm, _mark = _compose(_ALPHA_NATIVE_WIDTH, int(_ALPHA_NATIVE_WIDTH * 1.33))
        assert SamsungEngine().detect(wm).detected
        out, region = registry.get_mark("samsung").remove(wm, backend="cv2")
        assert region is not None
        assert not SamsungEngine().detect(out).detected

    def test_real_gray_capture_uses_a_sparse_detector_aligned_footprint(self):
        """The faint real glyph must not turn its whole text rectangle into a hole."""
        path = Path(__file__).parents[1] / "data/calibration/samsung/samsung_gray_1.png"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        eng = SamsungEngine()
        det = eng.detect(image)

        assert det.detected
        assert det.match_box is not None
        mask = eng.footprint_mask(image, detection=det)
        assert mask is not None
        assert np.count_nonzero(mask) < 10_000

        out = erase_cv2(image, mask)
        loc = eng.locate(image)
        roi = np.s_[loc.y : loc.y + loc.h, loc.x : loc.x + loc.w]
        clean_color = np.median(image[: image.shape[0] // 2].reshape(-1, 3), axis=0)
        before_mae = float(np.mean(np.abs(image[roi].astype(np.float32) - clean_color)))
        after_mae = float(np.mean(np.abs(out[roi].astype(np.float32) - clean_color)))
        assert after_mae < before_mae * 0.25
        assert not eng.detect(out).detected

    @pytest.mark.parametrize(
        ("w", "h"),
        [
            (_ALPHA_NATIVE_WIDTH, int(_ALPHA_NATIVE_WIDTH * 1.33)),  # captured width
            (2958, 4054),  # real-photo width (~2.7x native) -> template-free footprint generalizes
        ],
    )
    def test_fill_removes_and_leaves_far_region(self, w, h):
        wm, mark = _compose(w, h)
        assert float(np.abs(wm.astype(np.float32)[mark] - 100.0).mean()) > 15  # mark visible
        before = SamsungEngine().detect(wm)
        out, _ = registry.get_mark("samsung").remove(wm, backend="cv2")
        assert SamsungEngine().detect(out).confidence < before.confidence
        # The mark is bottom-left; the opposite (top-right) corner stays exact.
        assert np.array_equal(out[: h // 2, w // 2 :], wm[: h // 2, w // 2 :])


class TestDegenerateAndChannelInputs:
    """footprint_mask must not crash on degenerate sizes or non-3-channel inputs."""

    @pytest.mark.parametrize(("w", "h"), [(2048, 1), (1, 2048), (2048, 8)])
    def test_wide_short_does_not_raise(self, w, h):
        eng = SamsungEngine()
        img = np.zeros((h, w, 3), np.uint8)
        mask = eng.footprint_mask(img, force=True)
        assert mask is None or mask.shape == (h, w)

    def test_grayscale_2d_does_not_raise(self):
        eng = SamsungEngine()
        gray = np.zeros((1448, 1086), np.uint8)
        mask = eng.footprint_mask(gray, force=True)
        assert mask is None or mask.shape == (1448, 1086)

    def test_bgra_4channel_does_not_raise(self):
        eng = SamsungEngine()
        bgra = np.zeros((1448, 1086, 4), np.uint8)
        mask = eng.footprint_mask(bgra, force=True)
        assert mask is None or mask.shape == (1448, 1086)
