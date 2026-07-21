"""Tests for the Qwen (千问AI生成) visible-watermark engine (localize -> fill).

Every tuned constant in ``qwen_engine`` was measured on the 117-frame vendor
cohort (2026-07-21, ``scripts/vendor_mark_calibrate.py``); these tests pin the
load-bearing ones so a later "cleanup" cannot silently re-inherit Doubao's
geometry (the exact failure the calibration had to fix).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from remove_ai_watermarks import watermark_registry as registry
from remove_ai_watermarks.qwen_engine import (
    _ALPHA_HEIGHT_FRAC,
    _ALPHA_WIDTH_FRAC,
    _LADDER,
    QwenEngine,
    _alpha_template,
    _glyph_silhouette,
)

# The two measured size modes (fraction of the short side): a single fraction on
# the shared 3-rung ladder covers only ~75% of marks; the per-mark 2-rung ladder
# centres one rung on each mode.
_BIG_MODE, _SMALL_MODE = 0.203, 0.124
_MARGIN = 0.025  # measured right/bottom margin of the real mark


def _compose(w: int, h: int, mode: float = _BIG_MODE, bg: float = 100.0):
    """Composite the Qwen silhouette at a measured size mode onto a flat bg."""
    img = np.full((h, w, 3), bg, np.float32)
    at = _alpha_template()
    short = min(w, h)
    gw = int(mode * short)
    gh = max(4, int(mode * (_ALPHA_HEIGHT_FRAC / _ALPHA_WIDTH_FRAC) * short))
    margin = int(_MARGIN * short)
    ax = w - margin - gw
    ay = h - margin - gh
    amap = np.zeros((h, w), np.float32)
    amap[ay : ay + gh, ax : ax + gw] = cv2.resize(at, (gw, gh))
    a3 = amap[:, :, None]
    wm = (a3 * 255.0 + (1 - a3) * img).clip(0, 255).astype(np.uint8)
    return wm, amap > 0.2


class TestLocate:
    def test_box_anchored_bottom_right_off_the_corner(self):
        # The measured right margin (~0.025 of short) is wider than Doubao's 0.004;
        # inheriting Doubao's anchor clipped the first glyph (0.73 -> 0.26 NCC).
        eng = QwenEngine()
        img = np.zeros((2048, 2048, 3), np.uint8)
        loc = eng.locate(img)
        assert 2048 - (loc.x + loc.w) == pytest.approx(2048 * 0.0203, rel=0.15)
        assert 2048 - (loc.y + loc.h) == pytest.approx(2048 * 0.0218, rel=0.15)

    def test_box_scales_with_short_side_not_width(self):
        # scale_basis="short" (measured: frac_short CV 0.189 vs width 0.273).
        eng = QwenEngine()
        landscape = eng.locate(np.zeros((640, 1280, 3), np.uint8))
        wider = eng.locate(np.zeros((640, 2560, 3), np.uint8))
        assert wider.w == landscape.w  # same short side -> same box
        bigger = eng.locate(np.zeros((1280, 1920, 3), np.uint8))  # 2x the short side
        assert bigger.w == pytest.approx(landscape.w * 2, rel=0.05)


class TestConfig:
    def test_per_mark_ladder_and_shared_default_untouched(self):
        # Qwen's two size modes need their own 2-rung ladder; every other mark must
        # keep the shipped 3-rung default (the field's whole point is per-mark).
        assert _LADDER == (0.78, 1.27)
        assert QwenEngine().config.ladder == (0.78, 1.27)
        from remove_ai_watermarks.doubao_engine import _CONFIG as db
        from remove_ai_watermarks.jimeng_engine import _CONFIG as jm
        from remove_ai_watermarks.samsung_engine import _CONFIG as ss

        assert db.ladder == jm.ladder == ss.ladder == (0.8, 1.0, 1.25)

    def test_strict_only_no_provenance_relaxation(self):
        # The sub-gate band is dominated by non-Qwen banners on same-cohort frames,
        # so the relaxed arm was measured to be mostly false fills: factor pinned 1.0.
        assert QwenEngine().config.provenance_ncc_factor == 1.0

    def test_registry_row(self):
        mark = registry.get_mark("qwen")
        assert mark.location == "bottom-right"
        assert "千问AI生成" in mark.label
        assert mark.in_auto


class TestDetect:
    def test_clean_gradient_not_detected(self):
        eng = QwenEngine()
        ramp = np.tile(np.linspace(0, 255, 1024, dtype=np.uint8), (1024, 1))
        img = cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR)
        assert not eng.detect(img).detected

    def test_solid_blob_corner_not_detected(self):
        eng = QwenEngine()
        img = np.zeros((1024, 1024, 3), np.uint8)
        x, y, bw, bh = eng.locate(img).bbox
        img[y + bh // 4 : y + bh * 3 // 4, x : x + bw // 2] = 200
        assert not eng.detect(img).detected

    def test_silhouette_loads(self):
        sil = _glyph_silhouette()
        assert sil is not None
        assert set(np.unique(sil)).issubset({0, 255})

    @pytest.mark.parametrize("mode", [_BIG_MODE, _SMALL_MODE])
    def test_both_size_modes_detected(self, mode):
        # The registration's core claim: a mark at EITHER measured mode scores over
        # the gate (a single fraction on the shared ladder lost the small mode).
        # The floor is deliberately far above the gate: the synthetic mark is clean,
        # so it scores ~0.88/~0.94 when the geometry is right, but ~0.49 with Doubao's
        # box margins (the first glyph is clipped) and ~0.67 on the shared 3-rung
        # ladder -- the floor is what makes this test discriminate both regressions
        # (every variant passes a bare gate check on the synthetic).
        wm, _ = _compose(853, 640, mode=mode)
        det = QwenEngine().detect(wm)
        assert det.detected
        assert det.confidence >= 0.80

    def test_small_image_guarded(self):
        wm, _ = _compose(853, 640)
        eng = QwenEngine()
        assert eng.detect(wm).detected
        assert not eng.detect(cv2.resize(wm, (150, 112))).detected


class TestFootprintMaskAndRemoval:
    @pytest.mark.parametrize("mode", [_BIG_MODE, _SMALL_MODE])
    def test_removes_composed_mark_at_both_modes(self, mode):
        wm, mark = _compose(853, 640, mode=mode)
        assert float(np.abs(wm.astype(np.float32)[mark] - 100.0).mean()) > 15  # mark visible
        assert QwenEngine().detect(wm).detected
        out, region = registry.get_mark("qwen").remove(wm, backend="cv2")
        assert region is not None
        assert not QwenEngine().detect(out).detected
        h, w = wm.shape[:2]
        assert np.array_equal(out[: h // 2, : w // 2], wm[: h // 2, : w // 2])  # far region exact

    def test_footprint_mask_in_bottom_right(self):
        wm, _ = _compose(853, 640)
        mask = QwenEngine().footprint_mask(wm)
        assert mask is not None
        ys, xs = np.where(mask > 0)
        assert ys.mean() > wm.shape[0] / 2
        assert xs.mean() > wm.shape[1] / 2

    def test_clean_frame_produces_no_mask(self):
        clean = cv2.GaussianBlur(np.full((640, 853, 3), 120, np.uint8), (5, 5), 0)
        assert QwenEngine().footprint_mask(clean, force=False) is None
