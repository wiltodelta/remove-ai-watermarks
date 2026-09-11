"""Tests for the Kling CJK/Latin 3.0 visible-watermark engine (localize -> fill).

Every tuned constant in ``kling_engine`` was measured on the 30-frame vendor
cohort (2026-07-21, ``scripts/vendor_mark_calibrate.py``); these tests pin the
load-bearing ones so a later "cleanup" cannot silently re-inherit Doubao's
geometry or relax the measured strict-only gate.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from remove_ai_watermarks import watermark_registry as registry
from remove_ai_watermarks.kling_engine import (
    _ALPHA_HEIGHT_FRAC,
    _ALPHA_WIDTH_FRAC,
    _LATIN_CONFIG,
    KlingEngine,
    _alpha_template,
    _glyph_silhouette,
)

_MARK_FRAC = 0.12  # measured mark width, fraction of the short side (unimodal)
_MARGIN = 0.03  # measured right/bottom margin of the real mark


def _compose(w: int, h: int, mode: float = _MARK_FRAC, bg: float = 100.0):
    """Composite the Kling silhouette at the measured size onto a flat bg."""
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
    def test_box_anchored_bottom_right(self):
        eng = KlingEngine()
        img = np.zeros((2048, 2048, 3), np.uint8)
        loc = eng.locate(img)
        assert 2048 - (loc.x + loc.w) == pytest.approx(2048 * 0.03, rel=0.15)
        assert 2048 - (loc.y + loc.h) == pytest.approx(2048 * 0.023, rel=0.15)

    def test_box_scales_with_short_side_not_width(self):
        # scale_basis="short" (measured: mark_w/short 0.118-0.122 across orientations).
        eng = KlingEngine()
        landscape = eng.locate(np.zeros((640, 1280, 3), np.uint8))
        wider = eng.locate(np.zeros((640, 2560, 3), np.uint8))
        assert wider.w == landscape.w  # same short side -> same box
        bigger = eng.locate(np.zeros((1280, 1920, 3), np.uint8))  # 2x the short side
        assert bigger.w == pytest.approx(landscape.w * 2, rel=0.05)


class TestConfig:
    def test_shared_ladder_default(self):
        # The mark is unimodal at 0.12 of the short side, so Kling keeps the shared
        # 3-rung ladder (Qwen's per-mark ladder is the measured exception, not a norm).
        assert KlingEngine().config.ladder == (0.8, 1.0, 1.25)

    def test_strict_only_no_provenance_relaxation(self):
        # The sub-gate band (real Kling variants at 0.17-0.25) overlaps the clean
        # arm's top (p90 0.220), so a relaxed arm cannot separate: factor pinned 1.0.
        assert KlingEngine().config.provenance_ncc_factor == 1.0

    def test_gate_above_clean_arm_max(self):
        # Clean arm scored p99 0.304 / max 0.320 on 286 hand-labeled frames; the
        # gate must sit above that with margin.
        assert KlingEngine().config.detect_ncc_threshold > 0.32

    def test_latin_variant_keeps_its_measured_narrow_ladder(self):
        assert _LATIN_CONFIG.asset_name == "kling_latin_alpha.png"
        assert _LATIN_CONFIG.ladder == (0.9, 1.0, 1.1)
        assert _LATIN_CONFIG.detect_ncc_threshold == 0.40

    def test_registry_row(self):
        mark = registry.get_mark("kling")
        assert mark.location == "bottom-right"
        assert "可灵AI" in mark.label
        assert mark.in_auto


class TestDetect:
    def test_clean_gradient_not_detected(self):
        eng = KlingEngine()
        ramp = np.tile(np.linspace(0, 255, 1024, dtype=np.uint8), (1024, 1))
        img = cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR)
        assert not eng.detect(img).detected

    def test_solid_blob_corner_not_detected(self):
        eng = KlingEngine()
        img = np.zeros((1024, 1024, 3), np.uint8)
        x, y, bw, bh = eng.locate(img).bbox
        img[y + bh // 4 : y + bh * 3 // 4, x : x + bw // 2] = 200
        assert not eng.detect(img).detected

    def test_silhouette_loads(self):
        sil = _glyph_silhouette()
        assert sil is not None
        assert set(np.unique(sil)).issubset({0, 255})

    def test_composed_mark_detected(self):
        # The registration's core claim: a mark at the measured size scores over the
        # gate. The floor is deliberately far above the gate: the synthetic mark is
        # clean, so it scores high when the geometry is right.
        wm, _ = _compose(853, 640)
        det = KlingEngine().detect(wm)
        assert det.detected
        assert det.confidence >= 0.80

    def test_small_image_guarded(self):
        wm, _ = _compose(853, 640)
        eng = KlingEngine()
        assert eng.detect(wm).detected
        assert not eng.detect(cv2.resize(wm, (150, 112))).detected

    def test_metadata_only_kling_api_export_is_not_a_visible_mark(self):
        path = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "provenance" / "kling-image-v3.png"
        image = cv2.imread(str(path))
        assert image is not None
        assert not KlingEngine().detect(image).detected


class TestFootprintMaskAndRemoval:
    def test_removes_composed_mark(self):
        wm, mark = _compose(853, 640)
        assert float(np.abs(wm.astype(np.float32)[mark] - 100.0).mean()) > 15  # mark visible
        assert KlingEngine().detect(wm).detected
        out, region = registry.get_mark("kling").remove(wm, backend="cv2")
        assert region is not None
        assert not KlingEngine().detect(out).detected
        h, w = wm.shape[:2]
        assert np.array_equal(out[: h // 2, : w // 2], wm[: h // 2, : w // 2])  # far region exact

    def test_removes_complete_faint_canonical_mark(self):
        """A partial top-hat blob must not truncate the detector's full match."""
        path = Path(__file__).parents[1] / "data" / "fixtures" / "visible" / "kling" / "example.png"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert image is not None

        mark = registry.get_mark("kling")
        before = mark.detect(image)
        assert before.detected
        output, region = mark.remove(image, backend="cv2", detection=before)

        assert region is not None
        assert not KlingEngine().detect(output).detected

    def test_footprint_mask_in_bottom_right(self):
        wm, _ = _compose(853, 640)
        mask = KlingEngine().footprint_mask(wm)
        assert mask is not None
        ys, xs = np.where(mask > 0)
        assert ys.mean() > wm.shape[0] / 2
        assert xs.mean() > wm.shape[1] / 2

    def test_clean_frame_produces_no_mask(self):
        clean = cv2.GaussianBlur(np.full((640, 853, 3), 120, np.uint8), (5, 5), 0)
        assert KlingEngine().footprint_mask(clean, force=False) is None
