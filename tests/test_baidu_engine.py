"""Tests for the Baidu ("百度 AI生成") visible-watermark engine.

Every tuned constant in ``baidu_engine`` was measured on the 16-frame vendor
cohort (2026-07-22); these tests pin the load-bearing ones: detection keys on
the 百度 text run ONLY (the text+pill template was a measured bright-blob
magnet), the load-bearing Doubao rival margin, the strict-only gate, and the
corner-extended footprint (the tag's flat white interior gives no top-hat
response, so a blob-bbox mask leaves the tag as a ghost).
"""

from __future__ import annotations

import cv2
import numpy as np

from remove_ai_watermarks import watermark_registry as registry
from remove_ai_watermarks.baidu_engine import (
    _ALPHA_HEIGHT_FRAC,
    _ALPHA_WIDTH_FRAC,
    BaiduEngine,
    _alpha_template,
)

_TEXT_FRAC = 0.090  # measured 百度 text-run width, fraction of the short side
_TEXT_RIGHT = 0.099  # measured right margin of the text run (the tag is right of it)
_TAG_FRAC = 0.075  # the white tag's width, approx (text-right to corner)


def _compose(w: int, h: int, bg: float = 100.0):
    """Composite the 百度 text run + a solid white tag at the measured layout."""
    img = np.full((h, w, 3), bg, np.float32)
    at = _alpha_template()
    short = min(w, h)
    gw = int(_TEXT_FRAC * short)
    gh = max(4, int(_TEXT_FRAC * (_ALPHA_HEIGHT_FRAC / _ALPHA_WIDTH_FRAC) * short))
    margin_b = int(0.006 * short)
    ax = w - int(_TEXT_RIGHT * short) - gw
    ay = h - margin_b - gh
    amap = np.zeros((h, w), np.float32)
    amap[ay : ay + gh, ax : ax + gw] = cv2.resize(at, (gw, gh))
    # the white rounded tag between the text and the corner
    tx0 = w - int(0.015 * short) - int(_TAG_FRAC * short)
    amap[ay - gh // 8 : ay + gh + gh // 8, tx0 : w - int(0.015 * short)] = 1.0
    a3 = amap[:, :, None]
    wm = (a3 * 255.0 + (1 - a3) * img).clip(0, 255).astype(np.uint8)
    return wm, (ax, ay, gw, gh, tx0)


class TestLocate:
    def test_box_anchored_bottom_right(self):
        eng = BaiduEngine()
        img = np.zeros((2048, 2048, 3), np.uint8)
        loc = eng.locate(img)
        assert 2048 - (loc.x + loc.w) < 40
        assert 2048 - (loc.y + loc.h) < 40

    def test_box_scales_with_short_side(self):
        eng = BaiduEngine()
        landscape = eng.locate(np.zeros((640, 1280, 3), np.uint8))
        wider = eng.locate(np.zeros((640, 2560, 3), np.uint8))
        assert wider.w == landscape.w


class TestConfig:
    def test_tophat_frontend(self):
        assert BaiduEngine().config.detect_frontend == "tophat"

    def test_doubao_rival_margin(self):
        # 百度 vs 豆包 share a glyph and a corner: the candidate fires on 45.8% of
        # Doubao-marked frames at the gate, and the 0.10 margin suppresses ALL of
        # it at zero genuine-detection cost (crossfire, 2026-07-22).
        assert "doubao_alpha.png" in BaiduEngine().config.rivals

    def test_strict_only_no_provenance_relaxation(self):
        assert BaiduEngine().config.provenance_ncc_factor == 1.0

    def test_gate_above_clean_arm_max(self):
        # Compatibility testing separated true Baidu marks from unrelated
        # bottom-right text, so the gate sits at 0.48.
        assert BaiduEngine().config.detect_ncc_threshold >= 0.48

    def test_qwen_is_a_rival(self):
        # 百度 and 千问 are near-identical after binarization, so Qwen's template
        # must act as a rival.
        assert "qwen_alpha.png" in BaiduEngine().config.rivals

    def test_registry_row(self):
        mark = registry.get_mark("baidu")
        assert mark.location == "bottom-right"
        assert mark.in_auto


class TestDetectAndMask:
    def test_detects_composed_mark(self):
        eng = BaiduEngine()
        wm, _ = _compose(1024, 1024)
        det = eng.detect(wm)
        assert det.detected, f"composed mark missed (conf={det.confidence:.3f})"

    def test_clean_frame_stays_quiet(self):
        eng = BaiduEngine()
        img = np.full((1024, 1024, 3), 100, np.uint8)
        assert not eng.detect(img).detected

    def test_mask_extends_to_the_corner_tag(self):
        """Regression (2026-07-22): the tag's flat white interior gives no top-hat
        response, so a blob-bbox mask ended at the text run and the fill left the
        tag as a ghost. The footprint must extend right to the corner."""
        eng = BaiduEngine()
        wm, (ax, _ay, gw, _gh, tx0) = _compose(1024, 1024)
        mask = eng.footprint_mask(wm)
        assert mask is not None
        _ys, xs = np.where(mask > 0)
        assert xs.min() <= ax + int(0.1 * gw)  # covers the text run's left edge
        assert xs.max() >= tx0 + 10  # covers the white tag right of the text

    def test_no_mask_on_clean_frame(self):
        eng = BaiduEngine()
        img = np.full((1024, 1024, 3), 100, np.uint8)
        assert eng.footprint_mask(img) is None

    def test_force_masks_the_whole_locate_box_on_a_clean_frame(self):
        """``force`` takes priority over detection for this mark, unlike the base
        policy: a --no-detect caller named the mark, so the honest footprint is the
        whole geometry box even though nothing was detected."""
        eng = BaiduEngine()
        img = np.full((1024, 1024, 3), 100, np.uint8)
        mask = eng.footprint_mask(img, force=True)
        assert mask is not None
        bx, by, bw, bh = eng.locate(img).bbox
        ys, xs = np.where(mask > 0)
        assert xs.min() <= bx
        assert xs.max() >= bx + bw - 1
        assert ys.min() <= by
        assert ys.max() >= by + bh - 1
