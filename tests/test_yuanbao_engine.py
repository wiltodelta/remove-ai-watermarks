"""Tests for the Tencent Yuanbao (元宝 / AI生成) visible-watermark engine."""

from __future__ import annotations

import cv2
import numpy as np

from remove_ai_watermarks import watermark_registry as registry
from remove_ai_watermarks.yuanbao_engine import (
    _ALPHA_HEIGHT_FRAC,
    _ALPHA_WIDTH_FRAC,
    YuanbaoEngine,
    _alpha_template,
)

_MARK_WIDTH_FRAC = 0.08
_RIGHT_MARGIN_FRAC = 0.028
_BOTTOM_MARGIN_FRAC = 0.031


def _compose(w: int, h: int, *, bg: float, foreground: float):
    """Composite the synthetic Yuanbao silhouette at its measured geometry."""
    image = np.full((h, w, 3), bg, np.float32)
    alpha = _alpha_template()
    assert alpha is not None
    short = min(w, h)
    gw = int(_MARK_WIDTH_FRAC * short)
    gh = max(6, int((_ALPHA_HEIGHT_FRAC / _ALPHA_WIDTH_FRAC) * gw))
    ax = w - int(_RIGHT_MARGIN_FRAC * short) - gw
    ay = h - int(_BOTTOM_MARGIN_FRAC * short) - gh
    mark_alpha = np.zeros((h, w), np.float32)
    mark_alpha[ay : ay + gh, ax : ax + gw] = cv2.resize(alpha, (gw, gh))
    a3 = mark_alpha[:, :, None]
    composed = (a3 * foreground + (1 - a3) * image).clip(0, 255).astype(np.uint8)
    return composed, (ax, ay, gw, gh)


class TestConfig:
    def test_uses_two_polarity_contrast_frontend(self):
        assert YuanbaoEngine().config.detect_frontend == "contrast"

    def test_strict_only(self):
        assert YuanbaoEngine().config.provenance_ncc_factor == 1.0

    def test_registry_row(self):
        mark = registry.get_mark("yuanbao")
        assert mark.location == "bottom-right"
        assert "元宝" in mark.label
        assert mark.in_auto


class TestDetectAndMask:
    def test_detects_light_mark_on_dark_background(self):
        watermark, _ = _compose(1024, 1024, bg=60, foreground=230)
        detection = YuanbaoEngine().detect(watermark)
        assert detection.detected
        assert detection.confidence >= 0.80

    def test_detects_dark_mark_on_light_background(self):
        """Yuanbao switches mark polarity with the background.

        A white top-hat alone misses the dark-gray stamp used on pale scenes.
        """
        watermark, _ = _compose(1024, 1024, bg=230, foreground=110)
        detection = YuanbaoEngine().detect(watermark)
        assert detection.detected
        assert detection.confidence >= 0.80

    def test_clean_gradient_stays_quiet(self):
        ramp = np.tile(np.linspace(40, 220, 1024, dtype=np.uint8), (1024, 1))
        image = cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR)
        assert not YuanbaoEngine().detect(image).detected

    def test_match_must_hug_bottom_right_anchor(self):
        # Shift the mark 60 px in from the corner: still inside the locate window, so
        # the sweep finds it, but beyond the 0.04 anchor margin, so only the anchor
        # gate can demote it. A far shift would fall outside the window and pass
        # with or without the gate.
        class Ungated(YuanbaoEngine):
            def _post_gate(self, det, scan):
                return det

        watermark, (ax, ay, gw, gh) = _compose(1024, 1024, bg=60, foreground=230)
        assert YuanbaoEngine().detect(watermark).detected
        shifted = np.full_like(watermark, 60)
        shifted[ay - 60 : ay - 60 + gh, ax - 60 : ax - 60 + gw] = watermark[ay : ay + gh, ax : ax + gw]
        assert Ungated().detect(shifted).detected
        assert not YuanbaoEngine().detect(shifted).detected

    def test_mask_uses_detector_box_for_dark_mark(self):
        watermark, (ax, ay, gw, gh) = _compose(1024, 1024, bg=230, foreground=110)
        mask = YuanbaoEngine().footprint_mask(watermark)
        assert mask is not None
        ys, xs = np.where(mask > 0)
        assert xs.min() <= ax + int(0.05 * gw)
        assert xs.max() >= ax + gw - int(0.05 * gw)
        assert ys.min() <= ay + gh // 2 <= ys.max()

    def test_remove_clears_detector(self):
        watermark, _ = _compose(1024, 1024, bg=60, foreground=230)
        output, region = registry.get_mark("yuanbao").remove(watermark, backend="cv2")
        assert region is not None
        assert not YuanbaoEngine().detect(output).detected

    def test_corner_bars_and_caption_envelope_demoted_by_two_line_detail_gate(self):
        """Horizontal corner bars/captions inside the 0.04 anchor clear raw contrast NCC
        (>0.40) on 1D envelope alone, and must be demoted by YuanbaoEngine._post_gate."""
        from remove_ai_watermarks._text_mark_engine import BottomRightAnchoredEngine, TextMarkEngine

        class AnchorOnly(YuanbaoEngine):
            def _post_gate(self, det, scan):
                return BottomRightAnchoredEngine._post_gate(self, det, scan)

        h, w = 583, 780
        decoy = np.zeros((h, w, 3), np.uint8)
        decoy[:, :] = (20, 120, 245)  # saturated backdrop outside the low-sat shadow bars
        decoy[h - 33 : h - 23, w - 49 : w - 22] = (18, 18, 18)
        decoy[h - 19 : h - 9, w - 58 : w - 14] = (18, 18, 18)

        ungated = AnchorOnly().detect(decoy)
        assert ungated.detected
        assert ungated.template_asset == "yuanbao_alpha.png"
        assert ungated.confidence >= 0.40

        eng = YuanbaoEngine()
        base_strict, base_relaxed = TextMarkEngine.detect_both(eng, decoy)
        assert not base_strict.detected
        assert not base_relaxed.detected
        assert base_strict.confidence == ungated.confidence

        mark = registry.get_mark("yuanbao")
        strict, relaxed = mark.detect_both(decoy)
        det_strict = mark.detect(decoy, provenance=False)
        det_relaxed = mark.detect(decoy, provenance=True)
        assert not strict.detected
        assert (strict.detected, strict.confidence) == (det_strict.detected, det_strict.confidence)
        assert (relaxed.detected, relaxed.confidence) == (det_relaxed.detected, det_relaxed.confidence)
        assert mark.localize(decoy).mask is None
        assert registry.decide(registry._build_candidates(decoy), registry.Context()) == []
