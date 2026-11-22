"""Tests for the Genspark visible-watermark engine (localize -> fill).

All fixtures here are synthetic composites of the repository-rendered
``genspark_alpha.png`` silhouette (see ``scripts/render_vendor_silhouettes.py``
and ``genspark_engine.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest

from remove_ai_watermarks import watermark_registry as registry
from remove_ai_watermarks.genspark_engine import (
    _ALPHA_HEIGHT_FRAC,
    _ALPHA_NATIVE_SHORT_SIDE,
    _ALPHA_WIDTH_FRAC,
    DETECT_NCC_THRESHOLD,
    GensparkEngine,
    _alpha_template,
    _glyph_silhouette,
    _template_match_score,
)
from remove_ai_watermarks.identify import identify

if TYPE_CHECKING:
    from pathlib import Path


def _compose(w: int, h: int, bg: float = 100.0):
    """Composite the bundled alpha (scaled to the short side) at the frame center."""
    img = np.full((h, w, 3), bg, np.float32)
    at = _alpha_template()
    assert at is not None
    base = min(w, h)
    gw, gh = int(_ALPHA_WIDTH_FRAC * base), int(_ALPHA_HEIGHT_FRAC * base)
    ax = (w - gw) // 2
    ay = (h - gh) // 2
    amap = np.zeros((h, w), np.float32)
    amap[ay : ay + gh, ax : ax + gw] = cv2.resize(at, (gw, gh))
    a3 = amap[:, :, None] * 0.78
    wm = (a3 * 255.0 + (1.0 - a3) * img).clip(0, 255).astype(np.uint8)
    return wm, amap > 0.2


class TestLocate:
    def test_box_centered_in_frame(self):
        eng = GensparkEngine()
        loc = eng.locate(np.zeros((1280, 960, 3), np.uint8))
        cx, cy = loc.x + loc.w / 2, loc.y + loc.h / 2
        assert cx == pytest.approx(960 / 2, abs=960 * 0.02)
        assert cy == pytest.approx(1280 / 2, abs=1280 * 0.02)

    def test_box_scales_with_short_side(self):
        eng = GensparkEngine()
        small = eng.locate(np.zeros((960, 960, 3), np.uint8))
        large = eng.locate(np.zeros((1920, 1920, 3), np.uint8))
        assert large.w == pytest.approx(small.w * 2, rel=0.1)


class TestDetectAndRemove:
    def test_clean_and_random_texture_not_detected(self):
        eng = GensparkEngine()
        ramp = np.tile(np.linspace(0, 255, 960, dtype=np.uint8), (1280, 1))
        assert not eng.detect(cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR)).detected

        rng = np.random.default_rng(11)
        for _ in range(10):
            h, w = int(rng.integers(900, 1800)), int(rng.integers(900, 1800))
            noise = rng.integers(50, 210, (h, w, 3)).astype(np.uint8)
            assert not eng.detect(cv2.GaussianBlur(noise, (0, 0), sigmaX=7)).detected

    def test_silhouette_and_shape_match(self):
        sil = _glyph_silhouette()
        assert sil is not None
        assert set(np.unique(sil)).issubset({0, 255})
        h, w = sil.shape
        box = np.zeros((h + 12, w + 48), np.uint8)
        box[6 : 6 + h, 24 : 24 + w] = sil
        assert _template_match_score(box, _ALPHA_NATIVE_SHORT_SIDE) >= DETECT_NCC_THRESHOLD
        assert _template_match_score(np.full_like(box, 255), _ALPHA_NATIVE_SHORT_SIDE) < DETECT_NCC_THRESHOLD

    def test_detects_removes_and_promotes_identify(self, tmp_path: Path):
        wm, _mark = _compose(960, 1280)
        eng = GensparkEngine()
        before = eng.detect(wm)
        assert before.detected
        assert before.confidence >= DETECT_NCC_THRESHOLD
        assert not eng.detect(cv2.resize(wm, (150, 180))).detected

        img_path = tmp_path / "genspark.png"
        cv2.imwrite(str(img_path), wm)
        rep = identify(img_path, check_visible=True, check_invisible=False)
        assert rep.is_ai_generated is True
        assert rep.confidence == "medium"
        assert rep.platform == "Genspark (visible Genspark mark detected)"
        assert "visible_genspark" in {s.name for s in rep.signals}

        out, region = registry.get_mark("genspark").remove(wm, backend="cv2")
        assert region is not None
        assert not eng.detect(out).detected
        assert np.array_equal(out[:80, :80], wm[:80, :80])
