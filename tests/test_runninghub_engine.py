"""Tests for the RunningHub ("RunningHub AI生成") visible-watermark engine.

Every tuned constant in ``runninghub_engine`` was measured on the 73-frame
vendor cohort (2026-07-22, ``scripts/vendor_cohort_harvest.py`` +
``scripts/vendor_mark_calibrate.py``); these tests pin the load-bearing ones:
the top-left corner, the gray front-end, the exact-size tight ladder, the
strict-only gate, and the mask/coverage parity regression (the partial-blob
"Runni" miss).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from remove_ai_watermarks import watermark_registry as registry
from remove_ai_watermarks.runninghub_engine import (
    _ALPHA_HEIGHT_FRAC,
    _ALPHA_WIDTH_FRAC,
    RunningHubEngine,
    _alpha_template,
)

_MARK_FRAC = 0.32  # measured mark width, fraction of the frame WIDTH


def _compose(w: int, h: int, mode: float = _MARK_FRAC, bg: float = 100.0):
    """Composite the RunningHub silhouette at the measured size, top-left."""
    img = np.full((h, w, 3), bg, np.float32)
    at = _alpha_template()
    gw = int(mode * w)
    gh = max(4, int(mode * (_ALPHA_HEIGHT_FRAC / _ALPHA_WIDTH_FRAC) * w))
    ax, ay = int(0.008 * w), int(0.006 * h)
    amap = np.zeros((h, w), np.float32)
    amap[ay : ay + gh, ax : ax + gw] = cv2.resize(at, (gw, gh))
    a3 = amap[:, :, None]
    wm = (a3 * 255.0 + (1 - a3) * img).clip(0, 255).astype(np.uint8)
    return wm, (ax, ay, gw, gh)


class TestLocate:
    def test_box_anchored_top_left(self):
        eng = RunningHubEngine()
        img = np.zeros((2048, 1536, 3), np.uint8)
        loc = eng.locate(img)
        assert loc.x < 40  # hugs the left edge
        assert loc.y < 40  # hugs the top edge (corner="tl")

    def test_box_scales_with_width(self):
        # scale_basis="width" (measured: mark width is 0.32 of the frame width).
        eng = RunningHubEngine()
        narrow = eng.locate(np.zeros((2048, 1024, 3), np.uint8))
        wide = eng.locate(np.zeros((2048, 2048, 3), np.uint8))
        assert wide.w == pytest.approx(narrow.w * 2, rel=0.05)


class TestConfig:
    def test_gray_frontend(self):
        # The mark is a faint mid-gray the top-hat suppresses to clean-arm levels;
        # the raw-grayscale front-end is what separates (measured 2026-07-22).
        assert RunningHubEngine().config.detect_frontend == "gray"

    def test_tight_ladder(self):
        # The NCC comb is razor-sharp in size (0.537 on-size, 0.223 at +5.6%), so
        # the nominal sits exactly on the measured 0.32 with +-5% rungs.
        assert RunningHubEngine().config.ladder == (0.95, 1.0, 1.05)
        assert RunningHubEngine().config.alpha_width_frac == pytest.approx(0.32)

    def test_strict_only_no_provenance_relaxation(self):
        assert RunningHubEngine().config.provenance_ncc_factor == 1.0

    def test_gate_above_clean_arm_max(self):
        # Clean arm scored p99 0.273 / max 0.295 on 286 hand-labeled frames.
        assert RunningHubEngine().config.detect_ncc_threshold > 0.295

    def test_registry_row(self):
        mark = registry.get_mark("runninghub")
        assert mark.location == "top-left"
        assert mark.in_auto


class TestDetectAndMask:
    def test_detects_composed_mark(self):
        eng = RunningHubEngine()
        wm, _ = _compose(1080, 1620)
        det = eng.detect(wm)
        assert det.detected, f"composed mark missed (conf={det.confidence:.3f})"

    def test_clean_frame_stays_quiet(self):
        eng = RunningHubEngine()
        img = np.full((1620, 1080, 3), 100, np.uint8)
        assert not eng.detect(img).detected

    def test_mask_covers_the_whole_mark(self):
        """Regression (2026-07-22): the binary blob under-segments the faint head
        glyphs, so a blob-bbox mask left "Runni" unremoved. The gray front-end's
        mask must come from the detector's own match box and cover the mark."""
        eng = RunningHubEngine()
        wm, (ax, ay, gw, gh) = _compose(1080, 1620)
        mask = eng.footprint_mask(wm)
        assert mask is not None
        ys, xs = np.where(mask > 0)
        assert xs.min() <= ax + int(0.05 * gw)  # covers the LEFT edge of the mark
        assert xs.max() >= ax + gw - int(0.05 * gw)
        assert ys.min() <= ay + gh // 2 <= ys.max()

    def test_no_mask_on_clean_frame(self):
        eng = RunningHubEngine()
        img = np.full((1620, 1080, 3), 100, np.uint8)
        assert eng.footprint_mask(img) is None

    def test_anchor_window_rejects_off_corner_match(self):
        """The raw-gray front-end false-fires on text-like structure ANYWHERE in
        the box during compatibility testing; the
        anchor window is what keeps it about THIS mark. A composed mark placed
        off the measured corner anchor must NOT be detected."""
        eng = RunningHubEngine()
        wm, _ = _compose(1080, 1620)
        det = eng.detect(wm)
        assert det.detected  # on-anchor control
        # the same mark shifted right/down, off the anchor window
        shifted = np.full((1620, 1080, 3), 100, np.uint8)
        region = wm[10:60, 12:360]
        shifted[100 : 100 + region.shape[0], 200 : 200 + region.shape[1]] = region
        assert not eng.detect(shifted).detected
