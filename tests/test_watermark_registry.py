"""Tests for the known-visible-watermark registry (reverse-alpha only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from remove_ai_watermarks import watermark_registry as reg

DOUBAO_SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "doubao-1.png"


class TestCatalog:
    def test_keys(self):
        assert reg.mark_keys() == ["gemini", "doubao"]

    def test_all_in_auto(self):
        assert all(m.in_auto for m in reg.known_marks())

    def test_recovery_is_reverse_alpha(self):
        # Every catalogued mark is removed by exact reverse-alpha (no inpaint).
        assert all(m.recovery == "reverse-alpha" for m in reg.known_marks())

    def test_locations(self):
        by_key = {m.key: m for m in reg.known_marks()}
        assert by_key["gemini"].location == "bottom-right"
        assert by_key["doubao"].location == "bottom-right"

    def test_get_mark_unknown_raises(self):
        with pytest.raises(KeyError):
            reg.get_mark("nope")


class TestScan:
    def test_detect_marks_scans_all(self):
        img = np.zeros((256, 256, 3), np.uint8)
        keys = {d.key for d in reg.detect_marks(img)}
        assert keys == {"gemini", "doubao"}

    def test_blank_image_no_auto_mark(self):
        assert reg.best_auto_mark(np.zeros((256, 256, 3), np.uint8)) is None


@pytest.mark.skipif(not DOUBAO_SAMPLE.exists(), reason="doubao sample not present")
class TestRealSample:
    def test_doubao_sample_wins_auto(self):
        from remove_ai_watermarks.image_io import imread

        best = reg.best_auto_mark(imread(DOUBAO_SAMPLE))
        assert best is not None
        assert best.key == "doubao"

    def test_doubao_remove_returns_region(self):
        from remove_ai_watermarks.image_io import imread

        img = imread(DOUBAO_SAMPLE)  # 2048 wide -> reverse-alpha applies
        result, region = reg.get_mark("doubao").remove(img)
        assert region is not None
        assert result.shape == img.shape


class TestReverseAlphaOnly:
    def test_doubao_off_resolution_is_skipped(self):
        # No alpha capture for this width -> no inpaint fallback, image untouched.
        img = np.zeros((512, 512, 3), np.uint8)
        result, region = reg.get_mark("doubao").remove(img)
        assert region is None
        assert np.array_equal(result, img)
