"""Tests for the known-visible-watermark registry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from remove_ai_watermarks import watermark_registry as reg

DOUBAO_SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "doubao-1.png"


class TestCatalog:
    def test_keys(self):
        assert reg.mark_keys() == ["gemini", "doubao", "samsung"]

    def test_in_auto_flags(self):
        by_key = {m.key: m for m in reg.known_marks()}
        assert by_key["gemini"].in_auto is True
        assert by_key["doubao"].in_auto is True
        assert by_key["samsung"].in_auto is False  # explicit-only

    def test_recovery_strategy(self):
        by_key = {m.key: m for m in reg.known_marks()}
        assert by_key["gemini"].recovery == "reverse-alpha"  # exact background recovery
        assert by_key["doubao"].recovery == "inpaint"
        assert by_key["samsung"].recovery == "inpaint"

    def test_locations(self):
        by_key = {m.key: m for m in reg.known_marks()}
        assert by_key["gemini"].location == "bottom-right"
        assert by_key["doubao"].location == "bottom-right"
        assert by_key["samsung"].location == "bottom-left"

    def test_get_mark_unknown_raises(self):
        with pytest.raises(KeyError):
            reg.get_mark("nope")


class TestScan:
    def test_detect_marks_scans_all(self):
        img = np.zeros((256, 256, 3), np.uint8)
        keys = {d.key for d in reg.detect_marks(img, include_explicit=True)}
        assert keys == {"gemini", "doubao", "samsung"}

    def test_auto_scan_excludes_explicit_only(self):
        img = np.zeros((256, 256, 3), np.uint8)
        keys = {d.key for d in reg.detect_marks(img, include_explicit=False)}
        assert keys == {"gemini", "doubao"}  # samsung is explicit-only

    def test_blank_image_no_auto_mark(self):
        assert reg.best_auto_mark(np.zeros((256, 256, 3), np.uint8)) is None


@pytest.mark.skipif(not DOUBAO_SAMPLE.exists(), reason="doubao sample not present")
class TestRealSample:
    def test_doubao_sample_wins_auto(self):
        from remove_ai_watermarks.image_io import imread

        img = imread(DOUBAO_SAMPLE)
        best = reg.best_auto_mark(img)
        assert best is not None
        assert best.key == "doubao"

    def test_doubao_remove_returns_region(self):
        from remove_ai_watermarks.image_io import imread

        img = imread(DOUBAO_SAMPLE)
        result, region = reg.get_mark("doubao").remove(img)
        assert region is not None
        assert result.shape == img.shape

    def test_doubao_remove_lama_backend(self):
        from remove_ai_watermarks.image_io import imread
        from remove_ai_watermarks.region_eraser import lama_available

        if not lama_available():
            pytest.skip("onnxruntime (LaMa) not installed")
        img = imread(DOUBAO_SAMPLE)
        result, region = reg.get_mark("doubao").remove(img, backend="lama")
        assert region is not None
        assert result.shape == img.shape


class TestBackend:
    def test_default_backend_is_known(self):
        assert reg.default_backend() in ("cv2", "lama")

    def test_remove_accepts_backend_param(self):
        # cv2 backend works without any optional dep, on any image.
        result, region = reg.get_mark("doubao").remove(np.zeros((256, 256, 3), np.uint8), backend="cv2")
        assert result.shape == (256, 256, 3)
        assert region is None  # nothing detected on a blank image
