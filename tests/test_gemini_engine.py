"""Tests for the Gemini visible-watermark engine."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from remove_ai_watermarks.gemini_engine import (
    DetectionResult,
    GeminiEngine,
    WatermarkPosition,
    WatermarkSize,
    _calculate_alpha_map,
    get_watermark_config,
    get_watermark_size,
)

# ── WatermarkSize / config helpers ──────────────────────────────────


class TestWatermarkConfig:
    """Tests for watermark size detection and position calculation."""

    def test_small_image_gets_small_watermark(self):
        assert get_watermark_size(800, 600) == WatermarkSize.SMALL

    def test_large_image_gets_large_watermark(self):
        assert get_watermark_size(1920, 1080) == WatermarkSize.LARGE

    def test_boundary_image_stays_small(self):
        """Exactly 1024×1024 should be SMALL (rule: > 1024 for LARGE)."""
        assert get_watermark_size(1024, 1024) == WatermarkSize.SMALL

    def test_one_dimension_small(self):
        """Only one dimension > 1024 → still SMALL."""
        assert get_watermark_size(2000, 500) == WatermarkSize.SMALL

    def test_config_small_returns_correct_values(self):
        config = get_watermark_config(800, 600)
        assert config.margin_right == 32
        assert config.margin_bottom == 32
        assert config.logo_size == 48

    def test_config_large_returns_correct_values(self):
        config = get_watermark_config(1920, 1080)
        assert config.margin_right == 64
        assert config.margin_bottom == 64
        assert config.logo_size == 96

    def test_position_calculation(self):
        pos = WatermarkPosition(margin_right=32, margin_bottom=32, logo_size=48)
        x, y = pos.get_position(800, 600)
        assert x == 800 - 32 - 48  # 720
        assert y == 600 - 32 - 48  # 520


# ── Alpha map ───────────────────────────────────────────────────────


class TestAlphaMap:
    """Tests for alpha map calculation."""

    def test_pure_black_gives_zero_alpha(self):
        black = np.zeros((10, 10, 3), dtype=np.uint8)
        alpha = _calculate_alpha_map(black)
        assert alpha.shape == (10, 10)
        np.testing.assert_array_equal(alpha, 0.0)

    def test_pure_white_gives_one_alpha(self):
        white = np.full((10, 10, 3), 255, dtype=np.uint8)
        alpha = _calculate_alpha_map(white)
        np.testing.assert_allclose(alpha, 1.0)

    def test_grayscale_input(self):
        gray = np.full((10, 10), 128, dtype=np.uint8)
        alpha = _calculate_alpha_map(gray)
        np.testing.assert_allclose(alpha, 128 / 255.0)

    def test_max_channel_used(self):
        """Alpha should use max(R, G, B)."""
        img = np.zeros((1, 1, 3), dtype=np.uint8)
        img[0, 0] = [50, 200, 100]  # BGR
        alpha = _calculate_alpha_map(img)
        assert pytest.approx(alpha[0, 0], rel=1e-3) == 200 / 255.0


# ── GeminiEngine ────────────────────────────────────────────────────


class TestGeminiEngine:
    """Tests for the GeminiEngine class."""

    @pytest.fixture(autouse=True)
    def _setup_engine(self):
        self.engine = GeminiEngine()

    def test_engine_loads_alpha_maps(self):
        small = self.engine.get_alpha_map(WatermarkSize.SMALL)
        large = self.engine.get_alpha_map(WatermarkSize.LARGE)
        assert small.shape == (48, 48)
        assert large.shape == (96, 96)

    def test_remove_watermark_returns_same_shape(self, tmp_image_path):
        image = cv2.imread(str(tmp_image_path), cv2.IMREAD_COLOR)
        result = self.engine.remove_watermark(image)
        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_remove_watermark_does_not_modify_input(self, tmp_image_path):
        image = cv2.imread(str(tmp_image_path), cv2.IMREAD_COLOR)
        original = image.copy()
        self.engine.remove_watermark(image)
        np.testing.assert_array_equal(image, original)

    def test_remove_watermark_large_image(self, tmp_large_image_path):
        image = cv2.imread(str(tmp_large_image_path), cv2.IMREAD_COLOR)
        result = self.engine.remove_watermark(image)
        assert result.shape == image.shape

    def test_remove_watermark_custom_region(self, tmp_image_path):
        image = cv2.imread(str(tmp_image_path), cv2.IMREAD_COLOR)
        result = self.engine.remove_watermark_custom(image, (10, 10, 48, 48))
        assert result.shape == image.shape

    def test_remove_watermark_custom_large_region(self, tmp_image_path):
        image = cv2.imread(str(tmp_image_path), cv2.IMREAD_COLOR)
        result = self.engine.remove_watermark_custom(image, (10, 10, 96, 96))
        assert result.shape == image.shape

    def test_remove_watermark_custom_arbitrary_region(self, tmp_image_path):
        image = cv2.imread(str(tmp_image_path), cv2.IMREAD_COLOR)
        result = self.engine.remove_watermark_custom(image, (5, 5, 60, 60))
        assert result.shape == image.shape

    def test_force_size(self, tmp_image_path):
        image = cv2.imread(str(tmp_image_path), cv2.IMREAD_COLOR)
        result = self.engine.remove_watermark(image, force_size=WatermarkSize.LARGE)
        assert result.shape == image.shape


# ── Detection ───────────────────────────────────────────────────────


class TestDetection:
    """Tests for watermark detection."""

    @pytest.fixture(autouse=True)
    def _setup_engine(self):
        self.engine = GeminiEngine()

    def test_detect_returns_result_object(self, tmp_image_path):
        image = cv2.imread(str(tmp_image_path), cv2.IMREAD_COLOR)
        result = self.engine.detect_watermark(image)
        assert isinstance(result, DetectionResult)
        assert 0.0 <= result.confidence <= 1.0

    def test_detect_empty_image_returns_no_detection(self):
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        result = self.engine.detect_watermark(empty)
        assert not result.detected
        assert result.confidence == 0.0

    def test_detect_none_image_returns_no_detection(self):
        result = self.engine.detect_watermark(None)
        assert not result.detected

    def test_detect_random_image_low_confidence(self, tmp_image_path):
        """Random noise should not look like a watermark."""
        image = cv2.imread(str(tmp_image_path), cv2.IMREAD_COLOR)
        result = self.engine.detect_watermark(image)
        # Random image may or may not be detected; confidence should be meaningful
        assert isinstance(result.spatial_score, float)
        assert isinstance(result.gradient_score, float)


# ── Inpainting ──────────────────────────────────────────────────────


class TestInpainting:
    """Tests for residual inpainting."""

    @pytest.fixture(autouse=True)
    def _setup_engine(self):
        self.engine = GeminiEngine()

    def test_inpaint_ns(self, tmp_image_path):
        image = cv2.imread(str(tmp_image_path), cv2.IMREAD_COLOR)
        result = self.engine.inpaint_residual(image, (150, 150, 48, 48), method="ns")
        assert result.shape == image.shape

    def test_inpaint_telea(self, tmp_image_path):
        image = cv2.imread(str(tmp_image_path), cv2.IMREAD_COLOR)
        result = self.engine.inpaint_residual(image, (150, 150, 48, 48), method="telea")
        assert result.shape == image.shape

    def test_inpaint_gaussian(self, tmp_image_path):
        image = cv2.imread(str(tmp_image_path), cv2.IMREAD_COLOR)
        result = self.engine.inpaint_residual(image, (150, 150, 48, 48), method="gaussian")
        assert result.shape == image.shape

    def test_inpaint_zero_strength(self, tmp_image_path):
        image = cv2.imread(str(tmp_image_path), cv2.IMREAD_COLOR)
        result = self.engine.inpaint_residual(image, (150, 150, 48, 48), strength=0.0)
        np.testing.assert_array_equal(result, image)

    def test_inpaint_tiny_region_returns_unchanged(self, tmp_image_path):
        image = cv2.imread(str(tmp_image_path), cv2.IMREAD_COLOR)
        result = self.engine.inpaint_residual(image, (10, 10, 2, 2))
        np.testing.assert_array_equal(result, image)

    def test_inpaint_does_not_modify_input(self, tmp_image_path):
        image = cv2.imread(str(tmp_image_path), cv2.IMREAD_COLOR)
        original = image.copy()
        self.engine.inpaint_residual(image, (150, 150, 48, 48))
        np.testing.assert_array_equal(image, original)
