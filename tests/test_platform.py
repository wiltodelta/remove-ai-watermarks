"""Tests for cross-platform and cross-device compatibility.

Verifies that device detection, MPS fallback, and platform-specific
code paths work correctly on CPU, MPS (macOS), and CUDA (Linux/Windows).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from remove_ai_watermarks.noai.progress import is_mps_error
from remove_ai_watermarks.noai.utils import get_image_format, is_supported_format
from remove_ai_watermarks.noai.watermark_profiles import (
    detect_model_profile,
    get_model_id_for_profile,
    get_recommended_strength,
)
from remove_ai_watermarks.noai.watermark_remover import get_device, is_watermark_removal_available

# ── Device detection ────────────────────────────────────────────────


class TestDeviceDetection:
    """Tests for get_device() across platforms."""

    def test_returns_valid_device(self):
        device = get_device()
        assert device in ("cpu", "mps", "cuda")

    def test_cpu_fallback_when_no_gpu(self):
        """On CI / machines without GPU, should fall back to cpu or mps."""
        device = get_device()
        # Just verify it doesn't crash and returns a valid string
        assert isinstance(device, str)

    @patch("remove_ai_watermarks.noai.watermark_remover._HAS_TORCH", False)
    def test_no_torch_returns_cpu(self):
        assert get_device() == "cpu"


class TestMpsErrorDetection:
    """Tests for MPS error detection helper."""

    def test_detects_mps_error(self):
        err = RuntimeError("MPS backend out of memory")
        assert is_mps_error(err) is True

    def test_non_mps_error(self):
        err = RuntimeError("CUDA out of memory")
        assert is_mps_error(err) is False

    def test_generic_error(self):
        err = RuntimeError("something went wrong")
        assert is_mps_error(err) is False


# ── Model profiles ──────────────────────────────────────────────────


class TestModelProfiles:
    """Tests for watermark_profiles.py."""

    def test_default_profile(self):
        assert get_model_id_for_profile("default") == "Lykon/dreamshaper-8"

    def test_ctrlregen_profile(self):
        assert get_model_id_for_profile("ctrlregen") == "yepengliu/ctrlregen"

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError, match="Unknown model profile"):
            get_model_id_for_profile("nonexistent")

    def test_detect_default(self):
        assert detect_model_profile("Lykon/dreamshaper-8") == "default"

    def test_detect_ctrlregen(self):
        assert detect_model_profile("yepengliu/ctrlregen") == "ctrlregen"

    def test_recommended_strength_high(self):
        assert get_recommended_strength("treering") == 0.7

    def test_recommended_strength_low(self):
        assert get_recommended_strength("stablesignature") == 0.04

    def test_recommended_strength_medium(self):
        assert get_recommended_strength("unknown_type") == 0.35


# ── Format utilities ────────────────────────────────────────────────


class TestFormatUtils:
    """Tests for utils.py format helpers."""

    def test_supported_png(self, tmp_path):
        assert is_supported_format(tmp_path / "test.png")

    def test_supported_jpg(self, tmp_path):
        assert is_supported_format(tmp_path / "test.jpg")

    def test_supported_jpeg(self, tmp_path):
        assert is_supported_format(tmp_path / "test.jpeg")

    def test_supported_webp(self, tmp_path):
        assert is_supported_format(tmp_path / "test.webp")

    def test_unsupported_bmp(self, tmp_path):
        assert not is_supported_format(tmp_path / "test.bmp")

    def test_unsupported_gif(self, tmp_path):
        assert not is_supported_format(tmp_path / "test.gif")

    def test_get_format_png(self, tmp_path):
        assert get_image_format(tmp_path / "x.png") == "PNG"

    def test_get_format_jpg(self, tmp_path):
        assert get_image_format(tmp_path / "x.jpg") == "JPEG"

    def test_get_format_jpeg(self, tmp_path):
        assert get_image_format(tmp_path / "x.jpeg") == "JPEG"

    def test_get_format_webp_defaults_png(self, tmp_path):
        # .webp falls through to PNG in current implementation
        assert get_image_format(tmp_path / "x.webp") == "PNG"


# ── Availability checks ────────────────────────────────────────────


class TestAvailability:
    """Tests for dependency availability checks."""

    def test_watermark_removal_available(self):
        # In dev env with torch+diffusers installed
        assert is_watermark_removal_available() is True

    def test_invisible_is_available(self):
        from remove_ai_watermarks.invisible_engine import is_available

        assert is_available() is True


# ── Platform-specific path handling ─────────────────────────────────


class TestPlatformPaths:
    """Verify path handling works on current platform."""

    def test_pathlib_works_for_assets(self):
        from pathlib import Path

        asset_dir = Path(__file__).parent.parent / "src" / "remove_ai_watermarks" / "assets"
        assert (asset_dir / "gemini_bg_48.png").exists()
        assert (asset_dir / "gemini_bg_96.png").exists()

    def test_asset_loading_works(self):
        """Verify embedded assets load correctly (critical for packaging)."""
        from remove_ai_watermarks.gemini_engine import GeminiEngine

        engine = GeminiEngine()
        # If we get here without error, asset loading works
        assert engine._alpha_small.shape == (48, 48)
        assert engine._alpha_large.shape == (96, 96)
