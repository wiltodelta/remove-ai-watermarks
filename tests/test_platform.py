"""Tests for cross-platform and cross-device compatibility.

Verifies that device detection, MPS fallback, and platform-specific
code paths work correctly on CPU, MPS (macOS), and CUDA (Linux/Windows).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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
        assert device in ("cpu", "mps", "cuda", "xpu")

    def test_cpu_fallback_when_no_gpu(self):
        """On CI / machines without GPU, should fall back to cpu or mps."""
        device = get_device()
        # Just verify it doesn't crash and returns a valid string
        assert isinstance(device, str)

    @patch("remove_ai_watermarks.noai.watermark_remover._HAS_TORCH", False)
    def test_no_torch_returns_cpu(self):
        assert get_device() == "cpu"

    def test_xpu_selected_when_available(self):
        """An XPU-enabled torch (no CUDA) routes to the Intel GPU backend.

        The whole torch module is mocked so the smoke-test ops succeed without
        any real device; cuda must read False so the cuda branch is skipped.
        """
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        fake_torch.xpu.is_available.return_value = True
        with patch("remove_ai_watermarks.noai.watermark_remover.torch", fake_torch):
            assert get_device() == "xpu"
        fake_torch.tensor.assert_called_with([1.0], device="xpu")

    def test_init_accepts_xpu_and_selects_fp16(self):
        """WatermarkRemover accepts device='xpu' and picks fp16 (not fp32)."""
        if not is_watermark_removal_available():
            pytest.skip("torch/diffusers not installed")
        import torch

        from remove_ai_watermarks.noai.watermark_remover import WatermarkRemover

        remover = WatermarkRemover(device="xpu")
        assert remover.device == "xpu"
        assert remover.torch_dtype == torch.float16

    def test_seed_generator_falls_back_to_cpu_when_device_rng_unsupported(self):
        """A device with no RNG backend (e.g. some torch-xpu builds) falls back
        to a CPU generator instead of raising when --seed is used."""
        from remove_ai_watermarks.noai import watermark_remover as wr

        def fake_generator(device="cpu"):
            if device == "xpu":
                raise RuntimeError("Device type xpu is not supported for torch.Generator()")
            gen = MagicMock()
            gen.manual_seed.return_value = f"gen:{device}"
            return gen

        fake_torch = MagicMock()
        fake_torch.Generator.side_effect = fake_generator
        with patch.object(wr, "torch", fake_torch):
            assert wr._make_seed_generator("xpu", 123) == "gen:cpu"
            assert wr._make_seed_generator("cuda", 123) == "gen:cuda"


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
        assert get_model_id_for_profile("default") == "stabilityai/stable-diffusion-xl-base-1.0"

    def test_ctrlregen_profile(self):
        assert get_model_id_for_profile("ctrlregen") == "yepengliu/ctrlregen"

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError, match="Unknown model profile"):
            get_model_id_for_profile("nonexistent")

    def test_detect_default(self):
        assert detect_model_profile("stabilityai/stable-diffusion-xl-base-1.0") == "default"

    def test_detect_ctrlregen(self):
        assert detect_model_profile("yepengliu/ctrlregen") == "ctrlregen"

    def test_recommended_strength_high(self):
        assert get_recommended_strength("treering") == 0.7

    def test_recommended_strength_low(self):
        assert get_recommended_strength("stablesignature") == 0.04

    def test_recommended_strength_medium(self):
        assert get_recommended_strength("unknown_type") == 0.35

    @pytest.mark.parametrize("wm_type", ["stegastamp", "stegasamp", "treering", "ringid"])
    def test_high_perturbation_watermark_types(self, wm_type):
        """Robust spatial watermarks need aggressive (0.7) regeneration."""
        assert get_recommended_strength(wm_type) == 0.7

    @pytest.mark.parametrize("wm_type", ["stablesignature", "dwtectsvd", "rivagan", "ssl", "hidden"])
    def test_low_perturbation_watermark_types(self, wm_type):
        """Fragile frequency/latent watermarks break at low (0.04) strength."""
        assert get_recommended_strength(wm_type) == 0.04

    def test_strength_match_is_case_insensitive(self):
        assert get_recommended_strength("TreeRing") == 0.7
        assert get_recommended_strength("StableSignature") == 0.04

    def test_strength_matches_substring_in_descriptive_name(self):
        # e.g. a CLI passing "treering_v2" or "synthid-stegastamp" still maps.
        assert get_recommended_strength("treering_v2") == 0.7


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
        # Reflects the actual environment: True iff torch + diffusers (the gpu
        # extra) are importable. The core+dev CI env has no diffusers, so this
        # must not assume the full stack is present.
        import importlib.util

        expected = all(importlib.util.find_spec(m) is not None for m in ("torch", "diffusers"))
        assert is_watermark_removal_available() is expected

    def test_invisible_is_available(self):
        import importlib.util

        from remove_ai_watermarks.invisible_engine import is_available

        expected = all(importlib.util.find_spec(m) is not None for m in ("torch", "diffusers"))
        assert is_available() is expected


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
