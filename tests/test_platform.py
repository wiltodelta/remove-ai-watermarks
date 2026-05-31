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
    CTRLREGEN_DEFAULT_STRENGTH,
    DEFAULT_STRENGTH,
    detect_model_profile,
    get_model_id_for_profile,
    resolve_strength,
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


class TestResolveStrength:
    """resolve_strength applies the profile default only when strength is unset."""

    def test_none_default_profile_uses_sdxl_default(self):
        assert resolve_strength(None, "default") == DEFAULT_STRENGTH

    def test_none_ctrlregen_uses_clean_noise_default(self):
        # ctrlregen must NOT inherit the SDXL DEFAULT_STRENGTH (that makes it a no-op);
        # clean-noise regeneration is the lever against robust marks.
        assert resolve_strength(None, "ctrlregen") == CTRLREGEN_DEFAULT_STRENGTH
        assert CTRLREGEN_DEFAULT_STRENGTH > DEFAULT_STRENGTH

    def test_explicit_value_overrides_both_profiles(self):
        assert resolve_strength(0.3, "default") == 0.3
        assert resolve_strength(0.3, "ctrlregen") == 0.3

    def test_explicit_zero_is_respected_not_treated_as_unset(self):
        # 0.0 is falsy but explicit -- must not fall through to the profile default
        # (the old `strength or DEFAULT` bug would have). Range validation lives in
        # remove_watermark, not here.
        assert resolve_strength(0.0, "ctrlregen") == 0.0


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


class TestFp16VaeFix:
    """The plain SDXL img2img pipeline must swap in the fp16-fixed VAE on fp16
    GPUs to avoid the NaN/all-black decode (issue #29). Pure decision logic, no
    torch or model download needed."""

    DEFAULT = "stabilityai/stable-diffusion-xl-base-1.0"

    def test_default_sdxl_on_fp16_needs_fix(self):
        from remove_ai_watermarks.noai.watermark_remover import _needs_fp16_vae_fix

        assert _needs_fp16_vae_fix(self.DEFAULT, self.DEFAULT, is_fp16=True) is True

    def test_fp32_does_not_need_fix(self):
        """cpu/mps run fp32, where the stock SDXL VAE is fine."""
        from remove_ai_watermarks.noai.watermark_remover import _needs_fp16_vae_fix

        assert _needs_fp16_vae_fix(self.DEFAULT, self.DEFAULT, is_fp16=False) is False

    def test_non_default_model_keeps_own_vae(self):
        """A custom (non-SDXL) checkpoint must not get the SDXL-specific VAE."""
        from remove_ai_watermarks.noai.watermark_remover import _needs_fp16_vae_fix

        assert _needs_fp16_vae_fix("runwayml/stable-diffusion-v1-5", self.DEFAULT, is_fp16=True) is False
